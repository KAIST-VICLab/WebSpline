import numpy as np
import random
import os, sys
import cv2

import torch

from tqdm import tqdm

import torch.nn as nn

from pytorch3d.transforms import axis_angle_to_matrix
import torch.nn.functional as F


def rescale_camera_pose(
    opt, scene, robust_alignment_jitter_th_ratio=2.0
):
    train_cams = scene.getTrainCameras()
    viewpoint_stack = [i for i in train_cams]
    device = torch.device("cuda")
    T = len(viewpoint_stack)
    H, W = viewpoint_stack[0].image_height, viewpoint_stack[0].image_width
    s_track = scene.static_track[..., :2].clone().to(device)  # [T, N, 2]
    s_track_mask = scene.static_track_mask.clone().to(device)  # [T, N]
    _, N, _ = s_track.shape
    all_deps = torch.stack([cam.depth for cam in viewpoint_stack]).to(device)
    all_rgbs = torch.stack([cam.original_image for cam in viewpoint_stack]).to(device)
    L = float(min(H, W))
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    u_homo = 2.0 * x / L - W / L
    v_homo = 2.0 * y / L - H / L
    all_homos = torch.stack([u_homo, v_homo], dim=0).unsqueeze(0).expand(T, -1, -1, -1)  # [T, 2, H, W]
    u_int = torch.clamp(s_track[..., 0].round().long(), 0, W - 1)  # [T, N]
    v_int = torch.clamp(s_track[..., 1].round().long(), 0, H - 1)  # [T, N]
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(T, N)  # [T, N]
    if all_deps.ndim == 4:
        all_deps = all_deps.squeeze(1)  # [T, 1, H, W] -> [T, H, W]
    dep_list = all_deps[t_idx, v_int, u_int]  # [T, N]
    if all_rgbs.ndim == 4 and all_rgbs.shape[1] in [3, 4]:
        all_rgbs = all_rgbs.permute(0, 2, 3, 1)
    rgb_list = all_rgbs[t_idx, v_int, u_int]  # [T, N, 3]
    all_homos_permuted = all_homos.permute(0, 2, 3, 1)
    homo_list = all_homos_permuted[t_idx, v_int, u_int]  # [T, N, 2]
    dep_list[~s_track_mask] = -1.0
    homo_list[~s_track_mask] = 0.0
    focals = torch.tensor([cam.focal for cam in viewpoint_stack], device=device).float()
    neighbor_frame_mask = s_track_mask[1:] * s_track_mask[:-1]
    depth_diff = abs(dep_list[1:] - dep_list[:-1])
    large_depth_jitter_th = depth_diff[neighbor_frame_mask].median()
    jitter_mask = (
        depth_diff > large_depth_jitter_th * robust_alignment_jitter_th_ratio
    ) * neighbor_frame_mask
    neighbor_frame_mask = neighbor_frame_mask * (~jitter_mask)
    T, M = dep_list.shape
    d = dep_list.reshape(-1)[..., None]
    uv = homo_list.reshape(-1, 2)
    rel_focal_val = 2 * focals[0] / min(H, W)
    rel_focal = torch.tensor([rel_focal_val, rel_focal_val], device=device)
    cxcy = torch.zeros(2, device=device)
    xy = (uv - cxcy[None, :]) * d / rel_focal[None]
    z = d
    xyz = torch.cat([xy, z], dim=-1)
    point_cam = xyz.reshape(T, M, 3)
    point_cam = point_cam.reshape(T, M, 3)
    R_orig = torch.stack([torch.from_numpy(cam.R.T).float() for cam in viewpoint_stack]).to(device)
    T_orig = torch.stack([torch.from_numpy(cam.T).float() for cam in viewpoint_stack]).to(device)
    R_wc = R_orig.transpose(1, 2)
    t_wc = -torch.bmm(R_wc, T_orig.unsqueeze(2)).squeeze(2)
    point_ref_rot = torch.einsum("tij,tmj->tmi", R_wc, point_cam)
    #point_ref = point_ref_rot + t_wc[:, None]
    a = point_ref_rot[1:] - point_ref_rot[:-1]
    b = (t_wc[1:] - t_wc[:-1])[:, None].expand(-1, M, -1)
    a, b = a[neighbor_frame_mask], b[neighbor_frame_mask]
    s_optimal = float(-(a * b).sum() / (b * b).sum())
    if s_optimal < 0.00001:
        s_optimal = 1.0
    for t in range(T):
        old_info = scene.train_camera.dataset[t]
        new_T = old_info.T * s_optimal
        scene.train_camera.dataset[t] = old_info._replace(T=new_T)


def positive_th_gaussian_decay(x, th, sigma):
    w = torch.ones_like(x)
    mask = x > th
    w[mask] = torch.exp(-((x[mask] - th) ** 2) / (2 * sigma ** 2))
    return w

def bundle_adjustment(opt, scene, source_path):
    # Only datasets with a static track can be bundle-adjusted; DAVIS loads no
    # static track because its poses come from MegaSaM and are used as given.
    if not hasattr(scene, "static_track"):
        raise RuntimeError(
            f"bundle adjustment needs a static track, which the {scene.dataset_type} "
            f"loader does not produce -- this dataset's poses are used as given"
        )
    if scene.dataset_type == 'nvidia':
        rescale_camera_pose(opt,scene)
    train_cams = scene.getTrainCameras()
    viewpoint_stack = [i for i in train_cams]  # Large CPU usage
    max_t_per_step = 32
    total_steps = 2000
    switch_to_ind_step = 500
    depth_correction_after_step = 500

    lr_cam_q = 0.0001
    lr_cam_t = 0.0001
    # iPhone carries sensor depth, which is already metric -- refining a
    # per-frame scale/shift on top of it only lets BA drift.  The other
    # datasets have monocular depth with an arbitrary scale, so there the
    # scale and shift do need to be learned.  0 = excluded from training.
    lr_dep_s = 0.0 if scene.dataset_type == "iphone" else 0.001
    lr_dep_c = 0.0 if scene.dataset_type == "iphone" else 0.001
    lambda_flow = 1.0
    lambda_photo = 1.0  # replaces the original lambda_flow
    lambda_depth = 0.1
    lambda_small_correction = 0.03
    lambda_cam_smooth_trans = 0.0
    lambda_cam_smooth_rot = 0.0
    huber_delta = -1
    depth_decay_th = 2.0
    depth_decay_sigma = 1.0
    std_decay_th = 0.2
    std_decay_sigma = 0.2
    optimizer_class = torch.optim.Adam
    T = len(viewpoint_stack)
    device = torch.device("cuda")
    H, W = viewpoint_stack[0].image_height, viewpoint_stack[0].image_width

    # ------------------------------------------------------------------
    # Sample static points: pixels where the dynamic mask is 0, in place of tracks
    # ------------------------------------------------------------------
    # Stands in for the original s_track: static pixels sampled from frame 0
    s_track = scene.static_track[..., :2].clone().to(device)  # [T, N, 2]
    s_track_valid_mask = scene.static_track_mask.clone().to(device)  # [T, N]

    # Subsample at random when there are too many tracks
    max_num_of_tracks = 10000
    if s_track.shape[1] > max_num_of_tracks:
        choice = torch.randperm(s_track.shape[1])[:max_num_of_tracks]
        s_track = s_track[:, choice]
        s_track_valid_mask = s_track_valid_mask[:, choice]

    _, N, _ = s_track.shape
    uv_norm = s_track.clone()
    uv_norm[..., 0] = (uv_norm[..., 0] / (W - 1)) * 2 - 1
    uv_norm[..., 1] = (uv_norm[..., 1] / (H - 1)) * 2 - 1
    uv_norm = uv_norm.unsqueeze(2)  # [T, N, 1, 2]
    # Stack every depth map and image into [T, C, H, W]
    all_deps = torch.stack([cam.depth for cam in viewpoint_stack]).to(device)
    all_rgbs = torch.stack([cam.original_image for cam in viewpoint_stack]).to(device)
    L = float(min(H, W))
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    u_homo = 2.0 * x / L - W / L
    v_homo = 2.0 * y / L - H / L
    all_homos = torch.stack([u_homo, v_homo], dim=0).unsqueeze(0).expand(T, -1, -1, -1)  # [T, 2, H, W]
    dep_list = F.grid_sample(all_deps, uv_norm, align_corners=True, padding_mode="zeros").squeeze(3).squeeze(
        1)  # [T, N]
    rgb_list = F.grid_sample(all_rgbs, uv_norm, align_corners=True, padding_mode="zeros").squeeze(3).permute(0, 2,
                                                                                                            1)  # [T, N, 3]
    homo_list = F.grid_sample(all_homos, uv_norm, align_corners=True, padding_mode="zeros").squeeze(3).permute(0, 2,
                                                                                                            1)  # [T, N, 2
    # Mask out the invalid tracks
    dep_list[~s_track_valid_mask] = -1.0
    homo_list[~s_track_valid_mask] = 0.0
    param_scale = nn.Parameter(torch.ones(T, device=device))
    param_dep_corr = nn.Parameter(torch.zeros_like(dep_list, device=device))  # shape [T, N]
    delta_q = nn.Parameter(torch.zeros((T, 3), device=device))
    delta_t = nn.Parameter(torch.zeros((T, 3), device=device))

    def get_optim_list():
        lst = [
            {"params": [delta_q], "lr": lr_cam_q, "name": "cam_q"},
            {"params": [delta_t], "lr": lr_cam_t, "name": "cam_t"},
        ]
        if lr_dep_s > 0: lst.append({"params": [param_scale], "lr": lr_dep_s, "name": "cam_scale"})
        if lr_dep_c > 0: lst.append({"params": [param_dep_corr], "lr": lr_dep_c, "name": "dep_correction"})
        return lst

    optimizer = optimizer_class(get_optim_list())
    scheduler = None
    focals = torch.tensor([cam.focal for cam in viewpoint_stack], device=device).float()
    cxs = torch.tensor([cam.metadata.principal_point_x for cam in viewpoint_stack], device=device).float()
    cys = torch.tensor([cam.metadata.principal_point_y for cam in viewpoint_stack], device=device).float()

        # Normalised validity mask, used for the std below
    s_track_valid_mask_w = s_track_valid_mask.float()
    s_track_valid_mask_w = s_track_valid_mask_w / torch.clamp(s_track_valid_mask_w.sum(0), min=1e-6)
    R_orig = torch.stack([torch.from_numpy(cam.R.T).float() for cam in viewpoint_stack]).to(device)
    T_orig = torch.stack([torch.from_numpy(cam.T).float() for cam in viewpoint_stack]).to(device)
    for step in tqdm(range(total_steps)):
        if step == switch_to_ind_step:
            print("Switch to Independent Camera Optimization and Start Scheduling")
            optimizer = torch.optim.Adam(get_optim_list())
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, total_steps - switch_to_ind_step, eta_min=min(lr_cam_q, lr_cam_t) / 10.0
            )

        optimizer.zero_grad()
        dep_scale = param_scale.abs()
        dep_scale = dep_scale / torch.clamp(dep_scale.mean(), min=1e-6)

        scaled_depth_list = dep_list * dep_scale[:, None]
        if step > depth_correction_after_step:
            scaled_depth_list = scaled_depth_list + param_dep_corr
            dep_corr_loss = abs(param_dep_corr).mean()
        else:
            dep_corr_loss = torch.zeros_like(dep_scale[0])

        R_cw = torch.matmul(axis_angle_to_matrix(delta_q), R_orig)  # [T, 3, 3]
        t_cw = T_orig + delta_t  # [T, 3]

        # Invert the extrinsics to get camera-to-world
        R_wc = R_cw.transpose(1, 2)
        t_wc = -torch.bmm(R_wc, t_cw.unsqueeze(2)).squeeze(2)

        # 2D track pixels -> 3D points in camera space
        x_cam = (s_track[..., 0] - cxs[:, None]) * scaled_depth_list / focals[:, None]
        y_cam = (s_track[..., 1] - cys[:, None]) * scaled_depth_list / focals[:, None]
        point_cam = torch.stack([x_cam, y_cam, scaled_depth_list], dim=-1)  # [T, N, 3]

        # Camera space -> world space
        point_ref = torch.einsum("tij,tnj->tni", R_wc, point_cam) + t_wc[:, None, :]  # [Src, N, 3]

        # Build the target-frame minibatch
        if T > max_t_per_step:
            tgt_inds = torch.randperm(T)[:max_t_per_step].to(device)
        else:
            tgt_inds = torch.arange(T).to(device)

        R_cw_tgt = R_cw[tgt_inds]
        t_cw_tgt = t_cw[tgt_inds]
        focals_tgt = focals[tgt_inds]
        cxs_tgt = cxs[tgt_inds]
        cys_tgt = cys[tgt_inds]

        # Project every source frame's world points into every target frame: [Src, Tgt, N, 3]
        point_ref_to_every_frame = torch.einsum("vji,sni->svnj", R_cw_tgt, point_ref) + t_cw_tgt[None, :, None, :]
        warped_depth = point_ref_to_every_frame[..., 2]  # [Src, Tgt, N]
        safe_warped_depth = torch.clamp(warped_depth, min=1e-6)
        # Project onto the target camera's pixel (UV) coordinates
        u_tgt = (point_ref_to_every_frame[..., 0] * focals_tgt[None, :, None] / safe_warped_depth) + cxs_tgt[
            None, :, None]
        v_tgt = (point_ref_to_every_frame[..., 1] * focals_tgt[None, :, None] / safe_warped_depth) + cys_tgt[
            None, :, None]
        u_tgt_homo = 2.0 * u_tgt / L - W / L
        v_tgt_homo = 2.0 * v_tgt / L - H / L
        uv_src_to_every_frame = torch.stack([u_tgt_homo, v_tgt_homo], dim=-1)  # [Src, Tgt, N, 2]

        with torch.no_grad():
            projection_singular_mask = warped_depth < 1e-5
            inside_mask = (u_tgt >= 0) & (u_tgt < W - 1) & (v_tgt >= 0) & (v_tgt < H - 1)

            # Only points tracked in both the source and the target frame count
            cross_time_mask = s_track_valid_mask[:, None, :] * s_track_valid_mask[None, tgt_inds, :]
            cross_time_mask = cross_time_mask.float() * inside_mask.float() * (~projection_singular_mask).float()

            # (1) Robust weight on depth
            depth_robust_w = positive_th_gaussian_decay(abs(scaled_depth_list), depth_decay_th,
                                                        depth_decay_sigma)  # [Src, N]

            # (2) Robust weight on the spread (std) of the 3D points
            point_ref_mean = (point_ref * s_track_valid_mask_w[:, :, None]).sum(0)  # [N, 3]
            point_ref_std = (point_ref - point_ref_mean[None]).norm(dim=-1, p=2)  # [T, N]
            point_ref_std_robust_w = (point_ref_std * s_track_valid_mask_w).sum(0)  # [N]
            point_ref_std_robust_w = positive_th_gaussian_decay(point_ref_std_robust_w, std_decay_th,
                                                                std_decay_sigma)

            # Combine the weights
            robust_w = depth_robust_w * point_ref_std_robust_w[None, :]  # [Src, N]
            cross_robust_time_mask = robust_w[:, None, :] * robust_w[None, tgt_inds, :]  # [Src, Tgt, N]
            cross_time_mask = cross_time_mask * cross_robust_time_mask.detach()

        # ------------------------------------------------------------------
        # Losses
        # ------------------------------------------------------------------
        # 1) UV Loss (Homo Space Reprojection Error)
        uv_target = homo_list[tgt_inds].unsqueeze(0).expand(T, -1, -1, -1)  # [Src, Tgt, N, 2]
        uv_loss_i = torch.sqrt((uv_src_to_every_frame - uv_target).pow(2).sum(dim=-1) + 1e-8)

        uv_loss = (uv_loss_i * cross_time_mask).sum() / (cross_time_mask.sum() + 1e-6)

        # 2) Depth Loss (Symmetric Ratio)
        dep_target = scaled_depth_list[tgt_inds].unsqueeze(0).expand(T, -1, -1)  # [Src, Tgt, N]
        dep_consistency_i = 0.5 * abs(dep_target / torch.clamp(warped_depth, min=1e-6) - 1) + 0.5 * abs(
            warped_depth / torch.clamp(dep_target, min=1e-6) - 1)
        dep_loss = (dep_consistency_i * cross_time_mask).sum() / (cross_time_mask.sum() + 1e-6)

        # Total loss
        loss = (
                lambda_depth * dep_loss
                + lambda_flow * uv_loss
                + lambda_small_correction * dep_corr_loss
        )

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % 100 == 0 or step == total_steps - 1:
            print(
                f"[BA Step {step:04d}] Loss: {loss.item():.5f} | Flow(UV): {uv_loss.item():.5f} | Dep: {dep_loss.item():.5f}")
    from utils.graphics_utils import getWorld2View2

    with torch.no_grad():
        final_dep_scale = param_scale.abs() / param_scale.abs().mean()
        for t in range(T):
            cam = viewpoint_stack[t]

            # (1) Update the Camera object used for rendering (still tensors here)
            R_orig_tensor = torch.from_numpy(cam.R.T).float().to(device)
            T_orig_tensor = torch.from_numpy(cam.T).float().to(device)
            R_new = torch.matmul(axis_angle_to_matrix(delta_q[t]), R_orig_tensor)
            T_new = T_orig_tensor + delta_t[t]
            new_R = R_new.T.detach().cpu().numpy()
            new_T = T_new.detach().cpu().numpy()

            # Overwrite with plain numpy arrays
            cam.R = new_R
            cam.T = new_T
            cam.depth = cam.depth * final_dep_scale[t]
            ###########depth scale#########################
            valid_mask = s_track_valid_mask[t]  # [N]

            # Round the valid points' original 2D pixel coordinates to integers
            u_int = torch.clamp(s_track[t, valid_mask, 0].round().long(), 0, W - 1)
            v_int = torch.clamp(s_track[t, valid_mask, 1].round().long(), 0, H - 1)

            # Add the learned per-point offset (param_dep_corr) at those pixels
            # Handle cam.depth being either [H, W] or [1, H, W]
            corr_values = param_dep_corr[t, valid_mask].detach()
            if cam.depth.ndim == 3:
                cam.depth[0, v_int, u_int] += corr_values.to(cam.depth.device)
            else:
                cam.depth[v_int, u_int] += corr_values.to(cam.depth.device)
            ###########depth scale done#########################

            target_device = cam.projection_matrix.device
            cam.world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)).transpose(0, 1).to(target_device)
            cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]

            # (2) Replace the CameraInfo held by the dataset
            if hasattr(scene, "train_camera") and hasattr(scene.train_camera, "dataset"):
                old_info = scene.train_camera.dataset[t]
                # new_R and new_T are numpy arrays by this point
                ###############depth scale############################
                new_depth_ndarray = cam.depth.detach().cpu().numpy().transpose(1, 2, 0)
                scene.train_camera.dataset[t] = old_info._replace(R=new_R, T=new_T, depth=new_depth_ndarray)
    if scene.dataset_type == 'nvidia':
        if hasattr(scene, "test_camera") and hasattr(scene.test_camera, "dataset"):
                    # Take the refined R, T of train frame 0 as the reference
            target_R = scene.train_camera.dataset[0].R
            target_T = scene.train_camera.dataset[0].T

            for i in range(len(scene.test_camera.dataset)):
                old_test_info = scene.test_camera.dataset[i]
                        # Pin every test frame's pose to that of train frame 0
                scene.test_camera.dataset[i] = old_test_info._replace(R=target_R, T=target_T)


