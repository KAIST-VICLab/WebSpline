import os
import sys
import argparse
import re
import torch
import torch.nn.functional as F
sys.path.append(os.path.join(sys.path[0], '..'))
# from dataloading import get_dataloader, load_config
# import model as mdl
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
import glob
from scene import Scene, GaussianModel, dataset_readers
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.graphics_utils import getWorld2View, pts2pixel
from utils_poses.vis_cam_traj import draw_camera_frustum_geometry
from utils_poses.align_traj import align_ate_c2b_use_a2b
from utils_poses.comp_ate import compute_rpe, compute_ATE
from dycheck_metrics import compute_psnr, compute_ssim, compute_lpips
from eval_pck import test_pck
from scene import bundle_adjustment
from argparse import ArgumentParser, Namespace
from gaussian_renderer import render, render_tto, network_gui, interpolate_cubic_hermite
from utils.image_utils import psnr
import cv2
from utils.loss_utils import l1_loss, l2_loss, ssim, BinaryDiceLoss, sgt_smoothness, masked_l1_loss
import lpips
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
import imageio
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
import inspect
import copy


torch.manual_seed(0)


def save_tensor_image(tensor_chw, save_path):
    img = torch.clamp(tensor_chw, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def _extract_int_tokens(name):
    if name is None:
        return []
    return [int(tok) for tok in re.findall(r"\d+", str(name))]


def _extract_frame_index(cam):
    toks = _extract_int_tokens(getattr(cam, "image_name", ""))
    if not toks:
        return None
    return int(toks[-1])


def _load_mosca_trajectory(source_path):
    if not source_path:
        return None
    trajectory_file = os.path.join(
        source_path,
        "trajectory",
        "mosca_unidepth_splinegs_track_all_viz.pth",
    )
    if not os.path.exists(trajectory_file):
        print(f"[WARN] Trajectory file not found: {trajectory_file}")
        return None

    try:
        traj_data = torch.load(trajectory_file, map_location="cpu")
        if "dyn_trajectory" not in traj_data:
            print(f"[WARN] dyn_trajectory key not found in {trajectory_file}")
            return None
        all_tracks_3d = traj_data["dyn_trajectory"]
        if not torch.is_tensor(all_tracks_3d):
            all_tracks_3d = torch.as_tensor(all_tracks_3d)
        print(
            f"[*] Loaded MoSca trajectory from {trajectory_file} with shape {tuple(all_tracks_3d.shape)}"
        )
        return all_tracks_3d
    except Exception as exc:
        print(f"[WARN] Failed to load trajectory file {trajectory_file}: {exc}")
        return None


def _load_mosca_visibility(source_path):
    if not source_path:
        return None

    trajectory_file = os.path.join(
        source_path,
        "trajectory",
        "mosca_unidepth_splinegs_track_all_viz.pth",
    )
    if not os.path.exists(trajectory_file):
        return None

    try:
        traj_data = torch.load(trajectory_file, map_location="cpu")
        scaffold_state_dict = traj_data.get("scaffold_state_dict", None)
        if scaffold_state_dict is None:
            return None

        node_certain = scaffold_state_dict.get("_node_certain", None)
        if node_certain is None:
            return None

        if not torch.is_tensor(node_certain):
            node_certain = torch.as_tensor(node_certain)
        return node_certain
    except Exception as exc:
        print(f"[WARN] Failed to load scaffold visibility from {trajectory_file}: {exc}")
        return None


def _load_overlay_tracks_and_visibility(source_path=None, checkpoint_path=None):
    all_tracks_3d = None
    scaffold_certain = None

    if checkpoint_path:
        scaffold_file = os.path.join(checkpoint_path, "scaffold_state_dict.pt")
        if os.path.exists(scaffold_file):
            try:
                scaffold_state_dict = torch.load(scaffold_file, map_location="cpu")
                all_tracks_3d = scaffold_state_dict.get("_node_xyz", None)
                scaffold_certain = scaffold_state_dict.get("_node_certain", None)
                if all_tracks_3d is not None and not torch.is_tensor(all_tracks_3d):
                    all_tracks_3d = torch.as_tensor(all_tracks_3d)
                if scaffold_certain is not None and not torch.is_tensor(scaffold_certain):
                    scaffold_certain = torch.as_tensor(scaffold_certain)
                if all_tracks_3d is not None:
                    print(
                        f"[*] Loaded trained scaffold trajectory from {scaffold_file} with shape {tuple(all_tracks_3d.shape)}"
                    )
                else:
                    print(f"[WARN] _node_xyz key not found in {scaffold_file}, using source cache fallback")
            except Exception as exc:
                print(f"[WARN] Failed to load trained scaffold from {scaffold_file}: {exc}")

    if all_tracks_3d is None:
        all_tracks_3d = _load_mosca_trajectory(source_path)
    if scaffold_certain is None:
        scaffold_certain = _load_mosca_visibility(source_path)

    return all_tracks_3d, scaffold_certain


def _get_camera_matrices(viewpoint, device):
    if not hasattr(viewpoint, "R") or not hasattr(viewpoint, "T"):
        return None, None

    r = torch.as_tensor(viewpoint.R, dtype=torch.float32, device=device)
    t = torch.as_tensor(viewpoint.T, dtype=torch.float32, device=device).reshape(3)
    w2c = torch.eye(4, dtype=torch.float32, device=device)
    w2c[:3, :3] = r.transpose(0, 1)
    w2c[:3, 3] = t

    if hasattr(viewpoint, "K") and viewpoint.K is not None:
        k = torch.as_tensor(viewpoint.K, dtype=torch.float32, device=device)
        if k.shape == (3, 3):
            return w2c, k

    if not hasattr(viewpoint, "metadata"):
        return w2c, None

    focal = getattr(viewpoint, "focal", None)
    if focal is None:
        return w2c, None

    k = torch.zeros((3, 3), dtype=torch.float32, device=device)
    k[0, 0] = float(focal)
    k[1, 1] = float(focal)
    k[0, 2] = float(viewpoint.metadata.principal_point_x)
    k[1, 2] = float(viewpoint.metadata.principal_point_y)
    k[2, 2] = 1.0
    return w2c, k


def render_track_include_image(viewpoint, render_img, all_tracks_3d, scaffold_certain=None):
    frame_idx = _extract_frame_index(viewpoint)
    if frame_idx is None:
        return None

    if frame_idx < 0 or frame_idx >= all_tracks_3d.shape[1]:
        print(f"[WARN] Frame index {frame_idx} is out of range for trajectory length {all_tracks_3d.shape[1]}")
        return None

    device = render_img.device
    w2c, k = _get_camera_matrices(viewpoint, device)
    if w2c is None or k is None:
        return None

    h = int(render_img.shape[1])
    w = int(render_img.shape[2])
    pad_w, pad_h = w // 2, h // 2
    canvas_w, canvas_h = w + 2 * pad_w, h + 2 * pad_h

    pred = torch.clamp(render_img, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    pred_pil = Image.fromarray((pred * 255).astype(np.uint8))

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    canvas.paste(pred_pil, (pad_w, pad_h))

    pts_3d = all_tracks_3d[:, frame_idx, :].to(device=device, dtype=torch.float32)
    vis_mask = None
    if scaffold_certain is not None and frame_idx < scaffold_certain.shape[0]:
        vis_mask = scaffold_certain[frame_idx].to(device=device)
        if vis_mask.ndim > 1:
            vis_mask = vis_mask.reshape(-1)
        if vis_mask.shape[0] != pts_3d.shape[0]:
            vis_mask = None
    pts_3d_h = torch.cat([pts_3d, torch.ones_like(pts_3d[:, :1])], dim=-1)
    pts_cam_h = (w2c @ pts_3d_h.T).T
    z = pts_cam_h[:, 2:3]
    pts_2d_h = (k @ pts_cam_h[:, :3].T).T
    pts_2d = pts_2d_h[:, :2] / (z + 1e-8)

    final_mask = torch.isfinite(pts_2d).all(dim=-1) & torch.isfinite(z.squeeze(-1))
    padded_pts_2d = (
        pts_2d[final_mask] + torch.tensor([pad_w, pad_h], device=device, dtype=torch.float32)
    ).detach().cpu().numpy()

    draw = ImageDraw.Draw(canvas)
    dot_radius = 1
    pt_indices = torch.where(final_mask)[0].detach().cpu().numpy()
    for local_idx, pt in enumerate(padded_pts_2d):
        x, y = float(pt[0]), float(pt[1])
        if 0 <= x < canvas_w and 0 <= y < canvas_h:
            if vis_mask is not None:
                is_certain = bool(vis_mask[pt_indices[local_idx]].item())
                color = (0, 255, 0) if is_certain else (255, 0, 0)
            else:
                color = (160, 160, 160)
            draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill=color)

    if vis_mask is not None:
        true_count = int(vis_mask.bool().sum().item())
        total_count = int(vis_mask.numel())
        draw.text((20, 20), f"Frame: {frame_idx} | node_certain True:{true_count}/{total_count} (green/red)", fill="white")
    else:
        draw.text((20, 20), f"Frame: {frame_idx} | node_certain unavailable", fill="white")

    return canvas


def _build_train_camera_maps(train_cams):
    by_name = {}
    by_uid = {}
    by_last_token = {}
    by_first_token = {}

    for cam in train_cams:
        name = cam.image_name.split('_')[-1]
        uid = getattr(cam, "uid", None)

        if name and name not in by_name:
            by_name[name] = cam

        if uid is not None:
            by_uid[int(uid)] = cam

        toks = _extract_int_tokens(name)
        if toks:
            by_last_token.setdefault(toks[-1], cam)
            by_first_token.setdefault(toks[0], cam)

    return by_name, by_uid, by_last_token, by_first_token


def _match_train_camera(test_cam, maps):
    by_name, by_uid, by_last_token, by_first_token = maps
    t_name = test_cam.image_name.split('_')[-1]
    t_uid = getattr(test_cam, "uid", None)

    if t_name in by_name:
        return by_name[t_name]

    if t_uid is not None and int(t_uid) in by_uid:
        return by_uid[int(t_uid)]

    toks = _extract_int_tokens(t_name)
    if toks and toks[-1] in by_last_token:
        return by_last_token[toks[-1]]

    if toks and toks[0] in by_first_token:
        return by_first_token[toks[0]]

    return None


def _extract_cam_index(cam):
    toks = _extract_int_tokens(getattr(cam, "image_name", ""))
    if toks:
        return int(toks[0])
    uid = getattr(cam, "uid", None)
    if uid is not None:
        return int(uid)
    return 0


def _resolve_train_uid(cam, train_maps):
    matched_train_cam = _match_train_camera(cam, train_maps)
    if matched_train_cam is not None:
        matched_uid = getattr(matched_train_cam, "uid", None)
        if matched_uid is not None:
            return int(matched_uid)

    cam_uid = getattr(cam, "uid", None)
    if cam_uid is not None:
        return int(cam_uid)
    return None


def _build_output_name(cam, train_maps, fallback_idx):
    cam_idx = _extract_cam_index(cam)
    uid = _resolve_train_uid(cam, train_maps)
    if uid is None:
        uid = int(fallback_idx)
    return f"{cam_idx}_{uid:05d}.png"



def trbfunction(x): 
    return torch.exp(-1*x.pow(2))

def de_casteljau(betas, t):
    """ castle interpolation, for knights
        see: https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
        assumes t in [0,1]
    """
    t = t[:,None] # 1,1,T
    
    out = betas.clone()
    N = betas.shape[1] # number of points
    for i in range(1, N):
        out = out[:, :-1,:] * (1-t) + out[:, 1:,:] * t
    return out.squeeze(1)

def get_pixels(image_size_x, image_size_y, use_center = None):
    """Return the pixel at center or corner."""
    xx, yy = np.meshgrid(
        np.arange(image_size_x, dtype=np.float32),
        np.arange(image_size_y, dtype=np.float32),
    )
    offset = 0.5 if use_center else 0
    return np.stack([xx, yy], axis=-1) + offset

def eval(scene, gaussians, ATE_align=True):
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    N_imgs = len(train_cams)
    print("Number of train views: {}".format(N_imgs))

    '''Define camera frustums'''
    frustum_length = 0.1
    est_train_color = np.array([39, 125, 161], dtype=np.float32) / 255
    gt_train_color = np.array([249, 65, 68], dtype=np.float32) / 255
    gt_test_color = np.array([65, 222, 68], dtype=np.float32) / 255
    gt_test_aligned_color = np.array([65, 30, 240], dtype=np.float32) / 255

    est_train_poses = []
    gaussians._posenet.eval()
    with torch.no_grad():
        for cam in train_cams.dataset:
            time_in = torch.tensor(cam.time).float().cuda()
            pred_R, pred_T = gaussians._posenet(time_in.view(1, 1))
            R_ = torch.transpose(pred_R, 2, 1)
            t_ = pred_T
            Rt = getWorld2View(R_[0].detach().cpu().numpy(), t_[0].detach().cpu().numpy())
            C2W = np.linalg.inv(Rt)
            est_train_poses.append(C2W)
    est_train_poses = torch.tensor(np.array(est_train_poses))
    H, W = train_cams.dataset[0].height, train_cams.dataset[0].width
    fx, fy = train_cams.dataset[0].FovX, train_cams.dataset[0].FovY
    frustum_est_train_list = draw_camera_frustum_geometry(est_train_poses.cpu().numpy(), H, W,
                                                        fx*W, fy*H,
                                                        frustum_length, est_train_color)

    gt_train_poses = []
    for cam in train_cams.dataset:
        C2W = np.concatenate((cam.c2w, [[0, 0, 0, 1]]), axis=0)
        gt_train_poses.append(C2W)
    gt_train_poses = torch.tensor(np.array(gt_train_poses))

    frustum_gt_train_list = draw_camera_frustum_geometry(gt_train_poses.cpu().numpy(), H, W,
                                                        fx*W, fy*H,
                                                        frustum_length, gt_train_color)

    '''Align est traj to colmap traj'''
    if ATE_align:  # Align learned poses to colmap poses
        c2ws_est_aligned = align_ate_c2b_use_a2b(est_train_poses, gt_train_poses)  # (N, 4, 4)
        c2ws_est_to_draw_align2cmp = c2ws_est_aligned

        # compute ate
        ate = compute_ATE(gt_train_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        rpe_trans, rpe_rot = compute_rpe(gt_train_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        print("{0:.3f}".format(rpe_trans*100),'&' "{0:.3f}".format(rpe_rot * 180 / np.pi), '&', "{0:.3f}".format(ate))

    frustum_est_train_aligned_list = draw_camera_frustum_geometry(c2ws_est_to_draw_align2cmp.cpu().numpy(), H, W,
                                                        fx*W, fy*H,
                                                        frustum_length, est_train_color)

    H, W = test_cams.dataset[0].height, test_cams.dataset[0].width
    fx, fy = test_cams.dataset[0].FovX, test_cams.dataset[0].FovY
    gt_test_poses = []
    for cam in test_cams.dataset:
        C2W = np.concatenate((cam.c2w, [[0, 0, 0, 1]]), axis=0)
        gt_test_poses.append(C2W)
        if len(gt_test_poses) == N_imgs:
            break
    gt_test_poses = torch.tensor(np.array(gt_test_poses))

    frustum_gt_test_list = draw_camera_frustum_geometry(gt_test_poses.cpu().numpy(), H, W,
                                                        fx*W, fy*H,
                                                        frustum_length, gt_test_color)

    gt_test_aligned = align_ate_c2b_use_a2b(gt_train_poses, est_train_poses, gt_test_poses)  # (N, 4, 4)
    frustum_gt_test_aligned_list = draw_camera_frustum_geometry(gt_test_aligned.cpu().numpy(), H, W,
                                                        fx*W, fy*H,
                                                        frustum_length, gt_test_aligned_color)

    geometry_to_draw = []
    geometry_to_draw.append(frustum_est_train_list)
    # geometry_to_draw.append(frustum_gt_train_list)
    # geometry_to_draw.append(frustum_gt_test_list)
    geometry_to_draw.append(frustum_gt_test_aligned_list)

    '''o3d for line drawing'''
    t_gt_test_list = gt_test_poses[:, :3, 3]
    t_gt_test_aligned_list = gt_test_aligned[:, :3, 3]
    t_gt_train_list = gt_train_poses[:, :3, 3]
    t_est_train_list = c2ws_est_aligned[:, :3, 3]

    '''line set to note pose correspondence between two trajs'''
    # line_points = torch.cat([t_est_train_list, t_gt_train_list, t_gt_test_list, t_gt_test_aligned_list], dim=0).cpu().numpy()  # (4N, 3)
    line_points = torch.cat([t_est_train_list, t_gt_test_aligned_list], dim=0).cpu().numpy()  # (3N, 3)

    line_ends = [[i, i] for i in range(N_imgs)]  # (N, 3) connect two end points.

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_ends)
    unit_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=2)
    unit_sphere = o3d.geometry.LineSet.create_from_triangle_mesh(unit_sphere)
    unit_sphere.paint_uniform_color((0, 1, 0))
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame()

    geometry_to_draw.append(line_set)

    o3d.visualization.draw_geometries(geometry_to_draw)

def im2tensor(img):
    return torch.Tensor(img.transpose(2, 0, 1) / 127.5 - 1.0)[None, ...]

def normalize_image(img):
    return (2.0 * img - 1.0)[None, ...]

def draw_histogram(ctrl_num):
    font_size=18
    x_value = np.array(range(4,14,1))
    y_value = np.zeros_like(x_value)
    for idx, bin_value in enumerate(range(4,14,1)):
        current_num = ctrl_num[ctrl_num == bin_value].shape[0]
        y_value[idx] = current_num / ctrl_num.shape[0] * 100
    
    # Creating histogram
    px = 1/plt.rcParams['figure.dpi']
    fig, axs = plt.subplots(1, 1,
                            figsize =(480*px, 270*px), 
                            tight_layout = True, 
                            )
    
    
    # Remove axes splines 
    for s in ['top', 'bottom', 'left', 'right']: 
        axs.spines[s].set_visible(False) 
    
    # # Remove x, y ticks
    axs.xaxis.set_ticks_position('none') 
    axs.yaxis.set_ticks_position('none') 
    
    # # Add padding between axes and labels 
    # axs.xaxis.set_tick_params(pad = 5) 
    # axs.yaxis.set_tick_params(pad = 10) 
    axs.set_xticks(list(x_value[:-1] +0.5), [str(x) for x in x_value[:-1]])
    
    axs.set_facecolor('#EAEAF2')
    # Creating histogram
    N, bins, patches = axs.hist(np.array(ctrl_num.squeeze()), bins = x_value, color = '#4C72B0', density=True)
    
    plt.xlabel("# of Ctrl. Pts.", fontsize = font_size)
    plt.ylabel("% of Gaussians", fontsize = font_size)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    vals = np.array([0,0.2,0.4])
    axs.set_yticks(vals, ['{}%'.format(int(x*100)) for x in vals])
    # axs.set_yticklabels(['{}%'.format(int(x*100)) for x in vals])
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    
    plt.savefig('histogram.png')
    #breakpoint()

def training_report(
    scene,
    train_cams,
    render_func,
    render_args,
    stage,
    out_dir,
    source_path,
    checkpoint_path=None,
):
    validation_configs = [
        {"name": "train", "cameras": train_cams},
    ]

    os.makedirs(out_dir, exist_ok=True)
    all_tracks_3d, scaffold_certain = _load_overlay_tracks_and_visibility(
        source_path=source_path,
        checkpoint_path=checkpoint_path,
    )
    train_maps = _build_train_camera_maps(train_cams)
    lpips_loss = lpips.LPIPS(net="alex").cuda()

    with torch.no_grad():
        for config in validation_configs:
            cams = config["cameras"]
            if not cams or len(cams) == 0:
                continue

            split_dir = os.path.join(out_dir, config["name"])
            os.makedirs(split_dir, exist_ok=True)
            rendering_dir = os.path.join(split_dir, "rendering")
            track_include_dir = os.path.join(split_dir, "track_include")
            os.makedirs(rendering_dir, exist_ok=True)
            os.makedirs(track_include_dir, exist_ok=True)

            psnr_test = 0.0
            lpips_test = 0.0
            ssim_test = 0.0

            for idx, viewpoint in enumerate(cams):
                render_pkg = render_func(
                    viewpoint,
                    scene.stat_gaussians,
                    scene.dyn_gaussians,
                    render_args[1],
                    stage=stage,
                    get_static=True,
                )
                output_name = _build_output_name(viewpoint, train_maps, idx)

                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                save_tensor_image(image, os.path.join(rendering_dir, output_name))

                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                psnr_test += psnr(image, gt_image, mask=None).mean().double()
                lpips_test += lpips_loss.forward(normalize_image(image), normalize_image(gt_image)).item()
                ssim_test += ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item()

                if all_tracks_3d is not None:
                    track_include_img = render_track_include_image(
                        viewpoint=viewpoint,
                        render_img=render_pkg["render"],
                        all_tracks_3d=all_tracks_3d,
                        scaffold_certain=scaffold_certain,
                    )
                    if track_include_img is not None:
                        track_include_img.save(os.path.join(track_include_dir, output_name))

            n = len(cams)
            print(
                "\n[{}] Evaluating {}: PSNR {:.4f}, LPIPS {:.4f}, SSIM {:.4f}".format(
                    stage, config["name"], psnr_test / n, lpips_test / n, ssim_test / n
                )
            )

    torch.cuda.empty_cache()
    print(f"Saved rendered images to: {out_dir}")


def render_test_tto(
    H,
    W,
    scene,
    test_cams,
    save_dir,
    gt_rgb_dir,
    source_path,
    checkpoint_path=None,
    tto_steps=25,
    decay_start=15,
    lr_p=0.003,
    lr_q=0.003,
    lr_final=0.0001,
    use_sgd=False,
    loss_type="psnr",
    initialize_from_previous_camera=True,
    initialize_from_previous_step_factor=10,
    initialize_from_previous_lr_factor=0.1,
    fg_mask_th=0.1,
    local_viewdirs=None,
    batch_shape=None,
    renderArgs=None
):
    device = scene.stat_gaussians.get_xyz.device
    all_tracks_3d, scaffold_certain = _load_overlay_tracks_and_visibility(
        source_path=source_path,
        checkpoint_path=checkpoint_path,
    )
    pad_w, pad_h = W // 2, H // 2  
    new_W, new_H = W + 2 * pad_w, H + 2 * pad_h
    test_ret = []
    lpips_fn = lpips.LPIPS(net="alex").cuda()
    def _load_gt_rgb_tensor(camera):
        gt_rgb = imageio.imread(os.path.join(gt_rgb_dir, f"{camera.image_name}.png")) / 255.0
        gt_rgb = gt_rgb[..., :3]
        return torch.tensor(gt_rgb, device=device).float()

    def _optimize_pose(current_camera, gt_rgb, init_w2c, n_steps, lr_factor):
        t_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        t_init = torch.nn.Parameter(init_w2c[:3, 3].clone().detach(), requires_grad=True)
        q_init = torch.nn.Parameter(
            matrix_to_quaternion(init_w2c[:3, :3]).clone().detach(),
            requires_grad=True,
        )

        optimizer_type = torch.optim.SGD if use_sgd else torch.optim.Adam
        optimizer = optimizer_type(
            [
                {"params": t_init, "lr": lr_p * lr_factor},
                {"params": q_init, "lr": lr_q * lr_factor},
            ]
        )
        scheduler_tmax = max(1, n_steps - decay_start)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_tmax,
            eta_min=lr_final * lr_factor,
        )

        tto_viewpoint = copy.deepcopy(current_camera)
        k = tto_viewpoint.K
        cam_ray = tto_viewpoint.cam_ray
        image_width = tto_viewpoint.image_width
        image_height = tto_viewpoint.image_height
        time = tto_viewpoint.time

        def _compute_loss(t, q):
            w2c = torch.cat([quaternion_to_matrix(q), t[:, None]], 1)
            w2c = torch.cat([w2c, t_bottom[None]], 0)
            pkg = render_tto(w2c, k, cam_ray, image_width, image_height, time,
                             scene.stat_gaussians, scene.dyn_gaussians, renderArgs[1])
            pred = pkg["render"].permute(1, 2, 0)
            if loss_type == "psnr":
                mse = ((pred - gt_rgb) ** 2).mean()
                return -(20 * torch.log10(1.0 / torch.sqrt(torch.clamp(mse, min=1e-10)))), pred
            elif loss_type == "psnr_lpips":
                mse = ((pred - gt_rgb) ** 2).mean()
                psnr_val = 20 * torch.log10(1.0 / torch.sqrt(torch.clamp(mse, min=1e-10)))
                pred_chw = pred.permute(2, 0, 1)
                gt_chw = gt_rgb.permute(2, 0, 1)
                lpips_val = lpips_fn(normalize_image(pred_chw), normalize_image(gt_chw))
                return -psnr_val + lpips_val, pred
            else:
                raise ValueError(f"Unknown loss type {loss_type}")

        # Initialize best_loss from initial pose so TTO only saves improvements
        with torch.no_grad():
            init_loss, _ = _compute_loss(t_init.detach(), q_init.detach())
            best_loss = init_loss.item()
        best_t = t_init.clone().detach()
        best_q = q_init.clone().detach()

        for _step in range(n_steps):
            optimizer.zero_grad()
            rgb_loss, _ = _compute_loss(t_init, q_init)

            rgb_loss.backward()

            # Save pose BEFORE the step (loss corresponds to current pose)
            with torch.no_grad():
                if rgb_loss.item() < best_loss:
                    best_loss = rgb_loss.item()
                    best_t = t_init.clone().detach()
                    best_q = q_init.clone().detach()

            optimizer.step()

            with torch.no_grad():
                q_init.data = F.normalize(q_init.data, dim=0)

            if _step >= decay_start:
                scheduler.step()

        solved_pose = torch.cat([quaternion_to_matrix(best_q), best_t[:, None]], 1)
        solved_pose = torch.cat([solved_pose, t_bottom[None]], 0)
        return solved_pose.detach().cpu().numpy()

    num_frames = len(test_cams)
    cam_ids = [_extract_cam_index(cam) for cam in test_cams]

    pass1_solved_pose_list = []
    prev_camid = -1
    for i in tqdm(range(num_frames), desc="TTO pass1"):
        if initialize_from_previous_camera and i == 0:
            step_factor = initialize_from_previous_step_factor
            lr_factor = 1.0
        else:
            step_factor = 1
            lr_factor = initialize_from_previous_lr_factor

        current_camera = test_cams[i]
        gt_rgb = _load_gt_rgb_tensor(current_camera)
        current_cam_id = cam_ids[i]

        if initialize_from_previous_camera:
            if current_cam_id != prev_camid or i == 0:
                init_w2c = current_camera.world_view_transform.transpose(0, 1).to(device)
            else:
                init_w2c = torch.from_numpy(pass1_solved_pose_list[-1]).to(device)
        else:
            init_w2c = current_camera.world_view_transform.transpose(0, 1).to(device)

        n_steps = max(1, int(tto_steps * step_factor))
        solved_pose_np = _optimize_pose(current_camera, gt_rgb, init_w2c, n_steps, lr_factor)
        pass1_solved_pose_list.append(solved_pose_np)
        prev_camid = current_cam_id

    solved_pose_list = pass1_solved_pose_list

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    for i in tqdm(range(num_frames), desc="Render+Eval"):
        current_camera = test_cams[i]
        frame_idx = _extract_frame_index(current_camera)
        if frame_idx is None:
            frame_idx = int(i)
        gt_rgb = _load_gt_rgb_tensor(current_camera)

        with torch.no_grad():
            solved_pose = torch.from_numpy(solved_pose_list[i]).to(device=device, dtype=torch.float32)
            k = current_camera.K
            cam_ray = current_camera.cam_ray
            image_width = current_camera.image_width
            image_height = current_camera.image_height
            time = current_camera.time

            render_pkg = render_tto(
                solved_pose,
                k,
                cam_ray,
                image_width,
                image_height,
                time,
                scene.stat_gaussians,
                scene.dyn_gaussians,
                renderArgs[1],
            )

            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(gt_rgb.permute(2, 0, 1), 0.0, 1.0)

            img_psnr = psnr(image, gt_image, mask=None).mean().double().item()
            img_lpips = lpips_fn.forward(normalize_image(image), normalize_image(gt_image)).item()
            img_ssim = ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
            total_psnr += img_psnr
            total_ssim += img_ssim
            total_lpips += img_lpips

            pred = image.permute(1, 2, 0).detach().cpu().numpy()
            gt = gt_rgb.cpu().numpy()

            img = Image.fromarray((np.clip(pred, 0, 1) * 255).astype("uint8"))
            os.makedirs(save_dir + f"/test_refined_{loss_type}", exist_ok=True)
            img.save(save_dir + f"/test_refined_{loss_type}/{{}}".format(f"{current_camera.image_name}.png"))

            diff_amp = 3.0
            diff_map = np.clip(np.abs(pred - gt) * diff_amp, 0, 1)
            concat_img = np.concatenate([pred, gt, diff_map], axis=1)
            concat_img_uint8 = (concat_img * 255).astype("uint8")
            concat_img_pil = Image.fromarray(concat_img_uint8)

            metrics_text = f"PSNR: {img_psnr:.2f}  |  SSIM: {img_ssim:.3f}  |  LPIPS: {img_lpips:.3f}"

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
            except IOError:
                try:
                    font = ImageFont.truetype("arial.ttf", 30)
                except IOError:
                    print("Warning: no .ttf font found, falling back to the default font.")
                    font = ImageFont.load_default()

            bar_height = 60
            total_width = concat_img_pil.width
            total_height = concat_img_pil.height + bar_height
            final_img = Image.new("RGB", (total_width, total_height), color="black")
            final_img.paste(concat_img_pil, (0, 0))

            draw = ImageDraw.Draw(final_img)
            draw.text((15, concat_img_pil.height + 15), metrics_text, fill="white", font=font)

            concat_save_path = os.path.join(save_dir, f"test_refined_full_{loss_type}")
            os.makedirs(concat_save_path, exist_ok=True)
            final_img.save(os.path.join(concat_save_path, f"{current_camera.image_name}.png"))

            if all_tracks_3d is not None:
                w2c = solved_pose
                k_tensor = torch.tensor(k, device=device).float()
                pts_3d = all_tracks_3d[:, frame_idx, :].to(device=device, dtype=torch.float32)
                vis_mask = None
                if scaffold_certain is not None and frame_idx < scaffold_certain.shape[0]:
                    vis_mask = scaffold_certain[frame_idx].to(device=device)
                    if vis_mask.ndim > 1:
                        vis_mask = vis_mask.reshape(-1)
                    if vis_mask.shape[0] != pts_3d.shape[0]:
                        vis_mask = None

                pts_3d_homo = torch.cat([pts_3d, torch.ones_like(pts_3d[:, :1])], dim=-1)
                pts_cam = (w2c @ pts_3d_homo.T).T
                z = pts_cam[:, 2:3]
                pts_2d = (k_tensor @ pts_cam[:, :3].T).T
                pts_2d = pts_2d[:, :2] / (z + 1e-8)

                final_mask = torch.isfinite(pts_2d).all(dim=-1) & torch.isfinite(z.squeeze(-1))
                final_pts_2d = (
                    pts_2d[final_mask]
                    + torch.tensor([pad_w, pad_h], device=device, dtype=torch.float32)
                ).cpu().numpy()
                canvas = Image.new("RGB", (new_W, new_H), (30, 30, 30))
                pred_pil = Image.fromarray((pred * 255).astype("uint8"))
                canvas.paste(pred_pil, (pad_w, pad_h))

                draw = ImageDraw.Draw(canvas)
                dot_radius = 1
                pt_indices = torch.where(final_mask)[0].detach().cpu().numpy()
                for local_idx, pt in enumerate(final_pts_2d):
                    x, y = pt[0], pt[1]
                    if 0 <= x < new_W and 0 <= y < new_H:
                        if vis_mask is not None:
                            is_certain = bool(vis_mask[pt_indices[local_idx]].item())
                            color = (0, 255, 0) if is_certain else (255, 0, 0)
                        else:
                            color = (160, 160, 160)
                        draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill=color)

                if vis_mask is not None:
                    true_count = int(vis_mask.bool().sum().item())
                    total_count = int(vis_mask.numel())
                    draw.text(
                        (20, 20),
                        f"Frame: {frame_idx} | node_certain True:{true_count}/{total_count} (green/red)",
                        fill="white",
                    )
                else:
                    draw.text((20, 20), f"Frame: {frame_idx} | node_certain unavailable", fill="white")

                track_include_path = os.path.join(save_dir, f"test_track_include_{loss_type}")
                os.makedirs(track_include_path, exist_ok=True)
                canvas.save(os.path.join(track_include_path, f"{current_camera.image_name}.png"))

    print("Average psnr: ", total_psnr / num_frames)
    print("Average ssim: ", total_ssim / num_frames)
    print("Average lpips: ", total_lpips / num_frames)

    metrics = {
        "PSNR": total_psnr / num_frames,
        "SSIM": total_ssim / num_frames,
        "LPIPS": total_lpips / num_frames,
    }
    
    output_file = os.path.join(save_dir, f"tto_{loss_type}_metrics.txt")
    with open(output_file, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    np.save(os.path.join(save_dir, f"solved_poses_{loss_type}.npy"), np.stack(solved_pose_list, 0))
    return test_ret


def render_test_no_tto(scene, test_cams, save_dir, gt_rgb_dir, renderArgs):
    device = scene.stat_gaussians.get_xyz.device
    lpips_fn = lpips.LPIPS(net="alex").cuda()

    def _load_gt_rgb_tensor(camera):
        gt_rgb = imageio.imread(os.path.join(gt_rgb_dir, f"{camera.image_name}.png")) / 255.0
        return torch.tensor(gt_rgb[..., :3], device=device).float()

    os.makedirs(os.path.join(save_dir, "test_no_tto"), exist_ok=True)
    num_frames = len(test_cams)
    total_psnr = total_ssim = total_lpips = 0.0

    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Render (no TTO)"):
            cam = test_cams[i]
            gt_rgb = _load_gt_rgb_tensor(cam)
            w2c = cam.world_view_transform.transpose(0, 1).to(device)

            render_pkg = render_tto(
                w2c, cam.K, cam.cam_ray,
                cam.image_width, cam.image_height, cam.time,
                scene.stat_gaussians, scene.dyn_gaussians, renderArgs[1],
            )

            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(gt_rgb.permute(2, 0, 1), 0.0, 1.0)

            total_psnr += psnr(image, gt_image, mask=None).mean().double().item()
            total_lpips += lpips_fn.forward(normalize_image(image), normalize_image(gt_image)).item()
            total_ssim += ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item()

            pred = image.permute(1, 2, 0).detach().cpu().numpy()
            Image.fromarray((np.clip(pred, 0, 1) * 255).astype("uint8")).save(
                os.path.join(save_dir, "test_no_tto", f"{cam.image_name}.png")
            )

    metrics = {
        "PSNR": total_psnr / num_frames,
        "SSIM": total_ssim / num_frames,
        "LPIPS": total_lpips / num_frames,
    }
    print(f"(no TTO) PSNR {metrics['PSNR']:.4f}, SSIM {metrics['SSIM']:.4f}, LPIPS {metrics['LPIPS']:.4f}")
    with open(os.path.join(save_dir, "no_tto_metrics.txt"), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    return metrics


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[300] + [1000 * i for i in range(100)])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[1000, 3000, 4000, 5000, 6000, 7_000, 9000, 10000, 12000, 14000, 20000, 30_000, 45000,
                                 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pck", action="store_true")
    parser.add_argument("-render_process", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    dataset = lp.extract(args)
    hyper = hp.extract(args)
    dyn_gaussians = GaussianModel(dataset.sh_degree, hyper)
    stat_gaussians = GaussianModel(dataset.sh_degree, hyper)
    opt = op.extract(args)

    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)  # for other datasets rather than iPhone dataset
    bundle_adjustment(opt, scene, dataset.source_path)
    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras()) # pose network with instance scaling
    dyn_gaussians.training_setup(opt, stage="fine")
    stat_gaussians.training_setup(opt)
    
    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = pp.extract(args)

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    my_test_cams = [i for i in test_cams]
    viewpoint_stack = [i for i in train_cams]

    dyn_gaussians.load_ply(os.path.join(args.checkpoint, 'point_cloud.ply'))
    stat_gaussians.load_ply(os.path.join(args.checkpoint, 'point_cloud_static.ply'))

    dyn_gaussians.load_model(args.checkpoint)
    
    # Compute view_dir
    pixels = get_pixels(scene.train_camera.dataset[0].metadata.image_size_x, scene.train_camera.dataset[0].metadata.image_size_y, use_center=True)
    if pixels.shape[-1] != 2:
        raise ValueError("The last dimension of pixels must be 2.")
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    y = (pixels[..., 1] - scene.train_camera.dataset[0].metadata.principal_point_y) / scene.train_camera.dataset[0].metadata.focal_length
    x = (pixels[..., 0] - scene.train_camera.dataset[0].metadata.principal_point_x) / scene.train_camera.dataset[0].metadata.focal_length
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    local_viewdirs =  viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)
    
    for attr in inspect.getmembers(dyn_gaussians):
        try:
            attr[1].requires_grad = False
        except:
            pass
    for attr in inspect.getmembers(stat_gaussians):
        try:
            attr[1].requires_grad = False
        except:
            pass
        
    render_test_no_tto(
        scene=scene,
        test_cams=my_test_cams,
        save_dir=os.path.join("output", args.expname),
        gt_rgb_dir=os.path.join(dataset.source_path, "gt"),
        renderArgs=[pipe, background],
    )

    _tto_common = dict(
        gt_rgb_dir=os.path.join(dataset.source_path, "gt"),
        tto_steps=200,
        decay_start=100,
        lr_p=0.0005,
        lr_q=0.0005,
        lr_final=0.00001,
        use_sgd=False,
        H=test_cams[0].image_height,
        W=test_cams[0].image_width,
        scene=scene,
        save_dir=os.path.join("output", args.expname),
        test_cams=my_test_cams,
        initialize_from_previous_camera=True,
        initialize_from_previous_step_factor=10,
        initialize_from_previous_lr_factor=0.1,
        fg_mask_th=0.1,
        local_viewdirs=local_viewdirs,
        batch_shape=batch_shape,
        renderArgs=[pipe, background],
        source_path=dataset.source_path,
        checkpoint_path=args.checkpoint,
    )

    frames = render_test_tto(loss_type="psnr", **_tto_common)
    # frames = render_test_tto(loss_type="psnr_lpips", **_tto_common)
    if args.pck == True:
        test_pck(
            cam=viewpoint_stack,
            static_gaussian=stat_gaussians,
            dynamic_gaussian=dyn_gaussians,
            gt_npz_fn=glob.glob(os.path.join(dataset.source_path, "*pck.npz")),
            device=torch.device("cuda"),
            save_fn=os.path.join("output", args.expname),
        )