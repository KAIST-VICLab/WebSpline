import torch
from utils.spline_utils import interpolate_cubic_hermite
import os, os.path as osp
import logging
import numpy as np
from tqdm import tqdm
from typing import Literal, Optional, Tuple
from pytorch3d.ops import knn_points
from gsplat.rendering import rasterization
from tqdm import tqdm
import imageio
from matplotlib import pyplot as plt
import cv2 as cv
import time

####################
# * DyCheck PCK Eval
####################
def trbfunction(x):
    return torch.exp(-1*x.pow(2))
def compute_pck(
        kps0: np.ndarray,
        kps1: np.ndarray,
        img_wh: Tuple[int, int],
        ratio: float = 0.05,
        reduce: Optional[Literal["mean"]] = "mean",
) -> np.ndarray:
    """Compute PCK between two sets of keypoints given the threshold ratio.

    Canonical Surface Mapping via Geometric Cycle Consistency.
        Kulkarni et al., ICCV 2019.
        https://arxiv.org/abs/1907.10043

    Args:
        kps0 (jnp.ndarray): A set of keypoints of shape (J, 2) in float32.
        kps1 (jnp.ndarray): A set of keypoints of shape (J, 2) in float32.
        img_wh (Tuple[int, int]): Image width and height.
        ratio (float): A threshold ratios. Default: 0.05.
        reduce (Optional[Literal["mean"]]): Reduction method. Default: "mean".

    Returns:
        jnp.ndarray:
            if reduce == "mean", PCK of shape();
            if reduce is None, corrects of shape (J,).
    """
    dists = np.linalg.norm(kps0 - kps1, axis=-1)
    thres = ratio * max(img_wh)
    corrects = dists < thres
    if reduce == "mean":
        return corrects.mean()
    elif reduce is None:
        return corrects


def eval_pck(gt_list, pred_list, image_size, ratio):
    N = len(gt_list)
    assert N == len(pred_list)
    metrics = []
    for i in tqdm(range(N)):
        common_corrects = compute_pck(
            gt_list[i],
            pred_list[i],
            image_size,
            ratio,
            reduce=None,
        )
        metrics.append(common_corrects)
    mean_pck = np.mean(
        [it.mean() for it in metrics]
    )  # ! the teddy scene verified the mean is in this way, not the cat all mean, but first mean across all points and then across all paris
    return mean_pck, metrics


def load_gt_pck_data(gt_data_dict):
    gt_dst_pixel_list = [it["dst_pixel_gt"] for it in gt_data_dict]
    gt_src_pixel_list = [it["src_pixel"] for it in gt_data_dict]
    src_t_list = [it["src_t"] for it in gt_data_dict]
    dst_t_list = [it["dst_t"] for it in gt_data_dict]
    img_wh = gt_data_dict[0]["img_wh"]
    ratio = gt_data_dict[0]["ratio"]
    for it in gt_data_dict:
        assert (it["img_wh"] == img_wh).all()
        assert it["ratio"] == ratio
    return (
        gt_src_pixel_list,
        gt_dst_pixel_list,
        src_t_list,
        dst_t_list,
        img_wh,
        ratio,
    )


@torch.no_grad()
def test_pck(cam, static_gaussian, dynamic_gaussian, gt_npz_fn, device, save_fn=None):
    # laod gt
    src, dst_gt, src_t, dst_t, img_wh, ratio = load_gt_pck_data(
        np.load(gt_npz_fn[0], allow_pickle=True)["arr_0"]
    )

    cams = cam
    s_model = static_gaussian
    d_model = dynamic_gaussian

    H, W = int(cam[0].image_height), int(cam[0].image_width)
    assert img_wh[0] == W and img_wh[1] == H
    debug_img_dir=os.path.join(save_fn, "pck_image")
    os.makedirs(debug_img_dir, exist_ok=True)
    dst_pred = []
    for _st, _dt, _src ,_dst_gt in tqdm(zip(src_t, dst_t, src, dst_gt)):
        _st, _dt = int(_st), int(_dt)
        time_src = float(_st) / len(cams)
        time_dst = float(_dt) / len(cams)

        stat_means3D = s_model.get_xyz
        stat_opacity = s_model.get_opacity
        stat_colors_precomp = s_model.get_features_static
        stat_scales = s_model.get_scaling
        stat_rotations = s_model.get_rotation_stat
        scales = d_model._scaling
        rotations = d_model._rotation
        opacity = d_model.get_opacity

        pointtimes = torch.ones(
            (d_model.get_xyz.shape[0], 1),
            dtype=d_model.get_xyz.dtype,
            requires_grad=False,
            device="cuda",
        ) + 0

        basicfunction = trbfunction

        trbfcenter = d_model.get_trbfcenter
        trbfscale = d_model.get_trbfscale
        trbfdistanceoffset_src = time_src * pointtimes - trbfcenter
        trbfdistanceoffset_dst = time_dst * pointtimes - trbfcenter
        trbfdistance_src = trbfdistanceoffset_src / torch.exp(trbfscale)
        trbfdistance_dst = trbfdistanceoffset_dst / torch.exp(trbfscale)
        trbfoutput_src = basicfunction(trbfdistance_src)
        trbfoutput_dst = basicfunction(trbfdistance_dst)

        tforpoly_src = trbfdistanceoffset_src.detach()
        tforpoly_dst = trbfdistanceoffset_dst.detach()
        rotations_src = d_model.get_rotation_dy(rotations, tforpoly_src)
        rotations_dst = d_model.get_rotation_dy(rotations, tforpoly_dst)
        control_xyz = d_model.get_control_xyz.cuda()
        deform_means3D_src = interpolate_cubic_hermite(
            control_xyz.permute(0, 2, 1),
            torch.tensor(time_src).cuda()[None, None].expand(control_xyz.shape[0], 3, 1),
            N=d_model.current_control_num,
        )
        deform_means3D_dst = interpolate_cubic_hermite(
            control_xyz.permute(0, 2, 1),
            torch.tensor(time_dst).cuda()[None, None].expand(control_xyz.shape[0], 3, 1),
            N=d_model.current_control_num,
        )
        means3D_src = deform_means3D_src * 1e-2
        means3D_dst = deform_means3D_dst * 1e-2
        # activations
        scales = d_model.scaling_activation(scales)
        rotations_src = d_model.rotation_activation(rotations_src)
        rotations_dst = d_model.rotation_activation(rotations_dst)
        colors_precomp_src = d_model.get_features(tforpoly_src)
        colors_precomp_dst = d_model.get_features(tforpoly_dst)



        mu = torch.cat([stat_means3D, means3D_src], 0)
        fr = torch.cat([stat_rotations, rotations_src], 0)
        s = torch.cat([stat_scales, scales], 0)
        o = torch.cat([stat_opacity, opacity], 0)
        sph = torch.cat([stat_colors_precomp, colors_precomp_src], 0)
        xyz_dst = torch.cat([stat_means3D, means3D_dst], 0)
        target_cam = cams[_dt]
        w2c_target = target_cam.world_view_transform.transpose(0, 1).cuda()

        R_target = w2c_target[:3, :3]
        t_target = w2c_target[:3, 3]

        # World -> target camera space
        mu_dst_cam = torch.einsum("ij,nj->ni", R_target, xyz_dst) + t_target[None]
        colors_precomp_pc = mu_dst_cam.clone().detach()

        # ----------------------------------------------------
        # Rasterise with gsplat
        # No SH: precomputed colours only
        # ----------------------------------------------------
        source_cam = cams[_st]
        viewmat = source_cam.world_view_transform.transpose(0, 1).cuda()

        if hasattr(source_cam, 'K') and source_cam.K is not None:
            K_src = source_cam.K.cuda()
        else:
            K_src = torch.eye(3, device="cuda")
            K_src[0, 0], K_src[1, 1] = source_cam.focal, source_cam.focal
            K_src[0, 2], K_src[1, 2] = W / 2.0, H / 2.0

        rendered_image, _, _ = rasterization(
            means=mu,
            quats=fr,
            scales=s,
            opacities=o.squeeze(-1),
            colors=colors_precomp_pc,  # paint the camera-space 3D coordinate instead of RGB
            backgrounds=torch.zeros(3, device="cuda")[None],
            viewmats=viewmat[None],
            Ks=K_src[None],
            width=W,
            height=H,
            packed=False,
            render_mode="RGB",  # RGB mode passes `colors` straight through, even without SH
        )

        dst_xyz_map = rendered_image.squeeze(0)
        vis_map = dst_xyz_map.detach().cpu().numpy()
        v_min, v_max = vis_map.min(), vis_map.max()
        vis_map_norm = (vis_map - v_min) / (v_max - v_min + 1e-5)
        vis_map_uint8 = (vis_map_norm * 255).astype(np.uint8)
        #imageio.imwrite(os.path.join(debug_img_dir,f"st{_st}_dt{_dt}.png"), vis_map_uint8)
        # ----------------------------------------------------
        # Sample the rendered buffer at the query pixels
        # ----------------------------------------------------
        rounded_src = np.round(_src).astype(int)
        rounded_src[:, 0] = np.clip(rounded_src[:, 0], 0, W - 1)
        rounded_src[:, 1] = np.clip(rounded_src[:, 1], 0, H - 1)
        index = rounded_src[:, 1] * W + rounded_src[:, 0]

        # What comes back is a 3D coordinate in the target camera's frame
        dst_xyz_cam_sampled = dst_xyz_map.reshape(-1, 3)[index]

        dst_uv = project_to_pixels(dst_xyz_cam_sampled, cams[_dt])
        dst_x = (dst_uv[:, :1] + 1.0) / 2.0 * 360.0
        dst_y = (dst_uv[:, 1:] + 480.0 / 360.0) / 2.0 * 360.0
        _dst_pred = torch.cat([dst_x, dst_y], dim=1).cpu().numpy()
        dst_pred.append(_dst_pred)
        vis_img = np.ascontiguousarray(vis_map_uint8)
        threshold = 24.0
        for pt_src, pt_dst, pt_gt in zip(_src,_dst_pred, _dst_gt):
            x1, y1, x2, y2 = int(pt_src[0]), int(pt_src[1]), int(pt_dst[0]), int(pt_dst[1])
            x_g, y_g = int(pt_gt[0]), int(pt_gt[1])
            dist = np.sqrt((x2 - x_g)**2 + (y2 - y_g)**2)
            is_correct = dist < threshold
            cv.arrowedLine(vis_img, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=1,tipLength=0.2)
            #cv.line(vis_img, (x2, y2), (x_g, y_g), color=(255, 255, 0), thickness=1)
            cv.circle(vis_img, (x1, y1), 3, color=(255, 0, 0), thickness=-1)
            #cv.circle(vis_img, (x2, y2), 3, color=(0, 255, 0), thickness=-1)
            cv.circle(vis_img, (x_g, y_g), 4, color=(0, 0, 255), thickness=-1)
            cv.circle(vis_img, (x_g, y_g), int(threshold), color=(255, 255, 255), thickness=1)
            if is_correct:
                cv.line(vis_img, (x2, y2), (x_g, y_g), color=(0, 255, 255), thickness=1)
                cv.circle(vis_img, (x2, y2), 4, color=(0, 255, 0), thickness=-1)
            else:
                cv.line(vis_img, (x2, y2), (x_g, y_g), color=(0, 0, 255), thickness=2)
                cross_size = 4
                cv.line(vis_img, (x2-cross_size, y2-cross_size), (x2 + cross_size, y2+cross_size), color=(0, 0, 255), thickness=2)
                cv.line(vis_img, (x2 + cross_size, y2 - cross_size), (x2 - cross_size, y2 + cross_size),color=(0, 0, 255), thickness=2)
        imageio.imwrite(os.path.join(debug_img_dir,f"st{_st}_dt{_dt}.png"), vis_img)
    pck005, _ = eval_pck(dst_gt, dst_pred, img_wh, ratio)
    print(f"PCK@0.05: {pck005}")
    if save_fn is not None:
        metrics_file_path = os.path.join(save_fn, "tto_metrics.txt")
        # Append, so repeated runs accumulate
        with open(metrics_file_path, "a") as fp:
            fp.write(f"PCK@0.05: {pck005:.10f}\n")

    return pck005


def project_to_pixels(xyz_cam, cam, th=1e-5):
    """
    Project camera-space points (xyz_cam) to 2D pixel coordinates.
    Keeps the guards of the original __project__, without its hard-coded resolution.
    """
    xy = xyz_cam[..., :2]
    z = xyz_cam[..., 2:]

    # Guard against division by zero and points behind the camera
    z_close_mask = torch.abs(z) < th
    if z_close_mask.any():
        z_close_mask = z_close_mask.float()
        z = z * (1 - z_close_mask) + (1.0 * th) * z_close_mask
        assert not (abs(z) < th).any()
    # Intrinsics
    if hasattr(cam, 'metadata') and cam.metadata is not None:
        fx = float(cam.metadata.scale_factor_x)
        fy = float(cam.metadata.scale_factor_y)
        cx = float(cam.metadata.principal_point_x)
        cy = float(cam.metadata.principal_point_y)
    elif hasattr(cam, 'K') and cam.K is not None:
        fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
        cx, cy = float(cam.K[0, 2]), float(cam.K[1, 2])
    else:
        fx, fy = float(cam.focal), float(cam.focal)
        cx, cy = float(cam.image_width) / 2.0, float(cam.image_height) / 2.0
    _, H, W =cam.depth.shape
    L = min(H, W)
    rel_focal = torch.tensor([fx * 2.0/L, fy * 2.0/L],device="cuda")
    cxcy_ratio = torch.tensor([cx/W,cy/H],device="cuda") * 2.0 - 1.0
    # Project straight to pixel coordinates

    return (xy * rel_focal[None] / z) + cxcy_ratio[None,:]





