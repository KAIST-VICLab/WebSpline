#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import logging

import numpy as np
import random
import os, sys
import cv2
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, BinaryDiceLoss, get_anchor_loss, get_smoothness_loss
from utils.depth_loss_utils import loss_depth_smoothness
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, deformation
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from dycheck_metrics import compute_psnr, compute_ssim, compute_lpips
import torch.nn as nn
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from time import time as get_time
import copy
from PIL import Image
import torch.nn.functional as F
from scene import bundle_adjustment
from utils.graphics_utils import (
    BasicPointCloud, 
    pts2pixel, 
    project_gaussians_forward, 
    interp_traj_by_visibility, 
    _refine_vis_by_3d_jumps,
    get_valid_3d_points_with_class)
from utils_poses.align_traj import align_ate_c2b_use_a2b
from utils_poses.comp_ate import compute_rpe, compute_ATE
from scipy.spatial.transform import Rotation as sci_Rot
from main_utils import *


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.ops import ball_query, knn_points, knn_gather

from scene.cameras import Camera
import torch.distributions

from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)
from gaussian_renderer import interpolate_cubic_hermite
from typing import Optional
from pytorch3d.transforms import axis_angle_to_matrix
to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)
import lpips
lpips_model = lpips.LPIPS(net='vgg', spatial=True).to("cuda")
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from lib_mosca.mosca import MoSca
from lib_mosca.dynamic_solver import geometry_scf_init, line_segment_init


def get_pixels(image_size_x, image_size_y, use_center = None):
    """Return the pixel at center or corner."""
    xx, yy = np.meshgrid(
        np.arange(image_size_x, dtype=np.float32),
        np.arange(image_size_y, dtype=np.float32),
    )
    offset = 0.5 if use_center else 0
    return np.stack([xx, yy], axis=-1) + offset


def build_knn_graph_indices(points_xyz: torch.Tensor, k: int) -> torch.Tensor:
    n_points = points_xyz.shape[0]
    if n_points <= 1 or k <= 0:
        return torch.empty((n_points, 0), dtype=torch.long, device=points_xyz.device)

    k_eff = min(k, n_points - 1)
    knn = knn_points(points_xyz[None], points_xyz[None], K=k_eff + 1)
    # Exclude self-neighbor at index 0.
    return knn.idx[0, :, 1:]


def compute_control_arap_losses(
    control_xyz: torch.Tensor,
    knn_idx: torch.Tensor,
    temporal_diff_shift=(1, 2),
    temporal_diff_weight=(0.75, 0.25),
    max_time_samples=16,   # Cap on how many frames are evaluated at once
):
    if control_xyz.ndim != 3:
        raise ValueError("control_xyz must be [N, C, 3]")
    if len(temporal_diff_shift) != len(temporal_diff_weight):
        raise ValueError("temporal_diff_shift and temporal_diff_weight size mismatch")

    n_points, n_ctrl, _ = control_xyz.shape
    if n_points == 0 or n_ctrl <= 1 or knn_idx.numel() == 0:
        zero = control_xyz.new_zeros(())
        return zero, zero

    # Random sampling along the time axis
    # Take one random contiguous window instead of every frame
    max_shift = max(temporal_diff_shift) if temporal_diff_shift else 0
    time_window = max(max_time_samples, max_shift + 1)
    
    if n_ctrl > time_window:
        # Use Beta(0.5, 0.5) distribution for sampling start_t to favor edges
        beta_dist = torch.distributions.Beta(torch.tensor([0.5], device=control_xyz.device), torch.tensor([0.5], device=control_xyz.device))
        rand_val = beta_dist.sample().item()
        start_t = int(round(rand_val * (n_ctrl - time_window)))
        end_t = start_t + time_window
        
        # Shrink only the time dimension to [N, time_window, 3]; the node count is unchanged
        control_xyz = control_xyz[:, start_t:end_t, :]
        n_ctrl = time_window  # number of frames in the window

    # Build the neighbourhood over all nodes, without subsampling them
    neighbor_ctrl = control_xyz[knn_idx]             # [N, K, T_sampled, 3]
    center_ctrl = control_xyz[:, None, :, :]         # [N, 1, T_sampled, 3]

    # ARAP terms, over the sampled window
    rel = neighbor_ctrl - center_ctrl
    rel_len = torch.norm(rel, dim=-1)

    loss_coord = control_xyz.new_zeros(())
    loss_len = control_xyz.new_zeros(())

    for shift, weight in zip(temporal_diff_shift, temporal_diff_weight):
        if shift <= 0 or n_ctrl <= shift:
            continue
        d_rel = rel[:, :, shift:, :] - rel[:, :, :-shift, :]
        d_len = rel_len[:, :, shift:] - rel_len[:, :, :-shift]
        loss_coord = loss_coord + float(weight) * d_rel.abs().mean()
        loss_len = loss_len + float(weight) * d_len.abs().mean()

    return loss_coord, loss_len


# Per-dataset initialisation settings.
#
# These used to be inline `if dataset == "iphone" else ...` ternaries scattered
# through scene_initialization, which is how the comparison drifted out of sync
# with the string Scene actually stores (it was "iPhone", so every one of those
# ternaries silently took the non-iPhone branch).  Keeping them in one table
# means a new dataset is one entry, and the keys are checked at lookup.
#
# Only settings that genuinely differ per dataset belong here; anything derived
# from sequence length is computed from the data instead.
DATASET_PRESETS = {
    "iphone": dict(
        # iPhone poses come from COLMAP and benefit from refinement; DAVIS poses
        # come from MegaSaM and are used as-is.
        run_bundle_adjustment=True,
        # MoSca scaffold geometry.  The iPhone sequences are long (hundreds of
        # frames) and metric, so the node spacing is set in metres and the
        # multi-level ARAP spans a wide range of temporal offsets.
        scaffold_spatial_unit=0.02,
        scaffold_sigma_init_ratio=5.0,
        scaffold_sigma_max_ratio=10.0,
        mlevel_list=[1, 7, 15],
        mlevel_k_list=[16, 8, 8],
        mlevel_w_list=[0.4, 0.3, 0.3],
        temporal_diff_shift=[2, 8, 16],
        temporal_diff_weight=[0.6, 0.4, 0.3],
    ),
    "nvidia": dict(
        run_bundle_adjustment=True,
        # The RoDynRF clips are ~12 frames, so temporal offsets of 7 and 15 would
        # reach past the end of the sequence: the level lists are shortened and the
        # node support widened to compensate for the sparser trajectories.
        scaffold_spatial_unit=0.01,
        scaffold_sigma_init_ratio=7.0,
        scaffold_sigma_max_ratio=20.0,
        mlevel_list=[1, 4],
        mlevel_k_list=[8, 8],
        mlevel_w_list=[0.4, 0.3],
        temporal_diff_shift=[2, 4],
        temporal_diff_weight=[0.6, 0.4],
    ),
    "davis": dict(
        # MegaSaM poses, nothing to bundle-adjust against.
        run_bundle_adjustment=False,
        scaffold_spatial_unit=0.02,
        scaffold_sigma_init_ratio=5.0,
        scaffold_sigma_max_ratio=10.0,
        mlevel_list=[1, 7, 15],
        mlevel_k_list=[16, 8, 8],
        mlevel_w_list=[0.4, 0.3, 0.3],
        temporal_diff_shift=[2, 8, 16],
        temporal_diff_weight=[0.6, 0.4, 0.3],
    ),
}


def get_dataset_preset(dataset_type):
    """Initialisation settings for a dataset type, falling back to the iPhone ones.

    Unknown types (colmap, blender, ...) have never been tuned for this pipeline,
    so they get the iPhone settings and a warning rather than a crash.
    """
    if dataset_type in DATASET_PRESETS:
        return DATASET_PRESETS[dataset_type]
    print(f"[!] no preset for dataset type {dataset_type!r}; using the iphone settings")
    return DATASET_PRESETS["iphone"]


def scene_initialization(opt, scene, source_path):
    preset = get_dataset_preset(scene.dataset_type)
    viewpoint_stack = None
    train_cams = scene.getTrainCameras()
    viewpoint_stack = [i for i in train_cams]  # Large CPU usage
    
    with torch.no_grad():
        # Return point cloud at IDX
        points_list, colors_list = [], []
        stat_time, stat_color, stat_point = [], [], []
        total_frames = len(viewpoint_stack)
        # ~10% of the frames, but a short sequence (NVIDIA clips are ~12 frames)
        # would get one or two keyframes, which is not enough to seed the dynamic
        # point cloud, so floor it at 3.
        if total_frames < 30:
            num_keyframes = max(3, total_frames // 4)
        else:
            num_keyframes = int(total_frames * 0.1)
        kf_interval = max(1, total_frames // num_keyframes) # stride, so the keyframes are spread evenly
        kf_indices = torch.linspace(0,total_frames-1,steps=num_keyframes).long().tolist()
        kf_indices = sorted(list(set(kf_indices)))
        keyframe_dyn_data = [] # dynamic pixels collected from each keyframe
        # depths, extrinsics = [], []
        for IDX in range(len(viewpoint_stack)):
            image_tensor = viewpoint_stack[IDX].original_image[None].cuda()
            # B, C, H, W = image_tensor.shape

            _, _, H, W = image_tensor.shape

            depth = viewpoint_stack[IDX].depth[None].cuda()
            depth_mask = viewpoint_stack[IDX].depth_mask.cuda()
            depth_mask = depth_mask.squeeze(0).cpu().numpy().reshape(-1)

            pred_R = torch.from_numpy(viewpoint_stack[IDX].R.T[None]).cuda()
            pred_T = torch.from_numpy(viewpoint_stack[IDX].T[None]).cuda()

            K_tensor = torch.zeros(1, 3, 3).type_as(image_tensor)
            K_tensor[:, 0, 0] = float(viewpoint_stack[IDX].focal)
            K_tensor[:, 1, 1] = float(viewpoint_stack[IDX].focal)
            K_tensor[:, 0, 2] = float(viewpoint_stack[IDX].metadata.principal_point_x)
            K_tensor[:, 1, 2] = float(viewpoint_stack[IDX].metadata.principal_point_y)
            K_tensor[:, 2, 2] = float(1)
            w2c_target = torch.cat((pred_R, pred_T[:, :, None]), -1)

            points = deformation.points_from_DRTK(depth, w2c_target, K_tensor)
            points = torch.permute(points, (0, 2, 1))  # B, N, 3

            # Make point cloud
            colors = torch.permute(image_tensor, (0, 2, 3, 1))  # B, H, W, 3

            # error init
            colors_list.append(colors[0].detach().cpu().numpy())
            points_list.append(points[0].view(H, W, 3).detach().cpu().numpy())
            colors = colors[0].view(-1, 3).detach().cpu().numpy()
            points = points[0].view(-1, 3).detach().cpu().numpy()
            coords_2d = get_pixels(W, H)

            motion_mask = viewpoint_stack[IDX].mask.cuda()
            motion_error = motion_mask.squeeze(0).cpu().numpy().reshape(-1)

            stat_mask = (motion_error == 0) & (depth_mask == 1)
            if not stat_mask.any() and opt.static_placeholder_total > 0:
                # Every pixel is masked as moving, so there are no static pixels to
                # seed from.  The pipeline still needs a non-degenerate static model
                # (scene aabb, create_from_pcd, the fine_static stage), so seed a
                # small placeholder from valid-depth pixels and let the dynamic
                # branch carry the actual scene.
                valid = np.flatnonzero(depth_mask == 1)
                if valid.size > 0:
                    n_ph = max(1, opt.static_placeholder_total // max(len(viewpoint_stack), 1))
                    stat_mask = np.zeros_like(stat_mask)
                    stat_mask[np.random.choice(valid, size=min(n_ph, valid.size), replace=False)] = True
            stat_colors = colors[stat_mask, :]
            stat_points = points[stat_mask, :]
            if stat_colors.shape[0] > 0:
                stat_time.append(np.ones((stat_colors.shape[0], 1), dtype=np.float32) * float(viewpoint_stack[IDX].time))
                stat_color.append(stat_colors)
                stat_point.append(stat_points)
                
            if IDX in kf_indices:
                dyn_colors = colors[(motion_error == 1) & (depth_mask == 1), :]
                dyn_points = points[(motion_error == 1) & (depth_mask == 1), :]
                dyn_coords_2d = coords_2d.reshape(-1, 2)[(motion_error == 1) & (depth_mask == 1), :]
                num_extracted = dyn_colors.shape[0]
                
                if num_extracted > 0:
                    # Split the budget across the keyframes, at least one point each
                    pts_per_kf = max(1, num_extracted // num_keyframes)

                    # An all-dynamic mask makes this the whole image; cap it so the
                    # MoSca scaffold stays a tractable size.  0 = no cap.
                    if opt.max_dyn_init_pts > 0:
                        pts_per_kf = min(pts_per_kf, max(1, opt.max_dyn_init_pts // num_keyframes))

                    if num_extracted > pts_per_kf:
                        select_inds = random.sample(range(num_extracted), pts_per_kf)
                        dyn_colors = dyn_colors[select_inds]
                        dyn_points = dyn_points[select_inds]
                        dyn_coords_2d = dyn_coords_2d[select_inds]
                        
                keyframe_dyn_data.append({
                    'idx': IDX,
                    'time': viewpoint_stack[IDX].time,
                    'colors': dyn_colors,
                    'points': dyn_points,
                    'coords_2d': dyn_coords_2d
                })

        stat_time = np.concatenate(stat_time, axis=0)
        stat_color = np.concatenate(stat_color, axis=0)
        stat_point = np.concatenate(stat_point, axis=0)

        total_static_candidates = stat_point.shape[0]
        target_static_n = 100000
        target_static_n = max(1, target_static_n)
        if stat_point.shape[0] > target_static_n:
            select_inds = np.random.choice(stat_point.shape[0], size=target_static_n, replace=False)
            stat_time = stat_time[select_inds]
            stat_color = stat_color[select_inds]
            stat_point = stat_point[select_inds]
        print(f"[*] Static init candidates: {total_static_candidates}, selected: {stat_point.shape[0]} (target: {target_static_n})")

        depth_mask_stack_thw = torch.stack(
            [vp.depth_mask.squeeze(0) for vp in viewpoint_stack], dim=0
        ).cuda()  # [T,H,W]

        # compute dyn tracker
        tracklet = scene.dyn_track.cuda()
        track_visibility = scene.dyn_track_mask.cuda()
        trajectory_3d = get_valid_3d_points_with_class(viewpoint_stack, tracklet,track_visibility)
        track_3d = line_segment_init(track_visibility, trajectory_3d).permute(1,0,2)
        track_3d_mask = track_visibility.permute(1,0)
        N = track_3d.shape[0]
        T = track_3d.shape[1]
        device = track_3d.device

        # Index of the first frame where each track is visible
        first_valid_idx = torch.argmax(track_3d_mask.float(), dim=1) 

        # dyn_time_3d: [N, 1], the time each track first appears
        all_times = torch.tensor([vp.time for vp in viewpoint_stack], device=device)
        dyn_time_3d = all_times[first_valid_idx].unsqueeze(-1)

        # dyn_point_3d: [N, 3], each track's initial 3D position
        batch_indices = torch.arange(N, device=device)
        dyn_point_3d = track_3d[batch_indices, first_valid_idx]

        # dyn_color_3d: [N, 3], each track's initial colour
        dyn_color_list = []
        for i in range(N):
            t_start = first_valid_idx[i].item()
            # tracklet is [T, N, 2]
            u, v = tracklet[t_start, i]
            
            view = viewpoint_stack[t_start]
            u_idx = u.long().clamp(0, view.image_width - 1)
            v_idx = v.long().clamp(0, view.image_height - 1)
            
            # Read the colour at that pixel from original_image [3, H, W]
            color = view.original_image[:, v_idx, u_idx]
            dyn_color_list.append(color)

        dyn_color_3d = torch.stack(dyn_color_list, dim=0)
        chunk = 1000
        dyn_tracjectory = []
        dyn_confidences = []
        dyn_keyframes = []
        
        dyn_color_list = []
        dyn_point_list = []
        dyn_time_list = []
        dyn_mask_list = []
        
        # [T, 3, H, W] on the GPU, so grid_sample below does not re-upload each call
        points_list = torch.from_numpy(np.stack(points_list, axis=0)).permute(0, 3, 1, 2).contiguous().cuda()
        colors_list = torch.from_numpy(np.stack(colors_list, axis=0)).permute(0, 3, 1, 2).contiguous().cuda()

        # Zero out the points whose depth_mask is 0
        dm_thw = depth_mask_stack_thw
        if dm_thw.dtype != torch.float32:
            dm_thw = dm_thw.float()
        if dm_thw.max() > 1.0:
            dm_thw = (dm_thw > 0).float()
        dm_t1hw = dm_thw[:, None]  # [T,1,H,W]

        
        for kf_data in keyframe_dyn_data:
            k_idx = kf_data['idx']
            dyn_coord_2d = kf_data['coords_2d']
            
            # Match against the track positions at the current keyframe k_idx, not at frame 0
            start_tracklet = tracklet[k_idx]
            for i in range(0, dyn_coord_2d.shape[0], chunk):
                dyn_tracklet_index = torch.square(torch.from_numpy(dyn_coord_2d[i:i+chunk, None]).cuda() - start_tracklet[None]).sum(-1).argmin(-1)
                dyn_tracklet = torch.gather(tracklet[None].expand(dyn_coord_2d[i:i+chunk].shape[0], -1, -1, -1), 2, dyn_tracklet_index[:, None, None, None].expand(-1, tracklet.shape[0], -1, 2)).squeeze(2) # N_dyn_pts, N_time, 2
                dyn_tracklet = dyn_tracklet.permute(1, 0, 2)[:, None]
                dyn_tracklet[..., 0] /= W
                dyn_tracklet[..., 1] /= H
                norm_dyn_tracklet = dyn_tracklet * 2 - 1.0 # norm to -1, 1
                
                # Sample the points (points_list is already masked by depth_mask)
                traj_chunk = torch.nn.functional.grid_sample(
                    points_list,                 # already on the GPU
                    norm_dyn_tracklet,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                ).squeeze(-2).permute(2, 0, 1)  # [C,T,3]

                dm_vals = torch.nn.functional.grid_sample(
                    dm_t1hw,                      # [T,1,H,W]
                    norm_dyn_tracklet,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )  # [T,1,1,C]
                valid_tc = (dm_vals.squeeze(1).squeeze(1) > 0.5)  # [T,C] bool

                # AND with the track confidence to get the final visibility
                vis_tc = track_visibility[:, dyn_tracklet_index].to(traj_chunk.device).bool()  # [T,C]
                vis_tc = vis_tc & valid_tc
                vis_ct = vis_tc.permute(1, 0).contiguous()  # [C,T]
                dyn_mask_list.append(vis_ct)
                
                # Drop abrupt 3D jumps, then let the interpolation fill them back in
                vis_ct = _refine_vis_by_3d_jumps(
                    traj_chunk, vis_ct,
                    mad_k=6.0,
                    abs_step_max=None,  # strongest when set to an absolute value in scene units
                )

                # Zero the invalid entries so they cannot leak into the trajectory

                # Interpolate across the gaps where visibility is 0
                traj_chunk_filled = interp_traj_by_visibility(traj_chunk, vis_ct)

                dyn_tracjectory.append(traj_chunk_filled)
                dyn_keyframes.append(torch.ones(traj_chunk.shape[0], dtype=torch.long) * k_idx)
                
                chunk_len = traj_chunk_filled.shape[0]
                dyn_color_list.append(kf_data['colors'][i:i+chunk])
                dyn_point_list.append(kf_data['points'][i:i+chunk])
                dyn_time_list.append(torch.ones(chunk_len, 1) * kf_data['time'])
                dyn_mask = torch.cat(dyn_mask_list, dim=0)
                
        dyn_tracjectory = torch.cat(dyn_tracjectory, dim=0) # N_pts, N_times, 3
        dyn_keyframes = torch.cat(dyn_keyframes, dim=0) # N_pts
        
        dyn_color = np.concatenate(dyn_color_list, axis=0)
        dyn_point = np.concatenate(dyn_point_list, axis=0)
        dyn_time = torch.cat(dyn_time_list, dim=0)
        if dyn_tracjectory.shape[0] > args.resample_number:
            select_inds = random.sample(range(dyn_tracjectory.shape[0]), args.resample_number)
            dyn_tracjectory = dyn_tracjectory[select_inds]
            dyn_color = dyn_color[select_inds]
            dyn_point = dyn_point[select_inds]
            dyn_time = dyn_time[select_inds]
            dyn_mask = dyn_mask[select_inds]
    del points_list
    del colors_list
    del depth_mask_stack_thw
    del dm_thw
    del dm_t1hw
    torch.cuda.empty_cache()

    scaffold: MoSca = MoSca(
        node_xyz=dyn_tracjectory.permute(1, 0, 2).contiguous().detach().clone(),
        node_certain=dyn_mask.permute(1, 0).contiguous().detach().clone(),
        spatial_unit_factor=1.0,
        spatial_unit_hard_set=preset["scaffold_spatial_unit"],  # ! SET NEGATIVE IF WANT TO USE AUTO
        sigma_init_ratio=preset["scaffold_sigma_init_ratio"],
        sigma_max_ratio=preset["scaffold_sigma_max_ratio"],
        topo_dist_top_k=3,
        topo_th_ratio=5.0,
        topo_sample_T=100,
        skinning_k=16,
        skinning_method="dqb",
        mlevel_list=preset["mlevel_list"],
        mlevel_k_list=preset["mlevel_k_list"],
        mlevel_w_list=preset["mlevel_w_list"],
        mlevel_detach_nn_flag=True,  # ! this should be False but due to the old code behavior, set default to True to align with the submission version.
        mlevel_detach_self_flag=False,
        w_corr_maintain_sum_flag=False,
    )
    scaffold.compute_rotation_from_xyz()
    sampled_inds = scaffold.resample_node(resample_factor=1.0, use_mask=True)
    node_rgb = torch.from_numpy(dyn_color).float().to('cuda')[sampled_inds]

    logging.info(
        f"MoSca: get scaffold with M={scaffold.M} and unit={scaffold.spatial_unit}"
    )

    logging.info("*" * 20 + "MoSca Geo" + "*" * 20)
    # * Optimize the curve with ARAP
    
    scaffold = geometry_scf_init(
        scf=scaffold,
        lr_q=0.03,
        lr_p=0.03,
        lr_sig=0.0,
        total_steps=2000,
        max_time_window=len(viewpoint_stack) + 1,
        temporal_diff_shift=preset["temporal_diff_shift"],
        temporal_diff_weight=preset["temporal_diff_weight"],
        lambda_local_coord=1.0,
        lambda_metric_len=1.0,
        lambda_xyz_acc=0.3,
        lambda_q_acc=0.1,
        lambda_xyz_vel=0.3,
        lambda_q_vel=0.1,
        mlevel_resample_steps=100,
        update_full_topo=False,
        use_mask_topo=True,
        update_all_topo_steps=[],
        reline_steps=[],
        decay_start=500,
        decay_factor=30.0,
        viewpoint_stack=viewpoint_stack,
    )

    sampled_inds_np = sampled_inds.cpu().numpy()
    final_stat_point = stat_point
    final_stat_color = stat_color
    final_stat_time = stat_time

    final_dyn_point = dyn_point[sampled_inds_np, :]
    final_dyn_color = dyn_color[sampled_inds_np, :]
    final_dyn_time = dyn_time[sampled_inds.cpu(), :]
        
    # (T, N, 3) -> (N, T, 3)
    final_trajectory = scaffold._node_xyz.data.permute(1, 0, 2).contiguous()
    stat_pc = BasicPointCloud(colors=final_stat_color, points=final_stat_point, normals=None, times=final_stat_time)
    dyn_pc = BasicPointCloud(colors=final_dyn_color, points=final_dyn_point, normals=None, times=final_dyn_time)
    
    return stat_pc, dyn_pc, final_trajectory.cuda(), scaffold


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, render_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
                         dyn_gaussians, stat_gaussians, scene, stage, tb_writer, train_iter, timer, drop=False, check_seed=False, scaffold=None):

    BEST_PSNR, BEST_ITER = 0, 0
    first_iter = 0
    dyn_gaussians.training_setup(opt, stage=stage)
    stat_gaussians.training_setup(opt, stage=stage, is_static=True)

    if scaffold is not None:
        if stage == "fine":
            dyn_gaussians.update_parent_topology(scaffold)
        decay_start = 3000   # hold the initial step size for the first 3000 steps
        decay_factor = 100   # decay the learning rate to 1/100 of its initial value
        lr_p = 0.003         # Position LR
        lr_q = 0.003         # Quaternion(Rotation) LR

        scaffold_optimizer = torch.optim.Adam(
            scaffold.get_optimizable_list(lr_np=lr_p, lr_nq=lr_q, lr_nsig=0.0)
        )

        scaffold_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            scaffold_optimizer,
            (train_iter - decay_start),
            eta_min=min(lr_p, lr_q) / decay_factor,
        )

    if stage == "fine":
        bg_color = [1, 1, 1, -10] if dataset.white_background else [0, 0, 0, -10]
    else:
        bg_color = [1, 1, 1, 0] if dataset.white_background else [0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    viewpoint_stack_ids = []
    ema_loss_for_log_photo = 0.0
    ema_loss_for_log_reg = 0.0
    ema_psnr_for_log = 0.0

    lambda_ctrl_arap_coord = float(getattr(opt, "lambda_ctrl_arap_coord", 0.1))
    lambda_ctrl_arap_len = float(getattr(opt, "lambda_ctrl_arap_len", 0.1))
    ctrl_arap_knn_k = int(getattr(opt, "ctrl_arap_knn_k", 8))
    ctrl_arap_shift = (1, 2)
    ctrl_arap_weight = (0.75, 0.25)
    ctrl_knn_idx = None
    ctrl_knn_size = -1

    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    my_test_cams = [i for i in test_cams] # Large CPU usage
    viewpoint_stack = [i for i in train_cams] # Large CPU usage
    
    print(f"SSIM: {opt.lambda_dssim}")

    batch_size = opt.batch_size
    print("data loading done")

    mask_dice_loss = BinaryDiceLoss(from_logits=False)


    for iteration in range(first_iter, final_iter + 1):
        iter_start.record()
        dyn_gaussians.update_learning_rate(iteration)
        stat_gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and iteration > 2000:
            dyn_gaussians.oneupSHdegree()
            stat_gaussians.oneupSHdegree()

        # Store the imgs/error/depths/disps/normals to visualize in this dict (use detach())
        
        # Built once outside the loop: the full list of ids
        all_ids = list(range(len(viewpoint_stack)))

        everyN_ids = all_ids.copy()

        # Reset viewpoint_stack_ids
        if len(everyN_ids) == 0:
            everyN_ids = all_ids  # safety check

        # Sample the viewpoint cameras
        viewpoint_cams = []
        
        idx = 0
        while idx < batch_size:
            # Refill once the id pool runs out
            if not viewpoint_stack_ids:
                viewpoint_stack_ids = everyN_ids.copy()
                if stage == "fine":
                    viewpoint_stack_ids += [0, 0, 0, 0]  # keep the duplicate-handling behaviour

            pick = randint(0, len(viewpoint_stack_ids) - 1)
            cam_id = viewpoint_stack_ids.pop(pick)
            viewpoint_cams.append(viewpoint_stack[cam_id])
            idx += 1


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        s_images = []
        gt_images = []

        pred_normals = []
        gt_normals = []
        gt_depths = []
        gt_depth_masks = []

        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        depth_list = []
        alpha_list = []

        motion_masks = []
        d_alphas = []
        depth_masks = []

        s_alphas = []

        for n_batch in range(len(viewpoint_cams)):
            gt_image = viewpoint_cams[n_batch].original_image.cuda()
            gt_images.append(gt_image.unsqueeze(0))
            if viewpoint_cams[n_batch].a_chann is not None:
                alpha_list.append(viewpoint_cams[n_batch].a_chann[None].cuda())

            gt_normals.append(viewpoint_cams[n_batch].normal[None].cuda())
            gt_depths.append(viewpoint_cams[n_batch].depth[None].cuda())
            gt_depth_masks.append(viewpoint_cams[n_batch].depth_mask[None].cuda())

        if len(alpha_list) > 0:
            alpha_tensor = torch.cat(alpha_list, 0)
        else:
            alpha_tensor = 1
        gt_image_tensor = torch.cat(gt_images, 0)
        B, C, H, W = gt_image_tensor.shape
        gt_normal_tensor = torch.cat(gt_normals, 0)
        gt_depth_tensor = torch.cat(gt_depths, 0)
        gt_depth_mask_tensor = torch.cat(gt_depth_masks, 0)

        no_stat_gs = stat_gaussians.get_xyz.shape[0]
        no_dyn_gs = dyn_gaussians.get_xyz.shape[0]

        for n_batch, viewpoint_cam in enumerate(viewpoint_cams):
            camera_metadata = viewpoint_cam.metadata
            render_pkg = render(viewpoint_cam, stat_gaussians, dyn_gaussians, background, stage=stage, get_static=True, get_dynamic=True)

            s_images.append(render_pkg["s_render"].unsqueeze(0))
            pred_image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
            visibility_filter, radii = render_pkg["visibility_filter"], render_pkg["radii"]
            depth_list.append(render_pkg["depth"].unsqueeze(0))

            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

            images.append(pred_image.unsqueeze(0))

            pred_normal = get_normals(render_pkg["depth"] + 1e-6, camera_metadata)
            pred_normals.append(pred_normal)
            motion_masks.append(viewpoint_cam.mask.unsqueeze(0))

            if stage == "fine":
                d_alphas.append(render_pkg["d_alpha"].unsqueeze(0))

            depth_masks.append(viewpoint_cam.depth_mask.unsqueeze(0))
            s_alphas.append(render_pkg["s_alpha"].unsqueeze(0))

            if torch.isnan(pred_normal).any():
                print("NaN found in pred normal!")

        loss = 0
        radii = torch.cat(radii_list, 0).max(dim=0).values
        stat_radii = radii[:no_stat_gs]
        dyn_radii = radii[no_stat_gs:]

        visibility_filter = torch.cat(visibility_filter_list, 0).any(dim=0)
        stat_visibility_filter = visibility_filter[:no_stat_gs]
        dyn_visibility_filter = visibility_filter[no_stat_gs:]

        s_image_tensor = torch.cat(s_images, 0)
        image_tensor = torch.cat(images, 0)
        depth_tensor = torch.cat(depth_list, 0)

        normal_tensor = torch.cat(pred_normals, 0)
        motion_mask_tensor = torch.cat(motion_masks, 0)
        s_alpha_tensor = torch.cat(s_alphas, 0)
        depth_mask_tensor = torch.cat(depth_masks, 0)

        if stage == "fine":
            d_alpha_tensor = torch.cat(d_alphas, 0)

        # Main losses (L1 and SSIM) for GS densification
        if stage == "fine":
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
            psnr_ = psnr(image_tensor, gt_image_tensor).detach().mean().double()

            ssim_loss = 0
            if opt.lambda_dssim != 0:
                ssim_loss = 1.0 - ssim(image_tensor * alpha_tensor, gt_image_tensor * alpha_tensor)
        else:
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :], mask=1-motion_mask_tensor)
            psnr_ = psnr(image_tensor * (1-motion_mask_tensor), gt_image_tensor * (1-motion_mask_tensor)).detach().mean().double()
            ssim_loss = 0

        photo_loss = Ll1 + opt.lambda_dssim * ssim_loss
        photo_loss.backward(retain_graph=True)

        # Split static and dynamic gradients (we know their indices because of cat in render)
        if stage == "fine_static":
            stat_viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor.squeeze(0)[0:no_stat_gs])
            for grad_idx in range(0, len(viewspace_point_tensor_list)):
                stat_viewspace_point_tensor_grad += viewspace_point_tensor_list[grad_idx].grad.squeeze(0)[0:no_stat_gs]
            stat_viewspace_point_tensor_grad = stat_viewspace_point_tensor_grad / B
            stat_viewspace_point_tensor_grad[:, 0] *= W * 0.5
            stat_viewspace_point_tensor_grad[:, 1] *= H * 0.5

        if stage == "fine":
            dyn_viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor.squeeze(0)[no_stat_gs:no_stat_gs+no_dyn_gs])
            for grad_idx in range(0, len(viewspace_point_tensor_list)):
                dyn_viewspace_point_tensor_grad += viewspace_point_tensor_list[grad_idx].grad.squeeze(0)[no_stat_gs:no_stat_gs + no_dyn_gs]
            dyn_viewspace_point_tensor_grad = dyn_viewspace_point_tensor_grad / B
            dyn_viewspace_point_tensor_grad[:, 0] *= W * 0.5
            dyn_viewspace_point_tensor_grad[:, 1] *= H * 0.5

        reg_loss = 0
        if stage == "fine":
            dynamic_l1_loss = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :], mask=motion_mask_tensor)
            reg_loss += 1.0 * dynamic_l1_loss

            mask_loss = mask_dice_loss(d_alpha_tensor, motion_mask_tensor)
            reg_loss += 2 * mask_loss

            depth_loss = l1_loss(depth_tensor, gt_depth_tensor, mask=depth_mask_tensor)
            reg_loss += 1 * depth_loss

            normal_loss = l2_loss(normal_tensor, gt_normal_tensor, mask=depth_mask_tensor)
            reg_loss += 0.1 * normal_loss

            control_arap_coord_loss = torch.zeros((), device=image_tensor.device)
            control_arap_len_loss = torch.zeros((), device=image_tensor.device)
            if dyn_gaussians.control_xyz.shape[0] > 1 and ctrl_arap_knn_k > 0 and scaffold is not None:
                if ctrl_knn_idx is None or ctrl_knn_size != dyn_gaussians.control_xyz.shape[0]:
                    ctrl_knn_idx = build_knn_graph_indices(dyn_gaussians.control_xyz[:,0,:].detach(), ctrl_arap_knn_k)
                    ctrl_knn_size = dyn_gaussians.control_xyz.shape[0]
                if ctrl_knn_idx.shape[1] > 0:
                    control_arap_coord_loss, control_arap_len_loss = compute_control_arap_losses(
                        dyn_gaussians.control_xyz,
                        ctrl_knn_idx,
                        temporal_diff_shift=ctrl_arap_shift,
                        temporal_diff_weight=ctrl_arap_weight,
                    )
                    reg_loss += lambda_ctrl_arap_coord * control_arap_coord_loss
                    reg_loss += lambda_ctrl_arap_len * control_arap_len_loss
            control_p_arap_coord_loss = torch.zeros((), device=image_tensor.device)
            control_p_arap_len_loss = torch.zeros((), device=image_tensor.device)
            if dyn_gaussians.control_xyz.shape[0] > 1 and scaffold is not None and hasattr(dyn_gaussians, 'parent_local_knn_nodes'):
                device_gs = dyn_gaussians.control_xyz.device
                M_nodes = scaffold.M
                N_gs = dyn_gaussians.control_xyz.shape[0]
                parent_idx = dyn_gaussians.parent_node_idx.squeeze().long()
                    
                # Node id -> gaussian id, mapped at random
                rand_perm = torch.randperm(N_gs, device=device_gs)
                shuffled_parents = parent_idx[rand_perm]
                node_to_gs = torch.full((M_nodes,), -1, dtype=torch.long, device=device_gs)
                node_to_gs.scatter_(0, shuffled_parents, rand_perm)
                
                # Do not collapse unmapped entries onto gaussian 0; that silently biases the loss.
                
                # 2. Local ARAP Loss
                gs_local_knn = node_to_gs[dyn_gaussians.parent_local_knn_nodes.long()]

                # Point unmapped entries (-1) at the gaussian itself, so the relative term is zero.
                invalid_mask = (gs_local_knn == -1)
                self_ids = torch.arange(N_gs, device=device_gs).unsqueeze(1).expand_as(gs_local_knn)
                gs_local_knn = torch.where(invalid_mask, self_ids, gs_local_knn)
                
                if gs_local_knn.shape[1] > 0:
                    local_coord_loss, local_len_loss = compute_control_arap_losses(
                        dyn_gaussians.control_xyz, gs_local_knn,
                        temporal_diff_shift=ctrl_arap_shift, temporal_diff_weight=ctrl_arap_weight,
                    )
                    control_p_arap_coord_loss += local_coord_loss
                    control_p_arap_len_loss += local_len_loss

                    reg_loss += lambda_ctrl_arap_coord * control_p_arap_coord_loss
                    reg_loss += lambda_ctrl_arap_len * control_p_arap_len_loss

            loss += reg_loss

        elif stage == "fine_static":
            depth_loss = l1_loss(depth_tensor, gt_depth_tensor, mask=(1.0 - motion_mask_tensor) * depth_mask_tensor)
            reg_loss += 0.1 * depth_loss
            loss += reg_loss

        else:
            raise NotImplementedError
            
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, ending program now.")
            exit()
        iter_end.record()

        if stage == "fine" and scaffold is not None:
            if iteration % 100 == 0:
                scaffold.update_topology(curve_mask=None, verbose=True)
                dyn_gaussians.update_parent_topology(scaffold)

        with torch.no_grad():
            # Progress bar   
            ema_loss_for_log_photo = 0.4 * photo_loss.detach().item() + 0.6 * ema_loss_for_log_photo
            ema_loss_for_log_reg = 0.4 * reg_loss.detach().item() + 0.6 * ema_loss_for_log_reg
            ema_psnr_for_log = 0.4 * psnr_.detach() + 0.6 * ema_psnr_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({#"photo loss": f"{ema_loss_for_log_photo:.{6}f}",
                                        #"reg loss": f"{ema_loss_for_log_reg:.{6}f}",
                                        "psnr": f"{ema_psnr_for_log:.{2}f}",
                                        "Pts (static, dynamic)": f"{no_stat_gs}, {no_dyn_gs}",
                                        "Focal": f"{viewpoint_stack[0].focal}",
                                        "MinCtrl": f"{dyn_gaussians.current_control_num.min().item()}"})
                progress_bar.update(10)
            
            if iteration == opt.iterations:
                progress_bar.close()

        # Log and save
        timer.pause()
        with torch.no_grad():
            if iteration in testing_iterations and stage == "fine":
                test_psnr, cur_iter = training_report(tb_writer, iteration, Ll1, loss, depth_loss, mask_loss, normal_loss, l1_loss, iter_start.elapsed_time(iter_end),
                                testing_iterations, scene, my_test_cams, render, [pipe, background], stage,
                                scene.dataset_type, final_iter)
                
                if test_psnr > BEST_PSNR:
                    BEST_PSNR = test_psnr
                    BEST_ITER = cur_iter
                    scene.save_best_psnr(iteration, stage, scaffold=scaffold)

            if iteration in saving_iterations and stage == "fine":
                scene.save(iteration, stage, scaffold=scaffold)

            timer.start()

        # Optimizer step
        if iteration < opt.iterations:
            stat_gaussians.optimizer.step()
            with torch.no_grad():
                stat_gaussians.apply_scale_clamp()
            stat_gaussians.optimizer.zero_grad(set_to_none=True)

            dyn_gaussians.optimizer.step()
            with torch.no_grad():
                dyn_gaussians.apply_scale_clamp()
            dyn_gaussians.optimizer.zero_grad(set_to_none=True)

            if scaffold is not None and scaffold_optimizer is not None:
                scaffold_optimizer.step()
                scaffold_optimizer.zero_grad(set_to_none=True)
                if iteration > decay_start:
                    scaffold_scheduler.step()
                    
        # Densification
        with torch.no_grad():
            if iteration < opt.densify_until_iter :
                if stage == "fine":
                    dyn_gaussians.max_radii2D[dyn_visibility_filter] = torch.max(dyn_gaussians.max_radii2D[dyn_visibility_filter], dyn_radii[dyn_visibility_filter])
                    dyn_gaussians.add_densification_stats(dyn_viewspace_point_tensor_grad, dyn_visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        dyn_gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        dyn_gaussians.reset_opacity()

                else:
                    stat_gaussians.max_radii2D[stat_visibility_filter] = torch.max(stat_gaussians.max_radii2D[stat_visibility_filter], stat_radii[stat_visibility_filter])
                    stat_gaussians.add_densification_stats(stat_viewspace_point_tensor_grad, stat_visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        stat_gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        stat_gaussians.reset_opacity()
    if stage == "fine":
        scene.save(iteration, stage, scaffold=scaffold)
    
    if stage == "fine_static":
        return stat_gaussians, dyn_gaussians
    else:
        return BEST_PSNR, BEST_ITER


def training(dataset, hyper, opt, pipe, testing_iterations, render_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, expname, check_seed=False):
    tb_writer = prepare_output_and_logger(expname)
    stat_gaussians = GaussianModel(dataset.sh_degree, hyper)
    dyn_gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)  # large CPU usage
    timer.start()
    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras()) # pose network with instance scaling
    preset = get_dataset_preset(scene.dataset_type)
    if preset["run_bundle_adjustment"]:
        bundle_adjustment(opt, scene, dataset.source_path)
    else:
        print(f"[*] {scene.dataset_type}: skipping bundle adjustment (poses are used as given)")
    stat_pc, dyn_pc, dyn_tracjectory, scaffold = scene_initialization(opt, scene, dataset.source_path)

    stat_gaussians.create_from_pcd(pcd=stat_pc, spatial_lr_scale=5, time_line=0)
    dyn_gaussians.create_from_pcd_dynamic(pcd=dyn_pc, spatial_lr_scale=5, time_line=0, dyn_tracjectory=dyn_tracjectory)

    # getNerfppNorm derives cameras_extent from camera positions alone.  When the
    # camera barely translates (DAVIS) while the scene sits around unit depth, that
    # extent comes out far too small -- and densify_and_prune drops every gaussian
    # whose scale exceeds 0.1 * extent.  Allow deriving the extent from the actual
    # geometry, or setting it outright.
    if opt.densify_extent > 0:
        print(f"[*] densify extent: {scene.cameras_extent:.4f} -> {opt.densify_extent:.4f} (override)")
        scene.cameras_extent = opt.densify_extent
    elif opt.densify_extent_from_pcd:
        _pts = np.concatenate([np.asarray(stat_pc.points), np.asarray(dyn_pc.points)], axis=0)
        _r = float(np.linalg.norm(_pts - _pts.mean(0), axis=1).max())
        print(f"[*] densify extent: {scene.cameras_extent:.4f} -> {_r:.4f} (from init point cloud)")
        scene.cameras_extent = _r

    xyz_max = stat_pc.points.max(axis=0)
    xyz_min = stat_pc.points.min(axis=0)
    dyn_gaussians._deformation.deformation_net.set_aabb(xyz_max, xyz_min, ref_type=dyn_gaussians.get_xyz)

    stat_gaussians, dyn_gaussians = scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, render_iterations, saving_iterations,
                                    checkpoint_iterations, checkpoint, debug_from,
                                    dyn_gaussians, stat_gaussians, scene, "fine_static", tb_writer, opt.fine_static_iterations, timer, drop=True, check_seed=check_seed, scaffold=scaffold)
    
    BEST_PSNR, BEST_ITER = scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, render_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            dyn_gaussians, stat_gaussians, scene, "fine", tb_writer, opt.iterations, timer, drop=True, check_seed=check_seed, scaffold=scaffold)


    return BEST_PSNR, BEST_ITER


def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, depth_loss, mask_loss, normal_loss, l1_loss, elapsed, testing_iterations, scene: Scene, test_cams,
                    renderFunc, renderArgs, stage, dataset_type, final_iter):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train/depth_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train/mask_loss', mask_loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

    test_psnr = 0.0
    cur_iter = 0.0

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': test_cams},
                              {'name': 'train', 'cameras': []})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.stat_gaussians, scene.dyn_gaussians, renderArgs[1], stage=stage, get_static=True, get_dynamic=True)
                    image = render_pkg["render"]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/" + config['name'] + "_view_{}/render".
                                                 format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/" + config['name'] + "_view_{}/ground_truth".
                                                     format(viewpoint.image_name), gt_image[None],
                                                     global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    
                    gt = gt_image.permute(1, 2, 0).cpu().numpy()
                    pred = image.permute(1, 2, 0).detach().cpu().numpy()
                    gt_mask = None
                    if hasattr(viewpoint, 'covisible') and viewpoint.covisible is not None:
                        gt_mask = viewpoint.covisible.cpu().numpy().astype(np.float32)
                    psnr_test += compute_psnr(gt, pred, mask=gt_mask).mean()
                    lpips_test += compute_lpips(lpips_model ,gt, pred, mask=gt_mask).mean()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test))

            if config['name'] == 'test':
                test_psnr = psnr_test
                cur_iter = iteration
                
        torch.cuda.empty_cache()
        return test_psnr, cur_iter

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--check_seed", action="store_true")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--resample_number', type=int, default=20000)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[20000 * i for i in range(1000)])
    parser.add_argument("--render_iterations", nargs="+", type=int, default=[20000 * i for i in range(1000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20000 * i for i in range(1000)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # After the config merge, not before: the config is what sets `iterations`,
    # so appending it earlier used the argparse default instead and a config with
    # a non-default iteration count would never checkpoint its final iteration.
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    torch.set_num_threads(4)

    BEST_PSNR, BEST_ITER = training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
            args.render_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.check_seed)

    if args.check_seed:
        with open(os.path.join(args.model_path, "seed.txt"), 'a') as f:
            f.write("BEST PSNR : " + str(BEST_PSNR) + " SEED : " + str(args.seed) + "\n")

    # All done
    print("\nTraining complete.")
    print("BEST PSNR : ", BEST_PSNR)
    print("BEST ITER : ", BEST_ITER)
