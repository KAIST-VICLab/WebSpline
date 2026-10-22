# functions for initialize the missing scaffold
import torch
import logging
import os, sys, os.path as osp
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import imageio
import open3d as o3d
import torch.nn.functional as F
sys.path.append(osp.dirname(osp.abspath(__file__)))

from camera import MonocularCameras
from dynamic_solver_utils import prepare_track_buffers, get_world_points
from mosca import MoSca


def detect_sharp_changes_in_curve(track_mask, curve, max_vel_th, valid_type="and"):
    assert len(track_mask) >= 2, "too short!"
    diff = (curve[:-1] - curve[1:]).norm(dim=-1)  # T-1,N
    if valid_type == "and":
        valid = track_mask[:-1] * track_mask[1:]  # both side are valid
    elif valid_type == "or":
        valid = (track_mask[:-1] + track_mask[1:]) > 0  # one is valid
    to_next_diff = torch.cat([diff, torch.zeros_like(diff[:1])], 0)
    to_prev_diff = torch.cat([torch.zeros_like(diff[:1]), diff], 0)
    to_next_valid = torch.cat([valid, torch.ones_like(valid[:1])], 0)
    to_prev_valid = torch.cat([torch.ones_like(valid[:1]), valid], 0)
    # only when two points are all valid, the vel compute there is meaningful, other wise mask the diff to zero so it won't exceed the th
    to_next_diff = to_next_diff * to_next_valid.float()
    to_prev_diff = to_prev_diff * to_prev_valid.float()
    max_diff = torch.max(to_next_diff, to_prev_diff)
    invalid_mask = max_diff > max_vel_th
    # assert track_mask[invalid_mask].all()
    logging.info(
        f"curve velocity check th={max_vel_th} has {invalid_mask.sum()} ({invalid_mask.sum()/(track_mask.sum()+1e-6)*100.0:.2f}%) invalid slots"
    )
    new_track_mask = track_mask.clone()
    new_track_mask[invalid_mask] = False
    return new_track_mask, invalid_mask


@torch.no_grad()
def line_segment_init(track_mask, point_ref):
    logging.info("Naive Line Segment Init")
    # ! this function is a bad init, but this doesn't matter, later will directly optimize the curve

    tracl_mask_valid_cnt = track_mask.sum(0)
    working_mask = tracl_mask_valid_cnt > 0
    logging.info(f"Line Segment Init, invalid curve cnt={(~working_mask).sum()}")

    working_point_ref = point_ref.detach().clone()[:, working_mask]
    working_track_mask = track_mask[:, working_mask]

    T, N = track_mask.shape
    # point_ref # T,N,3
    # scan the T, for each empty slot, identify the right ends, and compute linear interpolation, if there is only one side, stay at the same position, if two end are empty, assert error, there shouldn't be an empty noodle!
    inverse_muti = torch.Tensor([i + 1 for i in range(T)][::-1]).to(working_point_ref)
    for t in tqdm(range(T)):
        to_fill_mask = ~working_track_mask[t]
        if not to_fill_mask.any():
            continue  # skip this time if everything is filled
        # identify the left and right nearest valid side

        if t == T - 1:  # if right end, use the previous one
            value = working_point_ref[t - 1, to_fill_mask].clone()
        else:
            # identify the right end, the left end must be filled in already
            to_fill_valid_curve = working_track_mask[t + 1 :, to_fill_mask]  # T,M
            # find the left most True slot
            to_fill_valid_curve = (
                to_fill_valid_curve.float() * inverse_muti[t + 1 :, None]
            )
            max_value, max_ind = to_fill_valid_curve.max(dim=0)
            # for no right mask case, use the left
            select_from = working_point_ref[t + 1 :, to_fill_mask]
            valid_right_end = torch.gather(
                select_from, 0, max_ind[None, :, None].expand(-1, -1, 3)
            )[
                0, max_value > 0
            ]  # valid when max_value > 0
            if t == 0:
                assert (
                    len(valid_right_end) == to_fill_valid_curve.shape[1]
                ), "empty noodle!"
                value = valid_right_end
            else:
                # must have a left end
                value = working_point_ref[t - 1, to_fill_mask].clone()
                valid_left_end = value[max_value > 0]
                delta_t = (
                    max_ind[max_value > 0] + 2
                )  # left valid, current, [0] in the max_ind
                delta_x = valid_right_end - valid_left_end
                inc = 1.0 * delta_x / delta_t[:, None]
                value[max_value > 0] = valid_left_end + inc
        working_point_ref[t, to_fill_mask] = value.clone()
    # np.savetxt("./debug/line_segment_init.xyz", point_ref.reshape(-1, 3).cpu().numpy())
    ret = point_ref.clone()
    ret[:, working_mask] = working_point_ref
    return ret.detach().clone()


def get_dynamic_curves(
    s2d,
    cams: MonocularCameras,
    return_all_curves=False,
    # filter of 2D tracks to avoid the fg-bg error track flickering
    refilter_2d_track_flag=True,
    refilter_2d_track_only_mask=False,  # if set true won't remove any curve, just mark the outlier as invalid
    refilter_min_valid_cnt=2,
    refilter_o3d_nb_neighbors=16,
    refilter_o3d_std_ratio=5.0,
    refilter_shaking_th=0.2,
    refilter_remove_shaking_curve=False,  # if true, any curve with shaking will be totally removed
    refilter_spatracker_consistency_th=0.2,
    #
    spatracker_original_curve=False,
    # additional mask, this is for the semantic label consistency mask
    fg_additional_mask=None,
    # spatracker 3D curve also have a choice to use line init
    enforce_line_init=False,
    # subsample t list
    t_list=None,
    # safe cfgs
    min_num_curves=0,
):
    device = s2d.rgb.device
    # * load track
    if return_all_curves:
        track = s2d.track.clone()
        track_mask = s2d.track_mask.clone()
    else:
        track = s2d.track[:, s2d.dynamic_track_mask].clone()
        track_mask = s2d.track_mask[:, s2d.dynamic_track_mask].clone()
    if fg_additional_mask is not None:
        assert fg_additional_mask.shape == track_mask.shape
        track_mask = track_mask * fg_additional_mask

    filter_mask = torch.ones(track.shape[1]).bool().to(device)
    if t_list is None:
        t_list = torch.arange(s2d.T).to(device)  # use all
    else:
        t_list = torch.as_tensor(t_list).to(device)

    track = track[t_list]
    track_mask = track_mask[t_list]
    # ! the subsample may lead to all empty track, which will be filtered out later by the min valid cnt!

    if track.shape[-1] == 3:
        logging.info(f"SpaT mode, direct use 3D Track")
        # spa tracker model
        # manually homo list
        homo_list = __int2homo_coord__(track[..., :2], s2d.H, s2d.W)
        dep_list = track[..., -1]
        node_xyz_spatracker = get_world_points(homo_list, dep_list, cams, t_list)

        # todo: align the spatrack curve to depth
        # 1. get inlier mask by checking the unproject points
        # 2. for inliers, aign the curve depth by interpolating the per-frame valid alignment

        gathered_homo_list, gathered_dep_list, rgb_list = prepare_track_buffers(
            s2d, track[..., :2], track_mask, t_list
        )
        node_xyz_unproject = get_world_points(
            gathered_homo_list, gathered_dep_list, cams, t_list
        )
        # for visible slot, use gathered_curve else use
        if spatracker_original_curve:
            curve_xyz = node_xyz_spatracker
        else:
            mix_mask = track_mask.float()[..., None].expand(-1, -1, 3)
            curve_xyz = node_xyz_unproject * mix_mask + node_xyz_spatracker * (
                1.0 - mix_mask
            )

        # * 3D curve still need to use filter to exclud wrong fetch
        if refilter_2d_track_flag and not return_all_curves:
            inlier_mask = slot_o3d_outlier_identifyication(
                curve_xyz,
                track_mask,
                nb_neighbors=refilter_o3d_nb_neighbors,
                std_ratio=refilter_o3d_std_ratio,
            )
            logging.info(
                f"O3d outlier ratio {(~inlier_mask).float().mean()*100.0:.2f}%"
            )
            track_mask = track_mask * inlier_mask

            inlier_mask = curve_shaking_identification(curve_xyz, refilter_shaking_th)
            logging.info(
                f"shaking outlier ratio {(~inlier_mask).float().mean()*100.0:.2f}%"
            )
            if refilter_remove_shaking_curve:
                has_shaking = track_mask * (~inlier_mask)  # valid but has outlier
                has_shaking = has_shaking.any(0, keepdim=True)
                logging.info(f"{has_shaking.sum()} curves has shaking, remove")
                inlier_mask = (~has_shaking).expand(len(track_mask), -1)
            track_mask = track_mask * inlier_mask

            fetch_spatracker_diff = (node_xyz_unproject - node_xyz_spatracker).norm(
                dim=-1
            )
            spatracker_consistent_mask = (
                fetch_spatracker_diff < refilter_spatracker_consistency_th
            )
            logging.info(
                f"spatracker consistency newly detect {(~spatracker_consistent_mask & track_mask).float().mean()*100.0:.2f}% with th={refilter_spatracker_consistency_th}"
            )
            track_mask = track_mask * spatracker_consistent_mask

            if not refilter_2d_track_only_mask:
                # find valid cnt
                valid_cnt = track_mask.sum(0)
                if filter_mask.sum() > min_num_curves:
                    filter_mask = (valid_cnt >= refilter_min_valid_cnt) & filter_mask
                    logging.info(
                        f"Refiltering 2D tracks, {(~filter_mask).sum()} curves has less than {refilter_min_valid_cnt} valid slots, remove"
                    )
                track = track[:, filter_mask]
                track_mask = track_mask[:, filter_mask]
                # * redo
                homo_list = __int2homo_coord__(track[..., :2], s2d.H, s2d.W)
                dep_list = track[..., -1]
                node_xyz_spatracker = get_world_points(
                    homo_list, dep_list, cams, t_list
                )
                gathered_homo_list, gathered_dep_list, rgb_list = prepare_track_buffers(
                    s2d, track[..., :2], track_mask, t_list
                )
                node_xyz_unproject = get_world_points(
                    gathered_homo_list, gathered_dep_list, cams, t_list
                )
                # for visible slot, use gathered_curve else use
                mix_mask = track_mask.float()[..., None].expand(-1, -1, 3)
                if spatracker_original_curve:
                    curve_xyz = node_xyz_spatracker
                else:
                    curve_xyz = node_xyz_unproject * mix_mask + node_xyz_spatracker * (
                        1.0 - mix_mask
                    )
            else:
                # if use non-original 3D curve, need to re-mix the curve to remove the noise samling
                if not spatracker_original_curve:
                    mix_mask = track_mask.float()[..., None].expand(-1, -1, 3)
                    curve_xyz = node_xyz_unproject * mix_mask + node_xyz_spatracker * (
                        1.0 - mix_mask
                    )
        if enforce_line_init:
            curve_xyz = line_segment_init(track_mask, curve_xyz)

    else:
        logging.info(f"2D track mode, use line segment to fill")
        homo_list, dep_list, rgb_list = prepare_track_buffers(
            s2d, track[..., :2], track_mask, t_list
        )
        curve_xyz = line_segment_init(
            track_mask, get_world_points(homo_list, dep_list, cams, t_list).clone()
        )
    return curve_xyz, track_mask, rgb_list, filter_mask


@torch.no_grad()
def slot_o3d_outlier_identifyication(
    curve_xyz, curve_mask, nb_neighbors=20, std_ratio=2.0
):
    # curve_xyz: T,N,3, tensor

    assert curve_xyz.ndim == 3
    T, N, _ = curve_xyz.shape
    ret_inlier_mask = np.ones((T, N)) < 0  # all false
    for t in tqdm(range(T)):
        if not curve_mask[t].any():
            continue
        fg_mask = curve_mask[t].cpu()
        fg_xyz = curve_xyz[t].cpu().numpy()[fg_mask]
        inlier_mask_buffer = np.zeros(len(fg_xyz)) > 0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fg_xyz)

        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        inlier_ind = np.asarray(ind)
        if len(inlier_ind) > 0:
            inlier_mask_buffer[inlier_ind] = True  # len(fg_xyz)
        _t_inlier_mask = ret_inlier_mask[t]
        _t_inlier_mask[fg_mask] = inlier_mask_buffer
        ret_inlier_mask[t] = _t_inlier_mask
    ret_inlier_mask = torch.from_numpy(ret_inlier_mask).bool().to(curve_xyz.device)
    logging.warning(
        f"O3D outlier has {ret_inlier_mask.sum()/curve_mask.sum()*100:.2f}% inliers ({ret_inlier_mask.sum()} inliers out of {curve_mask.sum()})"
    )
    return ret_inlier_mask


@torch.no_grad()
def curve_shaking_identification(curve_xyz, shacking_th=0.2):
    # the queried curve
    # T,N,3
    ref = (curve_xyz[2:] + curve_xyz[:-2]) / 2.0
    ref = torch.cat([curve_xyz[1:2], ref, curve_xyz[-2:-1]], 0)
    diff = (curve_xyz - ref).norm(dim=-1)
    inlier = diff < shacking_th
    return inlier


def __int2homo_coord__(track_uv, H, W):
    # the short side is [-1,1]
    H, W = float(H), float(W)
    L = min(H, W)
    homo_x = (track_uv[..., 0] + 0.5) / L * 2 - (W / L)
    homo_y = (track_uv[..., 1] + 0.5) / L * 2 - (H / L)
    homo = torch.stack([homo_x, homo_y], -1)
    return homo


def __compute_physical_losses__(
    scf,
    temporal_diff_shift: list,
    temporal_diff_weight: list,
    max_time_window: int,
    reduce="sum",
    square=False,
):
    if scf.T > max_time_window:
        start = torch.randint(0, scf.T - max_time_window + 1, (1,)).item()
        sup_tids = torch.arange(start, start + max_time_window)
    else:
        sup_tids = torch.arange(scf.T)
    sup_tids = sup_tids.to(scf.device)

    # * compute losses from the scaffold
    loss_coord, loss_len = scf.compute_arap_loss(
        tids=sup_tids,
        temporal_diff_shift=temporal_diff_shift,
        temporal_diff_weight=temporal_diff_weight,
        reduce_type=reduce,
        square=square,
    )
    loss_p_vel, loss_q_vel, loss_p_acc, loss_q_acc = scf.compute_vel_acc_loss(
        tids=sup_tids, reduce_type=reduce, square=square
    )
    return loss_coord, loss_len, loss_p_acc, loss_q_acc, loss_p_vel, loss_q_vel


def __compute_physical_losses_sup__(
    scf,
    sup_tids,
    temporal_diff_shift: list,
    temporal_diff_weight: list,
    reduce="sum",
    square=False,
):

    # * compute losses from the scaffold
    loss_coord, loss_len = scf.compute_arap_loss(
        tids=sup_tids,
        temporal_diff_shift=temporal_diff_shift,
        temporal_diff_weight=temporal_diff_weight,
        reduce_type=reduce,
        square=square,
    )
    loss_p_vel, loss_q_vel, loss_p_acc, loss_q_acc = scf.compute_vel_acc_loss(
        tids=sup_tids, reduce_type=reduce, square=square
    )
    return loss_coord, loss_len, loss_p_acc, loss_q_acc, loss_p_vel, loss_q_vel

def geometry_scf_init(
    #
    scf: MoSca,
    mlevel_resample_steps=32,
    #
    lr_q=0.1,
    lr_p=0.1,
    lr_sig=0.03,
    #
    total_steps=1000,
    max_time_window=200,
    # * Basic Phy losses
    temporal_diff_shift=[1],
    temporal_diff_weight=[1.0],
    lambda_local_coord=1.0,
    lambda_metric_len=0.0,
    lambda_xyz_acc=0.0,
    lambda_q_acc=0.1,
    lambda_xyz_vel=0.0,
    lambda_q_vel=0.0,
    # * stablize
    lambda_small_corr=0.0,
    hard_fix_valid=True,
    #
    square_loss_flag=False,
    ################
    # topo
    update_full_topo=False,
    use_mask_topo=True,
    update_all_topo_steps=[],
    reline_steps=[],
    #
    decay_start=400,
    decay_factor=10.0,
    # viz
    # save
    prefix="",
    #
    viewpoint_stack = None,           
    lambda_mask_consist=10,
):
    torch.cuda.empty_cache()

    if square_loss_flag:
        raise RuntimeError(
            "Square loss is not good! set [square_loss_flag] flag to false"
        )
    all_penalty_maps = prepare_all_penalty_maps(viewpoint_stack)
    T_frames = all_penalty_maps.shape[0]

    # * The small change is resp. to the init stage
    solid_mask = scf._node_certain
    solid_xyz = scf._node_xyz.clone().detach()

    optimizer = torch.optim.Adam(
        scf.get_optimizable_list(lr_np=lr_p, lr_nq=lr_q, lr_nsig=lr_sig)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        (total_steps - decay_start),
        eta_min=min(lr_p, lr_q) / decay_factor,
    )

    loss_list, loss_coord_list, loss_len_list = [], [], []
    loss_small_corr_list = []
    loss_p_acc_list, loss_q_acc_list = [], []
    loss_p_vel_list, loss_q_vel_list = [], []
    loss_mask_list = []
    loss_flow_xyz_list, loss_flow_nrm_list = [], []
    metric_flow_error_list, metric_normal_angle_list = [], []

    loss_drag_xyz_list, metric_drag_xyz_list = [], []
    loss_dense_flow_xyz_list, loss_dense_flow_nrm_list = [], []
    metric_dense_flow_error_list, metric_dense_normal_angle_list = [], []

    loss_sk_w_consist_list, metric_sk_w_consist_list = [], []
    loss_dense_sk_w_consist_list, metric_dense_sk_w_consist_list = [], []

    # before start, update topo
    scf.update_topology(curve_mask=solid_mask if use_mask_topo else None)

    logging.info(f"4DSCF-Solver-loop prefix=[{{prefix}}] summary: ")
    logging.info(
        f"total_steps={total_steps}, decay_start={decay_start}, hard_fix_valid_flag={hard_fix_valid}"
    )
    logging.info(f"lr_p={lr_p}, lr_q={lr_q}, lr_sig={lr_sig}")

    for step in tqdm(range(total_steps)):
        if step in reline_steps:
            logging.info(f"Warning, at step={step}, reline the curve")
            new_xyz = line_segment_init(scf._node_certain, scf._node_xyz.detach())
            with torch.no_grad():
                scf._node_xyz.data = new_xyz
            scf.compute_rotation_from_xyz()
        if step in update_all_topo_steps:
            logging.info(
                f"As specified, update full topo at step={step} without any mask"
            )
            scf.update_topology(curve_mask=None, verbose=True)
        elif step % mlevel_resample_steps == 0 and step > 0:
            if update_full_topo:
                scf.update_topology(
                    curve_mask=solid_mask if use_mask_topo else None, verbose=True
                )
            else:
                scf.update_multilevel_arap_topo(verbose=True)

        optimizer.zero_grad()
        progress = step / total_steps
        
        # Anneal the ARAP time window down to a 16-frame floor, then clamp to the
        # sequence: NVIDIA's ~12-frame clips are shorter than that floor, so they
        # keep the full window throughout.
        curr_window = max(16, int(max_time_window * (1.0 - 0.9 * progress)))
        curr_window = min(curr_window, int(max_time_window))
        
        # Shift weight towards the mask term later in the schedule, while keeping ARAP
        curr_lambda_arap = lambda_local_coord * (1.0 - 0.3 * progress)
        curr_lambda_mask = lambda_mask_consist * (1.0)
        curr_lambda_vel = lambda_xyz_acc * (1.0 + 1.0 * progress)
        dist = torch.distributions.Beta(torch.tensor([0.5]), torch.tensor([0.5]))

        # Pick the window
        if scf.T > curr_window:
        # Draw from a U-shaped distribution on [0, 1], favouring both ends
            rand_val = dist.sample().item()
        
        # Scale to the full index range
            max_start = scf.T - curr_window
            start = int(round(rand_val * max_start))
        
        # Clamp to a valid index range
            start = max(0, min(max_start, start))
        
            sup_tids = torch.arange(start, start + curr_window).to(scf.device)
        else:
            sup_tids = torch.arange(scf.T).to(scf.device)
        optimizer.zero_grad()
        loss_coord, loss_len, loss_p_acc, loss_q_acc, loss_p_vel, loss_q_vel = (
            __compute_physical_losses_sup__(
                scf,
                sup_tids,
                temporal_diff_shift,
                temporal_diff_weight,
                square=square_loss_flag,
            )
        )
        loss_mask = torch.tensor(0.0).to(scf.device)
        # Step two frames at a time within the window, for speed
        for t in sup_tids:
            # Fetch the precomputed penalty map, [1, H, W]
            penalty_map = all_penalty_maps[t].cuda()
            if penalty_map.ndim == 2:
                penalty_map.unsqueeze(0).unsqueeze(0)
            elif penalty_map.ndim == 3:
                penalty_map= penalty_map.unsqueeze(0)
            view = viewpoint_stack[t]
            node_xyz = scf._node_xyz[t]
            
            # Project 3D -> 2D, in NDC
            R_wc = torch.from_numpy(view.R).to(scf.device).float()
            t_cw = torch.from_numpy(view.T).to(scf.device).float()
            p_cam = torch.matmul(node_xyz, R_wc) + t_cw
            z = p_cam[:, 2]
            z_safe = torch.clamp(z, min=1e-5)
            
            u_ndc = ((p_cam[:, 0] / z_safe) * view.K[0,0] + view.K[0,2]) / view.image_width * 2.0 - 1.0
            v_ndc = ((p_cam[:, 1] / z_safe) * view.K[1,1] + view.K[1,2]) / view.image_height * 2.0 - 1.0
            
            # Assemble the coordinates for grid_sample
            grid = torch.stack([u_ndc, v_ndc], dim=-1).view(1, 1, -1, 2)
            
            # Sample the penalty at each node position (bilinear)
            # It peaks near the mask boundary and falls off both inside and far outside
            sampled_penalty = F.grid_sample(penalty_map, grid, align_corners=False, padding_mode="zeros").squeeze()
            
            # Only points in front of the camera contribute
            in_view = (z > 0.01) # & (u_ndc.abs() <= 1.0) & (v_ndc.abs() <= 1.0) # deliberately not clamped, so nodes may leave the frame
            
            # Accumulate the penalty
            loss_mask += (sampled_penalty * in_view.float()).sum()

        # Normalise by the window length and node count, then weight
        loss_mask = (loss_mask / (len(sup_tids))) * curr_lambda_mask * 10

        # loss of near original curve
        diff_to_solid_xyz = (scf._node_xyz - solid_xyz.detach()).norm(dim=-1) ** 2
        loss_small_corr = diff_to_solid_xyz[solid_mask].sum()  # ! use sum
        loss = (
            lambda_local_coord * loss_coord
            + lambda_metric_len * loss_len
            + lambda_xyz_acc * loss_p_acc
            + lambda_q_acc * loss_q_acc
            + lambda_small_corr * loss_small_corr
            + lambda_xyz_vel * loss_p_vel
            + lambda_q_vel * loss_q_vel
            + loss_mask
        )
        with torch.no_grad():
            loss_list.append(loss.item())
            loss_coord_list.append(loss_coord.item())
            loss_len_list.append(loss_len.item())
            loss_p_acc_list.append(loss_p_acc.item())
            loss_q_acc_list.append(loss_q_acc.item())
            loss_small_corr_list.append(loss_small_corr.item())
            loss_p_vel_list.append(loss_p_vel.item())
            loss_q_vel_list.append(loss_q_vel.item())
            loss_mask_list.append(loss_mask.item() if isinstance(loss_mask, torch.Tensor) else 0)

        loss.backward()
        if hard_fix_valid:
            scf.mask_xyz_grad(~solid_mask)
        optimizer.step()

        # * control
        if step > decay_start:
            scheduler.step()
        if step % 10 == 0:
            print(f"[Step {step:04d}] loss_coord={lambda_local_coord * loss_coord:.6f}, loss_len={lambda_metric_len * loss_len:.6f}, loss_p_vel={ lambda_xyz_vel * loss_p_vel:.6f}, loss_R_vel={lambda_q_vel * loss_q_vel:.6f}, loss_p_acc={lambda_xyz_acc * loss_p_acc:.6f}, loss_R_acc={lambda_q_acc * loss_q_acc:.6f}, loss_small_corr={loss_small_corr:.6f},loss_mask={loss_mask:.6f} | Win: {curr_window}")
        if step % 50 == 0:
            logging.info(f"step={step}, loss={loss:.6f}")
            msg = f"[{prefix}] loss_coord={loss_coord:.6f}, loss_len={loss_len:.6f}, loss_p_vel={loss_p_vel:.6f}, loss_R_vel={loss_q_vel:.6f}, loss_p_acc={loss_p_acc:.6f}, loss_R_acc={loss_q_acc:.6f}, loss_small_corr={loss_small_corr:.6f},loss_mask={loss_mask:.6f}"
            logging.info(msg)


    return scf


@torch.no_grad()
def __draw_graph__(t, scf: MoSca, line_N=16, level=-1):
    if level < 0:
        dst_ind = scf.topo_knn_ind
        src_ind = torch.arange(scf.M)[:, None].expand(-1, scf.topo_knn_ind.shape[1])
        mask = scf.topo_knn_mask
    else:
        src_ind = scf.multilevel_arap_edge_list[level][:, 0]
        dst_ind = scf.multilevel_arap_edge_list[level][:, 1]
        mask = scf.multilevel_arap_topo_w[level] > 0
    node_mu = scf._node_xyz[t]
    dst_xyz = node_mu[dst_ind]
    src_xyz = node_mu[src_ind]
    line_xyz = __draw_gs_point_line__(src_xyz[mask], dst_xyz[mask], n=line_N).reshape(
        -1, 3
    )
    return line_xyz


@torch.no_grad()
def __draw_gs_point_line__(start, end, n=32):
    # start, end is N,3 tensor
    line_dir = end - start
    xyz = (
        start[:, None]
        + torch.linspace(0, 1, n)[None, :, None].to(start) * line_dir[:, None]
    )
    return xyz


import cv2

@torch.no_grad()
def prepare_all_penalty_maps(viewpoint_stack, max_dist_threshold=0.2):
    """
    Precompute, for every frame, the penalty map that decides whether a node
    should move inside or outside the mask.
    max_dist_threshold: as a fraction of the longer image side; the penalty
    peaks at that distance and decays beyond it.
    """
    logging.info("Pre-computing Smart Guidance Maps for all frames...")
    penalty_maps = []
    
    for view in tqdm(viewpoint_stack, desc="Generating Guidance Maps"):
        mask_np = view.mask.detach().cpu().numpy()
        if mask_np.ndim == 3: mask_np = mask_np[0]
        
        h, w = mask_np.shape
        max_dim = max(h, w)
        
        # Distance from the mask, measured outside the object
        inv_mask = (1.0 - (mask_np > 0.5)).astype(np.uint8) * 255
        dist_map = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
        
        # Normalise the distance to [0, 1]
        dist_norm = dist_map / max_dim
        
        # Double-well potential
        # f(x) = x * exp(-x / sigma)
        # 0 inside the mask, peaking near x = sigma, decaying back to 0 far away
        sigma = max_dist_threshold / 2.0
        smart_penalty = dist_norm * np.exp(-dist_norm / sigma)
        
        # Rescale so the peak sits near 1.0
        if smart_penalty.max() > 0:
            smart_penalty = smart_penalty / (smart_penalty.max() + 1e-6)
            
        # Blur, for a smoother gradient
        smart_penalty = cv2.GaussianBlur(smart_penalty, (7, 7), 0)
        
        penalty_maps.append(torch.from_numpy(smart_penalty).float().to(view.data_device))
        
    # Returned as [T, 1, H, W]
    return torch.stack(penalty_maps, dim=0).unsqueeze(1)