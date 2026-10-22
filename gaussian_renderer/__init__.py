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

import torch
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)

from scene.gaussian_model import GaussianModel
from utils.spline_utils import interpolate_cubic_hermite
from utils.sh_utils import eval_sh
from time import time as get_time
import numpy as np
# from forward_full import GaussianRasterizationSettings, GaussianRasterizer
from gsplat.rendering import rasterization
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
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

# gsplat
def render(viewpoint_camera, stat_pc: GaussianModel, dyn_pc: GaussianModel, bg_color: torch.Tensor,
    stage="fine", get_static=False, get_dynamic=False, coherent=None, w2c=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    # ---------- Common camera / bg ----------
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).cuda()
    K = viewpoint_camera.K

    bg_color_3 = bg_color[:3]
    bg_color_3 = torch.concat([bg_color_3, bg_color_3, bg_color_3], dim=-1)  # keep your original behavior
    bg_color_10 = torch.cat([bg_color_3, torch.tensor([0.0]).cuda()], dim=-1)  # for RGB+ED mode

    # ---------- Static variables ----------
    stat_means3D = stat_pc.get_xyz
    no_stat_gs = stat_means3D.shape[0]
    stat_opacity = stat_pc.get_opacity
    stat_colors_precomp = stat_pc.get_features_static
    stat_scales = stat_pc.get_scaling
    stat_rotations = stat_pc.get_rotation_stat
    
    stat_ones = torch.ones((no_stat_gs, 1), dtype=stat_colors_precomp.dtype, device="cuda")
    stat_feat_with_alpha = torch.cat([stat_colors_precomp, stat_ones], dim=-1)

    # ============================================================
    # (A) EARLY RETURN: fine_static -> static-only rendering path
    # ============================================================
    if stage == "fine_static":
        # static rasterization only
        s_rendered_image, _, s_info = rasterization(
            means=stat_means3D,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=stat_feat_with_alpha,
            backgrounds=bg_color_10[None],
            viewmats=viewmat[None],
            Ks=K[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB+ED",
        )

        s_depth = s_rendered_image[..., -1]
        s_alpha = s_rendered_image[..., -2]
        s_features = s_rendered_image[..., :-2].permute(0, 3, 1, 2)  # (1,3,H,W)

        # NOTE: you still use dyn_pc.rgbdecoder for static (shared decoder design)
        s_rendered_image = dyn_pc.rgbdecoder(s_features, viewpoint_camera.cam_ray).squeeze(0)

        # optional: static alpha (only if get_static requested)
        # s_alpha = None
        # if get_static:
        #     s_alpha, _, _ = rasterization(
        #         means=stat_means3D,
        #         quats=stat_rotations,
        #         scales=stat_scales,
        #         opacities=stat_opacity.squeeze(-1),
        #         colors=torch.ones(stat_colors_precomp.shape[0], 1).cuda(),
        #         backgrounds=bg_color_3[0:1][None],
        #         viewmats=viewmat[None],
        #         Ks=K[None],
        #         width=int(viewpoint_camera.image_width),
        #         height=int(viewpoint_camera.image_height),
        #         packed=False,
        #         render_mode="RGB",
        #     )
        #     s_alpha = s_alpha[..., 0]

        radii = s_info["radii"].squeeze(0)
        viewspace_points = s_info["means2d"]  # [1, N_stat, 2]

        try:
            viewspace_points.retain_grad()
        except Exception:
            pass

        return {
            "render": s_rendered_image,
            "s_render": s_rendered_image,
            "s_depth": s_depth,
            "s_alpha": s_alpha,
            "d_render": None,
            "d_depth": None,
            "d_alpha": None,
            "d_means3d": None,
            "viewspace_points": viewspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": s_depth,
            "blending_factor": None,
            "world_coordinates": None,
            "splat_center": None,
            "means_3d_final": stat_means3D,          # static-only
            "colors_precomp_final": stat_colors_precomp,
        }

    # ============================================================
    # (B) NORMAL PATH: (fine / warm / etc.) -> dyn + (stat+dyn) render
    # ============================================================

    # ---------- Dynamic variables ----------
    # NOTE: your original code detaches dyn xyz
    scales = dyn_pc.get_scaling
    rotations = dyn_pc._rotation
    opacity = dyn_pc.get_opacity

    pointtimes = torch.ones(
        (dyn_pc.get_xyz.shape[0], 1),
        dtype=dyn_pc.get_xyz.dtype,
        requires_grad=False,
        device="cuda",
    ) + 0

    basicfunction = trbfunction

    trbfcenter = dyn_pc.get_trbfcenter
    trbfscale = dyn_pc.get_trbfscale
    trbfdistanceoffset = viewpoint_camera.time * pointtimes - trbfcenter
    trbfdistance = trbfdistanceoffset / torch.exp(trbfscale)
    trbfoutput = basicfunction(trbfdistance)

    tforpoly = trbfdistanceoffset.detach()
    rotations = dyn_pc.get_rotation_dy(rotations, tforpoly)

    control_xyz = dyn_pc.get_control_xyz.cuda()
    deform_means3D = interpolate_cubic_hermite(
        control_xyz.permute(0, 2, 1),
        torch.tensor(viewpoint_camera.time).cuda()[None, None].expand(control_xyz.shape[0], 3, 1),
        N=dyn_pc.current_control_num,
    )
    means3D = deform_means3D * 1e-2

    if coherent is not None:
        means3D = means3D + coherent

    # activations
    rotations = dyn_pc.rotation_activation(rotations)

    colors_precomp = dyn_pc.get_features(tforpoly)

    smeans3D_final, sscales_final, srotations_final, sopacity_final = (
        stat_means3D,
        stat_scales,
        stat_rotations,
        stat_opacity,
    )
    means3D_final, scales_final, rotations_final, opacity_final = (
        means3D,
        scales,
        rotations,
        opacity,
    )

    # ---------- (optional) dynamic-only render outputs ----------
    d_alpha = None
    d_image = None
    d_depth = None
    dmeans3D_final = None

    if get_dynamic:
        dmeans3D_final = means3D_final
        dscales_final = scales_final
        drotations_final = rotations_final
        dopacity_final = opacity_final
        dcolors_precomp = colors_precomp
        
        dones = torch.ones((dcolors_precomp.shape[0], 1), dtype=dcolors_precomp.dtype, device="cuda")
        dfeat_with_alpha = torch.cat([dcolors_precomp, dones], dim=-1)

        d_img, _, _ = rasterization(
            means=dmeans3D_final,
            quats=drotations_final,
            scales=dscales_final,
            opacities=dopacity_final.squeeze(-1),
            colors=dfeat_with_alpha,
            backgrounds=bg_color_10[None],
            viewmats=viewmat[None],
            Ks=K[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB+ED",
        )

        d_depth = d_img[..., -1]
        d_alpha = d_img[..., -2]
        d_features = d_img[..., :-2].permute(0, 3, 1, 2)
        d_image = dyn_pc.rgbdecoder(d_features, viewpoint_camera.cam_ray).squeeze(0)

        # d_alpha, _, _ = rasterization(
        #     means=dmeans3D_final,
        #     quats=drotations_final,
        #     scales=dscales_final,
        #     opacities=dopacity_final.squeeze(-1),
        #     colors=torch.ones(dcolors_precomp.shape[0], 1).cuda(),
        #     backgrounds=bg_color_3[0:1][None],
        #     viewmats=viewmat[None],
        #     Ks=K[None],
        #     width=int(viewpoint_camera.image_width),
        #     height=int(viewpoint_camera.image_height),
        #     packed=False,
        #     render_mode="RGB",
        # )
        # d_alpha = d_alpha[..., 0]

    # ---------- combine stat + dyn gaussians ----------
    means3D_final = torch.cat((smeans3D_final, means3D_final), 0)
    scales_final = torch.cat((sscales_final, scales_final), 0)
    rotations_final = torch.cat((srotations_final, rotations_final), 0)
    opacity_final = torch.cat((sopacity_final, opacity_final), 0)
    colors_precomp_final = torch.cat((stat_colors_precomp, colors_precomp), 0)

    # optional camera transform
    if w2c is not None:
        R_cw, t_cw = w2c[:3, :3], w2c[:3, 3]
        means3D_final = torch.einsum("ij,nj->ni", R_cw, means3D_final) + t_cw[None]
        rot_mat = quaternion_to_matrix(rotations_final)
        rot_mat = torch.einsum("ij,njk->nik", R_cw, rot_mat)
        rotations_final = matrix_to_quaternion(rot_mat)

    rendered_image, _, info = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=colors_precomp_final,
        backgrounds=bg_color_3[None],
        viewmats=viewmat[None],
        Ks=K[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB+ED",
    )

    depth = rendered_image[..., -1]
    rendered_image = rendered_image[..., :-1].permute(0, 3, 1, 2)
    radii = info["radii"].squeeze(0)
    viewspace_points = info["means2d"]

    rendered_image = dyn_pc.rgbdecoder(rendered_image, viewpoint_camera.cam_ray).squeeze(0)

    # ---------- optional static-only render outputs (for logging/loss) ----------
    s_rendered_image = None
    s_depth = None
    s_alpha = None

    if get_static:
        stat_ones = torch.ones((no_stat_gs, 1), dtype=stat_colors_precomp.dtype, device="cuda")
        stat_feat_with_alpha = torch.cat([stat_colors_precomp, stat_ones], dim=-1)
        
        s_rendered_image, _, s_info = rasterization(
            means=smeans3D_final,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=stat_feat_with_alpha,
            backgrounds=bg_color_10[None],
            viewmats=viewmat[None],
            Ks=K[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB+ED",
        )
        s_depth = s_rendered_image[..., -1]
        s_alpha = s_rendered_image[..., -2]
        s_features = s_rendered_image[..., :-2].permute(0, 3, 1, 2)
        s_rendered_image = dyn_pc.rgbdecoder(s_features, viewpoint_camera.cam_ray).squeeze(0)

        # s_alpha, _, _ = rasterization(
        #     means=smeans3D_final,
        #     quats=stat_rotations,
        #     scales=stat_scales,
        #     opacities=stat_opacity.squeeze(-1),
        #     colors=torch.ones(stat_colors_precomp.shape[0], 1).cuda(),
        #     backgrounds=bg_color_3[0:1][None],
        #     viewmats=viewmat[None],
        #     Ks=K[None],
        #     width=int(viewpoint_camera.image_width),
        #     height=int(viewpoint_camera.image_height),
        #     packed=False,
        #     render_mode="RGB",
        # )
        # s_alpha = s_alpha[..., 0]

    try:
        viewspace_points.retain_grad()  # [1, N, 2]
    except Exception:
        pass

    return {
        "render": rendered_image,
        "s_render": s_rendered_image,
        "s_depth": s_depth,
        "s_alpha": s_alpha,
        "d_render": d_image if get_dynamic else None,
        "d_depth": d_depth if get_dynamic else None,
        "d_alpha": d_alpha if get_dynamic else None,
        "d_means3d": dmeans3D_final if get_dynamic else None,
        "viewspace_points": viewspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "blending_factor": None,
        "world_coordinates": None,
        "splat_center": None,
        "means_3d_final": means3D_final,
        "colors_precomp_final": colors_precomp_final,
    }


# gsplat
def render_tto(viewmat, K, cam_ray, image_width, image_height, time, stat_pc: GaussianModel, dyn_pc: GaussianModel, bg_color: torch.Tensor):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    bg_color_3 = bg_color[:3]
    bg_color_3 = torch.concat([bg_color_3, bg_color_3, bg_color_3], dim=-1)  # keep your original behavior

    # ---------- Static variables ----------
    stat_means3D = stat_pc.get_xyz
    no_stat_gs = stat_means3D.shape[0]
    stat_opacity = stat_pc.get_opacity
    stat_colors_precomp = stat_pc.get_features_static
    stat_scales = stat_pc.get_scaling
    stat_rotations = stat_pc.get_rotation_stat

    # ---------- Dynamic variables ----------
    means3D = dyn_pc.get_xyz.detach()
    no_dyn_gs = means3D.shape[0]
    scales = dyn_pc.get_scaling
    rotations = dyn_pc._rotation
    opacity = dyn_pc.get_opacity

    pointtimes = torch.ones(
        (dyn_pc.get_xyz.shape[0], 1),
        dtype=dyn_pc.get_xyz.dtype,
        requires_grad=False,
        device="cuda",
    ) + 0

    basicfunction = trbfunction

    trbfcenter = dyn_pc.get_trbfcenter
    trbfscale = dyn_pc.get_trbfscale
    trbfdistanceoffset = time * pointtimes - trbfcenter
    trbfdistance = trbfdistanceoffset / torch.exp(trbfscale)
    trbfoutput = basicfunction(trbfdistance)

    tforpoly = trbfdistanceoffset.detach()
    rotations = dyn_pc.get_rotation_dy(rotations, tforpoly)

    control_xyz = dyn_pc.get_control_xyz.cuda()
    deform_means3D = interpolate_cubic_hermite(
        control_xyz.permute(0, 2, 1),
        torch.tensor(time).cuda()[None, None].expand(control_xyz.shape[0], 3, 1),
        N=dyn_pc.current_control_num,
    )
    means3D = deform_means3D * 1e-2

    # activations
    rotations = dyn_pc.rotation_activation(rotations)

    colors_precomp = dyn_pc.get_features(tforpoly)

    smeans3D_final, sscales_final, srotations_final, sopacity_final = (
        stat_means3D,
        stat_scales,
        stat_rotations,
        stat_opacity,
    )
    means3D_final, scales_final, rotations_final, opacity_final = (
        means3D,
        scales,
        rotations,
        opacity,
    )

    # ---------- combine stat + dyn gaussians ----------
    means3D_final = torch.cat((smeans3D_final, means3D_final), 0)
    scales_final = torch.cat((sscales_final, scales_final), 0)
    rotations_final = torch.cat((srotations_final, rotations_final), 0)
    opacity_final = torch.cat((sopacity_final, opacity_final), 0)
    colors_precomp_final = torch.cat((stat_colors_precomp, colors_precomp), 0)

    rendered_image, _, _ = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=colors_precomp_final,
        backgrounds=bg_color_3[None],
        viewmats=viewmat[None],
        Ks=K[None],
        width=int(image_width),
        height=int(image_height),
        packed=False,
        render_mode="RGB+ED",
    )
    rendered_image = rendered_image[..., :-1].permute(0, 3, 1, 2)
    rendered_image = dyn_pc.rgbdecoder(rendered_image, cam_ray).squeeze(0)

    return {"render": rendered_image}




