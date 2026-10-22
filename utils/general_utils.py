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
import sys
from datetime import datetime
import numpy as np
import random
import cv2

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    if np.array(resized_image_PIL).max()!=1:
        resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    else:
        resized_image = torch.from_numpy(np.array(resized_image_PIL))
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent, seed=0):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def laplacian_filter_depth(
    depths,
    threshold_ratio=0.5,
    ksize=5,
    open_ksize=3,
    *,
    threshold_mode: str = "depth",   # "depth" or "mad"
    mad_k: float = 3.0,
    do_median_outlier: bool = True,
    median_ksize: int = 5,
    outlier_abs: float = 0.02,
    outlier_rel: float = 0.05,
    # === strictness knobs ===
    boundary_dilate_ksize: int = 0,   # >0 dilates the boundary, invalidating more pixels
    boundary_dilate_iter: int = 0,
):
    """
    Build a mask that invalidates depth boundaries and noise.
    To be stricter:
      - lower mad_k (e.g. 2.0)
      - lower outlier_abs / outlier_rel
      - enable boundary_dilate_ksize / boundary_dilate_iter
    """
    dep_boundary_errors, dep_valid_masks = [], []
    ellip_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))

    dilate_kernel = None
    if boundary_dilate_ksize and boundary_dilate_ksize > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (boundary_dilate_ksize, boundary_dilate_ksize)
        )

    for dep in depths:
        dep = dep.astype(np.float32)

        base_valid = np.isfinite(dep) & (dep > 1e-6)

        if do_median_outlier:
            dep_med = cv2.medianBlur(dep, median_ksize)
            diff = np.abs(dep - dep_med)
            outlier = (diff > outlier_abs) | (diff > (outlier_rel * np.maximum(dep_med, 1e-6)))
        else:
            outlier = np.zeros_like(dep, dtype=bool)

        error = cv2.Laplacian(dep, cv2.CV_64F, ksize=ksize)
        error = np.abs(error).astype(np.float32)

        if threshold_mode == "mad":
            e = error[base_valid]
            if e.size == 0:
                threshold = 1e9
            else:
                med = np.median(e)
                mad = np.median(np.abs(e - med)) + 1e-12
                threshold = med + mad_k * mad
        else:
            d = dep[base_valid]
            if d.size == 0:
                threshold = 1e9
            else:
                threshold = np.median(d) * float(threshold_ratio)

        _, occlusion_boundaries = cv2.threshold(error, float(threshold), 255, cv2.THRESH_BINARY)
        boundaries = (occlusion_boundaries.astype(np.uint8) > 0)

        # Dilate the boundary outwards
        if dilate_kernel is not None:
            boundaries_u8 = boundaries.astype(np.uint8) * 255
            boundaries_u8 = cv2.dilate(boundaries_u8, dilate_kernel, iterations=int(boundary_dilate_iter))
            boundaries = boundaries_u8 > 0

        invalid = boundaries | outlier
        valid = (~invalid) & base_valid

        # Opening, to drop small speckles of valid pixels
        valid_opened = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN, ellip_kernel) > 0

        dep_valid_masks.append(valid_opened)
        dep_boundary_errors.append(error)

    dep_valid_masks = np.stack(dep_valid_masks, axis=0)
    dep_boundary_errors = np.stack(dep_boundary_errors, axis=0)
    return dep_valid_masks, dep_boundary_errors

def detect_depth_occlusion_boundaries(depth_map, threshold=10, ksize=5):
    error = cv2.Laplacian(depth_map, cv2.CV_64F, ksize=ksize)
    error = np.abs(error)
    _, occlusion_boundaries = cv2.threshold(error, threshold, 255, cv2.THRESH_BINARY)
    return occlusion_boundaries.astype(np.uint8), error