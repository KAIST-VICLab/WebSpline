#!/usr/bin/env python3

# ! jiahui 2024 April 1st copied from https://github.com/apple/ml-pgdvs/tree/main?tab=readme-ov-file

# ! warning, should calibrate with the apple paper!

# Modified from https://github.com/KAIR-BAIR/dycheck/blob/ddf77a4e006fdbc5aed28e0859c216da0de5aff5/dycheck/core/metrics/image.py

#
# File   : image.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
#
# Copyright 2022 Adobe. All rights reserved.
#
# This file is licensed to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR REPRESENTATIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import scipy
import numpy as np
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F

import lpips

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def masked_mean(
    x: Union[np.ndarray, jnp.ndarray],
    mask: Optional[Union[np.ndarray, jnp.ndarray]] = None,
) -> Union[np.ndarray, jnp.ndarray]:
    """Compute mean of masked values by soft blending.

    Support both jnp.ndarray and np.ndarray.

    Args:
        x (types.Array): Input array of shape (...,).
        mask (types.Array): Mask array in [0, 1]. Shape will be broadcasted to
            match x.

    Returns:
        types.Array: Masked mean of x of shape ().
    """
    eps = 1e-6

    broadcast_to = jnp.broadcast_to if isinstance(x, jnp.ndarray) else np.broadcast_to
    if mask is None:
        return x.mean()

    mask = broadcast_to(mask, x.shape)
    return (x * mask).sum() / mask.sum().clip(eps)  # type: ignore


def compute_psnr(
    img0: np.ndarray, img1: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute PSNR between two images.

    Args:
        img0 (np.ndarray): An image of shape (H, W, 3) in float32.
        img1 (np.ndarray): An image of shape (H, W, 3) in float32.
        mask (Optional[np.ndarray]): An optional forground mask of shape (H,
            W, 1) in float32 {0, 1}. The metric is computed only on the pixels
            with mask == 1.

    Returns:
        np.ndarray: PSNR in dB of shape ().
    """
    mse = (img0 - img1) ** 2
    return -10.0 / np.log(10.0) * np.log(masked_mean(mse, mask))


def compute_ssim(img0, img1, mask=None, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, max_val=1.0):
    """
    SSIM over numpy inputs, computed with torch on the GPU.
    Inputs:
        img0, img1: numpy arrays of shape (H, W, C)
        mask: optional numpy array of shape (H, W, 1)
    Returns:
        float: the SSIM value
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # numpy -> torch, with the channel axis moved to the front
    # NumPy (H, W, C) -> PyTorch (1, C, H, W)
    def to_tensor(arr):
        tensor = torch.from_numpy(arr).float().to(device)
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor

    # Images
    img0_t = to_tensor(img0)
    img1_t = to_tensor(img1)
    
    batch_size, channels, _, _ = img0_t.shape

    # Mask
    if mask is None:
        mask_t = torch.ones((batch_size, 1, img0_t.shape[2], img0_t.shape[3]), device=device)
    else:
        mask_t = to_tensor(mask)
        # The mask is usually single-channel; match it to the image
        if mask_t.shape[1] != 1:
            mask_t = mask_t[:, :1, :, :] # keep the first channel

    # 1D Gaussian filter
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    
    # built directly on the device
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Separable convolution kernels
    # horizontal, (channels, 1, 1, K)
    filt_x = filt.view(1, 1, 1, filter_size).repeat(channels, 1, 1, 1)
    # vertical, (channels, 1, K, 1)
    filt_y = filt.view(1, 1, filter_size, 1).repeat(channels, 1, 1, 1)
    
    # single-channel kernels, for the mask
    filt_x_mask = filt.view(1, 1, 1, filter_size)
    filt_y_mask = filt.view(1, 1, filter_size, 1)

    # ones kernel, for normalisation
    ones_x_mask = torch.ones_like(filt_x_mask)
    ones_y_mask = torch.ones_like(filt_y_mask)

    # Masked blur, mirroring the reference JAX implementation
    def masked_blur(z, m):
        # Horizontal pass
        z_m = z * m
        # groups=channels makes this a depthwise convolution
        blur_h = F.conv2d(z_m, filt_x, groups=channels, padding=0)
        m_conv_h = F.conv2d(m, ones_x_mask, padding=0)
        
        m_safe_h = m_conv_h.clone()
        m_safe_h[m_safe_h == 0] = 1.0 # avoid dividing by zero
        
        # Normalise over the valid region
        res_h = torch.where(m_conv_h != 0, blur_h * filter_size / m_safe_h, torch.zeros_like(blur_h))
        new_mask_h = (m_conv_h != 0).type(z.dtype)

        # Vertical pass
        z_m_v = res_h * new_mask_h
        blur_v = F.conv2d(z_m_v, filt_y, groups=channels, padding=0)
        m_conv_v = F.conv2d(new_mask_h, ones_y_mask, padding=0)
        
        m_safe_v = m_conv_v.clone()
        m_safe_v[m_safe_v == 0] = 1.0
        
        res_v = torch.where(m_conv_v != 0, blur_v * filter_size / m_safe_v, torch.zeros_like(blur_v))
        return res_v

    # SSIM statistics
    mu0 = masked_blur(img0_t, mask_t)
    mu1 = masked_blur(img1_t, mask_t)
    
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    
    sigma00 = masked_blur(img0_t**2, mask_t) - mu00
    sigma11 = masked_blur(img1_t**2, mask_t) - mu11
    sigma01 = masked_blur(img0_t * img1_t, mask_t) - mu01

    # Clip, then combine
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    
    ssim_map = numer / denom
    
    # Return a plain float
    return ssim_map.mean().item()


@torch.inference_mode()
def compute_lpips(
    lpips_model: torch.nn.Module,
    img0: np.ndarray,
    img1: np.ndarray,
    mask: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cuda"),
) -> np.array:
    """Compute LPIPS between two images.

    This function computes mean LPIPS over masked regions. The input images
    are also masked. The following previous works leverage this metric:

    [1] Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic
    Scenes.
        Li et al., CVPR 2021.
        https://arxiv.org/abs/2011.13084

    [2] Transforming and Projecting Images into Class-conditional
    Generative Networks.
        Huh et al., CVPR 2020.
        https://arxiv.org/abs/2005.01703

    [3] Controlling Perceptual Factors in Neural Style Transfer.
        Gatys et al., CVPR 2017.
        https://arxiv.org/abs/1611.07865

    Args:
        img0 (np.ndarray): An image of shape (H, W, 3) in float32.
        img1 (np.ndarray): An image of shape (H, W, 3) in float32.
        mask (Optional[np.ndarray]): An optional forground mask of shape
            (H, W, 1) in float32 {0, 1}. The metric is computed only on the
            pixels with mask == 1.

    Returns:
        np.ndarray: LPIPS in range [0, 1] in shape ().
    """
    if mask is None:
        mask = np.ones_like(img0[..., :1])
    img0 = lpips.im2tensor(np.array(img0 * mask), factor=1 / 2).to(device)
    img1 = lpips.im2tensor(np.array(img1 * mask), factor=1 / 2).to(device)
    return masked_mean(lpips_model(img0, img1).cpu().numpy()[0, 0, ..., None], mask)
