import torch
import numpy as np
from matplotlib import cm
from PIL import Image
import os

__all__ = ('get_normals',)


def flow2rgb(flow_map, max_value):
    _, h, w = flow_map.shape
    flow_map[:, (flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[0, :, :] += normalized_flow_map[0, :, :]
    rgb_map[1, :, :] -= 0.5 * (normalized_flow_map[0, :, :] + normalized_flow_map[1, :, :])
    rgb_map[2, :, :] += normalized_flow_map[1, :, :]
    return rgb_map.clip(0, 1)


def get_normals(z, camera_metadata):
    pixels = camera_metadata.get_pixels()
    y = (pixels[..., 1] - camera_metadata.principal_point_y) / camera_metadata.scale_factor_y
    x = (
                pixels[..., 0] - camera_metadata.principal_point_x - y * camera_metadata.skew
        ) / camera_metadata.scale_factor_x
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    viewdirs = torch.from_numpy(viewdirs).to(z.device)

    coords = viewdirs[None] * z[..., None]
    
    # coords = coords.permute(0, 3, 1, 2)

    # dxdu = coords[..., 0, :, 1:] - coords[..., 0, :, :-1]
    # dydu = coords[..., 1, :, 1:] - coords[..., 1, :, :-1]
    # dzdu = coords[..., 2, :, 1:] - coords[..., 2, :, :-1]
    # dxdv = coords[..., 0, 1:, :] - coords[..., 0, :-1, :]
    # dydv = coords[..., 1, 1:, :] - coords[..., 1, :-1, :]
    # dzdv = coords[..., 2, 1:, :] - coords[..., 2, :-1, :]

    # dxdu = torch.nn.functional.pad(dxdu, (0, 1), mode='replicate')
    # dydu = torch.nn.functional.pad(dydu, (0, 1), mode='replicate')
    # dzdu = torch.nn.functional.pad(dzdu, (0, 1), mode='replicate')

    # dxdv = torch.cat([dxdv, dxdv[..., -1:, :]], dim=-2)
    # dydv = torch.cat([dydv, dydv[..., -1:, :]], dim=-2)
    # dzdv = torch.cat([dzdv, dzdv[..., -1:, :]], dim=-2)

    # n_x = dydv * dzdu - dydu * dzdv
    # n_y = dzdv * dxdu - dzdu * dxdv
    # n_z = dxdv * dydu - dxdu * dydv

    # pred_normal = torch.stack([n_x, n_y, n_z], dim=-3)
    # pred_normal = torch.nn.functional.normalize(pred_normal, dim=-3)
    
    coords = coords.squeeze(0)
    hd, wd, _ = coords.shape
    bottom_point = coords[..., 2:hd, 1 : wd - 1, :]
    top_point = coords[..., 0 : hd - 2, 1 : wd - 1, :]
    right_point = coords[..., 1 : hd - 1, 2:wd, :]
    left_point = coords[..., 1 : hd - 1, 0 : wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    pred_normal = torch.nn.functional.pad(xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant")
    return pred_normal[None]

