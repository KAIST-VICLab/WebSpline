import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

import numpy as np
from scene import deformation
import torch.nn.functional as F

import copy
@torch.no_grad()
def filter_track(track, track_vis, dep_mask_list, min_valid_cnt=4):
        # query valid depth and visible mask
    track_dep_mask, _ = gather_track_from_buffer(
        track[..., :2].long(), track_vis, dep_mask_list
    )
    track_mask = track_dep_mask.squeeze(-1) * track_vis
        # check whether a track is visible and has valid depth more more thant min_valid_cnt times
    valid_cnt = track_mask.sum(0)
    filter_track_mask = valid_cnt >= min_valid_cnt

    track = track[:, filter_track_mask]
    track_mask = track_mask[:, filter_track_mask].clone()
    return track, track_mask

def gather_track_from_buffer(track, track_base_mask, buffer_list):
        # buffer_list: T, H, W, C
    if buffer_list.ndim == 3:
        buffer_list = buffer_list.unsqueeze(3)
    T, N = track_base_mask.shape
        # fill the ret with nan
    ret = torch.zeros(T, N, buffer_list.shape[-1], dtype=buffer_list.dtype)
    ret = ret.to(track.device)
    ret.fill_(np.nan)
    ret_mask = torch.zeros_like(track_base_mask) > 0
    for tid in range(T):
        _mask = track_base_mask[tid]
        _uv_int = track[tid][_mask]
        _value = query_image_buffer_by_pix_int_coord(buffer_list[tid], _uv_int)
        ret[tid][_mask] = _value
        ret_mask[tid][_mask] = True
    return ret, ret_mask

def query_image_buffer_by_pix_int_coord(buffer, pixel_int_coordinate):
    assert pixel_int_coordinate.ndim == 2 and pixel_int_coordinate.shape[-1] == 2
    assert (pixel_int_coordinate[..., 0] >= 0).all()
    assert (pixel_int_coordinate[..., 0] < buffer.shape[1]).all()
    assert (pixel_int_coordinate[..., 1] >= 0).all()
    assert (pixel_int_coordinate[..., 1] < buffer.shape[0]).all()
        # u is the col, v is the row
    col_id, row_id = pixel_int_coordinate[:, 0], pixel_int_coordinate[:, 1]
    H, W = buffer.shape[:2]
    index = col_id + row_id * W
    ret = buffer.reshape(H * W, *buffer.shape[2:])[index]
    if isinstance(ret, np.ndarray):
        ret = ret.copy()
    return ret

def query_buffers_by_track(image_buffer, track, track_mask, default_value=0.0):
    # image_buffer: T,H,W,C; track: T,N,2, track_mask: T,N
    assert image_buffer.ndim == 4 and track.ndim == 3 and track_mask.ndim == 2
    assert len(image_buffer) == len(track) == len(track_mask)
    T, H, W, C = image_buffer.shape
    N = track.shape[1]
    ret_buffer = torch.ones(T, N, C).to(image_buffer) * default_value

    for i in range(T):
        _uv = track[i][..., :2]
        _int_uv, _inside_mask = round_int_coordinates(_uv, H, W)
        _value = query_image_buffer_by_pix_int_coord(image_buffer[i].clone(), _int_uv)
        valid_mask = track_mask[i] & _inside_mask
        # for outside, put default value
        _value[~valid_mask] = default_value
        ret_buffer[i] = _value
    return ret_buffer

def round_int_coordinates(coord, H, W):
    ret = coord.round().long()
    valid_mask = (
        (ret[..., 0] >= 0) & (ret[..., 0] < W) & (ret[..., 1] >= 0) & (ret[..., 1] < H)
    )
    ret[..., 0] = torch.clamp(ret[..., 0], 0, W - 1)
    ret[..., 1] = torch.clamp(ret[..., 1], 0, H - 1)
    return ret, valid_mask