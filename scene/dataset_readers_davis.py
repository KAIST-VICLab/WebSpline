"""Scene reader for DAVIS sequences.

DAVIS has no ground-truth cameras and no held-out views, so this reader differs
from the iphone one in three ways:

  * poses come from MegaSaM (data/DAVIS/pose/480p/<scene>, c2w 4x4 + intrinsics),
  * depth comes from the MegaSaM CVD output, which is already in the same scale
    as those poses, so no per-frame alignment against a sensor depth is needed,
  * there is no val split -- the train cameras are reused as the "test" set so
    the reporting path stays intact (the PSNR is a training-fit number).

Expects the per-scene layout produced by scripts/prepare_davis.py.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import dycheck_geometry
from scene.dataset_readers import (
    CameraInfo,
    SceneInfo,
    getNerfppNorm,
    get_normals,
    fetchPly,
    storePly,
)
from utils.general_utils import laplacian_filter_depth
from utils.graphics_utils import focal2fov


def _frame_ids(images_dir):
    return sorted(int(Path(f).stem) for f in os.listdir(images_dir) if f.endswith(".jpg"))


def readDavisCameras(path, args):
    images_dir = os.path.join(path, "images")
    masks_dir = os.path.join(path, "masks")
    depth_dir = os.path.join(path, "depth")
    pose_dir = os.path.join(path, "pose")
    normal_dir = os.path.join(path, "uni_normal")
    os.makedirs(normal_dir, exist_ok=True)

    frames = _frame_ids(images_dir)
    max_time = len(frames) - 1

    K = np.load(os.path.join(pose_dir, "intrinsics.npy")).astype(np.float32)
    c2ws = np.stack([np.load(os.path.join(pose_dir, f"{i:05d}.npy")) for i in frames]).astype(np.float32)

    # MegaSaM's scale is arbitrary; normalise so the scene sits around unit depth.
    # Depth and camera positions are rescaled together so geometry stays consistent.
    depths_raw = [np.load(os.path.join(depth_dir, f"{i:05d}.npy")).astype(np.float32) for i in frames]
    mean_depth = float(np.mean(np.stack(depths_raw, 0)))
    print(f"[DAVIS] {len(frames)} frames, mean depth {mean_depth:.4f} -> normalising to 1.0")

    cam_infos = []
    for idx, fid in enumerate(frames):
        img_path = os.path.join(images_dir, f"{fid:05d}.jpg")
        curr_img = Image.fromarray(np.array(Image.open(img_path).convert("RGB")).astype(np.uint8))
        W, H = curr_img.size

        c2w = c2ws[idx]
        camera = dycheck_geometry.Camera(
            orientation=c2w[:3, :3].T,          # world-to-camera rotation
            position=c2w[:3, 3] / mean_depth,   # camera centre, rescaled
            focal_length=np.array([K[0, 0]]).astype(np.float32),
            principal_point=np.array([K[0, 2], K[1, 2]]).astype(np.float32),
            image_size=np.array([W, H]).astype(np.float32),
        )
        R = camera.orientation.T
        T = camera.translation
        focal = camera.focal_length
        FovX = focal2fov(focal, W)
        FovY = focal2fov(focal, H)

        depth = depths_raw[idx] / mean_depth                      # (H, W, 1)
        depth_mask = (depth > 1e-3) * (depth < 1000)
        lap_valid, _ = laplacian_filter_depth(depth.squeeze(-1)[None], threshold_mode="mad")
        depth_mask = depth_mask * lap_valid[0][..., None]

        normal_path = os.path.join(normal_dir, f"{fid:05d}.npy")
        if os.path.exists(normal_path):
            normal = np.load(normal_path)
        else:
            normal = get_normals(torch.from_numpy(depth).squeeze(-1)[None], camera)
            normal = normal.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            np.save(normal_path, normal)
        normal = F.avg_pool2d(
            torch.from_numpy(normal).permute(2, 0, 1)[None], 5, stride=1, padding=2
        ).squeeze(0).permute(1, 2, 0).numpy()

        mask = np.array(Image.open(os.path.join(masks_dir, f"{fid:05d}.png")))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)[..., None]           # 1 = dynamic

        cam_infos.append(
            CameraInfo(
                uid=idx, R=R.astype(np.float32), T=T.astype(np.float32),
                FovY=float(FovY), FovX=float(FovX), image=curr_img,
                image_path=img_path, image_name=f"{fid:05d}", max_time=float(max_time),
                width=W, height=H, time=float(idx) / max(max_time, 1),
                mask=mask, metadata=camera, normal=normal.astype(np.float32),
                depth=depth.astype(np.float32), depth_mask=depth_mask.astype(np.float32),
                sem_mask=None, instance_mask=mask[None], K=K,
            )
        )
    return cam_infos, max_time


def readDavisInfo(path, args):
    print(f"Reading DAVIS Info from {path}")
    train_cam_infos, max_time = readDavisCameras(path, args)
    # No held-out split in DAVIS: reuse the train views so the eval path still runs.
    test_cam_infos = list(train_cam_infos)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Gaussians are created from scene_initialization, so this is only a placeholder
    # to satisfy the SceneInfo contract (same as the iphone reader).
    num_frames = max_time + 1
    xyz = np.zeros((100, 3))
    rgb = np.zeros((100, 3))
    totalxyz = [xyz for _ in range(num_frames)]
    totalrgb = [rgb for _ in range(num_frames)]
    totaltime = [np.ones((xyz.shape[0], 1)) * i / num_frames for i in range(num_frames)]
    xyz = np.concatenate(totalxyz, axis=0)
    rgb = np.concatenate(totalrgb, axis=0)
    totaltime = np.concatenate(totaltime, axis=0)

    ply_path = os.path.join(path, "points3D.ply")
    storePly(ply_path, np.concatenate((xyz, totaltime), axis=1), rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        video_cameras=None,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=max_time,
    )
