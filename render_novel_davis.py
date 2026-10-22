"""Render two videos per DAVIS scene from a trained checkpoint.

  1. <scene>_original.mp4 -- the source frames, just concatenated.
  2. <scene>_novel.mp4    -- novel views from poses near the training ones.

Each video is also written out frame by frame, into <scene>_original/ and
<scene>_novel/ next to it, named by the source frame id so the two line up.
Pass --no_frames to skip that.

By default the novel video holds the camera at one training pose (the middle
frame) while time advances, so every frame is seen from a camera that did not
observe it.  --radius_scale layers a lateral orbit on top, but these
reconstructions smear badly under even small translations, so it defaults to 0.

    python render_novel_davis.py -s data/DAVIS/scenes/kite-surf \
        --configs arguments/davis/kite-surf.py \
        --checkpoint output/davis/kite-surf/point_cloud/iteration_30000 \
        --out_dir output/davis/kite-surf/videos
"""

import copy
import os
import sys
from argparse import ArgumentParser

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene


def get_pixels(image_size_x, image_size_y, use_center=True):
    xx, yy = np.meshgrid(np.arange(image_size_x, dtype=np.float32),
                         np.arange(image_size_y, dtype=np.float32))
    offset = 0.5 if use_center else 0
    return np.stack([xx, yy], axis=-1) + offset


def build_local_viewdirs(cam):
    """Camera-space ray directions for every pixel (intrinsics are shared)."""
    md = cam.metadata
    pixels = get_pixels(md.image_size_x, md.image_size_y, use_center=True)
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    y = (pixels[..., 1] - md.principal_point_y) / md.focal_length
    x = (pixels[..., 0] - md.principal_point_x) / md.focal_length
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    return viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True), batch_shape


def cam_center(cam):
    """World-space camera centre. Camera.R is c2w rotation, Camera.T is w2c translation."""
    return -np.asarray(cam.R) @ np.asarray(cam.T)


def write_video(frames, path, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=1)
    print(f"[*] wrote {path}  ({len(frames)} frames, {fps} fps)")


def write_frames(frames, out_dir, names):
    """Dump the same frames as individual PNGs, named by source frame id."""
    os.makedirs(out_dir, exist_ok=True)
    for img, name in zip(frames, names):
        imageio.imwrite(os.path.join(out_dir, f"{name}.png"), img)
    print(f"[*] wrote {len(frames)} frames to {out_dir}")


def frame_names(train_cams):
    """Source frame ids, so the original and novel dumps index the same way."""
    return [str(getattr(c, "image_name", i)) for i, c in enumerate(train_cams)]


def write_output(frames, out_path, fps, names, save_frames):
    write_video(frames, out_path, fps)
    if save_frames:
        write_frames(frames, os.path.splitext(out_path)[0], names)


def render_original(train_cams, out_path, fps, names, save_frames):
    frames = []
    for cam in train_cams:
        img = (cam.original_image.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        frames.append(img)
    write_output(frames, out_path, fps, names, save_frames)


@torch.no_grad()
def render_novel(train_cams, stat_gaussians, dyn_gaussians, background,
                 out_path, fps, radius, revolutions, anchor, names, save_frames):
    """Hold the camera at one training pose while time advances.

    Every frame t is rendered from the anchor frame's viewpoint, which is a
    different camera than the one that observed t -- a genuine novel view, but
    one that stays on the pose manifold the model was supervised on.  A lateral
    orbit can be layered on top via --radius_scale, though these reconstructions
    tolerate very little translation before they smear.
    """
    local_viewdirs, batch_shape = build_local_viewdirs(train_cams[0])
    n = len(train_cams)
    src = train_cams[anchor]
    R = np.asarray(src.R)
    base_center = cam_center(src)
    frames = []
    for i in tqdm(range(n), desc="rendering novel views"):
        theta = 2.0 * np.pi * revolutions * i / max(n - 1, 1)
        offset_cam = radius * np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)
        new_center = base_center + R @ offset_cam
        new_T = (-R.T @ new_center).astype(np.float32)

        cam = copy.deepcopy(src)
        cam.update_cam(R.astype(np.float32), new_T, local_viewdirs, batch_shape)
        cam.time = train_cams[i].time     # freeze the view, advance time

        pkg = render(cam, stat_gaussians, dyn_gaussians, background,
                     stage="fine", get_static=True, get_dynamic=True)
        img = pkg["render"].clamp(0, 1)
        frames.append((img * 255).byte().permute(1, 2, 0).cpu().numpy())
    write_output(frames, out_path, fps, names, save_frames)


if __name__ == "__main__":
    parser = ArgumentParser(description="DAVIS novel-view video rendering")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--revolutions", type=float, default=2.0)
    parser.add_argument("--radius_scale", type=float, default=0.0,
                        help="orbit radius as a multiple of the scene camera extent; "
                             "0 keeps the camera fixed at the anchor pose")
    parser.add_argument("--anchor", type=int, default=-1,
                        help="training frame whose pose the camera is held at (-1 = middle)")
    parser.add_argument("--no_frames", action="store_true",
                        help="only write the videos, not the per-frame PNGs")
    parser.add_argument("--expname", type=str, default="")
    args = parser.parse_args(sys.argv[1:])

    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, mmcv.Config.fromfile(args.configs))

    dataset, hyper, opt, pipe = lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args)
    dataset.model_path = args.checkpoint

    dyn_gaussians = GaussianModel(dataset.sh_degree, hyper)
    stat_gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)

    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())
    dyn_gaussians.training_setup(opt, stage="fine")
    stat_gaussians.training_setup(opt, is_static=True)

    dyn_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud.ply"))
    stat_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud_static.ply"))
    dyn_gaussians.load_model(args.checkpoint)
    print(f"[*] loaded {dyn_gaussians.get_xyz.shape[0]} dynamic / "
          f"{stat_gaussians.get_xyz.shape[0]} static gaussians from {args.checkpoint}")

    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = [c for c in scene.getTrainCameras()]
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    out_dir = args.out_dir or os.path.join("output", "davis", scene_name, "videos")

    centers = np.stack([cam_center(c) for c in train_cams])
    extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
    radius = args.radius_scale * extent
    print(f"[*] {scene_name}: {len(train_cams)} frames, camera extent {extent:.4f}, "
          f"orbit radius {radius:.4f}")

    names = frame_names(train_cams)
    save_frames = not args.no_frames

    render_original(train_cams, os.path.join(out_dir, f"{scene_name}_original.mp4"),
                    args.fps, names, save_frames)
    anchor = args.anchor if args.anchor >= 0 else len(train_cams) // 2
    print(f"[*] novel view: camera held at training frame {anchor}, time advancing over all frames")
    render_novel(train_cams, stat_gaussians, dyn_gaussians, background,
                 os.path.join(out_dir, f"{scene_name}_novel.mp4"),
                 args.fps, radius, args.revolutions, anchor, names, save_frames)
    print("done")
