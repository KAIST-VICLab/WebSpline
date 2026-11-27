"""Evaluate one trained DAVIS scene: reconstruction metrics + demo videos.

DAVIS has no held-out views, so the metrics are computed on every training view
against the input frames (full 480p image):
    psnr / ssim / lpips          whole image
    psnr_dynamic / psnr_static   inside / outside the motion mask (masks/)
    render_fps, num_gaussians
and the novel-view videos show what the reconstruction does off the input path.

Writes to --out_dir:
    recon.mp4        every training camera, input frame order
    novel_fixed.mp4  camera frozen at the middle frame's pose, time advancing
    novel_orbit.mp4  circular orbit (radius = orbit_frac * median rendered depth of
                     the middle frame) around the middle pose, aimed at the point at
                     median depth on the middle camera's optical axis
    recon_mid.png, novel_mid.png   stills
    metrics.json
All videos 15 fps, H.264 yuv420p.

If training ran bundle adjustment (DAVIS default), the refined poses saved next
to the checkpoint (<model>/ba_cams.npz) are used.

    python eval_davis.py -s data/DAVIS/scenes/goat --configs arguments/davis/goat.py \
        --checkpoint output/code_release_version/davis/goat/point_cloud/iteration_30000 \
        --out_dir output/code_release_version/davis/goat/eval
"""

import copy
import json
import os
import sys
import time
from argparse import ArgumentParser

import imageio.v2 as imageio
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams, apply_dataset_defaults
from gaussian_renderer import render
from scene import GaussianModel, Scene, detect_dataset_type


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


def write_h264(frames, path, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8,
                           macro_block_size=2, output_params=["-movflags", "+faststart"])
    for f in frames:
        w.append_data(f)
    w.close()
    print(f"[*] wrote {path} ({len(frames)} frames @ {fps} fps)")


def to_u8(img):
    return (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def render_cam(cam, stat_g, dyn_g, bg):
    return render(cam, stat_g, dyn_g, bg, stage="fine", get_static=True, get_dynamic=True)["render"].clamp(0, 1)


def posed_copy(src, R, center, time_, local_viewdirs, batch_shape):
    cam = copy.deepcopy(src)
    T = (-R.T @ center).astype(np.float32)
    cam.update_cam(R.astype(np.float32), T, local_viewdirs, batch_shape)
    cam.time = time_
    return cam


def look_at(R_ref, center, target):
    """c2w rotation (OpenCV: x right, y down, z forward) at `center` looking at `target`,
    keeping the reference camera's down vector as the roll reference."""
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(R_ref[:, 1], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def psnr_from_mse(mse):
    return float(-10.0 * np.log10(mse)) if mse > 0 else float("inf")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp, op, pp, hp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser), ModelHiddenParams(parser)
    parser.add_argument("--checkpoint", required=True, help="<model>/point_cloud/iteration_XXXX")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--configs", default="")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--orbit_frac", type=float, default=0.05)
    parser.add_argument("--revolutions", type=float, default=1.0)
    parser.add_argument("--train_minutes", type=float, default=None)
    parser.add_argument("--no_videos", action="store_true", help="metrics only")
    parser.add_argument("--port", type=int, default=6009)  # accepted for symmetry with train.py; unused
    args = parser.parse_args(sys.argv[1:])
    # same dataset-type defaults and config precedence as train.py
    args = apply_dataset_defaults(parser, args, detect_dataset_type(args.source_path, args.dataset_type))
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, mmcv.Config.fromfile(args.configs))

    dataset, hyper, opt = lp.extract(args), hp.extract(args), op.extract(args)
    dataset.model_path = args.checkpoint
    dyn_g = GaussianModel(dataset.sh_degree, hyper)
    stat_g = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, dyn_g, stat_g, load_coarse=None)
    ba_path = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(args.checkpoint))), "ba_cams.npz")
    if os.path.exists(ba_path):  # training ran bundle adjustment: render with the refined poses
        d = np.load(ba_path)
        ds = scene.train_camera.dataset
        for i in range(len(ds)):
            ds[i] = ds[i]._replace(R=d["R"][i].astype(np.float32), T=d["T"][i].astype(np.float32))
        print(f"[*] using BA cameras from {ba_path}")
    dyn_g.create_pose_network(hyper, scene.getTrainCameras())
    dyn_g.training_setup(opt, stage="fine")
    stat_g.training_setup(opt, is_static=True)
    dyn_g.load_ply(os.path.join(args.checkpoint, "point_cloud.ply"))
    stat_g.load_ply(os.path.join(args.checkpoint, "point_cloud_static.ply"))
    dyn_g.load_model(args.checkpoint)
    n_dyn, n_stat = int(dyn_g.get_xyz.shape[0]), int(stat_g.get_xyz.shape[0])
    print(f"[*] {n_dyn} dynamic / {n_stat} static gaussians")

    bg = torch.tensor([1] * 9 + [0] if dataset.white_background else [0] * 9 + [0], dtype=torch.float32, device="cuda")
    cams = list(scene.getTrainCameras())
    n = len(cams)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- reconstruction at every training camera: metrics + timing
    lpips_fn = lpips.LPIPS(net="alex").cuda()
    frames, psnrs, ssims, lps, times, mse_dyn, mse_stat = [], [], [], [], [], [], []
    for _ in range(3):  # untimed warm-up (CUDA init), then time every training camera
        render_cam(cams[0], stat_g, dyn_g, bg)
    for i, cam in enumerate(cams):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        img = render_cam(cam, stat_g, dyn_g, bg)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        gt = cam.original_image.cuda().clamp(0, 1)
        err = ((img - gt) ** 2).mean(0)                      # H, W
        psnrs.append(psnr_from_mse(float(err.mean())))
        if cam.mask is not None:
            m = cam.mask.cuda().reshape(err.shape) > 0.5     # 1 = dynamic
            if m.any():
                mse_dyn.append(float(err[m].mean()))
            if (~m).any():
                mse_stat.append(float(err[~m].mean()))
        frames.append(to_u8(img))
        # SSIM: skimage on uint8 RGB, data_range=255
        ssims.append(float(structural_similarity(to_u8(gt), frames[-1], channel_axis=2, data_range=255)))
        with torch.no_grad():
            lps.append(float(lpips_fn(img[None] * 2 - 1, gt[None] * 2 - 1).mean()))

    metrics = {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lps)),
        # masked PSNR: mean over frames of the per-frame masked MSE, then PSNR
        "psnr_dynamic": psnr_from_mse(float(np.mean(mse_dyn))) if mse_dyn else None,
        "psnr_static": psnr_from_mse(float(np.mean(mse_stat))) if mse_stat else None,
        "train_minutes": args.train_minutes,
        "render_fps": float(n / np.sum(times)),
        "num_gaussians": n_dyn + n_stat,
        "num_frames": n,
        "checkpoint": os.path.basename(os.path.normpath(args.checkpoint)),
        "notes": ("metrics on all training views vs input frames (full image); psnr = mean of per-frame PSNR; "
                  "SSIM skimage (uint8 RGB); LPIPS-AlexNet in [-1,1]; black background; "
                  f"{n_dyn} dynamic + {n_stat} static gaussians; render_fps = n / GPU-synced render() time."),
    }

    if not args.no_videos:
        write_h264(frames, os.path.join(args.out_dir, "recon.mp4"), args.fps)
        imageio.imwrite(os.path.join(args.out_dir, "recon_mid.png"),
                        np.concatenate([to_u8(cams[n // 2].original_image), frames[n // 2]], 1))

        # ---- novel views around the middle pose
        local_viewdirs, batch_shape = build_local_viewdirs(cams[0])
        src = cams[n // 2]
        R0 = np.asarray(src.R, dtype=np.float64)
        c0 = cam_center(src).astype(np.float64)
        with torch.no_grad():
            d = render(src, stat_g, dyn_g, bg, stage="fine", get_static=True, get_dynamic=True)["depth"].flatten()
        valid = torch.isfinite(d) & (d > 1e-4)
        med_depth = float(d[valid].median()) if valid.any() else 1.0
        radius = args.orbit_frac * med_depth
        target = c0 + R0[:, 2] * med_depth
        print(f"[*] anchor frame {n // 2}, median depth {med_depth:.4f}, orbit radius {radius:.4f}")

        fixed, orbit = [], []
        for i in range(n):
            cam = posed_copy(src, R0, c0, cams[i].time, local_viewdirs, batch_shape)
            fixed.append(to_u8(render_cam(cam, stat_g, dyn_g, bg)))
            th = 2.0 * np.pi * args.revolutions * i / n
            c = c0 + R0 @ np.array([radius * np.cos(th), radius * np.sin(th), 0.0])
            cam = posed_copy(src, look_at(R0, c, target), c, cams[i].time, local_viewdirs, batch_shape)
            orbit.append(to_u8(render_cam(cam, stat_g, dyn_g, bg)))
        write_h264(fixed, os.path.join(args.out_dir, "novel_fixed.mp4"), args.fps)
        write_h264(orbit, os.path.join(args.out_dir, "novel_orbit.mp4"), args.fps)
        imageio.imwrite(os.path.join(args.out_dir, "novel_mid.png"), np.concatenate([fixed[0], orbit[n // 4]], 1))
        metrics["orbit_radius"] = radius

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
