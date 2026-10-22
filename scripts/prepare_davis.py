"""Stage a DAVIS scene into the per-scene layout the training code expects.

DAVIS stores every modality in its own top-level folder (JPEGImages/480p/<scene>,
Annotations/480p/<scene>, ...), while the training code wants one directory per
scene.  This builds that directory out of symlinks and converts the
Shape-of-Motion style per-pair bootstapir tracks into the single
(T, N, 2) + (T, N) npz that Scene.load_track globs for.

    python scripts/prepare_davis.py --davis_root data/DAVIS --scene kite-surf
"""

import argparse
import os

import numpy as np
import torch


def build_links(davis_root, scene, out_dir):
    links = {
        "images": os.path.join(davis_root, "JPEGImages", "480p", scene),
        "masks": os.path.join(davis_root, "Annotations", "480p", scene),
        "depth": os.path.join(davis_root, "megasam_depth", "480p", scene),
        "pose": os.path.join(davis_root, "pose", "480p", scene),
        "bootstapir": os.path.join(davis_root, "bootstapir", "480p", scene),
    }
    os.makedirs(out_dir, exist_ok=True)
    for name, target in links.items():
        assert os.path.isdir(target), f"missing {target}"
        link = os.path.join(out_dir, name)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link) if os.path.islink(link) else None
        if not os.path.exists(link):
            os.symlink(os.path.abspath(target), link)
    return links


def convert_tracks(tracks_dir, num_frames, subsample, out_path):
    """Concatenate per-(query, target) npy files into (T, N, 2) + (T, N)."""
    all_tracks, all_vis = [], []
    for q in range(num_frames):
        per_target = []
        for t in range(num_frames):
            f = os.path.join(tracks_dir, f"{q:05d}_{t:05d}.npy")
            per_target.append(np.load(f).astype(np.float32))
        # (T, N_q, 4) -> subsample the point axis, matching the iphone reader
        stacked = torch.from_numpy(np.stack(per_target, axis=0))[:, ::subsample]
        occ, dist = stacked[..., 2], stacked[..., 3]
        visible = (1 - torch.sigmoid(occ)) * (1 - torch.sigmoid(dist)) > 0.5
        all_tracks.append(stacked[..., :2])
        all_vis.append(visible)
        print(f"  query {q:3d}/{num_frames}: {stacked.shape[1]:6d} pts", end="\r")

    tracks = torch.cat(all_tracks, dim=1).numpy()
    visibility = torch.cat(all_vis, dim=1).numpy()
    print(f"\n  tracks {tracks.shape}, visible ratio {visibility.mean():.3f}")
    np.savez_compressed(out_path, tracks=tracks, visibility=visibility)
    return tracks.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis_root", type=str, default="data/DAVIS")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--out_root", type=str, default=None)
    parser.add_argument("--subsample", type=int, default=10)
    args = parser.parse_args()

    out_root = args.out_root or os.path.join(args.davis_root, "scenes")
    out_dir = os.path.join(out_root, args.scene)
    links = build_links(args.davis_root, args.scene, out_dir)

    num_frames = len([f for f in os.listdir(links["images"]) if f.endswith(".jpg")])
    print(f"[{args.scene}] {num_frames} frames -> {out_dir}")

    npz_path = os.path.join(out_dir, "davis_bootstapir.npz")
    if os.path.exists(npz_path):
        print(f"  tracks already converted: {npz_path}")
    else:
        convert_tracks(links["bootstapir"], num_frames, args.subsample, npz_path)
    print("done")
