"""Precompute what WebSpline's DAVIS reader writes on its first run: <scene>/uni_normal/%05d.npy and the
placeholder <scene>/points3D.ply. Calls the repo's own readDavisInfo, so the files are identical to what
training would create (normals are scale invariant, so the depth normalisation does not matter).
    python make_normals.py <scene_dir>"""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from scene.dataset_readers_davis import readDavisInfo  # noqa: E402

info = readDavisInfo(sys.argv[1], types.SimpleNamespace(davis_depth_norm="median"))
print("ok", len(info.train_cameras), "frames")
