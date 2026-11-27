"""Check that a scene directory has the layout the WebSpline DAVIS loader expects.
    python check_scene.py <scene_dir>"""
import os, sys
import numpy as np

S = sys.argv[1]
ls = lambda d, ext: sorted(f for f in os.listdir(os.path.join(S, d)) if f.endswith(ext))
imgs = ls("images", ".jpg"); n = len(imgs)
assert imgs == [f"{i:05d}.jpg" for i in range(n)], "images must be 00000.jpg ... contiguous"
assert ls("masks", ".png") == [f"{i:05d}.png" for i in range(n)], "masks do not match the frames"
assert ls("depth", ".npy") == [f"{i:05d}.npy" for i in range(n)], "depth does not match the frames"
assert len([f for f in ls("pose", ".npy") if f != "intrinsics.npy"]) == n, "pose does not match the frames"
K = np.load(os.path.join(S, "pose", "intrinsics.npy")); assert K.shape == (3, 3)
d0 = np.load(os.path.join(S, "depth", "00000.npy"))
for k in ("uniform", "dynamic"):
    z = np.load(os.path.join(S, f"{k}_dep=uni_bootstapir_tap.npz"))
    assert z["tracks"].shape[0] == n and z["visibility"].shape == z["tracks"].shape[:2], k
    print(f"  {k} tracks {z['tracks'].shape}, visible {z['visibility'].mean():.2f}")
m = np.mean([np.array(__import__("PIL.Image", fromlist=["Image"]).open(os.path.join(S, "masks", f))) > 0 for f in ls("masks", ".png")[:: max(1, n // 10)]])
print(f"ok: {n} frames, depth {d0.shape}, fx {K[0, 0]:.1f}, dynamic area {m:.3f} -> {S}")
