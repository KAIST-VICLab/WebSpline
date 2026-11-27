"""Per-pixel motion evidence from the MegaSaM depth + cameras: photometric residual of warping each frame to its
neighbours (gaps 2, 5, 10) as if the scene were static, minimised over a small search window and over the two
directions, median over the gaps, thresholded. Only used by sam3_masks.py to decide which SAM3 objects move.
With --gt, pixels that land on a given dynamic mask in the other frame are ignored (moved-object ghosts).
    python motion_evidence.py <scene_dir> <out.npz> [--gt <binary mask dir>]
out.npz: raw = packbits(bool (T,H,W), axis=-1), shape = [T,H,W]"""
import argparse, os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("scene"); ap.add_argument("out"); ap.add_argument("--gt", default="")
a = ap.parse_args()
R = a.scene
TH = 0.04; SR = 2; GAPS = [2, 5, 10]; BLUR = 1.0
K = np.load(f"{R}/pose/intrinsics.npy")
names = sorted(f for f in os.listdir(f"{R}/images") if f.endswith(".jpg")); N = len(names)
imgs = [cv2.GaussianBlur(cv2.cvtColor(cv2.imread(f"{R}/images/{n}"), cv2.COLOR_BGR2RGB).astype(np.float32) / 255, (0, 0), BLUR) for n in names]
deps = [np.load(f"{R}/depth/{t:05d}.npy")[..., 0] for t in range(N)]
poses = [np.load(f"{R}/pose/{t:05d}.npy") for t in range(N)]
H, W = deps[0].shape
gts = [cv2.imread(f"{a.gt}/{t:05d}.png", 0) > 0 for t in range(N)] if a.gt else [np.zeros((H, W), bool)] * N
yy, xx = np.mgrid[0:H, 0:W]
rays = (np.stack([xx, yy, np.ones_like(xx)], -1).reshape(-1, 3).astype(np.float64)) @ np.linalg.inv(K).T


def res(t, s):
    X = rays * deps[t].reshape(-1, 1); Xw = X @ poses[t][:3, :3].T + poses[t][:3, 3]
    Ti = np.linalg.inv(poses[s]); Xs = Xw @ Ti[:3, :3].T + Ti[:3, 3]; uv = Xs @ K.T; z = uv[:, 2]
    u = (uv[:, 0] / np.maximum(z, 1e-6)).reshape(H, W).astype(np.float32)
    v = (uv[:, 1] / np.maximum(z, 1e-6)).reshape(H, W).astype(np.float32)
    ok = (z.reshape(H, W) > 0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    gs = cv2.remap(gts[s].astype(np.uint8), u, v, cv2.INTER_NEAREST) > 0
    r = np.full((H, W), np.inf, np.float32)
    for dx in range(-SR, SR + 1):
        for dy in range(-SR, SR + 1):
            r = np.minimum(r, np.abs(cv2.remap(imgs[s], u + dx, v + dy, cv2.INTER_LINEAR) - imgs[t]).mean(-1))
    r[~ok | gs] = np.nan
    return r


raws = []
for t in range(N):
    per = []
    for gp in GAPS:
        rr = [res(t, s) for s in (t - gp, t + gp) if 0 <= s < N]
        if rr:
            per.append(np.nanmin(np.stack(rr), 0))
    with np.errstate(all="ignore"):
        r = np.nanmedian(np.stack(per), 0)
    raws.append(np.nan_to_num(r, nan=0.0) > TH)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez_compressed(a.out, raw=np.packbits(np.stack(raws), axis=-1), shape=np.array([N, H, W]))
print(f"motion evidence: {N} frames, moving fraction {np.mean(raws):.3f} -> {a.out}")
