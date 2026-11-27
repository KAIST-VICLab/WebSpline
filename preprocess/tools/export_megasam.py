"""MegaSaM CVD output (outputs_cvd/<name>_sgd_cvd_hr.npz) -> WebSpline DAVIS layout.

  <depth_out>/%05d.npy        (H, W, 1) float32 depth at full frame resolution
  <pose_out>/%05d.npy         (4, 4) float32 camera-to-world
  <pose_out>/intrinsics.npy   (3, 3) float32 K at full resolution

MegaSaM works at a downscaled, 8-aligned crop (h1 x w1 resized, cropped to multiples of 8
at the bottom/right). Depth is resampled back to full resolution with the same pixel
mapping MegaSaM used to scale K (u_small = u_full * w1 / W0), in disparity space,
edge-replicating the few cropped rows/cols.

    python export_megasam.py <img_dir> <npz> <depth_out> <pose_out>
"""
import glob, os, sys
import cv2
import numpy as np

img_dir, npz, dout, pout = sys.argv[1:5]
frames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
H0, W0 = cv2.imread(frames[0]).shape[:2]
h1 = int(H0 * np.sqrt((384 * 512) / (H0 * W0)))
w1 = int(W0 * np.sqrt((384 * 512) / (H0 * W0)))

d = np.load(npz)
depths = d["depths"].astype(np.float32)        # (T, hc, wc)
K = d["intrinsic"].astype(np.float64)           # at (h1, w1) scale
c2w = d["cam_c2w"].astype(np.float32)           # (T, 4, 4)
T, hc, wc = depths.shape
assert T == len(frames), (T, len(frames))
assert (hc, wc) == (h1 - h1 % 8, w1 - w1 % 8), (hc, wc, h1, w1)
sx, sy = w1 / W0, h1 / H0

Kf = np.eye(3, dtype=np.float32)
Kf[0, 0], Kf[1, 1] = K[0, 0] / sx, K[1, 1] / sy
Kf[0, 2], Kf[1, 2] = K[0, 2] / sx, K[1, 2] / sy

uu, vv = np.meshgrid(np.arange(W0, dtype=np.float32), np.arange(H0, dtype=np.float32))
map_x, map_y = uu * sx, vv * sy

os.makedirs(dout, exist_ok=True)
os.makedirs(pout, exist_ok=True)
for t, f in enumerate(frames):
    fid = int(os.path.splitext(os.path.basename(f))[0])
    disp = 1.0 / np.clip(depths[t], 1e-3, None)
    disp_full = cv2.remap(disp, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    depth_full = (1.0 / np.clip(disp_full, 1e-6, None)).astype(np.float32)[..., None]
    np.save(os.path.join(dout, f"{fid:05d}.npy"), depth_full)
    np.save(os.path.join(pout, f"{fid:05d}.npy"), c2w[t])
np.save(os.path.join(pout, "intrinsics.npy"), Kf)
print(f"exported {T} frames; MegaSaM resolution {hc}x{wc}; K fx={Kf[0,0]:.1f} fy={Kf[1,1]:.1f} "
      f"cx={Kf[0,2]:.1f} cy={Kf[1,2]:.1f}; depth median {np.median(depths):.3f}; "
      f"camera travel {np.linalg.norm(c2w[-1,:3,3]-c2w[0,:3,3]):.3f}")
