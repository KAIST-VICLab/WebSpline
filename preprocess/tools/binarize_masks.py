"""Mask folder (e.g. DAVIS palette annotations) -> <out>/%05d.png, 0/255 (any value > 0 = dynamic).
Masks are taken in sorted filename order and must match the frame count.
    python binarize_masks.py <mask_dir> <out_dir> <num_frames>"""
import os, sys
import numpy as np
from PIL import Image

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sorted(f for f in os.listdir(src) if f.lower().endswith(".png"))
assert len(files) == n, f"{len(files)} masks in {src}, but {n} frames"
os.makedirs(dst, exist_ok=True)
for i, f in enumerate(files):
    m = np.array(Image.open(os.path.join(src, f)))
    if m.ndim == 3:
        m = m.max(-1)
    Image.fromarray(((m > 0) * 255).astype(np.uint8)).save(os.path.join(dst, f"{i:05d}.png"))
print(f"binarised {n} masks -> {dst}")
