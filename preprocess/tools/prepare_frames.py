"""Frame folder -> <out>/%05d.jpg at 480 px height (aspect kept, width rounded to even; 16:9 -> 854x480).
Frames are taken in sorted filename order. JPEGs that already have height 480 are copied byte for byte.
    python prepare_frames.py <frames_dir> <out_dir>"""
import os, shutil, sys
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
files = sorted(f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png")))
assert files, f"no .jpg/.png frames in {src}"
for i, f in enumerate(files):
    p, q = os.path.join(src, f), os.path.join(dst, f"{i:05d}.jpg")
    im = Image.open(p)
    if f.lower().endswith((".jpg", ".jpeg")) and im.height == 480:
        shutil.copyfile(p, q)
        continue
    w = int(round(im.width * 480 / im.height / 2)) * 2
    im.convert("RGB").resize((w, 480), Image.LANCZOS).save(q, quality=95)
print(f"{len(files)} frames -> {dst}")
