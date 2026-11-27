"""SAM3 text-prompted video segmentation -> dynamic masks <out>/%05d.png (0/255).

Each tracked object gets ONE keep/drop decision for the whole video (so masks do not flicker):
  - amorphous stuff (water, sea, wave, splash, smoke, dust, foam) is kept;
  - shadows are kept if they stay next to the anchor (the given masks, or without them the kept objects);
  - other objects are kept if they overlap the given masks (>30% of their area) or move (>25% of their area
    has static-reprojection motion evidence, see motion_evidence.py).
With --gt the result is the union with the given masks.
Needs transformers with Sam3VideoModel and access to the gated Hugging Face model facebook/sam3.
    python sam3_masks.py <scene_dir> "<prompt1,prompt2,...>" <motion_raw.npz> <out_dir> [--gt DIR] [--meta objects.json]"""
import argparse, json, os, time
import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("scene"); ap.add_argument("prompts"); ap.add_argument("motion"); ap.add_argument("out")
ap.add_argument("--gt", default=""); ap.add_argument("--meta", default=""); ap.add_argument("--model", default="facebook/sam3")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
prompts = [p.strip() for p in a.prompts.split(",") if p.strip()]
proc = Sam3VideoProcessor.from_pretrained(a.model)
model = Sam3VideoModel.from_pretrained(a.model).to(dev).eval()
AM = ("water", "sea", "wave", "splash", "smoke", "dust", "foam")
kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)); k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

t0 = time.time(); R = a.scene; os.makedirs(a.out, exist_ok=True)
names = sorted(f for f in os.listdir(f"{R}/images") if f.endswith(".jpg")); N = len(names)
fr = [cv2.cvtColor(cv2.imread(f"{R}/images/{n}"), cv2.COLOR_BGR2RGB) for n in names]
H, W = fr[0].shape[:2]
gts = [cv2.imread(f"{a.gt}/{t:05d}.png", 0) > 0 for t in range(N)] if a.gt else [np.zeros((H, W), bool)] * N
z = np.load(a.motion); raw = np.unpackbits(z["raw"], axis=-1)[..., :W].astype(bool)
mot = [cv2.blur(raw[t].astype(np.float32), (15, 15)) > 0.2 for t in range(N)]
sess = proc.init_video_session(video=fr, inference_device=dev, video_storage_device="cpu")
proc.add_text_prompt(sess, prompts)
per = {}; lab = {}; st = {}
with torch.no_grad():
    for out in model.propagate_in_video_iterator(sess):
        o = proc.postprocess_outputs(sess, out); t = out.frame_idx
        for c, ids in o["prompt_to_obj_ids"].items():
            for i in ids:
                lab[int(i)] = c
        ids = o["object_ids"].tolist(); ms = o["masks"].cpu().numpy() > 0.5 if len(ids) else []
        fdict = {}
        for i, m in zip(ids, ms):
            ar = m.sum()
            if ar < 30:
                continue
            fdict[i] = np.packbits(m, axis=-1)
            s = st.setdefault(i, [0, 0.0, 0.0])
            s[0] += 1; s[1] += (m & gts[t]).sum() / ar; s[2] += (m & mot[t]).sum() / ar
        per[t] = fdict
unpack = lambda pm: np.unpackbits(pm, axis=-1)[:, :W].astype(bool)
keep = {}
for i, (n, fgt, fmot) in st.items():   # pass 1: everything except shadows
    l = lab.get(i, ""); fgt /= n; fmot /= n
    if "shadow" in l:
        continue
    ok = any(k in l for k in AM) or fgt > 0.3 or fmot > 0.25
    keep[i] = dict(l=l, frames=n, gt=round(fgt, 2), mot=round(fmot, 2), keep=bool(ok))
# pass 2: shadows next to the anchor (given masks; without them, the objects kept in pass 1)
anchor = []
for t in range(N):
    m = gts[t].copy()
    if not a.gt:
        for i, pm in per.get(t, {}).items():
            if i in keep and keep[i]["keep"]:
                m |= unpack(pm)
    anchor.append(cv2.dilate(m.astype(np.uint8), kd) > 0)
near = {}
for t in range(N):
    for i, pm in per.get(t, {}).items():
        if "shadow" in lab.get(i, ""):
            m = unpack(pm); near[i] = near.get(i, 0.0) + (m & anchor[t]).sum() / m.sum()
for i, (n, fgt, fmot) in st.items():
    if "shadow" in lab.get(i, ""):
        fn = near.get(i, 0.0) / n
        keep[i] = dict(l=lab[i], frames=n, near=round(fn, 2), mot=round(fmot / n, 2), keep=bool(fn > 0.3))
ex = []
for t in range(N):
    m = np.zeros((H, W), bool)
    for i, pm in per.get(t, {}).items():
        if keep[i]["keep"]:
            m |= unpack(pm)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k7) > 0
    fin = m | gts[t]
    cv2.imwrite(f"{a.out}/{t:05d}.png", fin.astype(np.uint8) * 255); ex.append((fin & ~gts[t]).mean())
if a.meta:
    json.dump({str(k): v for k, v in keep.items()}, open(a.meta, "w"), indent=0)
print(f"SAM3 {prompts}: {N} frames, {len(keep)} objects, kept {sum(v['keep'] for v in keep.values())}, "
      f"area added to the given masks {np.mean(ex):.3f} ({time.time() - t0:.0f}s)")
