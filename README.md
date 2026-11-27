# WebSpline: Structure-Informed Splines for Real-Time 3D Gaussians from Monocular Videos

### [NeurIPS 2026] Official repository of WebSpline

Jongmin Park\*, Jeonghwan Yun\*, Minh-Quan Viet Bui, Munchurl Kim

\* Co-first authors (equal contribution) &nbsp;&nbsp; KAIST, South Korea

[![Project](https://img.shields.io/badge/Project-Page-blue)](https://kaist-viclab.github.io/webspline-site/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.02096-b31b1b.svg)](https://arxiv.org/abs/2606.02096)
[![GitHub Stars](https://img.shields.io/github/stars/KAIST-VICLab/WebSpline?style=social)](https://github.com/KAIST-VICLab/WebSpline)

<!-- TODO: architecture figure, e.g. ![architecture](assets/architecture.png) -->

WebSpline reconstructs a dynamic scene from a single moving camera. Each
Gaussian's motion is modelled by a learnable cubic Hermite spline over its own
control points, organised by a structural proxy graph — a
[MoSca](https://github.com/JiahuiLei/MoSca)-style deformation scaffold
optimised under an as-rigid-as-possible prior — and trained in two stages
(static, then dynamic).

This release covers
- the **iPhone (DyCheck)** benchmark: training on the monocular training view,
  and evaluation on the held-out views with test-time camera optimisation,
  reporting the masked mPSNR / mSSIM / mLPIPS of the DyCheck protocol;
- **DAVIS** casual videos: training on the whole sequence, reconstruction
  metrics on the input views and novel-view demo renders.

```bash
./train_iphone.sh          # seven iPhone scenes, train + evaluate
./train_davis.sh           # twenty DAVIS-2016 val scenes, train + evaluate + render
```

One code base and one command per dataset: the dataset type is detected from
the scene directory (`emf.json` = iPhone, `pose/intrinsics.npy` = DAVIS) and the
DAVIS-specific settings are switched on only for DAVIS; the iPhone path is the
one the iPhone numbers were produced with.

- [Environmental Setups](#environmental-setups) — conda, torch, the one CUDA extension
- [Data Preparations](#data-preparations) — preprocessed scenes, ready to untar
- [Get Started](#get-started) — training, evaluation, outputs
- [DAVIS](#davis) — data layout, training, evaluation, metrics
- [Preprocessing your own video](#preprocessing-your-own-video) — video → DAVIS-layout scene

## Environmental Setups

Developed and verified against:

| | |
| --- | --- |
| OS | Ubuntu 22.04 (Linux 5.15) |
| GPU | NVIDIA RTX A6000, 48 GB — ~20 GB peak, so a 24 GB card is enough |
| Python | 3.9.23 |
| PyTorch | 1.13.1, CUDA 11.7 |
| torchvision | 0.14.1 |
| gsplat | 1.4.0 |
| pytorch3d | 0.7.8 |

```bash
git clone --recursive https://github.com/KAIST-VICLab/WebSpline.git
cd WebSpline

conda create -n splinegs python=3.9
conda activate splinegs

# torch first, matched to your CUDA version
pip install torch==1.13.1 torchvision==0.14.1

pip install -r requirements.txt
pip install -e submodules/simple-knn
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

If you cloned without `--recursive`, fetch the one submodule that gets compiled:

```bash
git submodule update --init submodules/simple-knn
```

Rasterisation goes through [gsplat](https://github.com/nerfstudio-project/gsplat)
(pulled in by `requirements.txt`), so `simple-knn` is the only CUDA extension
that has to build locally.

Check the install before launching a run:

```bash
python -c "import torch; from simple_knn._C import distCUDA2; import gsplat, pytorch3d; print('ok')"
```

`import torch` has to come first — the extension links against torch's
`libc10.so` and fails with `ImportError: libc10.so` if it is imported on its own.

## Data Preparations

The preprocessed scenes are published on the
[`dataset` release](https://github.com/KAIST-VICLab/WebSpline/releases/tag/dataset).
Download the ones you need and unpack them under `data/iphone/`:

```bash
REL=https://github.com/KAIST-VICLab/WebSpline/releases/download/dataset
mkdir -p data/iphone

# one scene
wget $REL/iphone_apple.tar.gz
tar -xzf iphone_apple.tar.gz -C data/iphone/

# or all seven
for s in apple block paper-windmill space-out spin teddy wheel; do
    wget $REL/iphone_$s.tar.gz
    tar -xzf iphone_$s.tar.gz -C data/iphone/
done
```

| Scene | Archive | Size | Frames (train / held-out) |
| --- | --- | --- | --- |
| apple | `iphone_apple.tar.gz` | 1.75 GiB | 475 / 532 |
| block | `iphone_block.tar.gz` | 1.65 GiB | 447 / 667 |
| paper-windmill | `iphone_paper-windmill.tar.gz` | 1.04 GiB | 277 / 419 |
| space-out | `iphone_space-out.tar.gz` | 1.54 GiB | 429 / 681 |
| spin | `iphone_spin.tar.gz` | 1.63 GiB | 426 / 722 |
| teddy | `iphone_teddy.tar.gz` | 1.85 GiB | 472 / 688 |
| wheel | `iphone_wheel.tar.gz` | 1.54 GiB | 437 / 219 |

~11 GiB for all seven. Verify the downloads against the published checksums:

```bash
wget $REL/SHA256SUMS && sha256sum -c SHA256SUMS
```

The RGB frames, camera poses, LiDAR depth, masks, splits and co-visibility masks
come from [DyCheck](https://github.com/KAIR-BAIR/dycheck); the archives add the
monocular depth (UniDepth) and the 2D tracks this method needs, so no extra
preprocessing is required.

Each archive unpacks to one scene directory. This is everything the loader
reads — it ignores anything else present:

```
data/iphone/<scene>/
├── rgb/2x/                              RGB frames (training + held-out)
├── camera/                              per-frame COLMAP camera json
├── depth/2x/                            sensor (LiDAR) depth, the metric reference
├── uni_depth/                           UniDepth monocular depth
├── uni_normal/                          surface normals (generated on first run)
├── masks/cam0/                          dynamic-object masks
├── covisible/2x/val/                    co-visibility masks for the DyCheck metrics
├── splits/{train,val}.json              frame / camera ids per split
├── scene.json                           scene centre and scale
├── emf.json                             marker file: identifies the scene as DyCheck
├── <scene>_train_pck.npz                keypoints, only for `--pck`
├── uniform_dep=sensor_cotracker_tap.npz static tracks, consumed by bundle adjustment
└── dynamic_dep=sensor_cotracker_tap.npz dynamic tracks, consumed by the MoSca scaffold
```

`uni_normal/` is derived from `uni_depth/` and written back on the first run, so
it may be absent in a fresh copy.

## Get Started

All seven scenes, train then evaluate:

```bash
./train_iphone.sh                 # all seven on GPU 0
GPU=1 ./train_iphone.sh spin      # one scene on GPU 1
PCK=1 ./train_iphone.sh           # also report keypoint PCK
```

To split across two GPUs:

```bash
GPU=0 ./train_iphone.sh apple block paper-windmill &
GPU=1 ./train_iphone.sh space-out spin teddy wheel &
```

The script runs, per scene:

```bash
python train.py -s data/iphone/<scene> \
    --expname code_release_version/iphone/<scene> \
    --configs arguments/iphone/<scene>.py --port 6777

python eval.py -s data/iphone/<scene> \
    --expname code_release_version/iphone/<scene> \
    --configs arguments/iphone/<scene>.py \
    --checkpoint ./output/code_release_version/iphone/<scene>/point_cloud/iteration_20000/
```

Two runs on the same machine need different `--port` values; the port is only
used by the optional network viewer.

### Output

```
output/<expname>/
├── point_cloud/iteration_20000/     checkpoint (the one that evaluates best)
├── point_cloud/iteration_30000/     final checkpoint
├── point_cloud/fine_best/           best-scoring evaluated iteration
└── exp_log.txt                      gaussian counts per stage
```

Each checkpoint holds `point_cloud.ply` (dynamic) and `point_cloud_static.ply`
(static) plus the deformation network, the MoSca scaffold and the control points.

Evaluation optimises the test camera pose before rendering (test-time
optimisation), which is what the DyCheck protocol expects for a monocular
capture, and reports masked mPSNR / mSSIM / mLPIPS over the co-visible region.
Add `--pck` to also report keypoint PCK.

### What a run does

1. **Bundle adjustment** — refines the COLMAP cameras against the static tracks
   (2000 steps).
2. **Scaffold initialisation** — builds the structural proxy graph and fits it
   under the ARAP prior (2000 steps).
3. **`fine_static`** — fits the static Gaussians (15000 iterations).
4. **`fine`** — fits the deformable Gaussians and their control points
   (30000 iterations), evaluating along the way.

Steps 1 and 2 are not cached, so re-running a scene pays for them again.

### Config files override the command line

`arguments/**/*.py` are applied with `merge_hparams` *after* argparse, so a key
set in a config cannot be overridden by a flag — `--iterations 60` is ignored if
the config sets `iterations`. To change one of those, edit the config or write a
small one that inherits from it:

```python
_base_ = './default.py'
OptimizationParams = dict(iterations = 60)
```

## DAVIS

### Data layout

Each DAVIS scene is one directory (default root `data/DAVIS/scenes/`). This is
everything the DAVIS loader reads:

```
data/DAVIS/scenes/<scene>/
├── images/00000.jpg ...                 480p RGB frames (DAVIS JPEGImages/480p)
├── masks/00000.png ...                  dynamic-object masks, > 0 = dynamic (DAVIS Annotations/480p)
├── depth/00000.npy ...                  MegaSaM depth, (H, W, 1)
├── pose/00000.npy ...                   MegaSaM camera-to-world 4x4 per frame
├── pose/intrinsics.npy                  MegaSaM 3x3 intrinsics; marker: identifies the scene as DAVIS
├── uni_normal/                          surface normals (generated from depth on first run)
├── uniform_dep=uni_bootstapir_tap.npz   BootsTAPIR tracks on a uniform grid (static tracks, bundle adjustment)
└── dynamic_dep=uni_bootstapir_tap.npz   BootsTAPIR tracks seeded on the dynamic masks
```

Frame ids are contiguous and zero-based (`00000` ... `T-1`), shared by all
folders. The track files hold `tracks` (T, N, 2, pixels), `visibility`
(T, N) and `queries`; every `*bootstapir*.npz` in the directory is read for the dynamic
tracks, and a track counts as dynamic where it falls inside the mask. The
loader writes a placeholder `points3D.ply` into the scene directory.

### Getting the DAVIS-2016 scenes

Download DAVIS-2016 (`DAVIS-data.zip`, 480p) from the
[DAVIS site](https://davischallenge.org/davis2016/code.html) and stage each scene
with the preprocessing pipeline below (ground-truth masks):

```bash
for s in $(ls arguments/davis | grep -v default | sed 's/\.py$//'); do
    preprocess/run.sh --frames DAVIS/JPEGImages/480p/$s --masks DAVIS/Annotations/480p/$s --name $s --gpu 0
done
```

### Training and evaluation

```bash
./train_davis.sh                         # the 20 DAVIS-2016 val scenes on GPU 0
GPU=1 ./train_davis.sh goat dance-twirl  # some scenes on GPU 1
DATA_ROOT=/path/to/scenes ./train_davis.sh goat
```

Per scene the script runs:

```bash
python train.py -s data/DAVIS/scenes/<scene> \
    --expname code_release_version/davis/<scene> \
    --configs arguments/davis/<scene>.py --port 6778

python eval_davis.py -s data/DAVIS/scenes/<scene> \
    --configs arguments/davis/<scene>.py \
    --checkpoint ./output/code_release_version/davis/<scene>/point_cloud/iteration_30000 \
    --out_dir ./output/code_release_version/davis/<scene>/eval
```

Configs are provided for all 50 DAVIS-2016 scenes (20 val + 30 train) in
`arguments/davis/<scene>.py`. They only set the frame count (temporal plane
resolution, which also gives one spline control point per frame) on top of
`arguments/davis/default.py`. The DAVIS method settings are applied by the code
when the scene is detected as DAVIS (`DATASET_DEFAULTS["davis"]` in
`arguments/__init__.py`, `DATASET_PRESETS["davis"]` in `train.py`):

| Setting | DAVIS | iPhone / other |
| --- | --- | --- |
| bundle adjustment of the given poses | on (uniform static tracks) | on |
| depth normalisation | median depth = 1 | dataset loader default |
| scaffold nodes | every dynamic 3D track curve (o3d outlier / shaking / short-curve filters), dense resampling (0.05 x unit) | keyframe mask pixels snapped to tracks |
| scaffold ARAP fit | MoSca casual-video profile: 1 level (k=8), 4000 steps, reline at 1001 | 3 levels, 2000 steps |
| colour decoder | `sandwich_sgs`: sigmoid(albedo + MLP(spec, time, ray)) | `sandwich` |
| control points per Gaussian | one per frame | temporal resolution / 4 |
| control-point lr | 0.8 x (v / 0.00073)^0.5, v = median 5-frame speed of the dynamic curves | fixed (30 x position lr) |
| Gaussian size cap / world-size prune | none / off | 0.02 / on |
| densify extent | 1 (median-depth units) | from cameras |

Anything set in a config file still overrides these (precedence: config file >
command line > dataset default > class default).

### Output and metrics

```
output/<expname>/
├── point_cloud/iteration_30000/     final checkpoint (evaluated)
├── ba_cams.npz                      bundle-adjusted cameras, used by eval_davis.py
└── eval/
    ├── metrics.json
    ├── recon.mp4                    render at every input camera
    ├── novel_fixed.mp4              camera frozen at the middle frame, time advancing
    ├── novel_orbit.mp4              small orbit (5% of the median depth) around the middle camera
    └── recon_mid.png, novel_mid.png
```

DAVIS has no held-out views, so `metrics.json` reports reconstruction quality on
all input views at 480p: `psnr` (mean per-frame PSNR, full image), `ssim`
(skimage, uint8 RGB), `lpips` (AlexNet), `psnr_dynamic` / `psnr_static` (inside /
outside the motion mask), plus `render_fps`, `num_gaussians` and
`train_minutes`. The novel-view videos are for qualitative inspection.

## Preprocessing your own video

`preprocess/run.sh` turns a video (or a folder of frames) into a scene directory
with the [DAVIS layout](#data-layout) above, ready for `train_davis.sh`.

```bash
cp preprocess/env.sh.example preprocess/env.sh   # point it at the tools below
preprocess/run.sh --video clip.mp4 --name myclip --sam3 "person,shadow" --gpu 0
GPU=0 DATA_ROOT=data/DAVIS/scenes ./train_davis.sh myclip    # needs arguments/davis/myclip.py, see below
```

| Option | |
| --- | --- |
| `--video FILE` / `--frames DIR` | input; frames are extracted / resized to 480 px height, aspect kept (16:9 → 854x480 as DAVIS) |
| `--name NAME` | scene name; written to `--out`/NAME (default `data/DAVIS/scenes/NAME`) |
| `--masks DIR` | given dynamic masks (one `.png` per frame, > 0 = dynamic), e.g. DAVIS annotations |
| `--sam3 "p1,p2"` | SAM3 text prompts for the moving things (`davis` = the prompts we used, `preprocess/prompts_davis.json`); with `--masks` too, SAM3 objects are added to the given masks |
| `--gpu N`, `--max_frames N`, `--stride K`, `--normals`, `--work DIR` | device; video subsampling; precompute normals; intermediates directory |

Every stage is its own script under `preprocess/stages/` and skips itself when
its outputs exist, so an interrupted run can simply be restarted.

| Stage | Tool | Output in the scene directory |
| --- | --- | --- |
| 1 frames | ffmpeg / PIL | `images/%05d.jpg` |
| 2 depth + cameras | MegaSaM (Depth-Anything + UniDepth priors, camera tracking, consistent-video-depth optimisation with RAFT flow) | `depth/%05d.npy`, `pose/%05d.npy`, `pose/intrinsics.npy` |
| 3 2D tracks | MoSca precompute: UniDepth-v2 depth, RAFT flow, epipolar motion masks, BootsTAPIR tracking | `uniform_dep=uni_bootstapir_tap.npz`, `dynamic_dep=uni_bootstapir_tap.npz` |
| 4 masks | given masks, and/or SAM3 video segmentation + static-reprojection motion evidence (one keep/drop decision per object) | `masks/%05d.png` |
| 5 normals (optional) | WebSpline DAVIS loader | `uni_normal/`, `points3D.ply` (otherwise written by the first training run) |
| 6 check | `preprocess/tools/check_scene.py` | — |

Then write a scene config with the frame count (`ls images | wc -l`):

```python
# arguments/davis/myclip.py
_base_ = './default.py'
ModelHiddenParams = dict(kplanes_config = {'grid_dimensions': 2, 'input_coordinate_dim': 4,
                                           'output_coordinate_dim': 32, 'resolution': [64, 64, 64, NUM_FRAMES]})
```

### Installing the tools

The third-party tools are not vendored. Install each in its own environment
(the versions below are the ones the DAVIS results were produced with) and set
the variables in `preprocess/env.sh` (see `preprocess/env.sh.example`):
`MEGASAM_DIR`, `MEGASAM_PY`, `MOSCA_DIR`, `MOSCA_PY` (+ optional
`MOSCA_ACTIVATE`), `SAM3_PY`, `WEBSPLINE_PY`, `HF_HOME`, `TORCH_HOME`.

**MegaSaM** (stage 2): [mega-sam/mega-sam](https://github.com/mega-sam/mega-sam) at commit `a27b4e6`
(submodule `base` at `ee9ac6a`, its `thirdparty/lietorch` at `0fa9ce8`).

```bash
git clone --recursive https://github.com/mega-sam/mega-sam && cd mega-sam && git checkout a27b4e6 && git submodule update --init --recursive
# environment: follow its README (conda env create -f environment.yml, xformers optional with our patch)
# patches for recent PyTorch / no xformers (only needed there):
git apply  $WEBSPLINE/preprocess/patches/mega-sam.patch                     # UniDepth NystromAttention without xformers
(cd base && git apply $WEBSPLINE/preprocess/patches/mega-sam-base.patch)    # tensor.type() -> scalar_type()
(cd base/thirdparty/lietorch && git apply $WEBSPLINE/preprocess/patches/lietorch.patch)
# set your GPU arch in base/setup.py (-gencode=arch=compute_XX,code=sm_XX), then
(cd base && python setup.py install)
```

Checkpoints: `checkpoints/megasam_final.pth` ships with the repo;
[Depth-Anything ViT-L](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth)
→ `Depth-Anything/checkpoints/depth_anything_vitl14.pth`;
[RAFT things](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) → `cvd_opt/raft-things.pth`;
UniDepth-v2 ViT-L (`lpiccinelli/unidepth-v2-vitl14`, revision `1d0d3c5`) is fetched from Hugging Face on first use.

**MoSca** (stage 3): [JiahuiLei/MoSca](https://github.com/JiahuiLei/MoSca) at commit `02404d2`.

```bash
git clone https://github.com/JiahuiLei/MoSca && cd MoSca && git checkout 02404d2
bash install.sh                                          # its conda env
git apply $WEBSPLINE/preprocess/patches/mosca.patch      # gcc 13 <cstdint>, no torch_geometric, no forced UniDepth re-download
```

Weights (from the [MoSca README](https://github.com/JiahuiLei/MoSca#readme) bundle or the original sources):
`weights/tapnet/bootstapir_checkpoint_v2.pt` ([TAPNet BootsTAPIR](https://github.com/google-deepmind/tapnet)) and
`weights/raft_models/raft-things.pth` ([RAFT](https://github.com/princeton-vl/RAFT)); UniDepth-v2 ViT-L is loaded through
`torch.hub` (`lpiccinelli-eth/UniDepth`) into `TORCH_HOME` on first use.

**SAM3** (stage 4, only with `--sam3`): a Python env with `transformers` ≥ 4.57 (we used 5.24.1, torch 2.11),
`opencv-python`, `scipy`; the model [facebook/sam3](https://huggingface.co/facebook/sam3) is gated — request
access and `huggingface-cli login` (it is cached in `HF_HOME`).

Plus `ffmpeg` for `--video`.

### Output

```
<out>/<name>/                 the scene directory (DAVIS layout, see above)
<out>/_work/<name>/           intermediates: MoSca workspace (UniDepth, flow, EPI masks, tracks),
                              motion evidence, SAM3 object decisions (sam3_objects.json), given masks binarised
<MEGASAM_DIR>/{Depth-Anything/video_visualization,UniDepth/outputs,outputs,reconstructions,cache_flow,outputs_cvd}/<name>*
                              MegaSaM's own caches, keyed by the scene name
```

Re-running the pipeline on a DAVIS scene reproduces our inputs up to the
run-to-run noise of the tools (MegaSaM / MoSca are not bit-deterministic).

### Credits / third-party tools

The preprocessing builds on these works; please cite them as well if you use it.

- **MegaSaM** — Z. Li et al., *MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos*, CVPR 2025. [code](https://github.com/mega-sam/mega-sam) · [paper](https://arxiv.org/abs/2412.04463)
- **MoSca** — J. Lei et al., *MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds*, CVPR 2025. [code](https://github.com/JiahuiLei/MoSca) · [paper](https://arxiv.org/abs/2405.17421)
- **UniDepth** — L. Piccinelli et al., *UniDepth: Universal Monocular Metric Depth Estimation*, CVPR 2024; UniDepthV2, 2025. [code](https://github.com/lpiccinelli-eth/UniDepth)
- **Depth Anything** — L. Yang et al., *Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data*, CVPR 2024. [code](https://github.com/LiheYoung/Depth-Anything)
- **RAFT** — Z. Teed and J. Deng, *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, ECCV 2020. [code](https://github.com/princeton-vl/RAFT)
- **BootsTAPIR / TAPNet** — C. Doersch et al., *BootsTAP: Bootstrapped Training for Tracking-Any-Point*, ACCV 2024; *TAPIR*, ICCV 2023. [code](https://github.com/google-deepmind/tapnet)
- **SAM 3** — Meta AI, *SAM 3: Segment Anything with Concepts*, 2025. [model](https://huggingface.co/facebook/sam3) · [code](https://github.com/facebookresearch/sam3)
- **DAVIS** — F. Perazzi et al., *A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation*, CVPR 2016. [site](https://davischallenge.org/)
- **DyCheck** — H. Gao et al., *Monocular Dynamic View Synthesis: A Reality Check*, NeurIPS 2022. [code](https://github.com/KAIR-BAIR/dycheck)

Each tool keeps its own license; check them before redistributing their outputs or weights.

## Layout

```
train.py                    training
eval.py                     evaluation with test-time pose optimisation
eval_pck.py                 keypoint PCK, used by eval.py
dycheck_metrics.py          masked PSNR / SSIM / LPIPS
train_iphone.sh             runs both over all seven scenes
eval_davis.py               DAVIS metrics + demo renders
train_davis.sh              DAVIS train + eval_davis.py per scene
preprocess/                 video -> scene directory (run.sh, stages/, tools/, patches/)
scene/                      Scene, dataset readers, Gaussian model, bundle adjustment
gaussian_renderer/          rasterisation
lib_mosca/, lib_moca/       deformation scaffold and camera models
utils/                      geometry, losses, spline interpolation
arguments/iphone/           per-scene configs (iPhone)
arguments/davis/            per-scene configs (DAVIS-2016 train + val)
```

## Acknowledgments

Built on [4D Gaussians](https://github.com/hustvl/4DGaussians) and
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
with the deformation scaffold from [MoSca](https://github.com/JiahuiLei/MoSca)
and the evaluation protocol and data from
[DyCheck](https://github.com/KAIR-BAIR/dycheck).

<!-- TODO: funding statement, cf. other KAIST-VICLab repositories -->

## License

This code inherits the Gaussian-Splatting license of its upstream projects: free
for **non-commercial research and evaluation use only**. See [LICENSE.md](LICENSE.md).
<!-- TODO: confirm with the lab -- other KAIST-VICLab releases ship MIT, but a
     3DGS derivative normally has to carry Inria's non-commercial terms. -->

## Citing WebSpline

```bibtex
@inproceedings{park2026webspline,
  title     = {WebSpline: Structure-Informed Splines for Real-Time 3D Gaussians from Monocular Videos},
  author    = {Park, Jongmin and Yun, Jeonghwan and Bui, Minh-Quan Viet and Kim, Munchurl},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
