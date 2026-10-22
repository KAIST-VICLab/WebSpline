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

This release covers the **iPhone (DyCheck)** benchmark: training on the
monocular training view, and evaluation on the held-out views with test-time
camera optimisation, reporting the masked mPSNR / mSSIM / mLPIPS of the DyCheck
protocol.

```bash
./train_iphone.sh          # seven scenes, train + evaluate
```

- [Environmental Setups](#environmental-setups) — conda, torch, the one CUDA extension
- [Data Preparations](#data-preparations) — preprocessed scenes, ready to untar
- [Get Started](#get-started) — training, evaluation, outputs

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

## Layout

```
train.py                    training
eval.py                     evaluation with test-time pose optimisation
eval_pck.py                 keypoint PCK, used by eval.py
dycheck_metrics.py          masked PSNR / SSIM / LPIPS
train_iphone.sh             runs both over all seven scenes
scene/                      Scene, dataset readers, Gaussian model, bundle adjustment
gaussian_renderer/          rasterisation
lib_mosca/, lib_moca/       deformation scaffold and camera models
utils/                      geometry, losses, spline interpolation
arguments/iphone/           per-scene configs
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
