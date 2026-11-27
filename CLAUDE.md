# CLAUDE.md — guide for coding agents running WebSpline experiments

This file is for AI coding agents (Claude Code, Codex, ...) asked to reproduce
WebSpline or to run experiments on top of it. Humans should start with `README.md`.
`AGENTS.md` points here.

## 1. What this repo is

WebSpline is a dynamic 3D Gaussian Splatting method for monocular video. Each
dynamic Gaussian moves along a cubic Hermite spline. The spline control points are
organised by a MoSca-style deformation scaffold that is fitted under an ARAP prior.

One code base serves several datasets. `scene.detect_dataset_type(source_path)`
picks the dataset from the scene directory:

| Marker file in the scene dir | Dataset | Status |
|---|---|---|
| `emf.json` | iPhone (DyCheck) | released, reproduces the paper |
| `pose/intrinsics.npy` | DAVIS (casual video) | released, final configuration |
| `poses_bounds.npy` | NVIDIA (RoDynRF) | reader and configs exist, no released preprocessed data |
| `sparse/` | generic COLMAP | best effort |

Dataset-specific behaviour comes from two places:
- `DATASET_DEFAULTS[<type>]` in `arguments/__init__.py`, applied by `apply_dataset_defaults`.
- `DATASET_PRESETS[<type>]` in `train.py` (scaffold and bundle-adjustment settings).

iPhone behaviour must stay bit-for-bit what produced the paper numbers. Gate every
new behaviour behind a flag whose default keeps the old behaviour, or behind the
DAVIS/NVIDIA defaults.

## 2. Environment check

Run this before any experiment:

```bash
python -c "import torch; from simple_knn._C import distCUDA2; import gsplat, pytorch3d; print('ok')"
```

- `import torch` must come first, or you get `ImportError: libc10.so`.
- Peak GPU memory is about 20 GB per run. Run one job per GPU.
- Each concurrent run needs its own `--port`.

## 3. Reproduce the baselines first

Before changing anything, reproduce one scene per dataset and compare it with the
reference numbers below. A mismatch of more than about 0.1 dB means the environment
or the data differs. Fix that first.

### iPhone (DyCheck): masked mPSNR / mSSIM / mLPIPS on held-out views

```bash
GPU=0 ./train_iphone.sh paper-windmill   # about 1.5 h on an A6000
```

| scene | mPSNR | mSSIM | mLPIPS |
|---|---|---|---|
| apple | 19.65 | 0.816 | 0.315 |
| block | 19.23 | 0.722 | 0.241 |
| paper-windmill | 22.37 | 0.763 | 0.134 |
| space-out | 22.59 | 0.764 | 0.171 |
| spin | 20.98 | 0.728 | 0.201 |
| teddy | 16.55 | 0.650 | 0.298 |
| wheel | 19.03 | 0.720 | 0.209 |
| **mean** | **20.06** | **0.738** | **0.224** |

Notes:
- Metrics are printed at the end of `eval.py` as the `Average mPSNR / mSSIM / mLPIPS` lines.
- The evaluated checkpoint is `point_cloud/iteration_20000`.
- Run-to-run noise is about ±0.05 dB per scene.

### DAVIS-2016 val (20 scenes): reconstruction PSNR / SSIM / LPIPS on all input views

```bash
GPU=0 ./train_davis.sh goat               # about 30 min training + eval
cat output/code_release_version/davis/goat/eval/metrics.json
```

| scene | PSNR | SSIM | LPIPS |
|---|---|---|---|
| blackswan | 28.02 | 0.820 | 0.191 |
| bmx-trees | 27.87 | 0.850 | 0.171 |
| breakdance | 20.60 | 0.814 | 0.144 |
| camel | 29.18 | 0.925 | 0.062 |
| car-roundabout | 28.53 | 0.930 | 0.082 |
| car-shadow | 32.96 | 0.949 | 0.070 |
| cows | 27.67 | 0.917 | 0.057 |
| dance-twirl | 26.82 | 0.825 | 0.162 |
| dog | 29.70 | 0.836 | 0.266 |
| drift-chicane | 28.47 | 0.928 | 0.131 |
| drift-straight | 27.58 | 0.932 | 0.081 |
| goat | 26.03 | 0.812 | 0.194 |
| horsejump-high | 29.11 | 0.918 | 0.081 |
| kite-surf | 26.21 | 0.731 | 0.280 |
| libby | 33.97 | 0.932 | 0.080 |
| motocross-jump | 32.32 | 0.921 | 0.147 |
| paragliding-launch | 30.37 | 0.855 | 0.184 |
| parkour | 28.07 | 0.868 | 0.148 |
| scooter-black | 26.75 | 0.931 | 0.073 |
| soapbox | 30.97 | 0.917 | 0.087 |
| **mean** | **28.56** | | |

Notes:
- These numbers use ground-truth DAVIS masks.
- MegaSaM and MoSca preprocessing are not bit-deterministic. A freshly preprocessed
  scene can differ by a few tenths of a dB. Compare variants on the same staged scene.
- `metrics.json` also reports `psnr_dynamic` / `psnr_static`, which are PSNR inside
  and outside the motion mask. Report both when a change targets motion.

## 4. How to run an experiment

1. **Name the variant.** Use a short tag, for example `E1`, and write down in one
   line what it changes.
2. **Write configs. Do not edit the released configs.** Configs are applied after
   argparse, so a key set in a config cannot be overridden by a command-line flag.
   Put variants in a new folder and inherit from the released ones:
   ```python
   # arguments/exp/E1/goat.py
   _base_ = '../../davis/goat.py'
   OptimizationParams = dict(densify_until_iter = 15000)
   ```
   For a new code path, add a flag whose default keeps the current behaviour.
   Set the flag in the variant config.
3. **Run with its own expname and port, one job per GPU:**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python train.py -s data/DAVIS/scenes/goat \
       --expname exp/E1/davis/goat --configs arguments/exp/E1/goat.py --port 7001
   CUDA_VISIBLE_DEVICES=0 python eval_davis.py -s data/DAVIS/scenes/goat \
       --configs arguments/exp/E1/goat.py \
       --checkpoint output/exp/E1/davis/goat/point_cloud/iteration_30000 \
       --out_dir output/exp/E1/davis/goat/eval
   ```
   - For iPhone, use `eval.py` with `--checkpoint .../point_cloud/iteration_20000/`, as in `train_iphone.sh`.
   - `train_iphone.sh` and `train_davis.sh` read `GPU`, `PORT` and `EXP_ROOT` from the environment; `train_davis.sh` also reads `DATA_ROOT`.
   - Redirect logs to a file (`> log 2>&1`) and detach long runs (`setsid nohup ...`).
     A run is 0.5–2 h.
4. **Decide on averages over the whole benchmark,** not on one scene:
   - iPhone: all 7 scenes.
   - DAVIS: the 20 val scenes, or a fixed subset agreed up front.
   - Keep a variant only if the mean improves by more than the noise (about 0.05 dB
     on iPhone) and no metric gets clearly worse.
   - Record negative results too.
5. **Stack changes one at a time.** Build the next variant on top of the best
   previous one, so every gain can be attributed to one change.

### Already tried on iPhone (do not repeat without a new idea)

Each change was tried on top of the released iPhone config. Means are over the 7 scenes.

| change | mPSNR | result |
|---|---|---|
| released config | 20.06 | baseline |
| scene-scale densify/prune (`densify_extent 1`, no world-size prune) | 20.05 | neutral |
| motion-adaptive control-point lr | 19.75 | worse |
| one control point per frame | 19.87 | worse |
| SplineGS colour decoder (`sandwich_sgs`) | 19.88 | worse |
| no Gaussian size cap (`max_scale 1000`) | 20.00 | worse |
| DAVIS-style scaffold (track-curve nodes, `scaffold_preset davis_mosca`) | 19.11 | much worse |

Conclusion: the DAVIS defaults help on DAVIS, but none of them transfers to iPhone.

## 5. Things that bite

- **Bundle adjustment and scaffold fitting are not cached.** Every run pays for
  them again (a few minutes).
- **Stale `ba_cams.npz`.** `eval_davis.py` reads `ba_cams.npz` from the model
  directory. Do not reuse an output dir across variants: a stale file is picked up
  silently.
- **Flags that a dataset default overrides.** On a DAVIS scene, a command-line value
  equal to the argparse default (e.g. `--max_scale 0.02`) is treated as "not given",
  so the DAVIS default wins. Use a config file to force such values.
- **Dataset type from the CLI.** `dataset_type` is detected before the config merge.
  To force a type, pass `--dataset_type` on the command line.
- **DAVIS preprocessing** (`preprocess/run.sh`) needs MegaSaM, MoSca and optionally
  SAM3, each in its own environment (`preprocess/env.sh`). Every stage skips itself
  when its outputs exist. To redo a stage, delete its outputs.
- **Masks.** The DAVIS masks must have the frame resolution. `preprocess/tools/check_scene.py`
  checks this along with the rest of the layout.
- **Never commit** data, `output/`, or `preprocess/env.sh`.

## 6. Where things are

| what | where |
|---|---|
| training loop, presets | `train.py` (`DATASET_PRESETS`) |
| arguments, dataset defaults | `arguments/__init__.py` (`DATASET_DEFAULTS`, `apply_dataset_defaults`) |
| per-scene configs | `arguments/{iphone,davis,nvidia_rodynrf}/` |
| dataset readers | `scene/dataset_readers.py` (iPhone, NVIDIA), `scene/dataset_readers_davis.py` |
| tracks, scene setup | `scene/__init__.py` (`load_track`, track-file patterns per dataset) |
| bundle adjustment | `scene/bundle_adjustment.py` |
| Gaussians, splines, control points | `scene/gaussian_model.py`, `utils/` |
| scaffold | `lib_mosca/` |
| evaluation | `eval.py` + `dycheck_metrics.py` (iPhone), `eval_davis.py` (DAVIS), `eval_nvidia.py` (NVIDIA) |
| preprocessing | `preprocess/` (see README "Preprocessing your own video") |
