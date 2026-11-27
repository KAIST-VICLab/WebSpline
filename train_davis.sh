#!/usr/bin/env bash
# Train and evaluate DAVIS scenes.
#
#   ./train_davis.sh                      # the 20 DAVIS-2016 val scenes on GPU 0
#   GPU=1 ./train_davis.sh goat           # one scene on GPU 1
#   DATA_ROOT=/path/to/scenes ./train_davis.sh goat
#
# The DAVIS-specific settings are applied automatically because the scene
# directory is detected as DAVIS (pose/intrinsics.npy); arguments/davis/<scene>.py
# only sets the frame count. Training runs 30000 fine iterations; the final
# checkpoint is evaluated on every training view (DAVIS has no held-out views)
# and rendered into the demo videos (see eval_davis.py).
set -euo pipefail

GPU=${GPU:-0}
PORT=${PORT:-6778}
DATA_ROOT=${DATA_ROOT:-data/DAVIS/scenes}
EXP_ROOT=${EXP_ROOT:-code_release_version/davis}
ITER=${ITER:-30000}
PYTHON=${PYTHON:-python}

SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
    SCENES=(blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog
            drift-chicane drift-straight goat horsejump-high kite-surf libby motocross-jump
            paragliding-launch parkour scooter-black soapbox)
fi

for scene in "${SCENES[@]}"; do
    expname="$EXP_ROOT/$scene"
    out="./output/$expname"
    echo "=================== $scene -> $out ==================="

    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
        -s "$DATA_ROOT/$scene" \
        --expname "$expname" \
        --configs arguments/davis/"$scene".py \
        --port "$PORT"
    minutes=$(( ($(date +%s) - t0) / 60 ))

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON eval_davis.py \
        -s "$DATA_ROOT/$scene" \
        --configs arguments/davis/"$scene".py \
        --checkpoint "$out"/point_cloud/iteration_"$ITER" \
        --out_dir "$out"/eval \
        --train_minutes "$minutes"
done
