#!/usr/bin/env bash
# Train and evaluate the iPhone (DyCheck) scenes.
#
#   ./train_iphone.sh                 # all 7 scenes on GPU 0
#   GPU=1 ./train_iphone.sh spin      # one scene on GPU 1
#   PCK=1 ./train_iphone.sh           # also report keypoint PCK
#
# The configs run 30000 fine iterations and leave checkpoints at 20000 and
# 30000; 20000 is the one that evaluates best, so that is what eval.py reads.
set -euo pipefail

GPU=${GPU:-0}
PORT=${PORT:-6777}
EXP_ROOT=${EXP_ROOT:-code_release_version/iphone}
ITER=${ITER:-20000}

SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
    SCENES=(apple block paper-windmill space-out spin teddy wheel)
fi

for scene in "${SCENES[@]}"; do
    expname="$EXP_ROOT/$scene"
    echo "=================== $scene -> output/$expname ==================="

    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        -s data/iphone/"$scene" \
        --expname "$expname" \
        --configs arguments/iphone/"$scene".py \
        --port "$PORT"

    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        -s data/iphone/"$scene" \
        --expname "$expname" \
        --configs arguments/iphone/"$scene".py \
        --checkpoint ./output/"$expname"/point_cloud/iteration_"$ITER"/ \
        --port "$PORT" \
        ${PCK:+--pck}
done
