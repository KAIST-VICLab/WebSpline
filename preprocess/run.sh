#!/usr/bin/env bash
# Input video (or frame folder) -> a WebSpline DAVIS-layout scene directory.
#
#   preprocess/run.sh --video clip.mp4  --name myclip --sam3 "person,shadow" --gpu 0
#   preprocess/run.sh --frames DAVIS/JPEGImages/480p/dog --name dog --masks DAVIS/Annotations/480p/dog --gpu 0
#
# Options
#   --video FILE | --frames DIR   input (frames: any sorted .jpg/.png; resized to 480p height if needed)
#   --name NAME                   scene name (also the cache key inside the MegaSaM checkout: keep it unique)
#   --out DIR                     scene root, the scene is written to DIR/NAME (default data/DAVIS/scenes)
#   --masks DIR                   dynamic-object masks, one .png per frame in frame order (>0 = dynamic)
#   --sam3 "p1,p2,..."            SAM3 text prompts for the dynamic objects ("davis" = prompts_davis.json[NAME]);
#                                 with --masks as well, the SAM3 objects are added to the given masks
#   --gpu N                       CUDA device (default 0)
#   --max_frames N, --stride K    video only: keep every K-th frame, at most N frames
#   --normals                     also precompute uni_normal/ + points3D.ply (otherwise the first training run does)
#   --work DIR                    intermediates (default DIR_OUT/_work/NAME)
#
# Each stage skips itself when its outputs exist, so a failed run can simply be restarted.
# Tool locations come from environment variables, see preprocess/env.sh.example.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"

VIDEO=""; FRAMES=""; NAME=""; OUT=data/DAVIS/scenes; MASKS=""; SAM3=""; GPU=0
MAX_FRAMES=0; STRIDE=1; NORMALS=0; WORK=""
while [ $# -gt 0 ]; do
  case $1 in
    --video) VIDEO=$2; shift 2;;
    --frames) FRAMES=$2; shift 2;;
    --name) NAME=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --masks) MASKS=$2; shift 2;;
    --sam3) SAM3=$2; shift 2;;
    --gpu) GPU=$2; shift 2;;
    --max_frames) MAX_FRAMES=$2; shift 2;;
    --stride) STRIDE=$2; shift 2;;
    --normals) NORMALS=1; shift;;
    --work) WORK=$2; shift 2;;
    -h|--help) sed -n 2,24p "$0"; exit 0;;
    *) echo "unknown option $1"; exit 1;;
  esac
done
[ -n "$NAME" ] || { echo "--name is required"; exit 1; }
[ -n "$VIDEO$FRAMES" ] || { echo "--video or --frames is required"; exit 1; }
[ -n "$MASKS$SAM3" ] || { echo "--masks and/or --sam3 is required (the method needs dynamic masks)"; exit 1; }

mkdir -p "$OUT"
export SCENE=$(cd "$OUT" && pwd)/$NAME
export WORK=${WORK:-$(cd "$OUT" && pwd)/_work/$NAME}
export NAME GPU HERE VIDEO FRAMES MASKS SAM3 MAX_FRAMES STRIDE
export WEBSPLINE_PY=${WEBSPLINE_PY:-python}
mkdir -p "$SCENE" "$WORK"
echo "=== [$NAME] -> $SCENE (work $WORK), GPU $GPU, $(date '+%F %T')"

"$HERE"/stages/01_frames.sh
"$HERE"/stages/02_megasam.sh
"$HERE"/stages/03_tracks.sh
"$HERE"/stages/04_masks.sh
[ $NORMALS = 1 ] && "$HERE"/stages/05_normals.sh
"$HERE"/stages/06_check.sh
echo "=== [$NAME] done $(date '+%F %T')"
