# sourced by every stage (run.sh exports SCENE, WORK, NAME, GPU, HERE, ...)
set -euo pipefail
ts() { echo "=== [$NAME] $1 $(date '+%F %T')"; }
need() { for v in "$@"; do [ -n "${!v:-}" ] || { echo "environment variable $v is not set (see preprocess/env.sh.example)"; exit 1; }; done; }
IMG=$SCENE/images
count() { if [ -d "$1" ]; then ls "$1" | grep -c "$2" || true; else echo 0; fi; }   # files in $1 matching $2
nframes() { count "$IMG" '\.jpg$'; }
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-30}
