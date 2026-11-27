#!/usr/bin/env bash
# Stage 3  MoSca precompute -> BootsTAPIR long-term 2D tracks (UniDepth-v2 depth, RAFT flow and EPI motion masks are
#  computed on the way to place the dynamic tracks).
#  out: $SCENE/uniform_dep=uni_bootstapir_tap.npz  (uniform grid; static tracks for bundle adjustment)
#       $SCENE/dynamic_dep=uni_bootstapir_tap.npz  (resampled on the moving regions)
#       keys tracks (T,N,2) f32 pixel xy, visibility (T,N) bool, queries
source "$(dirname "$0")/common.sh"
U="$SCENE/uniform_dep=uni_bootstapir_tap.npz"; D="$SCENE/dynamic_dep=uni_bootstapir_tap.npz"
[ -f "$U" ] && [ -f "$D" ] && { ts "tracks exist, skip"; exit 0; }
need MOSCA_DIR MOSCA_PY
WS=$WORK/mosca_ws; N=$(nframes)
if [ ! -f "$WS/dynamic_dep=uni_bootstapir_tap.npz" ] || [ ! -f "$WS/uniform_dep=uni_bootstapir_tap.npz" ]; then
  mkdir -p "$WS/images"
  [ "$(count "$WS/images" '\.jpg$')" = "$N" ] || cp "$IMG"/*.jpg "$WS/images/"
  (
    [ -n "${MOSCA_ACTIVATE:-}" ] && source "$MOSCA_ACTIVATE"
    export CUDA_VISIBLE_DEVICES=$GPU
    cd "$MOSCA_DIR"
    $MOSCA_PY mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws "$WS" \
      --dep_mode=uni --tap_mode=bootstapir --boundary_enhance_th=-1.0 > "$WORK/mosca_precompute.log" 2>&1
  ) || { echo "MoSca precompute failed, see $WORK/mosca_precompute.log"; exit 1; }
fi
cp "$WS/uniform_dep=uni_bootstapir_tap.npz" "$U"
cp "$WS/dynamic_dep=uni_bootstapir_tap.npz" "$D"
ts "stage3 tracks done"
