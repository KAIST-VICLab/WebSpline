#!/usr/bin/env bash
# Stage 2  frames -> MegaSaM depth + cameras (Depth-Anything + UniDepth priors, DROID-style tracking, CVD optimisation).
#  out: $SCENE/depth/%05d.npy   (H,W,1) f32 depth, MegaSaM's arbitrary scale
#       $SCENE/pose/%05d.npy    (4,4) f32 camera-to-world (OpenCV axes)
#       $SCENE/pose/intrinsics.npy  (3,3) f32 K at full resolution
#  MegaSaM keeps its caches inside its checkout, keyed by NAME (Depth-Anything/video_visualization, UniDepth/outputs,
#  outputs, reconstructions, cache_flow, outputs_cvd).
source "$(dirname "$0")/common.sh"
[ -f "$SCENE/pose/intrinsics.npy" ] && [ "$(count "$SCENE/depth" '\.npy$')" = "$(nframes)" ] && { ts "megasam exists, skip"; exit 0; }
need MEGASAM_DIR MEGASAM_PY
N=$(nframes); PY=$MEGASAM_PY
export CUDA_VISIBLE_DEVICES=$GPU
(
  cd "$MEGASAM_DIR"
  if [ ! -f "outputs_cvd/${NAME}_sgd_cvd_hr.npz" ]; then
    [ "$(count "Depth-Anything/video_visualization/$NAME" .)" = "$N" ] || \
      $PY Depth-Anything/run_videos.py --encoder vitl --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
        --img-path "$IMG" --outdir "Depth-Anything/video_visualization/$NAME"
    [ "$(count "UniDepth/outputs/$NAME" .)" = "$N" ] || \
      PYTHONPATH="$MEGASAM_DIR/UniDepth" $PY UniDepth/scripts/demo_mega-sam.py --scene-name "$NAME" --img-path "$IMG" --outdir UniDepth/outputs
    $PY camera_tracking_scripts/test_demo.py --datapath="$IMG" --weights=checkpoints/megasam_final.pth --scene_name "$NAME" \
        --mono_depth_path "$MEGASAM_DIR/Depth-Anything/video_visualization" --metric_depth_path "$MEGASAM_DIR/UniDepth/outputs" --disable_vis
    $PY cvd_opt/preprocess_flow.py --datapath="$IMG" --model=cvd_opt/raft-things.pth --scene_name "$NAME" --mixed_precision
    $PY cvd_opt/cvd_opt.py --scene_name "$NAME" --w_grad 2.0 --w_normal 5.0
  fi
)
$WEBSPLINE_PY "$HERE/tools/export_megasam.py" "$IMG" "$MEGASAM_DIR/outputs_cvd/${NAME}_sgd_cvd_hr.npz" "$SCENE/depth" "$SCENE/pose"
ts "stage2 megasam done"
