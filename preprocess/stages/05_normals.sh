#!/usr/bin/env bash
# Stage 5 (optional)  uni_normal/%05d.npy + placeholder points3D.ply, exactly what the DAVIS loader writes on its first run.
source "$(dirname "$0")/common.sh"
[ "$(count "$SCENE/uni_normal" '\.npy$')" = "$(nframes)" ] && [ -f "$SCENE/points3D.ply" ] && { ts "normals exist, skip"; exit 0; }
CUDA_VISIBLE_DEVICES="" $WEBSPLINE_PY "$HERE/tools/make_normals.py" "$SCENE"
ts "stage5 normals done"
