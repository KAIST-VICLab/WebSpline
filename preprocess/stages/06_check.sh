#!/usr/bin/env bash
# Stage 6  sanity check of the finished scene directory.
source "$(dirname "$0")/common.sh"
$WEBSPLINE_PY "$HERE/tools/check_scene.py" "$SCENE"
