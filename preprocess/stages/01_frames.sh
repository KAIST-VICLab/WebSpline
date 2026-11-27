#!/usr/bin/env bash
# Stage 1  input -> $SCENE/images/%05d.jpg at 480p height (16:9 -> 854x480, as DAVIS), aspect kept.
source "$(dirname "$0")/common.sh"
if [ -d "$IMG" ] && [ "$(count "$IMG" '\.jpg$')" -gt 0 ]; then ts "frames exist ($(nframes)), skip"; exit 0; fi
mkdir -p "$IMG"
if [ -n "$VIDEO" ]; then
  sel="select='not(mod(n\,$STRIDE))',"
  lim=(); [ "$MAX_FRAMES" -gt 0 ] && lim=(-frames:v "$MAX_FRAMES")
  ffmpeg -loglevel error -i "$VIDEO" -vf "${sel}scale=-2:480:flags=area" -vsync vfr -q:v 2 -start_number 0 "${lim[@]}" "$IMG/%05d.jpg"
else
  $WEBSPLINE_PY "$HERE/tools/prepare_frames.py" "$FRAMES" "$IMG"
fi
ts "stage1 frames done ($(nframes) frames)"
