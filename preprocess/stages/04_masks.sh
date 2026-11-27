#!/usr/bin/env bash
# Stage 4  dynamic masks -> $SCENE/masks/%05d.png (0/255).
#   --masks only : the given masks, binarised (>0 = dynamic)
#   --sam3       : (a) static-scene reprojection residual from the MegaSaM depth + cameras = motion evidence,
#                  (b) SAM3 text-prompted video segmentation; each tracked object gets ONE keep/drop decision for
#                      the whole video (amorphous stuff kept; shadows kept next to the anchor objects; other objects
#                      kept if they overlap the given masks or move), union with the given masks if any.
source "$(dirname "$0")/common.sh"
N=$(nframes)
[ "$(count "$SCENE/masks" '\.png$')" = "$N" ] && { ts "masks exist, skip"; exit 0; }
GT=""
if [ -n "$MASKS" ]; then
  GT=$WORK/masks_given
  [ "$(count "$GT" '\.png$')" = "$N" ] || $WEBSPLINE_PY "$HERE/tools/binarize_masks.py" "$MASKS" "$GT" "$N"
fi
if [ -z "$SAM3" ]; then
  rm -rf "$SCENE/masks"; cp -r "$GT" "$SCENE/masks"
else
  need SAM3_PY
  PROMPTS=$SAM3
  if [ "$SAM3" = davis ]; then
    PROMPTS=$($SAM3_PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))[sys.argv[2]]))" "$HERE/prompts_davis.json" "$NAME")
  fi
  [ -f "$WORK/motion_raw.npz" ] || $SAM3_PY "$HERE/tools/motion_evidence.py" "$SCENE" "$WORK/motion_raw.npz" ${GT:+--gt "$GT"}
  CUDA_VISIBLE_DEVICES=$GPU $SAM3_PY "$HERE/tools/sam3_masks.py" "$SCENE" "$PROMPTS" "$WORK/motion_raw.npz" "$WORK/masks_sam3" \
      --meta "$WORK/sam3_objects.json" ${GT:+--gt "$GT"}
  rm -rf "$SCENE/masks"; cp -r "$WORK/masks_sam3" "$SCENE/masks"
fi
[ "$(count "$SCENE/masks" '\.png$')" = "$N" ]
ts "stage4 masks done"
