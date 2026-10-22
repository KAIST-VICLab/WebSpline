#!/usr/bin/env bash
# Build one tar.gz per iPhone scene containing exactly what the loader reads,
# i.e. the layout documented in the README. Extracts to data/iphone/<scene>/.
#
#   ./scripts/pack_iphone_release.sh [outdir] [scene ...]
set -euo pipefail

SRC=${SRC:-data/iphone}
OUT=${1:-/hdd/jmpark/iphone_release}
shift || true
SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
    SCENES=(apple block paper-windmill space-out spin teddy wheel)
fi

# Everything scene/dataset_readers.py opens, and nothing else.
PARTS=(
    rgb/2x
    camera
    depth/2x
    uni_depth
    uni_normal
    masks/cam0
    covisible/2x/val
    splits
    scene.json
    emf.json
)

mkdir -p "$OUT"
for scene in "${SCENES[@]}"; do
    [ -d "$SRC/$scene" ] || { echo "!! missing $SRC/$scene"; exit 1; }

    members=()
    for p in "${PARTS[@]}"; do
        [ -e "$SRC/$scene/$p" ] || { echo "!! $scene is missing $p"; exit 1; }
        members+=("$scene/$p")
    done
    # tracks + keypoints, matched by glob
    for f in "$SRC/$scene"/*cotracker_tap.npz "$SRC/$scene"/*_train_pck.npz; do
        [ -e "$f" ] || { echo "!! $scene is missing a tracks/pck npz"; exit 1; }
        members+=("$scene/$(basename "$f")")
    done

    echo "=== packing $scene (${#members[@]} entries)"
    tar -czf "$OUT/iphone_$scene.tar.gz" -C "$SRC" "${members[@]}"
    echo "    -> $(du -h "$OUT/iphone_$scene.tar.gz" | cut -f1)"
done

echo "=== checksums"
( cd "$OUT" && sha256sum iphone_*.tar.gz > SHA256SUMS && cat SHA256SUMS )
echo "=== total: $(du -sh "$OUT" | cut -f1) in $OUT"
