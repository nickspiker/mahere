#!/bin/sh
# Land ownership and motor vehicle use over an area's baked cells, a piece at a time (the pieces of data/areas/<area>/pieces.txt), from the shapefiles scripts/landuse-prep.py wrote into <src>.
# Usage: overlay-pieces.sh <area.vsf.zst> <area> <src> <out>
set -e
cd "$(dirname "$0")/.."
AREA_FILE=$1
AREA=$2
SRC=$3
OUT=$4
[ -n "$OUT" ] || { echo "usage: overlay-pieces.sh <area.vsf.zst> <area> <src> <out>" >&2; exit 2; }
cargo build --release -q -p mahere-tiles --bin mahere-overlay
while read -r name _; do
    target/release/mahere-overlay --out "$OUT" --area "$AREA_FILE" --src "$SRC" --under "$name" 2>&1 | grep -E "cells written|panicked|error" | sed "s/^/$name /"
done < "data/areas/$AREA/pieces.txt"
