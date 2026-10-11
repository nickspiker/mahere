#!/bin/sh
# Bake a large shared area a piece at a time: one loader run per depth-9 cell of the shape, each holding only that piece's sources, merged into one cell directory. Pieces are listed in data/areas/<area>/pieces.txt as `<cell name> <dem files, comma separated, preferred first> <naip files>`; a piece runs as soon as all its files are in <src>/dem and <src>/naip (so it can follow a fetch still running) and is marked done in <out>/.done.
# Usage: bake-pieces.sh <area.vsf.zst> <area dir name> <src dir> <out dir>; PBF names the OSM extract (default data/washington-latest.osm.pbf).
set -e
cd "$(dirname "$0")/.."
AREA_FILE=$1
AREA=$2
SRC=$3
OUT=$4
[ -n "$OUT" ] || { echo "usage: bake-pieces.sh <area.vsf.zst> <area> <src> <out>" >&2; exit 2; }
PBF=${PBF:-data/washington-latest.osm.pbf}
cargo build --release -q -p mahere-tiles --bin mahere-load
mkdir -p "$OUT/.done"
total=$(wc -l < "data/areas/$AREA/pieces.txt")
while :; do
    left=0
    progressed=0
    while read -r name dems naips; do
        [ -e "$OUT/.done/$name" ] && continue
        left=$((left + 1))
        ready=1
        for f in $(echo "$dems" | tr ',' ' '); do [ -s "$SRC/dem/$f" ] || ready=0; done
        for f in $(echo "$naips" | tr ',' ' '); do [ -s "$SRC/naip/$f" ] || ready=0; done
        [ $ready = 1 ] || continue
        DEM=$(echo "$dems" | tr ',' '\n' | sed "s|^|$SRC/dem/|" | tr '\n' ' ')
        NAIP=$(echo "$naips" | tr ',' '\n' | sed "s|^|$SRC/naip/|" | tr '\n' ' ')
        echo "$(date +%T) piece $name ($((total - left + 1)) of $total)"
        # shellcheck disable=SC2086
        target/release/mahere-load --pbf "$PBF" --out "$OUT" --area "$AREA_FILE" --under "$name" --vec-base 14 --dem-base 14 --min 9 --dem $DEM --naip $NAIP > "$OUT/.done/$name.log" 2>&1 || true
        # Done only when the loader got as far as writing; a failed piece stops the run with its log.
        if grep -q "cells written" "$OUT/.done/$name.log"; then
            grep "cells written" "$OUT/.done/$name.log"
            touch "$OUT/.done/$name"
        else
            tail -5 "$OUT/.done/$name.log"
            echo "piece $name failed" >&2
            exit 1
        fi
        progressed=1
    done < "data/areas/$AREA/pieces.txt"
    [ $left = 0 ] && break
    [ $progressed = 1 ] || sleep 60
done
echo "$(date +%T) all $total pieces baked"
du -sh "$OUT"
