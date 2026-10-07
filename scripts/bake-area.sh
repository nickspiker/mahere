#!/bin/sh
# Bake one area into a fresh cell directory and swap it in. The area's box and sources are declared below; vectors and terrain at depth 14 (1.2 m texels) across the whole box, imagery wherever NAIP quads are present. Run after scripts/fetch-area.sh, then scripts/publish-cells.sh.
set -e
cd "$(dirname "$0")/.."
AREA=$1
case "$AREA" in
    leavenworth)
        BOX=47.412343,-120.859099,47.648857,-120.571077
        DEMS=$(ls data/dem/USGS_1M_10_x6[678]y52[6789]_WA_CentralWildfire_D22.tif)
        NAIP=$(ls data/naip-$AREA/*.tif 2>/dev/null || true)
        ;;
    *) echo "unknown area $AREA" >&2; exit 2 ;;
esac
cargo build --release -p mahere-tiles
OUT=data/cells-next
rm -rf "$OUT"
IMAGERY=""
[ -n "$NAIP" ] && IMAGERY="$IMAGERY --naip $NAIP"
# shellcheck disable=SC2086
target/release/mahere-load --pbf data/washington-latest.osm.pbf --out "$OUT" --bbox "$BOX" --vec-base 14 --dem-base 14 --min 6 --dem $DEMS $IMAGERY
rm -rf data/cells-previous
[ -d data/cells ] && mv data/cells data/cells-previous
mv "$OUT" data/cells
du -sh data/cells
