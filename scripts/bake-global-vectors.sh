#!/bin/sh
# The global vector layers into the depth-8 cells, after the imagery bake has finished with them (two bakers must not write the same cells at once): land cover and water from ESA WorldCover's class map, rivers from HydroRIVERS, then, once the OSM planet has downloaded, its cut into the regions with osmium and the roads, rail, power and boundaries from each; the line, land and water pyramids above the regions last. Resumable: each stage keeps a done list in <out>.
# Usage: scripts/bake-global-vectors.sh <planet download log> <regions config json>
set -e
cd "$(dirname "$0")/.."
SRC=/mnt/Geoduck/mahere-sources
OUT=data/global
PLANET_LOG=$1
REGIONS_JSON=$2
cargo build --release -p mahere-tiles
# The imagery bake, if it is still running, finishes first.
while pgrep -f "mahere-globa[l] --out $OUT" >/dev/null; do
    sleep 120
done
echo "$(date +%T) land cover"
target/release/mahere-global --out "$OUT" --depth 8 --region 4 --land "$SRC/worldcover-map-2021-L2" --land-index "$SRC/worldcover-map-2021.index" 2>&1 | grep -v "^img tile" | tail -1
echo "$(date +%T) rivers"
target/release/mahere-global --out "$OUT" --depth 8 --region 4 --rivers "$SRC/hydrorivers/HydroRIVERS_v10_shp/HydroRIVERS_v10.shp" 2>&1 | tail -1
# The planet: wait for the download, cut it into the regions (one pass, the simple strategy: ways are clipped at a region's edge, which the margin in the boxes covers), join the pairs split at the antimeridian.
while ! grep -q "^exit 0" "$PLANET_LOG" 2>/dev/null; do
    sleep 300
done
if [ ! -f "$SRC/osm/regions/.cut-done" ]; then
    echo "$(date +%T) cutting the planet into $(grep -c '"output"' "$REGIONS_JSON") extracts"
    mkdir -p "$SRC/osm/regions"
    osmium extract -c "$REGIONS_JSON" -s simple --overwrite "$SRC/osm/planet-latest.osm.pbf"
    for a in "$SRC"/osm/regions/*-a.osm.pbf; do
        [ -f "$a" ] || continue
        b="${a%-a.osm.pbf}-b.osm.pbf"
        osmium merge --overwrite -o "${a%-a.osm.pbf}.osm.pbf" "$a" "$b" && rm -f "$a" "$b"
    done
    touch "$SRC/osm/regions/.cut-done"
fi
echo "$(date +%T) osm lines"
target/release/mahere-global --out "$OUT" --depth 8 --region 4 --osm "$SRC/osm/regions" 2>&1 | tail -1
echo "$(date +%T) the levels above the regions"
target/release/mahere-global --out "$OUT" --region 4 --osm "$SRC/osm/regions" --top 2>&1 | tail -1
du -sh "$OUT"
