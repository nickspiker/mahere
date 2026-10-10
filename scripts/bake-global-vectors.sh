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
    # osmium keeps an id set per extract sized by the planet's largest id, about 1.3 GB each, so a pass may carry twenty extracts at most: the planet is cut into 30°×60° chunks in passes of twenty, then each chunk into its regions in passes of twenty.
    python3 - "$REGIONS_JSON" "$SRC/osm" <<'PY'
import json, sys, os
r = json.load(open(sys.argv[1]))
src = sys.argv[2]
os.makedirs(f"{src}/chunks", exist_ok=True)
groups = {}
for e in r['extracts']:
    b = e['bbox']
    key = (int((b['bottom'] + b['top']) / 2 // 30), int((b['left'] + b['right']) / 2 // 60))
    groups.setdefault(key, []).append(e)
chunks = []
for i, (key, ex) in enumerate(sorted(groups.items())):
    bb = {'left': min(e['bbox']['left'] for e in ex), 'bottom': min(e['bbox']['bottom'] for e in ex), 'right': max(e['bbox']['right'] for e in ex), 'top': max(e['bbox']['top'] for e in ex)}
    chunks.append({'output': f"chunk{i}.osm.pbf", 'output_format': 'pbf,add_metadata=false', 'bbox': bb})
    for j in range(0, len(ex), 20):
        part = dict(r)
        part['extracts'] = ex[j:j + 20]
        json.dump(part, open(f"{src}/chunks/chunk{i}.part{j // 20:03}.json", 'w'))
for j in range(0, len(chunks), 20):
    json.dump({'directory': f"{src}/chunks", 'extracts': chunks[j:j + 20]}, open(f"{src}/chunks/chunks.pass{j // 20}.json", 'w'))
print(len(chunks), 'chunks')
PY
    for pass in "$SRC"/osm/chunks/chunks.pass*.json; do
        echo "$(date +%T) cutting the planet: $(basename "$pass")"
        osmium extract -c "$pass" -s simple --overwrite "$SRC/osm/planet-latest.osm.pbf"
    done
    for part in "$SRC"/osm/chunks/chunk*.part*.json; do
        chunk="${part%%.part*}.osm.pbf"
        echo "$(date +%T) extracting $(basename "$part") from $(basename "$chunk")"
        osmium extract -c "$part" -s simple --overwrite "$chunk"
    done
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
