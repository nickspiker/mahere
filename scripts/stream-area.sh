#!/bin/sh
# An area at 1 m without ever holding its sources (Nick 2026-11-21: "stage and pipe the selected region, imagery and all, so we don't fill up my SSD"): piece by piece in the order of data/areas/<area>/pieces.txt, each piece's lidar and NAIP fetched into a staging directory while the piece before it bakes, the piece baked (terrain, imagery, lines, names, routes), its sea and land use laid in, published into data/published and its changed files uploaded, and then every staged source no later piece needs deleted.
# Usage: stream-area.sh <area> <out dir>. Run area_sources first (the piece lists). Once, before the first piece: the OSM extracts (OSM_EXTRACTS, Geofabrik paths, merged into one), the land use (PAD-US states in PADUS_STATES, the national MVUM roads and trails) and the sea polygons are fetched and prepared under the staging directory; SEED names other areas' bake directories to link in where they overlap (scripts/../examples/area_seed).
# Resumable: a piece with its marker in <out>/.stream is skipped, a staged file is fetched under a .part name and renamed when whole. Copy, never sync.
set -e
cd "$(dirname "$0")/.."
AREA=$1
OUT=$2
[ -n "$OUT" ] || { echo "usage: stream-area.sh <area> <out dir>" >&2; exit 2; }
A=data/areas/$AREA
DOC=$A/$AREA.vsf
STAGE=${STAGE:-data/stage/$AREA}
OSM_EXTRACTS=${OSM_EXTRACTS:-north-america/us/washington}
PADUS_STATES=${PADUS_STATES:-WA}
BUCKET=r2:holdmyoscilloscope/mahere/cells
mkdir -p "$OUT/.stream" "$STAGE/dem" "$STAGE/naip"
cap() { systemd-run --user --scope -q -p MemoryMax=28G "$@"; }
log() { echo "$(date +%T) $*"; }

cargo build --release -q -p mahere-tiles --bin mahere-load --bin mahere-sea --bin mahere-overlay --bin mahere-landuse --bin mahere-publish --example area --example area_seed

# get <url> <file>: fetched whole or not at all.
get() {
    [ -s "$2" ] && return 0
    curl -sS -L --retry 5 --retry-all-errors -o "$2.part" "$1" && mv "$2.part" "$2"
}

# The one-time sources.
if [ ! -s "$STAGE/osm.pbf" ]; then
    mkdir -p "$STAGE/osm"
    for x in $OSM_EXTRACTS; do
        log "osm $x"
        get "https://download.geofabrik.de/$x/$(basename "$x")-latest.osm.pbf" "$STAGE/osm/$(basename "$x").osm.pbf"
    done
    if [ "$(ls "$STAGE/osm" | wc -l)" -gt 1 ]; then
        osmium merge "$STAGE"/osm/*.osm.pbf -o "$STAGE/osm.pbf.part" -f pbf --overwrite && mv "$STAGE/osm.pbf.part" "$STAGE/osm.pbf"
    else
        mv "$STAGE"/osm/*.osm.pbf "$STAGE/osm.pbf"
    fi
    rm -rf "$STAGE/osm"
fi
if [ ! -s "$STAGE/sea/water-polygons-split-4326/water_polygons.shp" ]; then
    log "sea polygons"
    mkdir -p "$STAGE/sea"
    get https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip "$STAGE/sea/water.zip"
    (cd "$STAGE/sea" && unzip -q -o water.zip && rm water.zip)
fi
if [ ! -s "$STAGE/landuse/own.shp" ]; then
    log "land use"
    mkdir -p "$STAGE/landuse/src"
    for st in $PADUS_STATES; do
        url=$(curl -sS "https://www.sciencebase.gov/catalog/item/6759abcfd34edfeb8710a004?format=json" | jq -r ".files[] | select(.name == \"PADUS4_1_State_${st}_GDB_KMZ.zip\") | .url")
        get "$url" "$STAGE/landuse/src/padus-$st.zip"
        (cd "$STAGE/landuse/src" && unzip -q -o "padus-$st.zip" && rm "padus-$st.zip")
    done
    for f in S_USA.Road_MVUM S_USA.Trail_MVUM; do
        get "https://data.fs.usda.gov/geodata/edw/edw_resources/shp/$f.zip" "$STAGE/landuse/src/$f.zip"
        (cd "$STAGE/landuse/src" && unzip -q -o "$f.zip" && rm "$f.zip")
    done
    BOX=$(target/release/examples/area "$DOC" bbox)
    # shellcheck disable=SC2046
    target/release/mahere-landuse "$BOX" "$STAGE/landuse" $(find "$STAGE/landuse/src" -maxdepth 2 -name '*.gdb' -type d) --mvum "$(find "$STAGE/landuse/src" -name 'S_USA.Road_MVUM.shp' | head -1)" "$(find "$STAGE/landuse/src" -name 'S_USA.Trail_MVUM.shp' | head -1)"
    rm -rf "$STAGE/landuse/src"
fi
if [ -n "$SEED" ] && [ ! -e "$OUT/.stream/seeded" ]; then
    # shellcheck disable=SC2086
    target/release/examples/area_seed "$DOC" "$OUT" $SEED
    touch "$OUT/.stream/seeded"
fi

# fetch <piece line>: the piece's lidar and imagery into the staging directory.
fetch() {
    dems=$(echo "$1" | cut -d' ' -f2 | tr ',' ' ')
    naips=$(echo "$1" | cut -d' ' -f3 | tr ',' ' ')
    for f in $dems; do
        [ -s "$STAGE/dem/$f" ] || echo "dem $(grep -F "/$f" "$A/dem-urls.txt" | head -1)"
    done
    for f in $naips; do
        [ -s "$STAGE/naip/$f" ] || echo "naip $(grep -F "/$f" "$A/naip-urls.txt" | head -1)"
    done
}
fetch_piece() {
    fetch "$1" | STAGE=$STAGE xargs -r -P 6 -L 1 sh -c '
        f="$STAGE/$0/$(basename "$1")"
        [ -s "$f" ] && exit 0
        url=$1
        [ "$0" = naip ] && url=$(curl -sS "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href=$1" | sed -n "s/.*\"href\":\"\([^\"]*\)\".*/\1/p")
        curl -sS -L --retry 5 --retry-all-errors -o "$f.part" "$url" && mv "$f.part" "$f"'
}

total=$(wc -l < "$A/pieces.txt")
n=0
prefetch=""
while read -r line; do
    n=$((n + 1))
    name=${line%% *}
    [ -e "$OUT/.stream/$name" ] && continue
    # This piece's files (waiting on the prefetch if it is fetching them), then the next undone piece's in the background.
    [ -n "$prefetch" ] && wait "$prefetch" || true
    log "piece $name ($n of $total): fetching"
    fetch_piece "$line"
    next=$(awk -v n="$n" 'NR > n' "$A/pieces.txt" | while read -r l; do [ -e "$OUT/.stream/${l%% *}" ] || { echo "$l"; break; }; done)
    prefetch=""
    if [ -n "$next" ]; then
        fetch_piece "$next" > /dev/null 2>&1 &
        prefetch=$!
    fi
    log "piece $name: baking ($(du -sh "$STAGE" | cut -f1) staged)"
    DEM=$(echo "$line" | cut -d' ' -f2 | tr ',' '\n' | sed "s|^|$STAGE/dem/|" | tr '\n' ' ')
    NAIP=$(echo "$line" | cut -d' ' -f3 | tr ',' '\n' | sed "s|^|$STAGE/naip/|" | tr '\n' ' ')
    # shellcheck disable=SC2086
    cap target/release/mahere-load --pbf "$STAGE/osm.pbf" --out "$OUT" --area "$DOC" --under "$name" --vec-base 14 --dem-base 14 --min 9 --dem $DEM --naip $NAIP > "$OUT/.stream/$name.log" 2>&1 || true
    if ! grep -q "cells written" "$OUT/.stream/$name.log"; then
        tail -5 "$OUT/.stream/$name.log"
        log "piece $name failed"
        exit 1
    fi
    grep "cells written" "$OUT/.stream/$name.log"
    cap target/release/mahere-sea --out "$OUT" --area "$DOC" --shp "$STAGE/sea/water-polygons-split-4326/water_polygons.shp" --under "$name" >> "$OUT/.stream/$name.log" 2>&1
    cap target/release/mahere-overlay --out "$OUT" --area "$DOC" --src "$STAGE/landuse" --under "$name" >> "$OUT/.stream/$name.log" 2>&1
    # Published and uploaded: every layer file the publish wrote for this piece.
    touch "$OUT/.stream/$name.since"
    cap target/release/mahere-publish "$OUT" data/published --min-depth 9 >> "$OUT/.stream/$name.log" 2>&1
    find data/published/base data/published/dem data/published/img data/published/nir -type f -newer "$OUT/.stream/$name.since" | sed 's|^data/published/||' > "$OUT/.stream/$name.files"
    log "piece $name: uploading $(wc -l < "$OUT/.stream/$name.files") files"
    rclone copy data/published "$BUCKET" --files-from "$OUT/.stream/$name.files" --no-traverse --checksum --header-upload Cache-Control:max-age=300 --header-upload Content-Type:application/octet-stream --transfers 16 --checkers 16 --retries 10 --low-level-retries 20 2>&1 | grep -E "ERROR" | head -5
    # The staged sources no piece after this one names.
    keep=$(awk -v n="$n" 'NR > n { gsub(",", "\n", $2); gsub(",", "\n", $3); print $2; print $3 }' "$A/pieces.txt" | sort -u)
    for d in dem naip; do
        ls "$STAGE/$d" | grep -v '\.part$' | while read -r f; do
            echo "$keep" | grep -qxF "$f" || rm -f "$STAGE/$d/$f"
        done
    done
    touch "$OUT/.stream/$name"
    log "piece $name done ($(du -sh "$OUT" | cut -f1) baked)"
done < "$A/pieces.txt"
[ -n "$prefetch" ] && wait "$prefetch" || true
log "all $total pieces streamed"
