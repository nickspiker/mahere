#!/bin/sh
# Fetch an area's sources: the 1 m DEM tiles straight from USGS, and the NAIP quads from the Planetary Computer with a signed URL each. Lists live in data/areas/<name>/: every dem*-urls.txt (one per flight; an area can need two where the newer flight has holes) and naip-urls.txt. Files land in data/dem and data/naip-<name>, skipped when already there.
# DEM_DIR and NAIP_DIR put the files elsewhere (a big area's sources on the big drive); a file lands under a .part name and is renamed when whole, so an interrupted run resumes cleanly.
set -e
cd "$(dirname "$0")/.."
AREA=$1
[ -n "$AREA" ] || { echo "usage: fetch-area.sh <area>" >&2; exit 2; }
DEM_DIR=${DEM_DIR:-data/dem}
NAIP_DIR=${NAIP_DIR:-data/naip-$AREA}
export DEM_DIR NAIP_DIR
mkdir -p "$DEM_DIR" "$NAIP_DIR"
cat "data/areas/$AREA"/dem*-urls.txt | while read -r url; do
    f="$DEM_DIR/$(basename "$url")"
    [ -s "$f" ] || echo "$url"
done | xargs -r -P 4 -I{} sh -c 'f="$DEM_DIR/$(basename {})"; curl -sS -L --retry 5 -o "$f.part" "{}" && mv "$f.part" "$f" && echo "dem $(basename {})"'
while read -r url; do
    f="$NAIP_DIR/$(basename "$url")"
    [ -s "$f" ] || echo "$url"
done < "data/areas/$AREA/naip-urls.txt" | xargs -r -P 4 -I{} sh -c 'f="$NAIP_DIR/$(basename {})"; signed=$(curl -sS "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={}" | sed -n "s/.*\"href\":\"\([^\"]*\)\".*/\1/p"); curl -sS -L --retry 5 -o "$f.part" "$signed" && mv "$f.part" "$f" && echo "naip $(basename {})"'
du -sh "$DEM_DIR" "$NAIP_DIR"
