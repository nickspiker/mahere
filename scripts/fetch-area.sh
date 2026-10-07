#!/bin/sh
# Fetch an area's sources: the 1 m DEM tiles straight from USGS, and the NAIP quads from the Planetary Computer with a signed URL each. Lists live in data/areas/<name>/dem-urls.txt and naip-urls.txt; files land in data/dem and data/naip-<name>, skipped when already there.
set -e
cd "$(dirname "$0")/.."
AREA=$1
[ -n "$AREA" ] || { echo "usage: fetch-area.sh <area>" >&2; exit 2; }
while read -r url; do
    f="data/dem/$(basename "$url")"
    [ -s "$f" ] || echo "$url"
done < "data/areas/$AREA/dem-urls.txt" | xargs -r -P 4 -I{} sh -c 'curl -sS -L -o "data/dem/$(basename {})" "{}" && echo "dem $(basename {})"'
while read -r url; do
    f="data/naip-$AREA/$(basename "$url")"
    [ -s "$f" ] || echo "$url"
done < "data/areas/$AREA/naip-urls.txt" | xargs -r -P 4 -I{} sh -c 'signed=$(curl -sS "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={}" | sed -n "s/.*\"href\":\"\([^\"]*\)\".*/\1/p"); curl -sS -L -o "data/naip-'"$AREA"'/$(basename {})" "$signed" && echo "naip $(basename {})"'
du -sh data/dem data/naip-$AREA
