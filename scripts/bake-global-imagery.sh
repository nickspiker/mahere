#!/bin/sh
# Global imagery at the terrain's depth: the ESA WorldCover Sentinel-2 composite (2021 median, red, green, blue, near-infrared; CC-BY 4.0), only its 37 m level pulled from each one-degree file, baked into the same depth-8 cells as the terrain once the terrain bake has finished (the two must not write the same cells at once).
# Usage: scripts/bake-global-imagery.sh <terrain download log> <terrain bake log>
# Waits for the terrain download to end (so the two pulls do not share the line), pulls, waits for the terrain bake to end, bakes. Resumable: pulled files are skipped, finished regions are listed in <out>/.global-img-done.
set -e
cd "$(dirname "$0")/.."
SRC=/mnt/Geoduck/mahere-sources
INDEX=$SRC/worldcover-s2-rgbnir-2021.index
TILES=$SRC/worldcover-s2-rgbnir-2021-L2
OUT=data/global
DL_LOG=$1
BAKE_LOG=$2
cargo build --release -p mahere-tiles
if [ ! -s "$INDEX" ]; then
    rclone lsf -R --files-only --s3-provider=AWS --s3-region=eu-central-1 --s3-env-auth=false --fast-list ':s3:esa-worldcover-s2/rgbnir/2021/' > "$INDEX"
fi
while ! grep -q "^exit " "$DL_LOG" 2>/dev/null; do
    sleep 300
done
echo "$(date +%T) pulling $(wc -l < "$INDEX") squares"
# A second round picks up anything that failed the first (cog-level lists failures in .failed).
pull() {
    rm -f "$TILES/.failed"
    target/release/cog-level --base https://esa-worldcover-s2.s3.eu-central-1.amazonaws.com/rgbnir/2021/ --keys "$INDEX" --level 2 --out "$TILES" --jobs "$1" 2>&1 | grep -v "^FAILED" | tail -1
}
pull 16
[ -f "$TILES/.failed" ] && pull 4
echo "$(date +%T) pulled: $(du -sh "$TILES" | cut -f1)"
# The terrain bake's last line is its size, "<n>G<tab>data/global".
while ! grep -q "$OUT\$" "$BAKE_LOG" 2>/dev/null; do
    sleep 300
done
echo "$(date +%T) $(target/release/mahere-global --out "$OUT" --depth 8 --region 4 --img "$TILES" --img-index "$INDEX" 2>&1 | grep -v "^img tile" | tail -1)"
target/release/mahere-global --out "$OUT" --region 4 --img "$TILES" --top 2>&1 | tail -1
du -sh "$OUT"
