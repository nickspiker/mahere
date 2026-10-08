#!/bin/sh
# Bake global terrain while the Copernicus download runs: every pass bakes the regions whose tiles have all arrived, then waits and goes again; once the download has finished, one last pass and the levels above the regions. Resumable at any point (finished regions are listed in <out>/.global-done).
# Usage: scripts/bake-global.sh <download log>
set -e
cd "$(dirname "$0")/.."
SRC=/mnt/Geoduck/mahere-sources/copernicus-dem-30m
INDEX=/mnt/Geoduck/mahere-sources/copernicus-dem-30m.index
OUT=data/global
LOG=$1
cargo build --release -p mahere-tiles
pass() {
    target/release/mahere-global --src "$SRC" --index "$INDEX" --out "$OUT" --depth 8 --region 4 --dem-loss 2.0 2>&1 | grep -v "^dem tile" | tail -1
}
while ! grep -q "^exit " "$LOG" 2>/dev/null; do
    echo "$(date +%T) pass: $(pass)"
    sleep 600
done
echo "$(date +%T) final pass: $(pass)"
target/release/mahere-global --out "$OUT" --region 4 --dem-loss 2.0 --top
du -sh "$OUT"
