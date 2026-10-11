#!/bin/sh
# Upload an area's bake to the bucket: the cells at depth 9 and finer only, so the planet's coarser cells stay as they are, with the short cache header publish-cells.sh uses.
# Usage: upload-area.sh <cells dir>
set -e
cd "$(dirname "$0")/.."
DIR=$1
[ -d "$DIR" ] || { echo "usage: upload-area.sh <cells dir>" >&2; exit 2; }
cargo build --release -q -p mahere-tiles --example cell_sizes
LIST=$(mktemp)
target/release/examples/cell_sizes "$DIR" names 9 > "$LIST"
echo "$(wc -l < "$LIST") cells at depth 9 and finer"
rclone copy "$DIR" r2:holdmyoscilloscope/mahere/cells --files-from "$LIST" \
    --header-upload "Cache-Control: max-age=300" --ignore-times \
    --transfers 32 --checkers 32 --stats 5m --stats-one-line
rm -f "$LIST"
