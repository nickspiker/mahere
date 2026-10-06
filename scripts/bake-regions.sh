#!/bin/sh
# Bake the test regions into a fresh cell directory and swap it in: Kitsap at 10 m, Mount St Helens at 1 m, and the Spirit Lake box with NAIP and lidar imagery. Run after a format or classification change, then scripts/publish-cells.sh.
# The three bakes merge into one directory (a 1 m region over a 10 m one composes), so the order is coarse first.
set -e
cd "$(dirname "$0")/.."
cargo build --release -p mahere-tiles
L=target/release/mahere-load
D=data
OUT=$D/cells-next
rm -rf "$OUT"
HELENS="$D/USGS_1M_10_x55y511_WA_FEMAHQ_2018_D18.tif $D/USGS_1M_10_x55y512_WA_FEMAHQ_2018_D18.tif $D/USGS_1M_10_x55y513_WA_FEMAHQ_2018_D18.tif $D/USGS_1M_10_x56y511_WA_FEMAHQ_2018_D18.tif $D/USGS_1M_10_x56y512_WA_FEMAHQ_2018_D18.tif $D/USGS_1M_10_x56y513_WA_FEMAHQ_2018_D18.tif $D/USGS_1M_10_x57y511_WA_FEMAHQ_2018_D18.tif $D/USGS_1M_10_x57y512_WA_FEMAHQ_2018_D18.tif $D/USGS_1M_10_x57y513_WA_FEMAHQ_2018_D18.tif"
$L --pbf $D/washington-latest.osm.pbf --out "$OUT" --bbox 47.30,-122.90,47.80,-122.30 --vec-base 13 --dem-base 10 --min 6 --dem $D/USGS_1_n48w123.tif $D/USGS_1_n47w123.tif
$L --pbf $D/washington-latest.osm.pbf --out "$OUT" --bbox 46.10,-122.32,46.30,-122.05 --vec-base 14 --dem-base 14 --min 6 --dem $HELENS
$L --pbf $D/washington-latest.osm.pbf --out "$OUT" --bbox 46.19,-122.22,46.27,-122.10 --vec-base 14 --dem-base 14 --min 6 --dem $HELENS --naip $D/naip/*.tif --laz $D/laz/*.laz
rm -rf $D/cells-previous
[ -d $D/cells ] && mv $D/cells $D/cells-previous
mv "$OUT" $D/cells
du -sh $D/cells
