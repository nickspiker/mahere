#!/bin/sh
# Copy the field-test DEM tiles (Mt Adams + Port Orchard coverage) into APK
# assets. Run once before building; assets/ is gitignored (110 MB).
set -e
cd "$(dirname "$0")"
mkdir -p app/src/main/assets/dem
rm -f app/src/main/assets/dem/USGS_1_n47w122.tif
cp ../data/USGS_13_n47w122.tif ../data/USGS_1_n48w123.tif app/src/main/assets/dem/
cp ../data/featpack-fieldtest.vsf app/src/main/assets/features.vsf
echo "staged:"; ls -la app/src/main/assets/dem app/src/main/assets/*.vsf
