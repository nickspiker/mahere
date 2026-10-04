#!/bin/sh
# Copy the field-test DEM tiles (Mt Adams + Port Orchard coverage) into APK
# assets. Run once before building; assets/ is gitignored (110 MB).
set -e
cd "$(dirname "$0")"
mkdir -p app/src/main/assets/dem
cp ../data/USGS_1_n47w122.tif ../data/USGS_1_n48w123.tif app/src/main/assets/dem/
echo "staged: $(ls app/src/main/assets/dem)"
