#!/bin/sh
# Ship the baked cell directory (the #pagetable renderer's entire diet) as
# APK assets. assets/ is gitignored.
set -e
cd "$(dirname "$0")"
rm -rf app/src/main/assets
mkdir -p app/src/main/assets
cp -r ../data/cells app/src/main/assets/cells
# Bake stamp: the app restages files/cells whenever it changes, so a
# rebaked cell with the same name never lingers on the phone.
echo "$(date +%s) $(du -sb ../data/cells | cut -f1)" > app/src/main/assets/cells/bake-stamp
echo "staged: $(du -sh app/src/main/assets/cells | cut -f1)"
