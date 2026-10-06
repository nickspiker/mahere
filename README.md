# mahere

An offline-first trail map, built from scratch.

mahere (Māori: *map*) is a Rust mapping app for backcountry and MTB use that
owns its whole stack: its own global coordinate system, its own binary tile
format, its own build pipeline from OpenStreetMap data, and its own
CPU-rendered cartography — fully user-configurable at render time, no
third-party basemaps, no protobuf, no GPU required.

## Why from scratch

Mainstream map stacks inherit Web Mercator's distortion, Mapbox Vector
Tiles' protobuf encoding, and styling baked into tiles at build time. mahere
replaces each deliberately:

- **Coordinates** ([docs/coord-spec.md](docs/coord-spec.md)): one `u64` is a
  position on Earth at ~6.6 mm resolution — 4 bits of icosahedral diamond ID
  plus 60 bits of Morton-interleaved face-local UV. Truncating low bits
  yields the enclosing quadtree cell, so a tile address is a coordinate
  prefix and containment tests are integer compares. Ground resolution is
  near-uniform globally: no pole singularities, no cos(latitude) anywhere.
  The closest relatives are Google's S2 (indexing only, ~2.1× cell-area
  spread) and astronomy's HEALPix/HiPS (equal-area diamonds serving all-sky
  imagery); mahere does the latter for Earth, on a better solid.
- **Tiles**: encoded in [VSF](https://github.com/nickspiker/vsf) instead of
  protobuf — O(1) skip, per-layer sections the renderer can ignore without
  parsing, bitpacked coordinate streams at exactly the bit width each tile
  needs.
- **Rendering**: CPU, front-to-back, via
  [fluor](https://github.com/nickspiker/fluor). Styling lives in a
  hot-reloadable text document, never in tiles; colour, stroke, transparency
  and effects are user-tweakable without retiling anything.

Data sources are all open: OpenStreetMap vectors (ODbL), USGS 3DEP elevation
and NAIP imagery (public domain).

## Status

Early. Working today:

- `mahere-coord` — the coordinate system, tested (round-trips < 2 cm
  globally, prefix/containment algebra, all diamonds reachable).
- `mahere-osm` — the ingest boundary: `.osm.pbf` in, codec-quantized
  coordinates out (a Washington-state extract loads in ~7 s).
- `mahere-dem` — the raster elevation boundary: USGS 3DEP GeoTIFFs in,
  bilinear elevation + gradient queries out.
- `mahere-tiles` + `mahere-load` — the bake: sources in, dymaxion **cells**
  out. A cell is one diamond-Morton rhombus whose texels are **triangles**:
  the icosahedron's own subdivision, 256×256 UV squares each split into a
  lower and an upper equilateral triangle (131072 texels), stored as a VSF
  file (whole-file zstd) at `{layer}/{cell}.vsf.zst` — the cell named by its
  flattened VSF value in base64url — a layout
  that is also the future object-store bucket. Triangles matter: the tiling
  has 6-fold symmetry and a line always crosses it edge-to-edge, so linework
  is isotropic (a rhombus grid draws +45° and −45° roads differently). The
  `dem` layer is elevation + gradient sampled at each triangle's centroid at
  the base depth, then a pyramid down to depth 6 where every texel is the
  mean of its four children — three corners and the inverted center; the
  `line` layer is every road, trail and stream stamped into the triangles it
  crosses at depth 13 as (class, coverage) texels, averaged up the same way —
  coverage up the pyramid is exact box filtering, so minor ways fade and
  towns glow with no styling tricks and no aliasing.
- `mahere-engine` — the `#pagetable` renderer. Pure raster: a frame is
  fetches. A 32 px block grid gets exact screen→diamond-UV corners; inside
  a block UV steps in Q30.16 fixed point; each block resolves its cell(s)
  once through a page table of decoded planes, falling back to resident
  parents while finer cells stream in from a loader thread. Per pixel:
  unpack a dem texel (u16 elevation, snorm16 normal), light it by the sun
  through a hypsometric LUT, lerp the line class colour by coverage. No
  vectors, no per-pixel hashing, no locks. Sun lives in screen space, so the
  terrain is lit from the top-left at any bearing.
- `mahere-app` (desktop, fluor) and `mahere-android` (chromeless, JNI) are
  thin frontends over the same `MapCore`: drag/pinch to pan, two-finger
  rotate, Q/E rotate, A/D/W/S move the sun, R goes home. Session and GPS
  tracks persist in a kete vault (`mahere-store`).

Measured: a 1024×768 frame of Mount Adams renders in ~1.4 ms on a desktop
CPU (the previous reservoir/vector engine took ~300 ms).

Next: landcover fills, hydro areas and contours as further layers (each
is a baker pass plus a compositor stage — the renderer core doesn't
change), the GUI cache and settings, and a Cloudflare R2 cell fetcher
behind the same `CellStore` trait.

## Building

```sh
# Grab an extract (any Geofabrik region works):
curl -L -o data/washington-latest.osm.pbf \
  https://download.geofabrik.de/north-america/us/washington-latest.osm.pbf

# And DEM tiles for the region (USGS 3DEP; 1/3 arc-second for 10 m):
curl -L -o data/USGS_13_n47w122.tif \
  "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n47w122/USGS_13_n47w122.tif"

# Bake cells for a lat/lon box (Mount Adams here): line pyramid 13..6,
# dem sampled at depth 11 (10 m source) with its pyramid down to 6.
cargo run --release -p mahere-tiles --bin mahere-load -- \
  --pbf data/washington-latest.osm.pbf --out data/cells \
  --bbox 46.0,-121.75,46.35,-121.30 --line-base 13 --line-min 6 \
  --dem-depths 11,6 --dem data/USGS_13_n47w122.tif

cargo run --release -p mahere-app   # Linux/macOS; reads data/cells if present
cargo test                          # coordinates, tiles, engine, store
```

Without a local `data/cells`, both frontends stream cells from the public
bucket (`https://brobdingnagian.holdmyoscilloscope.com/mahere/cells/…`,
the same `{layer}/{cell}.vsf.zst` layout) into the kete vault on
the device, and serve them from there afterwards — the Android APK carries
no map data. `scripts/publish-cells.sh` syncs a bake to the bucket.

## License

MIT or Apache-2.0, at your option.
