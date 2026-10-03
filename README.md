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
- `mahere-app` — a fluor window rendering the diamond/quadtree world view as
  a visual test of the codec.

Next, in order: the VSF tile spec (`mahere-tiles`), the OSM → tile build
pipeline (`mahere-tiler`), DEM-derived hillshade and contours, imagery.

## Building

```sh
cargo run -p mahere-app   # Linux; opens the world view
cargo test                # coordinate-system tests
```

## License

MIT or Apache-2.0, at your option.
