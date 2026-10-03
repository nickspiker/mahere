# mahere coordinate specification

One `u64` is a position on Earth at ~6.6 mm resolution. Truncating its low
bits yields the enclosing quadtree cell; a tile address is a coordinate
prefix. Implemented by `mahere-coord`, which is the **only** place the
lat/lon ↔ u64 bijection may exist.

## Bit layout (MSB first)

```
[ 4 bits diamond 0–9 ][ 60 bits Morton(u, v) ]
```

The icosahedron's 20 triangles are paired into 10 rhombic diamonds
`[a0, q, r, b0]`: lower triangle (a0, q, r) and upper triangle (b0, q, r)
share edge q–r, the diamond's diagonal. UV basis: origin a0, u along a0→q,
v along a0→r, b0 at (1, 1). Each axis is quantized to 2^30 steps and the two
30-bit values are Morton-interleaved from their MSBs down, u in the higher
bit of each pair.

Projection per face is gnomonic (ray through the origin to the face plane),
as in the `icosahedron` repo this ports. Face selection is
closest-face-centroid, which for a regular icosahedron is exactly the radial
projection of the faces. The upper-triangle parametrization
(u, v) = (1 − β_r, 1 − β_q) is continuous across the diagonal.

## Datum rule (locked — changing it re-tiles the world)

WGS84 **geodetic** latitude/longitude is treated as spherical:
`xyz = (cos φ cos λ, cos φ sin λ, sin φ)` with geodetic φ. Explicitly NOT
ECEF-normalized (that silently yields geocentric latitude, up to ~21 km of
ground away) and NOT authalic (the ≤0.7 % area polish is below the
projection's own distortion). Same convention as Web Mercator, chosen
deliberately: all source data (OSM, GPX, NAIP, 3DEP) is geodetic WGS84.

Elevation never enters the codec. (3DEP heights are NAVD88 orthometric, GPS
heights ellipsoidal, ~20 m apart in WA — a display-side concern.)

## Orientation (locked 2026-10-03 — changing it re-tiles the world)

**Pole-vertex, Greenwich-aligned**: vertex 0 is the north pole, vertices
1–5 form the upper ring at lat atan(1/2) ≈ 26.565° (lons 0°, 72°, 144°,
216°, 288°), vertices 6–10 the lower ring at −26.565° (lons 36°, 108°, …),
vertex 11 the south pole. Shared verbatim with `vsf::WorldCoord`. The five
northern diamonds hang from the north pole to the lower ring, the five
southern rise to the upper ring, interlocking at the waist; cap-face edges
are meridian arcs, so the whole construction is 72°-periodic in longitude.

Chosen on mathematical grounds, Nick's call: the three canonical choices
put the poles on a 5-fold (vertex), 3-fold (face center) or 2-fold (edge
midpoint) symmetry axis. Vertex wins twice over — lat/lon's rotational
symmetry shares its largest cyclic subgroup with the solid (C5 > C3 > C2),
and the coordinate system's polar degeneracy lands exactly on the grid's
two degree-5 points, spending both singular structures on the same two
spots (Arctic Ocean, Antarctic plateau) instead of four. Land-weighted
distortion is orientation-insensitive (~1% spread, measured vs the prior
accidental golden/z-up orientation), so the symmetry argument decided.
Conversion cost is unaffected either way: encode/decode are
transcendental-dominated, and face-finding (the only orientation-sensitive
step) is already a cheap 20-way max-dot — the 72° periodicity permits a
fold-to-fundamental-sector fast path if ever wanted.

Consequences: the ten ring vertices (distortion maxima, ~2.0× area vs face
center; five-grid meeting points) sit at ±26.565°; nearest land vertices
are ~Algeria, ~Rajasthan, ~southern Mozambique; 1.65% of land is within
500 km of any vertex. Washington is a single diamond (D3), all of New
Zealand a single diamond (D6). The face pairing remains a free perfect
matching: a future region on an ugly seam can be re-buried by re-pairing
(cheaper than re-orienting, still a re-tile).

## Cells and anchoring

- A cell at depth d (0–30) is the top `4 + 2d` bits; depth 0 is a whole
  diamond, each depth quarters it.
- Containment is `(coord >> shift) == prefix`. Cell boundaries nest.
- Cells are **corner-anchored**: prefix + zeros = minimum-UV corner.
- To degrade a *point's* precision, never bare-truncate (it biases toward the
  corner): truncate and re-center — set the next bit of each axis
  (`Cell::center`), giving uniform ±half-cell error.

## Depth → ground size (cell edge, approximate; gnomonic variation a few %)

| depth | edge | depth | edge |
|------:|-----:|------:|-----:|
| 0 | 7,054 km | 16 | 108 m |
| 8 | 27.6 km | 20 | 6.7 m |
| 10 | 6.9 km | 25 | 21 cm |
| 12 | 1.7 km | 30 | 6.6 mm |

Cells are diamond-aligned rhombi on the ground, not north-aligned squares;
only the renderer's view transform cares.

## Not yet specified

- Cross-edge neighbor rules (needed for the raster warper and label
  collision across diamond seams).
- Tile-local coordinate convention for `mahere-tiles` (the next N Morton
  bits below the tile prefix — quantization is truncation).
