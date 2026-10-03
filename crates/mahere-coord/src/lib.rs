//! mahere-coord: the canonical coordinate system for mahere.
//!
//! A position on Earth is one `u64` ([`Coord`]): 4 bits of diamond ID plus
//! 2×30 bits of Morton-interleaved face-local UV on an icosahedron whose 20
//! triangles are paired into 10 rhombic diamonds. Resolution at full depth is
//! ~6.6 mm of ground. Truncating the low bits yields the enclosing quadtree
//! cell ([`Cell`]); a tile address is a coordinate prefix.
//!
//! Datum rule (locked — changing it re-tiles the world): WGS84 *geodetic*
//! latitude/longitude is treated as spherical. The sphere point is
//! `(cos φ cos λ, cos φ sin λ, sin φ)` with geodetic φ — explicitly NOT
//! ECEF-normalized (that would be geocentric latitude, a different mapping,
//! up to ~21 km of ground away). This bijection lives here and only here.
//!
//! Anchoring: a [`Cell`] identifies an area and is corner-anchored (prefix +
//! zeros = minimum UV corner). To degrade a *point*, use [`Cell::center`],
//! which re-centers instead of biasing toward the corner.
//!
//! Elevation never enters this codec; it is a separately sampled DEM value.
//!
//! Projection geometry is ported from Nick's `icosahedron` codec
//! (face-finding via closest centroid, gnomonic projection, barycentric UV),
//! re-laid-out from face × BASE² packing to diamond-Morton prefix coding.

/// Pole-vertex icosahedron, unit vertices (locked orientation): vertex 0 is
/// the north pole, 1–5 the upper ring at lat atan(1/2) ≈ 26.565° (lons 0°,
/// 72°, …), 6–10 the lower ring at −atan(1/2) (lons 36°, 108°, …), 11 the
/// south pole. Chosen 2026-10-03 on symmetry grounds: the polar axis is a
/// 5-fold axis, so lat/lon's rotational symmetry shares its largest cyclic
/// subgroup with the solid, and the coordinate system's polar degeneracy
/// coincides with the grid's two degree-5 points. Ring components are
/// cos/sin of 72° multiples scaled by 2/√5, z = ±1/√5.
const VERTICES: [[f64; 3]; 12] = [
    [0., 0., 1.0],
    [0.8944271909999159, 0., 0.4472135954999579],
    [0.27639320225002106, 0.8506508083520399, 0.4472135954999579],
    [-0.7236067977499788, 0.5257311121191337, 0.4472135954999579],
    [-0.723606797749979, -0.5257311121191335, 0.4472135954999579],
    [0.27639320225002084, -0.85065080835204, 0.4472135954999579],
    [0.7236067977499789, 0.5257311121191336, -0.4472135954999579],
    [-0.27639320225002095, 0.85065080835204, -0.4472135954999579],
    [-0.8944271909999159, 0., -0.4472135954999579],
    [-0.2763932022500211, -0.8506508083520399, -0.4472135954999579],
    [0.7236067977499788, -0.5257311121191338, -0.4472135954999579],
    [0., 0., -1.0],
];

/// The 10 diamonds as vertex indices `[a0, q, r, b0]`: lower triangle
/// (a0, q, r) and upper triangle (b0, q, r) share the edge q–r (the diamond's
/// diagonal). UV basis: origin a0, u along a0→q, v along a0→r; b0 sits at
/// (1, 1). Pairing covers all 20 icosahedron faces exactly once.
const DIAMONDS: [[usize; 4]; 10] = [
    [0, 1, 2, 6],
    [0, 2, 3, 7],
    [0, 3, 4, 8],
    [0, 4, 5, 9],
    [0, 1, 5, 10],
    [11, 6, 7, 2],
    [11, 7, 8, 3],
    [11, 8, 9, 4],
    [11, 9, 10, 5],
    [11, 6, 10, 1],
];

/// Each diamond's two triangles as `(vertex indices, diamond, is_upper)` —
/// the face table the encoder searches. Derived from [`DIAMONDS`].
const FACES: [([usize; 3], u8, bool); 20] = {
    let mut faces = [([0usize; 3], 0u8, false); 20];
    let mut d = 0;
    while d < 10 {
        let [a0, q, r, b0] = DIAMONDS[d];
        faces[d * 2] = ([a0, q, r], d as u8, false);
        faces[d * 2 + 1] = ([b0, q, r], d as u8, true);
        d += 1;
    }
    faces
};

/// Quadtree depth at which one UV step is one coordinate unit.
pub const MAX_DEPTH: u8 = 30;
const UV_BITS: u32 = 30;
const UV_STEPS: u64 = 1 << UV_BITS; // 2^30 per axis
const MORTON_BITS: u32 = 60;

/// Mean Earth radius in meters (spherical datum, consistent with the rule
/// that geodetic lat/lon is treated as spherical).
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Arc length of an icosahedron edge on the unit sphere: atan(2).
const EDGE_ARC_RAD: f64 = 1.107148717794090503;

/// Approximate ground length of a cell edge at `depth`, in meters.
/// Depth 0 is a whole diamond (~7,054 km); each depth halves it; depth 30 is
/// ~6.6 mm. Gnomonic distortion varies this by a few percent across a face.
pub fn cell_edge_m(depth: u8) -> f64 {
    EDGE_ARC_RAD * EARTH_RADIUS_M / (1u64 << depth) as f64
}

/// A point on Earth at full (~6.6 mm) resolution.
///
/// Bit layout, MSB first: `[4 bits diamond 0–9][60 bits Morton(u, v)]`, where
/// the Morton field interleaves u and v from their MSBs down, u in the
/// higher bit of each pair.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Coord(u64);

/// A quadtree cell: the first `4 + 2·depth` bits of a [`Coord`], low bits
/// zero. Depth 0 is a whole diamond; depth `d` cells are the 4^d children.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Cell {
    bits: u64,
    depth: u8,
}

impl Coord {
    /// Encode WGS84 geodetic latitude/longitude (degrees), treated as
    /// spherical per the datum rule.
    pub fn from_lat_lon(lat_deg: f64, lon_deg: f64) -> Coord {
        let (phi, lam) = (lat_deg.to_radians(), lon_deg.to_radians());
        Coord::from_xyz([phi.cos() * lam.cos(), phi.cos() * lam.sin(), phi.sin()])
    }

    /// Encode a direction from the sphere's center (need not be unit length).
    pub fn from_xyz(p: [f64; 3]) -> Coord {
        // Containing face = face whose centroid direction is nearest: for a
        // regular icosahedron the spherical Voronoi of face centroids is
        // exactly the radial projection of the faces.
        let mut best = 0;
        let mut best_dot = f64::NEG_INFINITY;
        for (i, (verts, _, _)) in FACES.iter().enumerate() {
            let c = centroid(verts);
            let dot = c[0] * p[0] + c[1] * p[1] + c[2] * p[2];
            if dot > best_dot {
                best_dot = dot;
                best = i;
            }
        }
        let (verts, diamond, upper) = FACES[best];
        let (s, t) = gnomonic_barycentric(p, verts);
        // Lower triangle (a0,q,r): UV = barycentric directly. Upper triangle
        // (b0,q,r): fp = βb·b0 + βq·q + βr·r with (βq, βr) = (s, t), and the
        // continuous diamond parametrization is u = 1−βr, v = 1−βq.
        let (u, v) = if upper { (1. - t, 1. - s) } else { (s, t) };
        let iu = quantize(u);
        let iv = quantize(v);
        Coord(((diamond as u64) << MORTON_BITS) | (spread(iu) << 1) | spread(iv))
    }

    /// Decode to a unit vector (the center of the finest cell).
    pub fn to_xyz(self) -> [f64; 3] {
        let [a0, q, r, b0] = DIAMONDS[self.diamond() as usize];
        let (iu, iv) = self.uv();
        let u = (iu as f64 + 0.5) / UV_STEPS as f64;
        let v = (iv as f64 + 0.5) / UV_STEPS as f64;
        let fp = if u + v <= 1. {
            lerp3(VERTICES[a0], VERTICES[q], VERTICES[r], 1. - u - v, u, v)
        } else {
            lerp3(VERTICES[b0], VERTICES[q], VERTICES[r], u + v - 1., 1. - v, 1. - u)
        };
        normalize(fp)
    }

    /// Decode to WGS84 geodetic (latitude, longitude) in degrees.
    pub fn to_lat_lon(self) -> (f64, f64) {
        let [x, y, z] = self.to_xyz();
        (z.asin().to_degrees(), y.atan2(x).to_degrees())
    }

    /// Diamond ID, 0–9.
    pub fn diamond(self) -> u8 {
        (self.0 >> MORTON_BITS) as u8
    }

    /// Face-local (u, v), each 0..2^30.
    pub fn uv(self) -> (u64, u64) {
        (compact(self.0 >> 1), compact(self.0))
    }

    /// The cell containing this coordinate at `depth` (≤ [`MAX_DEPTH`]).
    pub fn cell(self, depth: u8) -> Cell {
        assert!(depth <= MAX_DEPTH);
        Cell { bits: self.0 & prefix_mask(depth), depth }
    }

    pub fn raw(self) -> u64 {
        self.0
    }

    /// Rebuild from a raw value (diamond bits must be 0–9).
    pub fn from_raw(raw: u64) -> Coord {
        assert!(raw >> MORTON_BITS <= 9, "diamond ID out of range");
        Coord(raw)
    }
}

impl Cell {
    /// True if `coord` lies inside this cell: one shift and compare.
    pub fn contains(self, coord: Coord) -> bool {
        coord.0 & prefix_mask(self.depth) == self.bits
    }

    /// True if `other` is this cell or a descendant of it.
    pub fn contains_cell(self, other: Cell) -> bool {
        other.depth >= self.depth && other.bits & prefix_mask(self.depth) == self.bits
    }

    /// The enclosing cell one level up. Depth 0 has no parent.
    pub fn parent(self) -> Option<Cell> {
        (self.depth > 0).then(|| Cell {
            bits: self.bits & prefix_mask(self.depth - 1),
            depth: self.depth - 1,
        })
    }

    /// Child `i` (0–3): the next Morton bit pair.
    pub fn child(self, i: u8) -> Cell {
        assert!(i < 4 && self.depth < MAX_DEPTH);
        Cell {
            bits: self.bits | ((i as u64) << (MORTON_BITS - 2 * (self.depth as u32 + 1))),
            depth: self.depth + 1,
        }
    }

    /// The cell's center as a full-resolution coordinate (next bit of each
    /// axis set — the re-centering rule for degraded points).
    pub fn center(self) -> Coord {
        if self.depth == MAX_DEPTH {
            return Coord(self.bits);
        }
        Coord(self.bits | (0b11 << (MORTON_BITS - 2 * (self.depth as u32 + 1))))
    }

    pub fn depth(self) -> u8 {
        self.depth
    }

    /// Raw prefix value (low bits zero).
    pub fn raw(self) -> u64 {
        self.bits
    }
}

/// Bits kept by a cell at `depth`: the diamond ID plus `2·depth` Morton bits.
fn prefix_mask(depth: u8) -> u64 {
    // depth 30 keeps everything; !0 << 0 would also work but be explicit.
    if depth as u32 >= MAX_DEPTH as u32 {
        !0
    } else {
        !0 << (MORTON_BITS - 2 * depth as u32)
    }
}

fn quantize(x: f64) -> u64 {
    ((x * UV_STEPS as f64) as i64).clamp(0, UV_STEPS as i64 - 1) as u64
}

/// Spread the low 30 bits of `x` into the even bit positions.
fn spread(x: u64) -> u64 {
    let mut x = x & 0x3FFF_FFFF;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

/// Inverse of [`spread`]: gather the even bit positions into the low 30 bits.
fn compact(x: u64) -> u64 {
    let mut x = x & 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF;
    x & 0x3FFF_FFFF
}

fn centroid(verts: &[usize; 3]) -> [f64; 3] {
    let (a, b, c) = (VERTICES[verts[0]], VERTICES[verts[1]], VERTICES[verts[2]]);
    [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]
}

/// Gnomonic projection of direction `p` onto the plane of triangle `verts`,
/// returned as barycentric (s, t): fp = A + s·(B−A) + t·(C−A).
fn gnomonic_barycentric(p: [f64; 3], verts: [usize; 3]) -> (f64, f64) {
    let (a, b, c) = (VERTICES[verts[0]], VERTICES[verts[1]], VERTICES[verts[2]]);
    let v0 = sub(b, a);
    let v1 = sub(c, a);
    let n = cross(v0, v1);
    // Ray origin→p meets the plane n·x = n·a at t = (n·a)/(n·p); the face
    // always subtends the ray for points chosen by closest-centroid.
    let scale = dot(n, a) / dot(n, p);
    let fp = [p[0] * scale, p[1] * scale, p[2] * scale];
    let v2 = sub(fp, a);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    ((d11 * d20 - d01 * d21) / denom, (d00 * d21 - d01 * d20) / denom)
}

fn lerp3(a: [f64; 3], b: [f64; 3], c: [f64; 3], wa: f64, wb: f64, wc: f64) -> [f64; 3] {
    [
        wa * a[0] + wb * b[0] + wc * c[0],
        wa * a[1] + wb * b[1] + wc * c[1],
        wa * a[2] + wb * b[2] + wc * c[2],
    ]
}

fn normalize(p: [f64; 3]) -> [f64; 3] {
    let m = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    [p[0] / m, p[1] / m, p[2] / m]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ground_error_m(lat: f64, lon: f64, coord: Coord) -> f64 {
        let (phi, lam) = (lat.to_radians(), lon.to_radians());
        let p = [phi.cos() * lam.cos(), phi.cos() * lam.sin(), phi.sin()];
        let q = coord.to_xyz();
        // Chord distance, not acos(dot): acos has a ~1.5e-8 rad (~10 cm)
        // precision floor near zero angle, far above the codec's resolution.
        let d = sub(p, q);
        dot(d, d).sqrt() * EARTH_RADIUS_M
    }

    #[test]
    fn morton_round_trips() {
        for x in [0u64, 1, 2, 0x3FFF_FFFF, 0x2AAA_AAAA, 0x1555_5555, 123_456_789] {
            assert_eq!(compact(spread(x)), x);
        }
        // u and v don't bleed into each other
        let c = Coord(((3u64) << 60) | (spread(0x3FFF_FFFF) << 1));
        assert_eq!(c.uv(), (0x3FFF_FFFF, 0));
        assert_eq!(c.diamond(), 3);
    }

    #[test]
    fn lat_lon_round_trips_below_a_centimeter_ish() {
        let mut worst: f64 = 0.;
        let mut lat = -89.5;
        while lat <= 89.5 {
            let mut lon = -180.;
            while lon < 180. {
                let c = Coord::from_lat_lon(lat, lon);
                worst = worst.max(ground_error_m(lat, lon, c));
                lon += 3.7;
            }
            lat += 2.9;
        }
        // Max error is half a cell diagonal plus gnomonic stretch; 2 cm is
        // comfortably above that and far below any data source's precision.
        assert!(worst < 0.02, "worst round-trip error {worst} m");
    }

    #[test]
    fn special_points_round_trip() {
        for (lat, lon) in [
            (90., 0.),
            (-90., 0.),
            (0., 0.),
            (0., 180.),
            (0., -180.),
            (47.6, -122.3),  // Seattle
            (-41.3, 174.8),  // Wellington
            (26.57, 100.),   // near an icosahedron vertex latitude
        ] {
            let c = Coord::from_lat_lon(lat, lon);
            let err = ground_error_m(lat, lon, c);
            assert!(err < 0.02, "({lat}, {lon}) error {err} m");
        }
    }

    #[test]
    fn cells_contain_their_coords_at_every_depth() {
        let c = Coord::from_lat_lon(47.6, -122.3);
        for depth in 0..=MAX_DEPTH {
            let cell = c.cell(depth);
            assert!(cell.contains(c));
            assert_eq!(cell.depth(), depth);
            if depth > 0 {
                let parent = cell.parent().unwrap();
                assert!(parent.contains_cell(cell));
                assert!(parent.contains(c));
                // cell is one of its parent's four children
                assert!((0..4).any(|i| parent.child(i) == cell));
            }
        }
        assert!(c.cell(0).parent().is_none());
    }

    #[test]
    fn distinct_cells_do_not_contain_each_other() {
        let seattle = Coord::from_lat_lon(47.6, -122.3);
        let wellington = Coord::from_lat_lon(-41.3, 174.8);
        assert!(!seattle.cell(5).contains(wellington));
        assert!(!seattle.cell(0).contains_cell(wellington.cell(10)));
    }

    #[test]
    fn cell_center_stays_inside_and_near() {
        let c = Coord::from_lat_lon(47.6, -122.3);
        for depth in [4, 10, 16, 22, 28] {
            let cell = c.cell(depth);
            let center = cell.center();
            assert!(cell.contains(center));
            let (clat, clon) = center.to_lat_lon();
            let err = ground_error_m(clat, clon, c);
            // center is within one cell diagonal of the original point
            assert!(err < 1.5 * cell_edge_m(depth), "depth {depth}: {err} m");
        }
    }

    #[test]
    fn all_ten_diamonds_and_both_halves_are_reachable() {
        let mut seen = [[false; 2]; 10];
        let mut lat = -88.;
        while lat <= 88. {
            let mut lon = -180.;
            while lon < 180. {
                let c = Coord::from_lat_lon(lat, lon);
                let (iu, iv) = c.uv();
                let upper = iu + iv >= UV_STEPS;
                seen[c.diamond() as usize][upper as usize] = true;
                lon += 2.;
            }
            lat += 2.;
        }
        for (d, halves) in seen.iter().enumerate() {
            assert!(halves[0] && halves[1], "diamond {d} missing a half: {halves:?}");
        }
    }

    #[test]
    fn nearby_points_get_distinct_fine_cells() {
        // Two trailheads ~150 m apart in the Issaquah Alps. They may straddle
        // a boundary at any depth (boundaries are nested), so sharing a coarse
        // cell is never guaranteed — but 150 m exceeds a depth-18 cell's
        // diagonal (~40 m), so distinctness at depth 18 is.
        let a = Coord::from_lat_lon(47.5290, -121.9960);
        let b = Coord::from_lat_lon(47.5300, -121.9945);
        assert_ne!(a.cell(18), b.cell(18));
        assert_ne!(a, b);
    }

    #[test]
    fn resolution_floor_is_sub_centimeter() {
        assert!(cell_edge_m(MAX_DEPTH) < 0.01);
        assert!(cell_edge_m(MAX_DEPTH) > 0.004);
    }
}
