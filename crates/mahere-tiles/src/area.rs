//! An area: a selection's shape, snapshotted (Nick 2026-10-10: "snapshot and share the shape").
//!
//! The shape is the selection's own coverage planes, copied at the depths they were drawn, so it survives exactly and the cells it covers at any depth can be derived later: a cell under a shape cell maps to a square block of the shape's texels, exact for eight levels down, loose only beyond.
//! An area document is VSF then zstd, like a cell: a name, when it was made, the detail wanted (a depth), and the planes. It is what the phone keeps, what is shared, and what the pipeline reads to bake imagery for just those cells.

use std::collections::BTreeSet;

use mahere_coord::{Coord, morton_spread, uv_to_lat_lon};
use vsf::types::Tensor;
use vsf::{VsfBuilder, VsfType};

use crate::{CellKey, CovCell, TEX, TEX_BITS, TRI, tri_idx};

/// The coarsest and finest detail an area may ask for: the global base, and the finest anything is baked at.
pub const AREA_MIN_DEPTH: u8 = 8;
pub const AREA_MAX_DEPTH: u8 = 14;

/// A selection's shape with a name and the detail wanted.
#[derive(Clone, Debug, PartialEq)]
pub struct Area {
    pub name: String,
    /// Eagle Time.
    pub created: i64,
    /// The depth the area is downloaded to.
    pub depth: u8,
    /// The shape: coverage planes at the depths they were drawn, disjoint (a finer plane's block is cleared from the coarser).
    pub cells: Vec<(CellKey, CovCell)>,
    /// On this device: when the download last completed and how many bytes it found; never shared.
    pub fetched: Option<(i64, u64)>,
}

impl PartialEq for CovCell {
    fn eq(&self, other: &CovCell) -> bool {
        self.cov == other.cov
    }
}

impl std::fmt::Debug for CovCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CovCell({} covered)", self.cov.iter().filter(|&&c| c > 0).count())
    }
}

/// Mean bytes of a cell at a depth, measured over the Leavenworth bake with imagery and the global base (2026-10-10): what an estimate rests on until the cells are on the device.
pub fn mean_cell_bytes(depth: u8) -> u64 {
    match depth {
        0..=5 => 50_000,
        6..=9 => 100_000,
        10 => 135_000,
        11 => 170_000,
        12 => 185_000,
        13 => 180_000,
        _ => 160_000,
    }
}

/// The texel squares a plane covers in either half: TEX × TEX booleans, by (ty << TEX_BITS) | tx.
fn squares(cov: &CovCell) -> Vec<bool> {
    (0..TEX * TEX).map(|i| cov.cov[2 * i] > 0 || cov.cov[2 * i + 1] > 0).collect()
}

/// Whether any square in the block (tx0..tx0+b, ty0..ty0+b) is covered.
fn block_any(sq: &[bool], tx0: usize, ty0: usize, b: usize) -> bool {
    (ty0..ty0 + b).any(|ty| sq[ty * TEX + tx0..ty * TEX + tx0 + b].iter().any(|&s| s))
}

/// The cell `k` levels under `key` with grid (cu, cv) in that level's 2^k × 2^k grid, u in the higher bit of each pair.
fn descendant(key: CellKey, k: u8, cu: u64, cv: u64) -> CellKey {
    let m = ((morton_spread(cu << (30 - k as u32)) << 1) | morton_spread(cv << (30 - k as u32))) >> (60 - 2 * k as u32);
    CellKey { depth: key.depth + k, prefix: (key.prefix << (2 * k)) | m }
}

/// The shape of one plane, at a depth: the cells it covers (exact for eight levels down, every descendant of a covered square beyond), or only how many.
fn plane_cells(key: CellKey, cov: &CovCell, depth: u8, mut out: Option<&mut BTreeSet<CellKey>>) -> u64 {
    let sq = squares(cov);
    if !sq.iter().any(|&s| s) {
        return 0;
    }
    if depth <= key.depth {
        if let Some(out) = out.as_mut() {
            out.insert(key.ancestor(depth));
        }
        return 1;
    }
    let k = depth - key.depth;
    let (kb, extra) = if k <= TEX_BITS { (k, 0) } else { (TEX_BITS, k - TEX_BITS) };
    let b = TEX >> kb;
    let n = 1usize << kb;
    let mut count = 0u64;
    for cv in 0..n {
        for cu in 0..n {
            if !block_any(&sq, cu * b, cv * b, b) {
                continue;
            }
            count += 1u64 << (2 * extra);
            if let Some(out) = out.as_mut() {
                let base = descendant(key, kb, cu as u64, cv as u64);
                for j in 0..(1u64 << (2 * extra)) {
                    out.insert(CellKey { depth, prefix: (base.prefix << (2 * extra)) | j });
                }
            }
        }
    }
    count
}

impl Area {
    /// The cells the shape covers at `depth`, sorted.
    pub fn cells_at(&self, depth: u8) -> Vec<CellKey> {
        let mut set = BTreeSet::new();
        for (key, cov) in &self.cells {
            plane_cells(*key, cov, depth, Some(&mut set));
        }
        set.into_iter().collect()
    }

    /// How many cells the shape covers at `depth`, without listing them (an upper bound where planes meet at a coarser depth).
    pub fn count_at(&self, depth: u8) -> u64 {
        if depth <= self.cells.iter().map(|(k, _)| k.depth).max().unwrap_or(0) {
            return self.cells_at(depth).len() as u64;
        }
        self.cells.iter().map(|(k, c)| plane_cells(*k, c, depth, None)).sum()
    }

    /// Every cell from the top of the pyramid down to `depth`: what a download fetches, so the map has a frame at every zoom offline.
    pub fn all_cells_to(&self, depth: u8) -> Vec<CellKey> {
        (0..=depth).flat_map(|d| self.cells_at(d)).collect()
    }

    /// Cells and bytes the shape asks for down to `depth`, estimated from the mean cell size at each depth.
    pub fn estimate_to(&self, depth: u8) -> (u64, u64) {
        let mut cells = 0;
        let mut bytes = 0;
        for d in 0..=depth {
            let n = self.count_at(d);
            cells += n;
            bytes += n * mean_cell_bytes(d);
        }
        (cells, bytes)
    }

    /// Covered texel centres as latitude and longitude, every `step`th square, for the bounds and the centre.
    fn samples(&self, step: usize) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        for (key, cov) in &self.cells {
            let sq = squares(cov);
            let (u0, v0, size) = key.uv_rect();
            let d = key.diamond();
            for ty in (0..TEX).step_by(step) {
                for tx in (0..TEX).step_by(step) {
                    if sq[ty * TEX + tx] {
                        out.push(uv_to_lat_lon(d, u0 + (tx as f64 + 0.5) / TEX as f64 * size, v0 + (ty as f64 + 0.5) / TEX as f64 * size));
                    }
                }
            }
        }
        out
    }

    /// The shape's centre: the mean of its covered texels on the sphere.
    pub fn centre(&self) -> Option<(f64, f64)> {
        // Every fourth square; a shape too thin for that, every square.
        let mut pts = self.samples(4);
        if pts.is_empty() {
            pts = self.samples(1);
        }
        if pts.is_empty() {
            return None;
        }
        let mut s = [0.0f64; 3];
        for &(lat, lon) in &pts {
            let p = Coord::from_lat_lon(lat, lon).to_xyz();
            s[0] += p[0];
            s[1] += p[1];
            s[2] += p[2];
        }
        Some(Coord::from_xyz(s).to_lat_lon())
    }

    /// A latitude and longitude box around the shape (lat0, lon0, lat1, lon1), half a texel loose, for a baker that fetches sources by box.
    pub fn bounds(&self) -> Option<(f64, f64, f64, f64)> {
        let pts = self.samples(1);
        if pts.is_empty() {
            return None;
        }
        let pad = self.cells.iter().map(|(k, _)| k.texel_m()).fold(0.0, f64::max) / 111_320.0;
        let (mut lat0, mut lon0, mut lat1, mut lon1) = (90.0f64, 180.0f64, -90.0f64, -180.0f64);
        for &(lat, lon) in &pts {
            lat0 = lat0.min(lat);
            lat1 = lat1.max(lat);
            lon0 = lon0.min(lon);
            lon1 = lon1.max(lon);
        }
        Some((lat0 - pad, lon0 - pad, lat1 + pad, lon1 + pad))
    }

    /// The document: VSF then zstd. Shared without the device's own fetch record.
    pub fn encode(&self, share: bool) -> Vec<u8> {
        let n = self.cells.len();
        let depth: Vec<u8> = self.cells.iter().map(|(k, _)| k.depth).collect();
        let cell: Vec<u64> = self.cells.iter().map(|(k, _)| k.raw()).collect();
        let mut cov = Vec::with_capacity(n * TRI);
        for (_, c) in &self.cells {
            cov.extend_from_slice(&c.cov);
        }
        let mut b = VsfBuilder::new()
            .add_section(
                "area",
                vec![
                    ("name".to_string(), VsfType::x(self.name.clone())),
                    ("created".to_string(), VsfType::t_i6(Tensor::new(vec![1], vec![self.created]))),
                    ("depth".to_string(), VsfType::u(self.depth as usize, false)),
                ],
            )
            .add_section(
                "shape",
                vec![
                    ("depth".to_string(), VsfType::t_u3(Tensor::new(vec![n], depth))),
                    ("cell".to_string(), VsfType::t_u6(Tensor::new(vec![n], cell))),
                    ("cov".to_string(), VsfType::t_u3(Tensor::new(vec![n, TRI], cov))),
                ],
            );
        if let (false, Some((when, bytes))) = (share, self.fetched) {
            b = b.add_section("device", vec![("fetched".to_string(), VsfType::t_i6(Tensor::new(vec![1], vec![when]))), ("bytes".to_string(), VsfType::t_u6(Tensor::new(vec![1], vec![bytes])))]);
        }
        let raw = match b.build() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("area encode: {e:?}");
                Vec::new()
            }
        };
        zstd::encode_all(raw.as_slice(), 9).unwrap_or(raw)
    }

    pub fn decode(bytes: &[u8]) -> Option<Area> {
        let raw = zstd::decode_all(bytes).ok()?;
        let (header, end) = vsf::VsfHeader::decode(&raw).map_err(|e| eprintln!("area header: {e:?}")).ok()?;
        let sections = header.sections(&raw, end).map_err(|e| eprintln!("area sections: {e:?}")).ok()?;
        let mut name = String::new();
        let mut created = 0i64;
        let mut depth = AREA_MAX_DEPTH;
        let mut depths: Vec<u8> = Vec::new();
        let mut cells: Vec<u64> = Vec::new();
        let mut cov: Vec<u8> = Vec::new();
        let mut fetched: Option<(i64, u64)> = None;
        let mut fetched_at: Option<i64> = None;
        let mut fetched_bytes: Option<u64> = None;
        for s in sections {
            for f in s.fields {
                let v = f.values.into_iter().next();
                match (s.name.as_str(), f.name.as_str(), v) {
                    ("area", "name", Some(VsfType::x(t) | VsfType::a(t))) => name = t,
                    ("area", "created", Some(VsfType::t_i6(t))) => created = t.data.first().copied().unwrap_or(0),
                    ("area", "created", Some(VsfType::v_i6(t))) => created = t.data.first().copied().unwrap_or(0),
                    ("area", "depth", Some(VsfType::u(d, _))) => depth = d as u8,
                    ("area", "depth", Some(VsfType::u3(d))) => depth = d,
                    ("shape", "depth", Some(VsfType::t_u3(t))) => depths = t.data,
                    ("shape", "depth", Some(VsfType::v_u3(t))) => depths = t.data,
                    ("shape", "cell", Some(VsfType::t_u6(t))) => cells = t.data,
                    ("shape", "cell", Some(VsfType::v_u6(t))) => cells = t.data,
                    ("shape", "cov", Some(VsfType::t_u3(t))) => cov = t.data,
                    ("shape", "cov", Some(VsfType::v_u3(t))) => cov = t.data,
                    ("device", "fetched", Some(VsfType::t_i6(t))) => fetched_at = t.data.first().copied(),
                    ("device", "fetched", Some(VsfType::v_i6(t))) => fetched_at = t.data.first().copied(),
                    ("device", "bytes", Some(VsfType::t_u6(t))) => fetched_bytes = t.data.first().copied(),
                    ("device", "bytes", Some(VsfType::v_u6(t))) => fetched_bytes = t.data.first().copied(),
                    _ => {}
                }
            }
        }
        if let (Some(a), Some(b)) = (fetched_at, fetched_bytes) {
            fetched = Some((a, b));
        }
        let n = depths.len().min(cells.len());
        if cov.len() < n * TRI {
            return None;
        }
        let planes = (0..n)
            .map(|i| {
                let d = depths[i];
                let key = CellKey { depth: d, prefix: cells[i] >> (60 - 2 * d as u32) };
                (key, CovCell { cov: cov[i * TRI..(i + 1) * TRI].to_vec() })
            })
            .collect();
        Some(Area { name, created, depth: depth.clamp(AREA_MIN_DEPTH, AREA_MAX_DEPTH), cells: planes, fetched })
    }
}

/// The planes a selection is made of, made disjoint: where a finer plane exists under a coarser one the coarser's block is cleared, as the finest cell decides in the view. Empty planes are dropped.
pub fn disjoint_planes(mut planes: Vec<(CellKey, CovCell)>) -> Vec<(CellKey, CovCell)> {
    planes.sort_by_key(|(k, _)| (k.depth, k.prefix));
    let keys: Vec<CellKey> = planes.iter().map(|(k, _)| *k).collect();
    for (key, cov) in planes.iter_mut() {
        for fine in keys.iter().filter(|f| f.depth > key.depth && f.ancestor(key.depth) == *key) {
            let k = (fine.depth - key.depth).min(TEX_BITS);
            let (cu, cv) = fine.ancestor(key.depth + k).grid();
            let b = TEX >> k;
            let (tx0, ty0) = ((cu as usize & ((1 << k) - 1)) * b, (cv as usize & ((1 << k) - 1)) * b);
            for ty in ty0..ty0 + b {
                for tx in tx0..tx0 + b {
                    cov.cov[tri_idx(tx, ty, 0)] = 0;
                    cov.cov[tri_idx(tx, ty, 1)] = 0;
                }
            }
        }
    }
    planes.retain(|(_, c)| !c.is_empty());
    planes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_with(squares: &[(usize, usize)]) -> CovCell {
        let mut c = CovCell::new();
        for &(tx, ty) in squares {
            c.cov[tri_idx(tx, ty, 0)] = 255;
            c.cov[tri_idx(tx, ty, 1)] = 255;
        }
        c
    }

    fn key(depth: u8, cu: u64, cv: u64) -> CellKey {
        CellKey::from_grid(3, depth, cu, cv)
    }

    #[test]
    fn cells_under_a_plane_follow_its_squares() {
        let k = key(10, 5, 9);
        let a = Area { name: "t".into(), created: 0, depth: 12, cells: vec![(k, plane_with(&[(0, 0), (255, 255)]))], fetched: None };
        // Two levels down: the first and last of the 4 × 4 blocks.
        let c = a.cells_at(12);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].ancestor(10), k);
        assert_eq!(c[0].grid(), (5 * 4, 9 * 4));
        assert_eq!(c[1].grid(), (5 * 4 + 3, 9 * 4 + 3));
        assert_eq!(a.count_at(12), 2);
        // Eight levels down a square is a cell; ten levels down each has sixteen descendants.
        assert_eq!(a.count_at(18), 2);
        assert_eq!(a.cells_at(20).len(), 32);
        assert_eq!(a.count_at(20), 32);
        // Above the plane, its ancestors.
        assert_eq!(a.cells_at(7), vec![k.ancestor(7)]);
        assert_eq!(a.cells_at(10), vec![k]);
        assert_eq!(a.all_cells_to(12).len(), 11 + 2 + 2);
    }

    #[test]
    fn a_finer_plane_clears_the_block_above_it() {
        let coarse = key(10, 5, 9);
        let fine = coarse.child(3).child(0);
        let planes = disjoint_planes(vec![(coarse, plane_with(&[(130, 130), (10, 10)])), (fine, plane_with(&[(1, 1)]))]);
        assert_eq!(planes.len(), 2);
        let (_, c) = planes.iter().find(|(k, _)| *k == coarse).unwrap();
        assert_eq!(c.cov[tri_idx(130, 130, 0)], 0, "under the finer plane: cleared");
        assert_eq!(c.cov[tri_idx(10, 10, 0)], 255, "elsewhere: kept");
    }

    #[test]
    fn an_area_round_trips_without_its_device_record_when_shared() {
        let a = Area { name: "spruce otter".into(), created: 123_456_789, depth: 13, cells: vec![(key(11, 1, 2), plane_with(&[(3, 4), (100, 7)]))], fetched: Some((5, 6)) };
        let kept = Area::decode(&a.encode(false)).unwrap();
        assert_eq!(kept, a);
        let shared = Area::decode(&a.encode(true)).unwrap();
        assert_eq!(shared.fetched, None);
        assert_eq!(shared.cells, a.cells);
        assert!(a.encode(true).len() < 4096, "a near-empty plane compresses to almost nothing");
        let (lat0, lon0, lat1, lon1) = a.bounds().unwrap();
        let (lat, lon) = a.centre().unwrap();
        assert!(lat0 <= lat && lat <= lat1 && lon0 <= lon && lon <= lon1);
    }
}
