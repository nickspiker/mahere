//! The cell pipeline: bake source data into dymaxion-cell rasters — the
//! #pagetable renderer's entire diet. Two layers today:
//!
//! - **dem**: per-cell elevation + gradient planes (f32), sampled from the
//!   source DEM at bake time (so cell-edge gradients are exact — the baker
//!   holds the whole source; no aprons needed).
//! - **line**: the linework pyramid. Every line feature stamped 1 texel wide
//!   at the base depth, then parents built by averaging the 4 children of
//!   each texel up the triangle tree — coverage IS box filtering, so opacity
//!   at every zoom is computed, not styled. No vectors exist at render time.
//!
//! **Texels are triangles.** A cell is one rhombus of the diamond-Morton
//! grid (`depth`), and its texels are the triangular subdivision of that
//! rhombus's two faces: 256×256 UV squares, each split along `u+v = k` into
//! a lower and an upper equilateral triangle. The triangular tiling has
//! 6-fold symmetry and a line always crosses it edge-to-edge, so linework
//! is isotropic; a square/rhombus texel grid is not (a line along one
//! diagonal touches rhombi tip-to-tip, along the other obtuse-to-obtuse).
//! Each triangle subdivides into four — three corners and the inverted
//! center — and that is the pyramid's box filter.
//!
//! In memory a cell's planes are indexed `((ty << 8 | tx) << 1) | half`
//! (the renderer's stepping order). On disk they are in triangle-path
//! order, `[face half][2 bits per level]` — the triangle code — so a
//! parent texel's four children are contiguous. [`disk_to_mem`] /
//! [`mem_to_disk`] convert.
//!
//! Texels are (class, coverage) so styling stays a draw-time LUT. Cells are
//! written as zstd'd VSF at `{layer}/{depth:02}/{prefix:016x}.vsf.zst` —
//! a directory layout that is byte-for-byte the future R2 bucket layout.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use mahere_coord::{Coord, morton_compact, morton_spread, uv_to_lat_lon};
use mahere_dem::DemStore;
use mahere_osm::Road;
use rayon::prelude::*;
use vsf::types::{Tensor, WorldCell};
use vsf::{VsfBuilder, VsfType};

/// base64url without padding — how bytes are spelled when they must be a
/// name (the same alphabet tohu uses for vault file names).
pub fn base64url(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().fold(0u32, |acc, &b| (acc << 8) | b as u32) << (8 * (3 - chunk.len()));
        for i in 0..=chunk.len() {
            out.push(A[((n >> (18 - 6 * i)) & 63) as usize] as char);
        }
    }
    out
}

/// Cells are TEX x TEX UV squares; TEX_BITS of Morton depth below the cell.
pub const TEX: usize = 256;
pub const TEX_BITS: u8 = 8;
/// Texels per cell: two triangles per UV square.
pub const TRI: usize = 2 * TEX * TEX;

/// A cell address: dymaxion Morton prefix (diamond in the top 4 bits of the
/// full-resolution coordinate, right-aligned here) at `depth`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CellKey {
    pub depth: u8,
    pub prefix: u64,
}

impl CellKey {
    /// The cell containing `c` at `depth`: diamond + 2*depth Morton bits,
    /// right-aligned (so prefixes sort and shift like integers).
    pub fn containing(c: Coord, depth: u8) -> CellKey {
        CellKey { depth, prefix: c.raw() >> (60 - 2 * depth as u32) }
    }

    pub fn parent(self) -> CellKey {
        CellKey { depth: self.depth - 1, prefix: self.prefix >> 2 }
    }

    /// Ancestor at a shallower `depth` (`depth <= self.depth`).
    pub fn ancestor(self, depth: u8) -> CellKey {
        CellKey { depth, prefix: self.prefix >> (2 * (self.depth - depth) as u32) }
    }

    /// Child 0-3 (Morton bit pair: u high, v low).
    pub fn child(self, q: u64) -> CellKey {
        CellKey { depth: self.depth + 1, prefix: (self.prefix << 2) | q }
    }

    pub fn diamond(self) -> u8 {
        (self.prefix >> (2 * self.depth)) as u8
    }

    /// Cell-grid coordinates (cu, cv) within the diamond, each `depth` bits.
    pub fn grid(self) -> (u64, u64) {
        let m = self.prefix & ((1u64 << (2 * self.depth)) - 1);
        // morton helpers operate on the 60-bit layout: left-align first.
        let aligned = m << (60 - 2 * self.depth as u32);
        (
            morton_compact(aligned >> 1) >> (30 - self.depth as u32),
            morton_compact(aligned) >> (30 - self.depth as u32),
        )
    }

    /// The full-resolution Morton value this cell is a prefix of.
    pub fn raw(self) -> u64 {
        self.prefix << (60 - 2 * self.depth as u32)
    }

    /// The cell as VSF values, flattened: its depth (`u`) and the world
    /// cell (`wm`). This is the cell's identity everywhere a key is needed —
    /// no delimiters, no numerals, the type tags are the structure.
    pub fn vsf_bytes(self) -> Vec<u8> {
        let mut b = VsfType::u(self.depth as usize, false).flatten();
        b.extend(VsfType::wm(WorldCell::from_raw(self.raw())).flatten());
        b
    }

    /// File / object name: the VSF bytes spelled base64url.
    pub fn name(self) -> String {
        base64url(&self.vsf_bytes())
    }

    pub fn path(self, layer: &str) -> String {
        format!("{layer}/{}.vsf.zst", self.name())
    }

    /// Diamond-UV rectangle covered by this cell.
    pub fn uv_rect(self) -> (f64, f64, f64) {
        let (cu, cv) = self.grid();
        let size = 1.0 / (1u64 << self.depth) as f64;
        (cu as f64 * size, cv as f64 * size, size)
    }
}

// ==================== TRIANGLE TEXELS ====================

/// Memory index of triangle texel (tx, ty, half) in a cell.
#[inline(always)]
pub fn tri_idx(tx: usize, ty: usize, half: usize) -> usize {
    (((ty << TEX_BITS) | tx) << 1) | half
}

/// Triangle texel containing a point given in texel units within the cell
/// (or any grid): the UV square plus which side of `u+v = k` it lies on.
#[inline(always)]
pub fn tri_at(gx: f64, gy: f64) -> (usize, usize, usize) {
    let (tx, ty) = (gx.floor(), gy.floor());
    let half = ((gx - tx) + (gy - ty) >= 1.0) as usize;
    (tx as usize, ty as usize, half)
}

/// UV position of a triangle texel's centroid, in texel units.
#[inline(always)]
pub fn tri_centroid(tx: usize, ty: usize, half: usize) -> (f64, f64) {
    let off = if half == 0 { 1.0 / 3.0 } else { 2.0 / 3.0 };
    (tx as f64 + off, ty as f64 + off)
}

/// The four children of triangle (tx, ty, half) on the next-finer grid, in
/// code-digit order: 0 apex, 1 toward r (+v), 2 toward q (+u), 3 the
/// inverted center — which lives in the apex's UV square with the opposite
/// orientation. A lower triangle's apex is its square's (0,0) corner, an
/// upper's is (1,1).
#[inline]
pub fn tri_children(tx: usize, ty: usize, half: usize) -> [(usize, usize, usize); 4] {
    let (x, y) = (2 * tx, 2 * ty);
    if half == 0 {
        [(x, y, 0), (x, y + 1, 0), (x + 1, y, 0), (x, y, 1)]
    } else {
        [(x + 1, y + 1, 1), (x, y + 1, 1), (x + 1, y, 1), (x + 1, y + 1, 0)]
    }
}

/// Parent-cell texel -> its four children as (child cell Morton digit,
/// memory index in that child cell).
#[inline]
fn child_cell_texels(tx: usize, ty: usize, half: usize) -> [(u64, usize); 4] {
    tri_children(tx, ty, half).map(|(cx, cy, ch)| {
        let q = (((cx >> TEX_BITS) as u64) << 1) | (cy >> TEX_BITS) as u64;
        (q, tri_idx(cx & (TEX - 1), cy & (TEX - 1), ch))
    })
}

/// Disk order (triangle path) -> memory index, both halves.
fn tri_order() -> &'static [u32] {
    static ORDER: OnceLock<Box<[u32]>> = OnceLock::new();
    ORDER.get_or_init(|| {
        let mut to_mem = vec![0u32; TRI].into_boxed_slice();
        fn descend(level: u8, tx: usize, ty: usize, half: usize, path: usize, out: &mut [u32]) {
            if level == TEX_BITS {
                out[path] = tri_idx(tx, ty, half) as u32;
                return;
            }
            for (d, (cx, cy, ch)) in tri_children(tx, ty, half).into_iter().enumerate() {
                descend(level + 1, cx, cy, ch, (path << 2) | d, out);
            }
        }
        for face in 0..2 {
            descend(0, 0, 0, face, face, &mut to_mem);
        }
        to_mem
    })
}

pub fn mem_to_disk<T: Copy>(mem: &[T]) -> Vec<T> {
    debug_assert_eq!(mem.len(), TRI);
    tri_order().iter().map(|&m| mem[m as usize]).collect()
}

pub fn disk_to_mem<T: Copy + Default>(disk: &[T]) -> Vec<T> {
    debug_assert_eq!(disk.len(), TRI);
    let mut mem = vec![T::default(); TRI];
    for (d, &m) in tri_order().iter().enumerate() {
        mem[m as usize] = disk[d];
    }
    mem
}

/// Global texel coordinates of a Coord at a given base depth: the texel grid
/// is the cell grid times TEX.
fn texel_of(c: Coord, depth: u8) -> (u8, u64, f64, f64) {
    let (iu, iv) = c.uv();
    let shift = 30 - depth as u32 - TEX_BITS as u32;
    let scale = 1.0 / (1u64 << shift) as f64;
    (c.diamond(), c.raw() >> (60 - 2 * depth as u32), iu as f64 * scale, iv as f64 * scale)
}

// ==================== LINE LAYER ====================

pub struct LineCell {
    pub class: Vec<u8>, // 0 = empty; else RoadClass as u8 + 1
    pub cov: Vec<u8>,
}

impl LineCell {
    fn new() -> LineCell {
        LineCell { class: vec![0; TRI], cov: vec![0; TRI] }
    }
}

/// Stamp every feature 1 texel wide into base-depth cells, then build the
/// pyramid up to `min_depth`. Returns cells keyed by (depth, prefix).
pub fn bake_lines(
    feats: &[Road],
    base_depth: u8,
    min_depth: u8,
) -> HashMap<CellKey, LineCell> {
    assert!(base_depth as u32 + TEX_BITS as u32 <= 30);
    let mut cells: HashMap<CellKey, LineCell> = HashMap::new();
    let mut cross_diamond = 0usize;
    for road in feats {
        let class = road.class as u8 + 1;
        let mut prev: Option<(u8, f64, f64)> = None;
        for &(lat, lon) in &road.pts {
            let c = Coord::from_lat_lon(lat as f64, lon as f64);
            let (d, _, gu, gv) = texel_of(c, base_depth);
            if let Some((pd, pu, pv)) = prev {
                if pd == d {
                    stamp_segment(&mut cells, base_depth, d, pu, pv, gu, gv, class);
                } else {
                    cross_diamond += 1;
                }
            }
            prev = Some((d, gu, gv));
        }
    }
    if cross_diamond > 0 {
        eprintln!("  (skipped {cross_diamond} cross-diamond segments)");
    }
    // Pyramid: each parent texel the mean coverage of its four triangle
    // children (missing children are empty), keeping the most major class.
    let mut depth = base_depth;
    while depth > min_depth {
        let mut parents: Vec<CellKey> =
            cells.keys().filter(|k| k.depth == depth).map(|k| k.parent()).collect();
        parents.sort_by_key(|k| k.prefix);
        parents.dedup();
        let built: Vec<(CellKey, LineCell)> = parents
            .par_iter()
            .map(|&p| {
                let kids: [Option<&LineCell>; 4] = std::array::from_fn(|q| cells.get(&p.child(q as u64)));
                let mut cell = LineCell::new();
                for ty in 0..TEX {
                    for tx in 0..TEX {
                        for half in 0..2 {
                            let mut covsum = 0u16;
                            let mut best = 0u8;
                            for (q, i) in child_cell_texels(tx, ty, half) {
                                let Some(child) = kids[q as usize] else { continue };
                                covsum += child.cov[i] as u16;
                                let cl = child.class[i];
                                if cl != 0 && (best == 0 || cl < best) {
                                    best = cl;
                                }
                            }
                            let i = tri_idx(tx, ty, half);
                            cell.cov[i] = (covsum / 4) as u8;
                            cell.class[i] = best;
                        }
                    }
                }
                (p, cell)
            })
            .collect();
        cells.extend(built);
        depth -= 1;
    }
    cells
}

/// Walk the triangles a segment crosses — every crossing of a `u = k`,
/// `v = k` or `u + v = k` lattice line enters a new triangle — and stamp
/// each one. Edge-to-edge in every direction: no corner-touching diagonals.
#[allow(clippy::too_many_arguments)]
fn stamp_segment(
    cells: &mut HashMap<CellKey, LineCell>,
    depth: u8,
    diamond: u8,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    class: u8,
) {
    let extent = ((1u64 << depth) * TEX as u64) as f64;
    let mut ts = vec![0.0f64, 1.0];
    for (a0, a1) in [(x0, x1), (y0, y1), (x0 + y0, x1 + y1)] {
        if a0 == a1 {
            continue;
        }
        let (lo, hi) = (a0.min(a1), a0.max(a1));
        for k in (lo.floor() as i64 + 1)..=(hi.ceil() as i64 - 1) {
            let t = (k as f64 - a0) / (a1 - a0);
            if t > 0.0 && t < 1.0 {
                ts.push(t);
            }
        }
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for w in ts.windows(2) {
        if w[1] - w[0] < 1e-12 {
            continue;
        }
        let t = (w[0] + w[1]) * 0.5;
        let gx = x0 + (x1 - x0) * t;
        let gy = y0 + (y1 - y0) * t;
        if gx < 0.0 || gy < 0.0 || gx >= extent || gy >= extent {
            continue;
        }
        let (tx, ty, half) = tri_at(gx, gy);
        let (cu, cv) = ((tx / TEX) as u64, (ty / TEX) as u64);
        let m = (morton_spread(cu << (30 - depth as u32)) << 1)
            | morton_spread(cv << (30 - depth as u32));
        let prefix = ((diamond as u64) << (2 * depth)) | (m >> (60 - 2 * depth as u32));
        let cell = cells
            .entry(CellKey { depth, prefix })
            .or_insert_with(LineCell::new);
        let i = tri_idx(tx % TEX, ty % TEX, half);
        cell.cov[i] = 255;
        if cell.class[i] == 0 || class < cell.class[i] {
            cell.class[i] = class;
        }
    }
}

// ==================== DEM LAYER ====================

#[derive(Clone)]
pub struct DemCell {
    pub elev: Vec<f32>,
    pub ge: Vec<f32>,
    pub gn: Vec<f32>,
}

impl DemCell {
    fn new() -> DemCell {
        DemCell { elev: vec![f32::NAN; TRI], ge: vec![0.0; TRI], gn: vec![0.0; TRI] }
    }
}

/// Bake dem cells covering `keys` from the source store: each triangle
/// texel sampled at its centroid.
pub fn bake_dem(dem: &DemStore, keys: &[CellKey]) -> Vec<(CellKey, DemCell)> {
    keys.par_iter()
        .map(|&key| {
            let (u0, v0, size) = key.uv_rect();
            let d = key.diamond();
            let step = size / TEX as f64;
            let mut cell = DemCell::new();
            for ty in 0..TEX {
                for tx in 0..TEX {
                    for half in 0..2 {
                        let (cx, cy) = tri_centroid(tx, ty, half);
                        let (lat, lon) = uv_to_lat_lon(d, u0 + cx * step, v0 + cy * step);
                        if let Some((e, (ge, gn))) = dem.elev_and_gradient(lat, lon) {
                            let i = tri_idx(tx, ty, half);
                            cell.elev[i] = e;
                            cell.ge[i] = ge;
                            cell.gn[i] = gn;
                        }
                    }
                }
            }
            (key, cell)
        })
        .collect()
}

/// Every depth from `base.depth - 1` down to `min_depth`, each texel the
/// mean of its four triangle children (no-data ignored). Building the
/// pyramid from the base set is what guarantees every ancestor of a baked
/// cell exists.
pub fn dem_pyramid(base: &[(CellKey, DemCell)], min_depth: u8) -> Vec<(CellKey, DemCell)> {
    let mut out: Vec<(CellKey, DemCell)> = Vec::new();
    // Start index of the most recent level inside `out` (None = base).
    let mut last: Option<usize> = None;
    loop {
        let cur: &[(CellKey, DemCell)] = match last {
            None => base,
            Some(i) => &out[i..],
        };
        let Some(&(k0, _)) = cur.first() else { break };
        if k0.depth <= min_depth {
            break;
        }
        let mut parents: Vec<CellKey> = cur.iter().map(|(k, _)| k.parent()).collect();
        parents.sort_by_key(|k| k.prefix);
        parents.dedup();
        let by_key: HashMap<(u8, u64), &DemCell> =
            cur.iter().map(|(k, c)| ((k.depth, k.prefix), c)).collect();
        let next: Vec<(CellKey, DemCell)> = parents
            .par_iter()
            .map(|&p| {
                let kids: [Option<&&DemCell>; 4] =
                    std::array::from_fn(|q| by_key.get(&(p.depth + 1, p.child(q as u64).prefix)));
                let mut cell = DemCell::new();
                for ty in 0..TEX {
                    for tx in 0..TEX {
                        for half in 0..2 {
                            let (mut e, mut ge, mut gn, mut n) = (0.0f32, 0.0f32, 0.0f32, 0u32);
                            for (q, i) in child_cell_texels(tx, ty, half) {
                                let Some(child) = kids[q as usize] else { continue };
                                if !child.elev[i].is_nan() {
                                    e += child.elev[i];
                                    ge += child.ge[i];
                                    gn += child.gn[i];
                                    n += 1;
                                }
                            }
                            if n > 0 {
                                let i = tri_idx(tx, ty, half);
                                cell.elev[i] = e / n as f32;
                                cell.ge[i] = ge / n as f32;
                                cell.gn[i] = gn / n as f32;
                            }
                        }
                    }
                }
                (p, cell)
            })
            .collect();
        last = Some(out.len());
        out.extend(next);
    }
    out
}

/// All cells at `depth` that intersect a lat/lon bbox. Scanned at half-cell
/// resolution, never coarser than an eighth of the box — a slanted rhombus
/// cell can cut through a box without containing any corner of it.
pub fn cells_covering(lat0: f64, lon0: f64, lat1: f64, lon1: f64, depth: u8) -> Vec<CellKey> {
    let mut keys = std::collections::HashSet::new();
    // Half a cell edge in degrees of latitude, as the scan step.
    let step = ((7054_000.0 / (1u64 << depth) as f64) / 111_320.0 / 2.0)
        .min((lat1 - lat0) / 8.0)
        .min((lon1 - lon0) / 8.0)
        .max(1e-4);
    let mut lat = lat0;
    while lat <= lat1 + step {
        let mut lon = lon0;
        while lon <= lon1 + step {
            keys.insert(CellKey::containing(
                Coord::from_lat_lon(lat.min(lat1), lon.min(lon1)),
                depth,
            ));
            lon += step;
        }
        lat += step;
    }
    let mut v: Vec<CellKey> = keys.into_iter().collect();
    v.sort_by_key(|k| k.prefix);
    v
}

// ==================== VSF I/O ====================

fn plane_u8(mem: &[u8]) -> VsfType {
    VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], mem_to_disk(mem)))
}

fn plane_f32(mem: &[f32]) -> VsfType {
    VsfType::t_f5(Tensor::new(vec![2, TEX * TEX], mem_to_disk(mem)))
}

pub fn write_line_cell(out: &Path, key: CellKey, cell: &LineCell) -> Result<(), String> {
    let bytes = VsfBuilder::new()
        .add_section(
            "linecell",
            vec![
                ("depth".to_string(), VsfType::u(key.depth as usize, false)),
                ("prefix".to_string(), VsfType::u(key.prefix as usize, false)),
                ("class".to_string(), plane_u8(&cell.class)),
                ("cov".to_string(), plane_u8(&cell.cov)),
            ],
        )
        .build()
        .map_err(|e| format!("linecell build: {e:?}"))?;
    write_file(out, &key.path("line"), &bytes)
}

pub fn write_dem_cell(out: &Path, key: CellKey, cell: &DemCell) -> Result<(), String> {
    let bytes = VsfBuilder::new()
        .add_section(
            "demcell",
            vec![
                ("depth".to_string(), VsfType::u(key.depth as usize, false)),
                ("prefix".to_string(), VsfType::u(key.prefix as usize, false)),
                ("elev".to_string(), plane_f32(&cell.elev)),
                ("ge".to_string(), plane_f32(&cell.ge)),
                ("gn".to_string(), plane_f32(&cell.gn)),
            ],
        )
        .build()
        .map_err(|e| format!("demcell build: {e:?}"))?;
    write_file(out, &key.path("dem"), &bytes)
}

fn write_file(out: &Path, rel: &str, bytes: &[u8]) -> Result<(), String> {
    let path = out.join(rel);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    // Whole-file zstd: VSF internals untouched, the R2 layout keeps the
    // .vsf.zst names, and near-empty line cells shrink ~100x.
    let z = zstd::encode_all(bytes, 3).map_err(|e| e.to_string())?;
    std::fs::write(&path, z).map_err(|e| format!("{}: {e}", path.display()))
}

/// Read any cell file's fields (width-agnostic, per VSF doctrine).
pub fn read_cell_fields(path: &Path) -> Result<HashMap<String, VsfType>, String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode_cell_fields(&data).map_err(|e| format!("{}: {e}", path.display()))
}

/// Decode a cell from raw file bytes (zstd or plain VSF) — the loader
/// thread's entry point; no filesystem coupling.
pub fn decode_cell_fields(data: &[u8]) -> Result<HashMap<String, VsfType>, String> {
    let plain: Vec<u8>;
    let data = if data.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
        plain = zstd::decode_all(data).map_err(|e| e.to_string())?;
        &plain[..]
    } else {
        data
    };
    let (header, end) = vsf::VsfHeader::decode(data).map_err(|e| e.to_string())?;
    let section = header.primary_section(data, end).map_err(|e| e.to_string())?;
    Ok(section
        .fields
        .into_iter()
        .filter_map(|f| f.values.into_iter().next().map(|v| (f.name, v)))
        .collect())
}

/// A u8 plane in memory order, or None if missing/malformed.
pub fn plane_u8_mem(v: &VsfType) -> Option<Vec<u8>> {
    let disk = match v {
        VsfType::t_u3(t) => &t.data,
        VsfType::v_u3(t) => &t.data,
        _ => return None,
    };
    (disk.len() == TRI).then(|| disk_to_mem(disk))
}

/// An f32 plane in memory order, or None if missing/malformed.
pub fn plane_f32_mem(v: &VsfType) -> Option<Vec<f32>> {
    let disk: Vec<f32> = match v {
        VsfType::t_f5(t) => t.data.clone(),
        VsfType::v_f5(t) => t.data.clone(),
        VsfType::t_f6(t) => t.data.iter().map(|&x| x as f32).collect(),
        _ => return None,
    };
    (disk.len() == TRI).then(|| disk_to_mem(&disk))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_key_round_trips() {
        let c = Coord::from_lat_lon(46.2024, -121.4909);
        for depth in [6u8, 10, 13] {
            let key = CellKey::containing(c, depth);
            assert_eq!(key.diamond(), c.diamond());
            let (u0, v0, size) = key.uv_rect();
            let (iu, iv) = c.uv();
            let (u, v) = (iu as f64 / 2f64.powi(30), iv as f64 / 2f64.powi(30));
            assert!(u >= u0 && u < u0 + size, "u {u} not in [{u0}, {})", u0 + size);
            assert!(v >= v0 && v < v0 + size);
            assert_eq!(key.parent().child((key.prefix & 3) as u64), key);
        }
    }

    /// Children partition the parent: every triangle at the fine grid is
    /// the child of exactly one triangle at the coarse grid.
    #[test]
    fn triangle_children_partition() {
        let n = 8usize;
        let mut seen = vec![0u8; 2 * (2 * n) * (2 * n)];
        for ty in 0..n {
            for tx in 0..n {
                for half in 0..2 {
                    for (cx, cy, ch) in tri_children(tx, ty, half) {
                        assert!(cx < 2 * n && cy < 2 * n);
                        seen[((cy * 2 * n) + cx) * 2 + ch] += 1;
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s == 1), "children overlap or leave gaps");
        // The center child really is the apex square with flipped orientation,
        // and its centroid is the parent's centroid.
        let (px, py) = tri_centroid(3, 5, 0);
        let (cx, cy, ch) = tri_children(3, 5, 0)[3];
        let (qx, qy) = tri_centroid(cx, cy, ch);
        assert!(((qx / 2.0) - px).abs() < 1e-12 && ((qy / 2.0) - py).abs() < 1e-12);
    }

    #[test]
    fn disk_order_is_a_bijection() {
        let order = tri_order();
        let mut hit = vec![false; TRI];
        for &m in order.iter() {
            assert!(!hit[m as usize]);
            hit[m as usize] = true;
        }
        assert!(hit.iter().all(|&h| h));
        // A parent texel's four children are contiguous in disk order.
        let mem: Vec<u32> = (0..TRI as u32).collect();
        let disk = mem_to_disk(&mem);
        assert_eq!(disk_to_mem(&disk), mem);
        // Disk index 0..4 of face 0 are the four children of the level-7
        // apex chain: they share the (0,0) square or its neighbors.
        let first: Vec<usize> = disk[..4].iter().map(|&m| m as usize).collect();
        let (tx, ty, h) = (0, 0, 0);
        let kids: Vec<usize> = tri_children(tx, ty, h).iter().map(|&(x, y, c)| tri_idx(x, y, c)).collect();
        assert_eq!(first, kids);
    }

    #[test]
    fn stamped_line_survives_pyramid() {
        // One diagonal trail across a cell at depth 12 -> coverage must
        // appear at 12 and fade (not vanish) at 10.
        let road = Road {
            class: mahere_osm::RoadClass::Path,
            pts: (0..200)
                .map(|i| {
                    let t = i as f32 / 199.0;
                    (46.15 + t * 0.02, -121.52 + t * 0.03)
                })
                .collect(),
        };
        let cells = bake_lines(&[road], 12, 10);
        let base_cov: u64 = cells
            .iter()
            .filter(|(k, _)| k.depth == 12)
            .map(|(_, c)| c.cov.iter().map(|&x| x as u64).sum::<u64>())
            .sum();
        let top_cov: u64 = cells
            .iter()
            .filter(|(k, _)| k.depth == 10)
            .map(|(_, c)| c.cov.iter().map(|&x| x as u64).sum::<u64>())
            .sum();
        assert!(base_cov > 0, "base stamping produced nothing");
        assert!(top_cov > 0, "pyramid lost the line");
        // Box filtering conserves total coverage (up to edge rounding).
        let ratio = base_cov as f64 / top_cov as f64;
        assert!(
            (0.7..=1.5).contains(&(ratio / 16.0)),
            "coverage not conserved through 2 levels: base {base_cov} top {top_cov}"
        );
    }

    /// Lines in the two diagonal directions stamp the same number of
    /// triangles per unit length — the isotropy the rhombus grid lacked.
    #[test]
    fn diagonal_lines_are_isotropic() {
        let mut cells = HashMap::new();
        let len = 100.0;
        stamp_segment(&mut cells, 12, 3, 10.0, 10.0, 10.0 + len, 10.0 + len, 1);
        let plus: usize = cells.values().map(|c| c.cov.iter().filter(|&&x| x > 0).count()).sum();
        let mut cells = HashMap::new();
        stamp_segment(&mut cells, 12, 3, 10.0, 110.0, 10.0 + len, 110.0 - len, 1);
        let minus: usize = cells.values().map(|c| c.cov.iter().filter(|&&x| x > 0).count()).sum();
        assert!(plus > 0 && minus > 0);
        // In UV units (1,1) is the rhombus's long diagonal, sqrt(3) times
        // the ground length of (1,-1): per unit of ground the two stamp the
        // same number of triangles within the lattice's inherent 2/sqrt(3).
        let r = plus as f64 / 3f64.sqrt() / minus as f64;
        assert!((0.8..=1.25).contains(&r), "+45 stamped {plus} triangles, -45 stamped {minus}");
    }
}
