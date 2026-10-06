//! The cell pipeline: bake source data into dymaxion-cell rasters — the
//! #pagetable renderer's entire diet.
//!
//! **One object per cell, every layer inside it.** A cell is one rhombus of
//! the diamond-Morton grid at `depth`; its file carries a VSF section per
//! layer present at that depth, and the client decides what to decode,
//! draw and style:
//!
//! - **dem**: elevation (u16, 0.25 m steps from -500 m) + unit normal
//!   (snorm16 ×3), sampled at triangle centroids from the source DEM at
//!   the base depth, then a pyramid of means.
//! - **line**: every road, trail, rail, power line and waterway stamped at
//!   its physical width as (class, coverage) texels; waterways carry their
//!   upstream-network weight as width and coverage.
//! - **land**: land cover (class, coverage) from OSM polygons.
//! - **water**: lakes, ponds, reservoirs, riverbanks as coverage.
//!
//! **Texels are triangles.** A cell's 256×256 UV squares are each split
//! along `u+v = k` into a lower and an upper equilateral triangle. The
//! triangular tiling has 6-fold symmetry and a line always crosses it
//! edge-to-edge, so linework is isotropic. Each triangle subdivides into
//! four — three corners and the inverted center — and that is the
//! pyramid's box filter: coverage up the pyramid IS area, so minor
//! features fade and dense ones glow with no styling.
//!
//! In memory a cell's planes are indexed `((ty << 8 | tx) << 1) | half`
//! (the renderer's stepping order). On disk they are in triangle-path
//! order — the triangle code's digits — so a parent texel's four children
//! are contiguous. [`disk_to_mem`] / [`mem_to_disk`] convert.
//!
//! Files are zstd'd VSF at `{name}.vsf.zst` where the name is the cell's
//! flattened VSF value (`u` depth, `wm` cell) in base64url: no delimiters,
//! no numerals — a directory layout that is byte-for-byte the bucket.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use mahere_coord::{Coord, morton_compact, morton_spread, uv_to_lat_lon};
use mahere_dem::DemStore;
use mahere_osm::{Area, Road};
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

/// Elevation quantization: 0.25 m steps from -500 m; 0xFFFF = no data.
pub const ELEV_NODATA: u16 = 0xFFFF;
pub fn quantize_elev(e: f32) -> u16 {
    if e.is_nan() { ELEV_NODATA } else { ((e + 500.0) * 4.0).clamp(0.0, 65534.0) as u16 }
}
pub fn dequantize_elev(q: u16) -> f32 {
    q as f32 / 4.0 - 500.0
}

/// A cell address: dymaxion Morton prefix (diamond in the top 4 bits of the
/// full-resolution coordinate, right-aligned here) at `depth`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
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

    /// The cell at `depth` with grid coordinates (cu, cv) in `diamond`.
    pub fn from_grid(diamond: u8, depth: u8, cu: u64, cv: u64) -> CellKey {
        let m = (morton_spread(cu << (30 - depth as u32)) << 1) | morton_spread(cv << (30 - depth as u32));
        CellKey { depth, prefix: ((diamond as u64) << (2 * depth)) | (m >> (60 - 2 * depth as u32)) }
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

    pub fn path(self) -> String {
        format!("{}.vsf.zst", self.name())
    }

    /// Diamond-UV rectangle covered by this cell.
    pub fn uv_rect(self) -> (f64, f64, f64) {
        let (cu, cv) = self.grid();
        let size = 1.0 / (1u64 << self.depth) as f64;
        (cu as f64 * size, cv as f64 * size, size)
    }

    /// Rhombus edge length of one UV square (texel) in this cell, meters.
    pub fn texel_m(self) -> f64 {
        7_054_000.0 / (1u64 << self.depth) as f64 / TEX as f64
    }
}

/// Rhombus edge of a texel at `depth`, meters.
pub fn texel_m(depth: u8) -> f64 {
    7_054_000.0 / (1u64 << depth) as f64 / TEX as f64
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

/// Centroid offset within the UV square for each half.
#[inline(always)]
pub fn tri_off(half: usize) -> f64 {
    if half == 0 { 1.0 / 3.0 } else { 2.0 / 3.0 }
}

/// UV position of a triangle texel's centroid, in texel units.
#[inline(always)]
pub fn tri_centroid(tx: usize, ty: usize, half: usize) -> (f64, f64) {
    let off = tri_off(half);
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

/// Squared ground distance (in rhombus-edge units) of a UV offset: the UV
/// axes meet at 60°, so |du e_u + dv e_v|² = du² + dv² + du·dv.
#[inline(always)]
fn uv_dist2(du: f64, dv: f64) -> f64 {
    du * du + dv * dv + du * dv
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
fn texel_of(c: Coord, depth: u8) -> (u8, f64, f64) {
    let (iu, iv) = c.uv();
    let shift = 30 - depth as u32 - TEX_BITS as u32;
    let scale = 1.0 / (1u64 << shift) as f64;
    (c.diamond(), iu as f64 * scale, iv as f64 * scale)
}

// ==================== PLANES ====================

/// (class, coverage) planes: lines and land cover.
#[derive(Clone)]
pub struct ClassCell {
    pub class: Vec<u8>, // 0 = empty; else class + 1
    pub cov: Vec<u8>,
}

impl ClassCell {
    pub fn new() -> ClassCell {
        ClassCell { class: vec![0; TRI], cov: vec![0; TRI] }
    }
    pub fn is_empty(&self) -> bool {
        self.cov.iter().all(|&c| c == 0)
    }
}

impl Default for ClassCell {
    fn default() -> Self {
        Self::new()
    }
}

/// Coverage-only plane: water areas.
#[derive(Clone)]
pub struct CovCell {
    pub cov: Vec<u8>,
}

impl CovCell {
    pub fn new() -> CovCell {
        CovCell { cov: vec![0; TRI] }
    }
    pub fn is_empty(&self) -> bool {
        self.cov.iter().all(|&c| c == 0)
    }
}

impl Default for CovCell {
    fn default() -> Self {
        Self::new()
    }
}

/// Baker-side dem planes (f32 so the pyramid averages at full precision).
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

/// Everything a cell can carry. Vector layers at a vector depth are always
/// all present (empty planes compress to nothing), so a reader finding one
/// of them knows the cell's vector truth is complete.
#[derive(Clone, Default)]
pub struct Cell {
    pub dem: Option<DemCell>,
    pub line: Option<ClassCell>,
    pub land: Option<ClassCell>,
    pub water: Option<CovCell>,
}

// ==================== LINE LAYER ====================

/// Stamp every feature at its physical width into base-depth cells, then
/// build the pyramid up to `min_depth` (major class wins per texel).
pub fn bake_lines(feats: &[Road], base_depth: u8, min_depth: u8) -> HashMap<CellKey, ClassCell> {
    assert!(base_depth as u32 + TEX_BITS as u32 <= 30);
    let mut cells: HashMap<CellKey, ClassCell> = HashMap::new();
    let mut cross_diamond = 0usize;
    let tm = texel_m(base_depth);
    for road in feats {
        let class = road.class as u8 + 1;
        let r = (road.width_m() as f64 / 2.0 / tm).max(0.5);
        let cov_max = road.cov_max();
        let mut prev: Option<(u8, f64, f64)> = None;
        for &(lat, lon) in &road.pts {
            let c = Coord::from_lat_lon(lat as f64, lon as f64);
            let (d, gu, gv) = texel_of(c, base_depth);
            if let Some((pd, pu, pv)) = prev {
                if pd == d {
                    stamp_segment(&mut cells, base_depth, d, (pu, pv), (gu, gv), r, class, cov_max);
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
    pyramid_class(cells, base_depth, min_depth, ClassMerge::Major)
}

/// Stamp a segment as a band of radius `r` texels (ground metric): walk the
/// centreline in half-texel steps and cover every triangle whose centroid
/// is within the band, coverage feathered over the last texel. Coverage
/// combines by max; class follows the strongest coverage.
#[allow(clippy::too_many_arguments)]
fn stamp_segment(
    cells: &mut HashMap<CellKey, ClassCell>,
    depth: u8,
    diamond: u8,
    a: (f64, f64),
    b: (f64, f64),
    r: f64,
    class: u8,
    cov_max: u8,
) {
    let extent = ((1u64 << depth) * TEX as u64) as f64;
    let len = uv_dist2(b.0 - a.0, b.1 - a.1).sqrt();
    let steps = (len * 2.0).ceil().max(1.0) as usize;
    // Bounding box of a ground disc of radius r in UV units (the 60° basis
    // stretches it by up to 2/sqrt(3)).
    let reach = (r * 1.16 + 1.0).ceil() as i64;
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let (px, py) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
        let (cx, cy) = (px.floor() as i64, py.floor() as i64);
        for ty in (cy - reach)..=(cy + reach) {
            if ty < 0 || ty as f64 >= extent {
                continue;
            }
            for tx in (cx - reach)..=(cx + reach) {
                if tx < 0 || tx as f64 >= extent {
                    continue;
                }
                for half in 0..2 {
                    let off = tri_off(half);
                    let d = uv_dist2(tx as f64 + off - px, ty as f64 + off - py).sqrt();
                    let cov = (cov_max as f64 * (r + 0.5 - d).clamp(0.0, 1.0)) as u8;
                    if cov == 0 {
                        continue;
                    }
                    let (tx, ty) = (tx as usize, ty as usize);
                    let key = CellKey::from_grid(diamond, depth, (tx / TEX) as u64, (ty / TEX) as u64);
                    let cell = cells.entry(key).or_default();
                    let i = tri_idx(tx % TEX, ty % TEX, half);
                    if cov > cell.cov[i] || (cov == cell.cov[i] && class < cell.class[i]) {
                        cell.cov[i] = cov;
                        cell.class[i] = class;
                    }
                }
            }
        }
    }
}

// ==================== AREA LAYERS ====================

/// Rasterize land cover and water polygons at base depth (even-odd fill,
/// sampled at triangle centroids), then pyramid both to `min_depth`.
pub fn bake_areas(
    areas: &[Area],
    base_depth: u8,
    min_depth: u8,
) -> (HashMap<CellKey, ClassCell>, HashMap<CellKey, CovCell>) {
    let mut land: HashMap<CellKey, ClassCell> = HashMap::new();
    let mut water: HashMap<CellKey, CovCell> = HashMap::new();
    let mut cross_diamond = 0usize;
    for area in areas {
        let mut rings: Vec<Vec<(f64, f64)>> = Vec::new();
        let mut diamond: Option<u8> = None;
        let mut bad = false;
        for ring in &area.rings {
            let mut out = Vec::with_capacity(ring.len());
            for &(lat, lon) in ring {
                let c = Coord::from_lat_lon(lat as f64, lon as f64);
                let (d, gu, gv) = texel_of(c, base_depth);
                if diamond.get_or_insert(d) != &d {
                    bad = true;
                }
                out.push((gu, gv));
            }
            rings.push(out);
        }
        if bad {
            cross_diamond += 1;
            continue;
        }
        let Some(d) = diamond else { continue };
        let class = area.class as u8 + 1;
        if area.class == mahere_osm::AreaClass::Water {
            fill_rings(&rings, base_depth, |tx, ty, half| {
                let key = CellKey::from_grid(d, base_depth, (tx / TEX) as u64, (ty / TEX) as u64);
                water.entry(key).or_default().cov[tri_idx(tx % TEX, ty % TEX, half)] = 255;
            });
        } else {
            fill_rings(&rings, base_depth, |tx, ty, half| {
                let key = CellKey::from_grid(d, base_depth, (tx / TEX) as u64, (ty / TEX) as u64);
                let cell = land.entry(key).or_default();
                let i = tri_idx(tx % TEX, ty % TEX, half);
                if cell.class[i] == 0 || class > cell.class[i] {
                    cell.class[i] = class;
                    cell.cov[i] = 255;
                }
            });
        }
    }
    if cross_diamond > 0 {
        eprintln!("  (skipped {cross_diamond} cross-diamond areas)");
    }
    (
        pyramid_class(land, base_depth, min_depth, ClassMerge::Dominant),
        pyramid_cov(water, base_depth, min_depth),
    )
}

/// Even-odd scanline fill of rings given in global texel coordinates,
/// visiting every triangle whose centroid is inside.
fn fill_rings(rings: &[Vec<(f64, f64)>], depth: u8, mut visit: impl FnMut(usize, usize, usize)) {
    let extent = ((1u64 << depth) * TEX as u64) as f64;
    let (mut v0, mut v1) = (f64::MAX, f64::MIN);
    for r in rings {
        for p in r {
            v0 = v0.min(p.1);
            v1 = v1.max(p.1);
        }
    }
    if v0 == f64::MAX {
        return;
    }
    let ty0 = v0.floor().max(0.0) as i64;
    let ty1 = v1.ceil().min(extent) as i64;
    let mut xs: Vec<f64> = Vec::new();
    for ty in ty0..ty1 {
        for half in 0..2 {
            let off = tri_off(half);
            let vl = ty as f64 + off;
            xs.clear();
            for r in rings {
                for w in r.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    if (a.1 <= vl) != (b.1 <= vl) {
                        xs.push(a.0 + (vl - a.1) * (b.0 - a.0) / (b.1 - a.1));
                    }
                }
            }
            if xs.len() < 2 {
                continue;
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for pair in xs.chunks(2) {
                if pair.len() < 2 {
                    break;
                }
                // Centroid u = tx + off inside [x0, x1).
                let tx_min = ((pair[0] - off).ceil().max(0.0)) as i64;
                let tx_max = ((pair[1] - off).ceil() - 1.0).min(extent - 1.0) as i64;
                for tx in tx_min..=tx_max {
                    visit(tx as usize, ty as usize, half);
                }
            }
        }
    }
}

// ==================== PYRAMIDS ====================

#[derive(Clone, Copy)]
pub enum ClassMerge {
    /// Lines: the most major class present among the children wins.
    Major,
    /// Land cover: the class with the most coverage among the children.
    Dominant,
}

/// Parent texel = mean coverage of its four triangle children (missing
/// children are empty), class per `merge`.
pub fn pyramid_class(
    mut cells: HashMap<CellKey, ClassCell>,
    base_depth: u8,
    min_depth: u8,
    merge: ClassMerge,
) -> HashMap<CellKey, ClassCell> {
    let mut depth = base_depth;
    while depth > min_depth {
        let mut parents: Vec<CellKey> = cells.keys().filter(|k| k.depth == depth).map(|k| k.parent()).collect();
        parents.sort();
        parents.dedup();
        let built: Vec<(CellKey, ClassCell)> = parents
            .par_iter()
            .map(|&p| {
                let kids: [Option<&ClassCell>; 4] = std::array::from_fn(|q| cells.get(&p.child(q as u64)));
                let mut cell = ClassCell::new();
                for ty in 0..TEX {
                    for tx in 0..TEX {
                        for half in 0..2 {
                            let mut covsum = 0u32;
                            let mut best = 0u8;
                            let mut best_cov = 0u32;
                            let mut per_class = [0u32; 32];
                            for (q, i) in child_cell_texels(tx, ty, half) {
                                let Some(child) = kids[q as usize] else { continue };
                                let cv = child.cov[i] as u32;
                                covsum += cv;
                                let cl = child.class[i];
                                if cl != 0 && cv > 0 {
                                    match merge {
                                        ClassMerge::Major => {
                                            if best == 0 || cl < best {
                                                best = cl;
                                            }
                                        }
                                        ClassMerge::Dominant => {
                                            let slot = (cl as usize).min(31);
                                            per_class[slot] += cv;
                                            if per_class[slot] > best_cov {
                                                best_cov = per_class[slot];
                                                best = cl;
                                            }
                                        }
                                    }
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

pub fn pyramid_cov(mut cells: HashMap<CellKey, CovCell>, base_depth: u8, min_depth: u8) -> HashMap<CellKey, CovCell> {
    let mut depth = base_depth;
    while depth > min_depth {
        let mut parents: Vec<CellKey> = cells.keys().filter(|k| k.depth == depth).map(|k| k.parent()).collect();
        parents.sort();
        parents.dedup();
        let built: Vec<(CellKey, CovCell)> = parents
            .par_iter()
            .map(|&p| {
                let kids: [Option<&CovCell>; 4] = std::array::from_fn(|q| cells.get(&p.child(q as u64)));
                let mut cell = CovCell::new();
                for ty in 0..TEX {
                    for tx in 0..TEX {
                        for half in 0..2 {
                            let mut covsum = 0u32;
                            for (q, i) in child_cell_texels(tx, ty, half) {
                                if let Some(child) = kids[q as usize] {
                                    covsum += child.cov[i] as u32;
                                }
                            }
                            cell.cov[tri_idx(tx, ty, half)] = (covsum / 4) as u8;
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

// ==================== DEM LAYER ====================

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
        .filter(|(_, c)| c.elev.iter().any(|e| !e.is_nan()))
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
        parents.sort();
        parents.dedup();
        let by_key: HashMap<(u8, u64), &DemCell> = cur.iter().map(|(k, c)| ((k.depth, k.prefix), c)).collect();
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
            keys.insert(CellKey::containing(Coord::from_lat_lon(lat.min(lat1), lon.min(lon1)), depth));
            lon += step;
        }
        lat += step;
    }
    let mut v: Vec<CellKey> = keys.into_iter().collect();
    v.sort();
    v
}

// ==================== ASSEMBLY ====================

/// Union the layers into cells. Wherever any vector layer exists at a
/// depth, all three are present (empty planes for the missing ones) so
/// readers never have to climb for one layer but not another.
pub fn assemble(
    dem: Vec<(CellKey, DemCell)>,
    line: HashMap<CellKey, ClassCell>,
    land: HashMap<CellKey, ClassCell>,
    water: HashMap<CellKey, CovCell>,
) -> HashMap<CellKey, Cell> {
    let mut cells: HashMap<CellKey, Cell> = HashMap::new();
    for (k, d) in dem {
        cells.entry(k).or_default().dem = Some(d);
    }
    let mut vec_keys: Vec<CellKey> = line.keys().chain(land.keys()).chain(water.keys()).copied().collect();
    vec_keys.sort();
    vec_keys.dedup();
    let mut line = line;
    let mut land = land;
    let mut water = water;
    for k in vec_keys {
        let c = cells.entry(k).or_default();
        c.line = Some(line.remove(&k).unwrap_or_default());
        c.land = Some(land.remove(&k).unwrap_or_default());
        c.water = Some(water.remove(&k).unwrap_or_default());
    }
    cells
}

// ==================== VSF I/O ====================

/// Decoded, quantized planes in memory order — what the loader hands the
/// renderer and what a merge-on-write reads back.
#[derive(Clone, Default)]
pub struct CellPlanes {
    pub dem: Option<DemPlanes>,
    pub line: Option<ClassCell>,
    pub land: Option<ClassCell>,
    pub water: Option<CovCell>,
}

#[derive(Clone)]
pub struct DemPlanes {
    pub elev: Vec<u16>,
    pub nx: Vec<i16>,
    pub ny: Vec<i16>,
    pub nz: Vec<i16>,
}

impl DemCell {
    /// Quantize: 0.25 m elevation steps, unit normal from the gradient as
    /// snorm16 (i16, not u8: u8 bands on gentle slopes, exactly where
    /// hillshade banding shows).
    pub fn quantize(&self) -> DemPlanes {
        let mut p = DemPlanes { elev: vec![0; TRI], nx: vec![0; TRI], ny: vec![0; TRI], nz: vec![0; TRI] };
        for i in 0..TRI {
            p.elev[i] = quantize_elev(self.elev[i]);
            let inv = 1.0 / (1.0 + self.ge[i] * self.ge[i] + self.gn[i] * self.gn[i]).sqrt();
            p.nx[i] = (-self.ge[i] * inv * 32767.0) as i16;
            p.ny[i] = (-self.gn[i] * inv * 32767.0) as i16;
            p.nz[i] = (inv * 32767.0) as i16;
        }
        p
    }
}

impl Cell {
    pub fn quantize(&self) -> CellPlanes {
        CellPlanes {
            dem: self.dem.as_ref().map(|d| d.quantize()),
            line: self.line.clone(),
            land: self.land.clone(),
            water: self.water.clone(),
        }
    }
}

impl CellPlanes {
    /// Overlay `self` (a new bake) onto `old`. A bake's footprint is where
    /// it has elevation: inside it the new vector planes win, outside the
    /// old ones stay; without a dem the new planes replace wholesale.
    pub fn merge_over(self, old: CellPlanes) -> CellPlanes {
        let mut out = self;
        let CellPlanes { dem: old_dem, line: old_line, land: old_land, water: old_water } = old;
        let (mut old_line, mut old_land, mut old_water) = (old_line, old_land, old_water);
        // The new bake's footprint, if it has one.
        let mask: Option<Vec<bool>> = out.dem.as_ref().map(|d| d.elev.iter().map(|&e| e != ELEV_NODATA).collect());
        match (&mut out.dem, old_dem) {
            (Some(new), Some(old_dem)) => {
                let mask = mask.as_ref().unwrap();
                for i in 0..TRI {
                    if !mask[i] {
                        new.elev[i] = old_dem.elev[i];
                        new.nx[i] = old_dem.nx[i];
                        new.ny[i] = old_dem.ny[i];
                        new.nz[i] = old_dem.nz[i];
                    }
                }
            }
            (None, Some(old_dem)) => out.dem = Some(old_dem),
            _ => {}
        }
        if let Some(mask) = &mask {
            fn keep_old_class(new: &mut Option<ClassCell>, old: Option<ClassCell>, mask: &[bool]) {
                match (new, old) {
                    (Some(n), Some(o)) => {
                        for i in 0..TRI {
                            if !mask[i] {
                                n.class[i] = o.class[i];
                                n.cov[i] = o.cov[i];
                            }
                        }
                    }
                    (n @ None, Some(o)) => *n = Some(o),
                    _ => {}
                }
            }
            keep_old_class(&mut out.line, old_line.take(), mask);
            keep_old_class(&mut out.land, old_land.take(), mask);
            match (&mut out.water, old_water.take()) {
                (Some(n), Some(o)) => {
                    for i in 0..TRI {
                        if !mask[i] {
                            n.cov[i] = o.cov[i];
                        }
                    }
                }
                (n @ None, Some(o)) => *n = Some(o),
                _ => {}
            }
        }
        if out.line.is_none() {
            out.line = old_line;
        }
        if out.land.is_none() {
            out.land = old_land;
        }
        if out.water.is_none() {
            out.water = old_water;
        }
        out
    }

    pub fn encode(&self) -> Result<Vec<u8>, String> {
        let mut b = VsfBuilder::new();
        if let Some(d) = &self.dem {
            b = b.add_section(
                "dem",
                vec![
                    ("elev".to_string(), VsfType::t_u4(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&d.elev)))),
                    ("nx".to_string(), VsfType::t_i4(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&d.nx)))),
                    ("ny".to_string(), VsfType::t_i4(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&d.ny)))),
                    ("nz".to_string(), VsfType::t_i4(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&d.nz)))),
                ],
            );
        }
        for (name, planes) in [("line", &self.line), ("land", &self.land)] {
            if let Some(p) = planes {
                b = b.add_section(
                    name,
                    vec![
                        ("class".to_string(), VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&p.class)))),
                        ("cov".to_string(), VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&p.cov)))),
                    ],
                );
            }
        }
        if let Some(w) = &self.water {
            b = b.add_section(
                "water",
                vec![("cov".to_string(), VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&w.cov))))],
            );
        }
        b.build().map_err(|e| format!("cell build: {e:?}"))
    }
}

/// Write a cell, merged over whatever is already on disk under that key —
/// so regional bakes compose instead of clobbering each other.
pub fn write_cell(out: &Path, key: CellKey, cell: &Cell) -> Result<(), String> {
    let path = out.join(key.path());
    let mut planes = cell.quantize();
    if let Ok(existing) = std::fs::read(&path) {
        if let Ok(old) = decode_cell(&existing) {
            planes = planes.merge_over(old);
        }
    }
    let bytes = planes.encode()?;
    write_file(&path, &bytes)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    // Whole-file zstd: VSF internals untouched, the bucket keeps the
    // .vsf.zst names, and empty planes shrink to nothing.
    let z = zstd::encode_all(bytes, 3).map_err(|e| e.to_string())?;
    std::fs::write(path, z).map_err(|e| format!("{}: {e}", path.display()))
}

/// Decode a cell from raw file bytes (zstd or plain VSF) — the loader
/// thread's entry point; no filesystem coupling. Width-agnostic reads.
pub fn decode_cell(data: &[u8]) -> Result<CellPlanes, String> {
    let plain: Vec<u8>;
    let data = if data.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
        plain = zstd::decode_all(data).map_err(|e| e.to_string())?;
        &plain[..]
    } else {
        data
    };
    let (header, end) = vsf::VsfHeader::decode(data).map_err(|e| e.to_string())?;
    let sections = header.sections(data, end).map_err(|e| e.to_string())?;
    let mut out = CellPlanes::default();
    for s in sections {
        let fields: HashMap<String, VsfType> =
            s.fields.into_iter().filter_map(|f| f.values.into_iter().next().map(|v| (f.name, v))).collect();
        match s.name.as_str() {
            "dem" => {
                if let (Some(elev), Some(nx), Some(ny), Some(nz)) = (
                    fields.get("elev").and_then(plane_u16_mem),
                    fields.get("nx").and_then(plane_i16_mem),
                    fields.get("ny").and_then(plane_i16_mem),
                    fields.get("nz").and_then(plane_i16_mem),
                ) {
                    out.dem = Some(DemPlanes { elev, nx, ny, nz });
                }
            }
            "line" | "land" => {
                if let (Some(class), Some(cov)) =
                    (fields.get("class").and_then(plane_u8_mem), fields.get("cov").and_then(plane_u8_mem))
                {
                    let planes = Some(ClassCell { class, cov });
                    if s.name == "line" {
                        out.line = planes;
                    } else {
                        out.land = planes;
                    }
                }
            }
            "water" => {
                if let Some(cov) = fields.get("cov").and_then(plane_u8_mem) {
                    out.water = Some(CovCell { cov });
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// A u8 plane in memory order, or None if missing/malformed.
pub fn plane_u8_mem(v: &VsfType) -> Option<Vec<u8>> {
    let disk: Vec<u8> = match v {
        VsfType::t_u3(t) => t.data.clone(),
        VsfType::v_u3(t) => t.data.clone(),
        VsfType::t_u0(t) => t.data.iter().map(|&b| b as u8).collect(),
        _ => return None,
    };
    (disk.len() == TRI).then(|| disk_to_mem(&disk))
}

pub fn plane_u16_mem(v: &VsfType) -> Option<Vec<u16>> {
    let disk: Vec<u16> = match v {
        VsfType::t_u4(t) => t.data.clone(),
        VsfType::v_u4(t) => t.data.clone(),
        VsfType::t_u3(t) => t.data.iter().map(|&x| x as u16).collect(),
        VsfType::v_u3(t) => t.data.iter().map(|&x| x as u16).collect(),
        _ => return None,
    };
    (disk.len() == TRI).then(|| disk_to_mem(&disk))
}

pub fn plane_i16_mem(v: &VsfType) -> Option<Vec<i16>> {
    let disk: Vec<i16> = match v {
        VsfType::t_i4(t) => t.data.clone(),
        VsfType::v_i4(t) => t.data.clone(),
        VsfType::t_i3(t) => t.data.iter().map(|&x| x as i16).collect(),
        VsfType::v_i3(t) => t.data.iter().map(|&x| x as i16).collect(),
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
            let (cu, cv) = key.grid();
            assert_eq!(CellKey::from_grid(key.diamond(), depth, cu, cv), key);
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
        let mem: Vec<u32> = (0..TRI as u32).collect();
        let disk = mem_to_disk(&mem);
        assert_eq!(disk_to_mem(&disk), mem);
    }

    fn trail(class: mahere_osm::RoadClass, weight: f32) -> Road {
        Road {
            class,
            weight,
            pts: (0..200)
                .map(|i| {
                    let t = i as f32 / 199.0;
                    (46.15 + t * 0.02, -121.52 + t * 0.03)
                })
                .collect(),
        }
    }

    #[test]
    fn stamped_line_survives_pyramid() {
        let cells = bake_lines(&[trail(mahere_osm::RoadClass::Path, 0.0)], 12, 10);
        let sum = |d: u8| -> u64 {
            cells.iter().filter(|(k, _)| k.depth == d).map(|(_, c)| c.cov.iter().map(|&x| x as u64).sum::<u64>()).sum()
        };
        let (base_cov, top_cov) = (sum(12), sum(10));
        assert!(base_cov > 0, "base stamping produced nothing");
        assert!(top_cov > 0, "pyramid lost the line");
        let ratio = base_cov as f64 / top_cov as f64;
        assert!((0.7..=1.5).contains(&(ratio / 16.0)), "coverage not conserved: base {base_cov} top {top_cov}");
    }

    /// A motorway covers more ground than a path along the same line, in
    /// proportion to its width.
    #[test]
    fn width_scales_coverage() {
        let path = bake_lines(&[trail(mahere_osm::RoadClass::Path, 0.0)], 13, 13);
        let mway = bake_lines(&[trail(mahere_osm::RoadClass::Motorway, 0.0)], 13, 13);
        let total = |m: &HashMap<CellKey, ClassCell>| -> u64 {
            m.values().map(|c| c.cov.iter().map(|&x| x as u64).sum::<u64>()).sum()
        };
        let r = total(&mway) as f64 / total(&path) as f64;
        assert!(r > 8.0 && r < 40.0, "motorway/path coverage ratio {r}");
    }

    /// A square lake fills its interior (and only its interior), and the
    /// pyramid conserves its area.
    #[test]
    fn polygon_fill_is_area_exact() {
        let (lat0, lon0, lat1, lon1) = (46.20f32, -121.50f32, 46.22f32, -121.47f32);
        let area = Area {
            class: mahere_osm::AreaClass::Water,
            rings: vec![vec![(lat0, lon0), (lat0, lon1), (lat1, lon1), (lat1, lon0), (lat0, lon0)]],
        };
        let (_, water) = bake_areas(&[area], 12, 10);
        let sum = |d: u8| -> u64 {
            water.iter().filter(|(k, _)| k.depth == d).map(|(_, c)| c.cov.iter().map(|&x| x as u64).sum::<u64>()).sum()
        };
        let base = sum(12);
        // Expected count: the lake's area over the LOCAL triangle area (the
        // gnomonic face mapping varies texel size across a face, so measure
        // one texel's parallelogram on the ground at the lake's centre).
        let c = Coord::from_lat_lon(46.21, -121.485);
        let (iu, iv) = c.uv();
        let unit = 1.0 / (1u64 << 30) as f64;
        let (u, v) = (iu as f64 * unit, iv as f64 * unit);
        let du = 1.0 / (1u64 << (12 + 8)) as f64;
        let m = |lat: f64, lon: f64| -> (f64, f64) { (lon * 111_320.0 * 46.21f64.to_radians().cos(), lat * 111_320.0) };
        let p0 = uv_to_lat_lon(c.diamond(), u, v);
        let pu = uv_to_lat_lon(c.diamond(), u + du, v);
        let pv = uv_to_lat_lon(c.diamond(), u, v + du);
        let (o, a, b) = (m(p0.0, p0.1), m(pu.0, pu.1), m(pv.0, pv.1));
        let para = ((a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)).abs();
        let lake_m2 = (0.02 * 111_320.0) * (0.03 * 111_320.0 * 46.21f64.to_radians().cos());
        let expect_texels = lake_m2 / (para / 2.0);
        let got_texels = base as f64 / 255.0;
        assert!((got_texels / expect_texels - 1.0).abs() < 0.05, "lake texels {got_texels} vs {expect_texels}");
        let top = sum(10);
        assert!((base as f64 / top as f64 / 16.0 - 1.0).abs() < 0.1, "pyramid lost water: {base} vs {top}");
    }

    #[test]
    fn cell_round_trips_through_vsf_and_merges() {
        let mut dem = DemCell::new();
        for i in 0..TRI / 2 {
            dem.elev[i] = 1000.0 + (i % 7) as f32;
            dem.ge[i] = 0.1;
        }
        let mut line = ClassCell::new();
        line.class[5] = 3;
        line.cov[5] = 200;
        let cell = Cell { dem: Some(dem), line: Some(line), land: Some(ClassCell::new()), water: None };
        let bytes = cell.quantize().encode().unwrap();
        let back = decode_cell(&bytes).unwrap();
        let d = back.dem.as_ref().unwrap();
        assert_eq!(d.elev[3], quantize_elev(1003.0));
        assert_eq!(d.elev[TRI - 1], ELEV_NODATA);
        assert!(d.nx[3] < 0 && d.nz[3] > 30000);
        assert_eq!(back.line.as_ref().unwrap().cov[5], 200);
        assert!(back.water.is_none());

        // Merge: a second bake covering the OTHER half keeps our half.
        let mut dem2 = DemCell::new();
        for i in TRI / 2..TRI {
            dem2.elev[i] = 50.0;
        }
        let mut line2 = ClassCell::new();
        line2.cov[5] = 1; // outside dem2's footprint: must NOT win
        line2.cov[TRI - 1] = 9;
        let cell2 = Cell { dem: Some(dem2), line: Some(line2), land: None, water: None };
        let merged = cell2.quantize().merge_over(back);
        let d = merged.dem.as_ref().unwrap();
        assert_eq!(d.elev[3], quantize_elev(1003.0));
        assert_eq!(d.elev[TRI - 1], quantize_elev(50.0));
        let l = merged.line.as_ref().unwrap();
        assert_eq!(l.cov[5], 200);
        assert_eq!(l.cov[TRI - 1], 9);
        assert!(merged.land.is_some(), "a layer absent from the new bake survives from the old");
    }
}
