//! The cell pipeline: bake source data into dymaxion-cell rasters — the #pagetable renderer's entire diet.
//!
//! **One object per cell, every layer inside it.** A cell is one rhombus of the diamond-Morton grid at `depth`; its file carries a VSF section per layer present at that depth, and the client decides what to decode, draw and style:
//!
//! - **dem**: elevation sampled at triangle centroids from the source DEM at the base depth, then a pyramid of means; stored through the pyramid codec ([`pyr`]) at an adaptive step of at least 5 cm, normals derived on load from a one-texel apron of the neighbours.
//! - **line**: every road, trail, rail, power line and waterway stamped at its physical width as (class, coverage, magnitude, use) texels; waterways carry their upstream-network weight as width and coverage.
//! - **land**: land cover (class, coverage) from OSM polygons.
//! - **water**: lakes, ponds, reservoirs, riverbanks as coverage.
//! - **img**: NAIP red, green, blue and near-infrared, 8-bit, box-filtered into the texel, through the pyramid codec, never finer than depth 14.
//!
//! **Texels are triangles.** A cell's 256×256 UV squares are each split along `u+v = k` into a lower and an upper equilateral triangle. The triangular tiling has 6-fold symmetry and a line always crosses it edge-to-edge, so linework is isotropic. Each triangle subdivides into four — three corners and the inverted center — and that is the pyramid's box filter: coverage up the pyramid IS area, so minor features fade and dense ones glow with no styling.
//!
//! In memory a cell's planes are indexed `((ty << 8 | tx) << 1) | half` (the renderer's stepping order). On disk they are in triangle-path order — the triangle code's digits — so a parent texel's four children are contiguous. [`disk_to_mem`] / [`mem_to_disk`] convert.
//!
//! Files are zstd'd VSF at `{name}.vsf.zst` where the name is the cell's flattened VSF value (`u` depth, `wm` cell) in base64url: no delimiters, no numerals — a directory layout that is byte-for-byte the bucket.

pub mod pyr;

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use mahere_coord::{Coord, morton_compact, morton_spread, uv_to_lat_lon};
use mahere_dem::DemStore;
use mahere_osm::{Area, Road};
use rayon::prelude::*;
use vsf::types::{Tensor, WorldCell};
use vsf::{VsfBuilder, VsfType};

/// base64url without padding — how bytes are spelled when they must be a name (the same alphabet tohu uses for vault file names).
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

/// A cell address: dymaxion Morton prefix (diamond in the top 4 bits of the full-resolution coordinate, right-aligned here) at `depth`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct CellKey {
    pub depth: u8,
    pub prefix: u64,
}

impl CellKey {
    /// The cell containing `c` at `depth`: diamond + 2*depth Morton bits, right-aligned (so prefixes sort and shift like integers).
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

    /// The cell as VSF values, flattened: its depth (`u`) and the world cell (`wm`). This is the cell's identity everywhere a key is needed — no delimiters, no numerals, the type tags are the structure.
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

/// Triangle texel containing a point given in texel units within the cell (or any grid): the UV square plus which side of `u+v = k` it lies on.
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

/// The four children of triangle (tx, ty, half) on the next-finer grid, in code-digit order: 0 apex, 1 toward r (+v), 2 toward q (+u), 3 the inverted center — which lives in the apex's UV square with the opposite orientation. A lower triangle's apex is its square's (0,0) corner, an upper's is (1,1).
#[inline]
pub fn tri_children(tx: usize, ty: usize, half: usize) -> [(usize, usize, usize); 4] {
    let (x, y) = (2 * tx, 2 * ty);
    if half == 0 {
        [(x, y, 0), (x, y + 1, 0), (x + 1, y, 0), (x, y, 1)]
    } else {
        [(x + 1, y + 1, 1), (x, y + 1, 1), (x + 1, y, 1), (x + 1, y + 1, 0)]
    }
}

/// Parent-cell texel -> its four children as (child cell Morton digit, memory index in that child cell).
#[inline]
fn child_cell_texels(tx: usize, ty: usize, half: usize) -> [(u64, usize); 4] {
    tri_children(tx, ty, half).map(|(cx, cy, ch)| {
        let q = (((cx >> TEX_BITS) as u64) << 1) | (cy >> TEX_BITS) as u64;
        (q, tri_idx(cx & (TEX - 1), cy & (TEX - 1), ch))
    })
}

/// Squared ground distance (in rhombus-edge units) of a UV offset: the UV axes meet at 60°, so |du e_u + dv e_v|² = du² + dv² + du·dv.
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

/// Global texel coordinates of a Coord at a given base depth: the texel grid is the cell grid times TEX.
fn texel_of(c: Coord, depth: u8) -> (u8, f64, f64) {
    let (iu, iv) = c.uv();
    let shift = 30 - depth as u32 - TEX_BITS as u32;
    let scale = 1.0 / (1u64 << shift) as f64;
    (c.diamond(), iu as f64 * scale, iv as f64 * scale)
}

// ==================== PLANES ====================

/// (class, coverage) planes: lines and land cover. Lines also carry `mag` (a log-scaled magnitude whose meaning is per class — catchment for water, voltage for power, lanes/traffic for roads) and `uses` (the [`mahere_osm::use_bits`]); both stay empty for land cover.
#[derive(Clone)]
pub struct ClassCell {
    pub class: Vec<u8>, // 0 = empty; else class + 1
    pub cov: Vec<u8>,
    pub mag: Vec<u8>,
    pub uses: Vec<u8>,
}

impl ClassCell {
    pub fn new() -> ClassCell {
        ClassCell { class: vec![0; TRI], cov: vec![0; TRI], mag: Vec::new(), uses: Vec::new() }
    }
    /// A line cell: with the attribute planes.
    pub fn new_line() -> ClassCell {
        ClassCell { class: vec![0; TRI], cov: vec![0; TRI], mag: vec![0; TRI], uses: vec![0; TRI] }
    }
    pub fn is_empty(&self) -> bool {
        self.cov.iter().all(|&c| c == 0)
    }
    #[inline(always)]
    pub fn mag_at(&self, i: usize) -> u8 {
        self.mag.get(i).copied().unwrap_or(0)
    }
    #[inline(always)]
    pub fn uses_at(&self, i: usize) -> u8 {
        self.uses.get(i).copied().unwrap_or(0)
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

/// Baker-side dem planes: elevation only (f32 so the pyramid averages at full precision), plus a one-texel apron copied from neighbour cells so the loader can derive exact normals at the cell edge. Normals are never stored — they're six of the eight bytes a texel used to cost.
#[derive(Clone)]
pub struct DemCell {
    pub elev: Vec<f32>,
    /// Neighbour edge texels: `[side][half][index]`, sides west (tx=-1), east (tx=256), south (ty=-1), north (ty=256); NaN where unknown.
    pub apron: Vec<f32>,
}

pub const APRON: usize = 4 * 2 * TEX;

impl DemCell {
    fn new() -> DemCell {
        DemCell { elev: vec![f32::NAN; TRI], apron: vec![f32::NAN; APRON] }
    }
}

/// Apron slot for a texel just outside the cell.
#[inline(always)]
pub fn apron_idx(side: usize, half: usize, i: usize) -> usize {
    (side * 2 + half) * TEX + i
}

/// Imagery bands, 8-bit, 0 = no data: NAIP red, green, blue and near-infrared (~650 / ~550 / ~450 / ~850 nm). Never baked finer than depth 14 (IMG_MAX_DEPTH).
#[derive(Clone)]
pub struct ImgCell {
    pub red: Vec<u8>,
    pub green: Vec<u8>,
    pub blue: Vec<u8>,
    pub nir: Vec<u8>,
}

impl ImgCell {
    pub fn new() -> ImgCell {
        ImgCell { red: vec![0; TRI], green: vec![0; TRI], blue: vec![0; TRI], nir: vec![0; TRI] }
    }
    pub fn is_empty(&self) -> bool {
        self.bands().iter().all(|b| b.iter().all(|&v| v == 0))
    }
    fn bands(&self) -> [&Vec<u8>; 4] {
        [&self.red, &self.green, &self.blue, &self.nir]
    }
    fn bands_mut(&mut self) -> [&mut Vec<u8>; 4] {
        [&mut self.red, &mut self.green, &mut self.blue, &mut self.nir]
    }
}

impl Default for ImgCell {
    fn default() -> Self {
        Self::new()
    }
}

/// Imagery is never baked finer than this (1.2 m texels): NAIP's 60 cm is a downsample into it, never an upsample.
pub const IMG_MAX_DEPTH: u8 = 14;

/// Everything a cell can carry. Vector layers at a vector depth are always all present (empty planes compress to nothing), so a reader finding one of them knows the cell's vector truth is complete.
#[derive(Clone, Default)]
pub struct Cell {
    pub dem: Option<DemCell>,
    pub line: Option<ClassCell>,
    pub land: Option<ClassCell>,
    pub water: Option<CovCell>,
    pub img: Option<ImgCell>,
}

// ==================== LINE LAYER ====================

/// Stamp every feature ONCE, at its physical width, into base-depth cells; the pyramid does the rest. Lines are measure-zero features, so a parent texel's coverage is the SUM of its four children (saturating): a road stays a texel wide at every depth instead of averaging away, and dense networks saturate into a glow. Major class wins per texel.
pub fn bake_lines(feats: &[Road], base_depth: u8, min_depth: u8) -> HashMap<CellKey, ClassCell> {
    assert!(base_depth as u32 + TEX_BITS as u32 <= 30);
    let mut cells: HashMap<CellKey, ClassCell> = HashMap::new();
    let mut cross_diamond = 0usize;
    let tm = texel_m(base_depth);
    for road in feats {
        let class = road.class_id();
        let r = (road.width_m() as f64 / 2.0 / tm).max(0.5);
        let cov_max = road.cov_max();
        let attrs = (road.mag_byte(), road.uses);
        let mut prev: Option<(u8, f64, f64)> = None;
        for &(lat, lon) in &road.pts {
            let c = Coord::from_lat_lon(lat as f64, lon as f64);
            let (d, gu, gv) = texel_of(c, base_depth);
            if let Some((pd, pu, pv)) = prev {
                if pd == d {
                    stamp_segment(&mut cells, base_depth, d, (pu, pv), (gu, gv), r, class, cov_max, attrs);
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

/// Stamp a segment as a band of radius `r` texels (ground metric): walk the centreline in half-texel steps and cover every triangle whose centroid is within the band, coverage feathered over the last texel. Coverage combines by max; class follows the strongest coverage.
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
    attrs: (u8, u8),
) {
    let extent = ((1u64 << depth) * TEX as u64) as f64;
    let len = uv_dist2(b.0 - a.0, b.1 - a.1).sqrt();
    let steps = (len * 2.0).ceil().max(1.0) as usize;
    // Bounding box of a ground disc of radius r in UV units (the 60° basis stretches it by up to 2/sqrt(3)).
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
                    let cell = cells.entry(key).or_insert_with(ClassCell::new_line);
                    let i = tri_idx(tx % TEX, ty % TEX, half);
                    if cov > cell.cov[i] || (cov == cell.cov[i] && class < cell.class[i]) {
                        cell.cov[i] = cov;
                        cell.class[i] = class;
                        cell.mag[i] = attrs.0;
                    }
                    cell.uses[i] |= attrs.1;
                }
            }
        }
    }
}

// ==================== AREA LAYERS ====================

/// Rasterize land cover and water polygons at base depth (even-odd fill, sampled at triangle centroids), then pyramid both to `min_depth`.
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

/// Even-odd scanline fill of rings given in global texel coordinates, visiting every triangle whose centroid is inside.
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
    /// Lines: coverage is the SUM of the children (a line has no area, so its amount adds); the most major class present wins.
    Major,
    /// Land cover: coverage is the MEAN of the children (an area fraction); the class with the most coverage wins.
    Dominant,
}

/// Parent texel from its four triangle children (missing children are empty): coverage summed or averaged, class per `merge`.
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
        let with_attrs = matches!(merge, ClassMerge::Major);
        let built: Vec<(CellKey, ClassCell)> = parents
            .par_iter()
            .map(|&p| {
                let kids: [Option<&ClassCell>; 4] = std::array::from_fn(|q| cells.get(&p.child(q as u64)));
                let mut cell = if with_attrs { ClassCell::new_line() } else { ClassCell::new() };
                for ty in 0..TEX {
                    for tx in 0..TEX {
                        for half in 0..2 {
                            let mut covsum = 0u32;
                            let mut best = 0u8;
                            let mut best_cov = 0u32;
                            let mut per_class = [0u32; 32];
                            let mut mag = 0u8;
                            let mut uses = 0u8;
                            for (q, i) in child_cell_texels(tx, ty, half) {
                                let Some(child) = kids[q as usize] else { continue };
                                let cv = child.cov[i] as u32;
                                covsum += cv;
                                if with_attrs {
                                    mag = mag.max(child.mag_at(i));
                                    uses |= child.uses_at(i);
                                }
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
                            cell.cov[i] = match merge {
                                ClassMerge::Major => covsum.min(255) as u8,
                                ClassMerge::Dominant => (covsum / 4) as u8,
                            };
                            cell.class[i] = best;
                            if with_attrs {
                                cell.mag[i] = mag;
                                cell.uses[i] = uses;
                            }
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

/// Bake dem cells covering `keys` from the source store: each triangle texel's elevation sampled at its centroid.
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
                        if let Some(e) = dem.elevation(lat, lon) {
                            cell.elev[tri_idx(tx, ty, half)] = e;
                        }
                    }
                }
            }
            (key, cell)
        })
        .filter(|(_, c)| c.elev.iter().any(|e| !e.is_nan()))
        .collect()
}

/// Every depth from `base.depth - 1` down to `min_depth`, each texel the mean of its four triangle children (no-data ignored). Building the pyramid from the base set is what guarantees every ancestor of a baked cell exists.
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
                            let (mut e, mut n) = (0.0f32, 0u32);
                            for (q, i) in child_cell_texels(tx, ty, half) {
                                let Some(child) = kids[q as usize] else { continue };
                                if !child.elev[i].is_nan() {
                                    e += child.elev[i];
                                    n += 1;
                                }
                            }
                            if n > 0 {
                                cell.elev[tri_idx(tx, ty, half)] = e / n as f32;
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

// ==================== IMAGERY LAYER ====================

/// Sample NAIP (red, nir) and lidar intensity at every triangle centroid of `keys` (depth <= IMG_MAX_DEPTH); cells with nothing are dropped.
/// Sample the imagery at every triangle centroid: the four NAIP bands, each the mean of four samples spread a third of a texel around the centroid, so 60 cm pixels are box-filtered into a 1.2 m texel instead of point-picked. 0 stays no data.
pub fn bake_img(naip: Option<&mahere_dem::ImgStore>, keys: &[CellKey]) -> Vec<(CellKey, ImgCell)> {
    keys.par_iter()
        .map(|&key| {
            let (u0, v0, size) = key.uv_rect();
            let d = key.diamond();
            let step = size / TEX as f64;
            let mut cell = ImgCell::new();
            let Some(n) = naip else { return (key, cell) };
            for ty in 0..TEX {
                for tx in 0..TEX {
                    for half in 0..2 {
                        let (cx, cy) = tri_centroid(tx, ty, half);
                        let i = tri_idx(tx, ty, half);
                        let (mut acc, mut hits) = ([0u32; 4], 0u32);
                        for (ou, ov) in [(-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)] {
                            let (lat, lon) = uv_to_lat_lon(d, u0 + (cx + ou) * step, v0 + (cy + ov) * step);
                            if let Some(px) = n.sample(lat, lon) {
                                for b in 0..4 {
                                    acc[b] += px[b] as u32;
                                }
                                hits += 1;
                            }
                        }
                        if hits > 0 {
                            for (b, plane) in cell.bands_mut().into_iter().enumerate() {
                                plane[i] = ((acc[b] + hits / 2) / hits).clamp(1, 255) as u8;
                            }
                        }
                    }
                }
            }
            (key, cell)
        })
        .filter(|(_, c)| !c.is_empty())
        .collect()
}

/// Parent texel = mean of the lit children per band (0 = no data, ignored).
pub fn img_pyramid(base: &[(CellKey, ImgCell)], min_depth: u8) -> Vec<(CellKey, ImgCell)> {
    let mut out: Vec<(CellKey, ImgCell)> = Vec::new();
    let mut last: Option<usize> = None;
    loop {
        let cur: &[(CellKey, ImgCell)] = match last {
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
        let by_key: HashMap<CellKey, &ImgCell> = cur.iter().map(|(k, c)| (*k, c)).collect();
        let next: Vec<(CellKey, ImgCell)> = parents
            .par_iter()
            .map(|&p| {
                let kids: [Option<&&ImgCell>; 4] = std::array::from_fn(|q| by_key.get(&p.child(q as u64)));
                let mut cell = ImgCell::new();
                for ty in 0..TEX {
                    for tx in 0..TEX {
                        for half in 0..2 {
                            let mut acc = [0u32; 4];
                            let mut n = [0u32; 4];
                            for (q, i) in child_cell_texels(tx, ty, half) {
                                let Some(child) = kids[q as usize] else { continue };
                                for (b, plane) in child.bands().into_iter().enumerate() {
                                    if plane[i] != 0 {
                                        acc[b] += plane[i] as u32;
                                        n[b] += 1;
                                    }
                                }
                            }
                            let i = tri_idx(tx, ty, half);
                            for (b, plane) in cell.bands_mut().into_iter().enumerate() {
                                if n[b] > 0 {
                                    plane[i] = (acc[b] / n[b]) as u8;
                                }
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

/// All cells at `depth` that intersect a lat/lon bbox. Scanned at half-cell resolution, never coarser than an eighth of the box — a slanted rhombus cell can cut through a box without containing any corner of it.
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

/// Copy each cell's neighbours' edge texels into its apron (same depth, same diamond; a missing neighbour leaves NaN and the loader uses a one-sided gradient there).
pub fn fill_aprons(mut dem: Vec<(CellKey, DemCell)>) -> Vec<(CellKey, DemCell)> {
    let edges: HashMap<CellKey, [Vec<f32>; 4]> = dem
        .iter()
        .map(|(k, c)| {
            // What THIS cell offers its neighbours: its own edge texels, indexed so the neighbour can copy them straight in.
            let mut west = vec![f32::NAN; 2 * TEX]; // this cell's tx=0 column -> east neighbour's... see below
            let mut east = vec![f32::NAN; 2 * TEX];
            let mut south = vec![f32::NAN; 2 * TEX];
            let mut north = vec![f32::NAN; 2 * TEX];
            for i in 0..TEX {
                for half in 0..2 {
                    west[half * TEX + i] = c.elev[tri_idx(0, i, half)];
                    east[half * TEX + i] = c.elev[tri_idx(TEX - 1, i, half)];
                    south[half * TEX + i] = c.elev[tri_idx(i, 0, half)];
                    north[half * TEX + i] = c.elev[tri_idx(i, TEX - 1, half)];
                }
            }
            (*k, [west, east, south, north])
        })
        .collect();
    let n = |k: CellKey, du: i64, dv: i64| -> Option<CellKey> {
        let (cu, cv) = k.grid();
        let lim = 1i64 << k.depth;
        let (nu, nv) = (cu as i64 + du, cv as i64 + dv);
        (nu >= 0 && nv >= 0 && nu < lim && nv < lim).then(|| CellKey::from_grid(k.diamond(), k.depth, nu as u64, nv as u64))
    };
    for (k, c) in dem.iter_mut() {
        // My west apron (tx = -1) is the west neighbour's east column, etc.
        let pairs = [(n(*k, -1, 0), 0usize, 1usize), (n(*k, 1, 0), 1, 0), (n(*k, 0, -1), 2, 3), (n(*k, 0, 1), 3, 2)];
        for (nk, side, their_edge) in pairs {
            let Some(src) = nk.and_then(|nk| edges.get(&nk)) else { continue };
            for half in 0..2 {
                for i in 0..TEX {
                    c.apron[apron_idx(side, half, i)] = src[their_edge][half * TEX + i];
                }
            }
        }
    }
    dem
}

/// Union the layers into cells. Wherever any vector layer exists at a depth, all three are present (empty planes for the missing ones) so readers never have to climb for one layer but not another.
pub fn assemble(
    dem: Vec<(CellKey, DemCell)>,
    line: HashMap<CellKey, ClassCell>,
    land: HashMap<CellKey, ClassCell>,
    water: HashMap<CellKey, CovCell>,
    img: Vec<(CellKey, ImgCell)>,
) -> HashMap<CellKey, Cell> {
    let mut cells: HashMap<CellKey, Cell> = HashMap::new();
    let dem = fill_aprons(dem);
    for (k, d) in dem {
        cells.entry(k).or_default().dem = Some(d);
    }
    for (k, i) in img {
        cells.entry(k).or_default().img = Some(i);
    }
    let mut vec_keys: Vec<CellKey> = line.keys().chain(land.keys()).chain(water.keys()).copied().collect();
    vec_keys.sort();
    vec_keys.dedup();
    let mut line = line;
    let mut land = land;
    let mut water = water;
    for k in vec_keys {
        let c = cells.entry(k).or_default();
        c.line = Some(line.remove(&k).unwrap_or_else(ClassCell::new_line));
        c.land = Some(land.remove(&k).unwrap_or_default());
        c.water = Some(water.remove(&k).unwrap_or_default());
    }
    cells
}

// ==================== VSF I/O ====================

/// Decoded, quantized planes in memory order — what the loader hands the renderer and what a merge-on-write reads back.
#[derive(Clone, Default)]
pub struct CellPlanes {
    pub dem: Option<DemPlanes>,
    pub line: Option<ClassCell>,
    pub land: Option<ClassCell>,
    pub water: Option<CovCell>,
    pub img: Option<ImgCell>,
}

/// Decoded dem: elevation (NaN = no data) and the apron, memory order.
#[derive(Clone)]
pub struct DemPlanes {
    pub elev: Vec<f32>,
    pub apron: Vec<f32>,
}

impl DemCell {
    pub fn planes(&self) -> DemPlanes {
        DemPlanes { elev: self.elev.clone(), apron: self.apron.clone() }
    }
}

/// On-disk dem coding: per-cell `base` + `step` (adaptive, >= 5 cm, so the cell's range fits 16 bits), the quantised elevations through the pyramid codec (`pyr`, disk order, holes filled), a bit mask of the texels that had data when any were missing, and the apron as u16.
const MIN_STEP: f32 = 0.05;

/// How much a bake may throw away. Zero everywhere is lossless at the base quantisation (5 cm, 1 level of 255).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Loss {
    /// Finest-level quantiser for elevation diffs, metres.
    pub dem_m: f32,
    /// Finest-level quantiser for imagery diffs, in 8-bit levels.
    pub img: u32,
}

impl Loss {
    pub const LOSSLESS: Loss = Loss { dem_m: 0.0, img: 0 };
}

/// Quantise a plane of f32 (NaN = no data) to integers at `base` + k·`step`, memory order, holes filled; returns the plane and, when any texel was missing, its mask.
fn quantise_plane(mem: &[f32], base: f32, step: f32) -> (Vec<i32>, Option<Vec<bool>>) {
    let q: Vec<i32> = mem.iter().map(|&e| if e.is_nan() { 0 } else { ((e - base) / step).round() as i32 }).collect();
    let valid: Vec<bool> = mem.iter().map(|e| !e.is_nan()).collect();
    if valid.iter().all(|&v| v) {
        return (q, None);
    }
    let mut disk = mem_to_disk(&q);
    pyr::fill_holes(&mut disk, &mem_to_disk(&valid));
    (disk_to_mem(&disk), Some(valid))
}

fn dem_section(d: &DemPlanes, loss: &Loss) -> vsf::VsfSection {
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for &e in d.elev.iter().chain(d.apron.iter()) {
        if !e.is_nan() {
            lo = lo.min(e);
            hi = hi.max(e);
        }
    }
    if lo == f32::MAX {
        lo = 0.0;
        hi = 0.0;
    }
    let step = ((hi - lo) / 32000.0).max(MIN_STEP);
    let steps = pyr::Steps::tapered((loss.dem_m / step).round() as u32);
    let (plane, mask) = quantise_plane(&d.elev, lo, step);
    let apron: Vec<u16> = d.apron.iter().map(|&e| if e.is_nan() { u16::MAX } else { (((e - lo) / step).round() as i32).clamp(0, 65534) as u16 }).collect();
    let mut s = vsf::VsfSection::new("dem");
    s.add_field("base", VsfType::f5(lo));
    s.add_field("step", VsfType::f5(step));
    s.add_field("loss", pyr::steps_vsf(&steps));
    s.add_field_multi("elev", pyr::encode(&plane, &steps));
    if let Some(m) = mask {
        s.add_field("mask", pyr::mask_vsf(&m));
    }
    s.add_field("apron", VsfType::t_u4(Tensor::new(vec![4, 2, TEX], apron)));
    s
}

fn decode_dem(fields: &HashMap<String, Vec<VsfType>>) -> Option<DemPlanes> {
    let base = fields.get("base").and_then(|v| v.first()).and_then(scalar_f32)?;
    let step = fields.get("step").and_then(|v| v.first()).and_then(scalar_f32)?;
    if let Some(residual) = fields.get("residual").and_then(|v| v.first()).and_then(raw_i16) {
        return decode_dem_legacy(fields, base, step, residual);
    }
    let steps = pyr::steps_from_vsf(fields.get("loss").and_then(|v| v.first()));
    let plane = pyr::decode(fields.get("elev")?, &steps)?;
    let mask = match fields.get("mask").and_then(|v| v.first()) {
        Some(m) => Some(pyr::mask_from_vsf(m)?),
        None => None,
    };
    let apron = fields.get("apron").and_then(|v| v.first()).and_then(raw_u16)?;
    if apron.len() != APRON {
        return None;
    }
    let mut elev: Vec<f32> = plane.iter().map(|&v| base + v as f32 * step).collect();
    if let Some(m) = mask {
        for i in 0..TRI {
            if !m[i] {
                elev[i] = f32::NAN;
            }
        }
    }
    let apron = apron.iter().map(|&v| if v == u16::MAX { f32::NAN } else { base + v as f32 * step }).collect();
    Some(DemPlanes { elev, apron })
}

/// The epoch-3 dem coding, read-only so old bakes can be transcoded: LOCO-I MED prediction over the half-plane in row-major order, i16 residuals, a u8 `valid` plane when any texel was missing.
fn decode_dem_legacy(fields: &HashMap<String, Vec<VsfType>>, base: f32, step: f32, residual: Vec<i16>) -> Option<DemPlanes> {
    fn med(a: i32, b: i32, c: i32) -> i32 {
        if c >= a.max(b) { a.min(b) } else if c <= a.min(b) { a.max(b) } else { a + b - c }
    }
    let valid: Option<Vec<u8>> = match fields.get("valid").and_then(|v| v.first()) {
        Some(VsfType::t_u3(t)) => Some(t.data.clone()),
        _ => None,
    };
    let apron = fields.get("apron").and_then(|v| v.first()).and_then(raw_u16)?;
    if residual.len() != TRI || apron.len() != APRON || valid.as_ref().is_some_and(|v| v.len() != TRI) {
        return None;
    }
    let mut q = vec![0i32; TRI];
    let mut elev = vec![f32::NAN; TRI];
    for half in 0..2 {
        for ty in 0..TEX {
            for tx in 0..TEX {
                let i = tri_idx(tx, ty, half);
                let pred = match (tx, ty) {
                    (0, 0) => 0,
                    (0, _) => q[tri_idx(0, ty - 1, half)],
                    (_, 0) => q[tri_idx(tx - 1, 0, half)],
                    _ => med(q[tri_idx(tx - 1, ty, half)], q[tri_idx(tx, ty - 1, half)], q[tri_idx(tx - 1, ty - 1, half)]),
                };
                let v = pred + residual[i] as i32;
                q[i] = v;
                if valid.as_ref().is_none_or(|m| m[i] != 0) {
                    elev[i] = base + v as f32 * step;
                }
            }
        }
    }
    let apron = apron.iter().map(|&v| if v == u16::MAX { f32::NAN } else { base + v as f32 * step }).collect();
    Some(DemPlanes { elev, apron })
}

/// Imagery bands through the pyramid codec: 0 is no data (masked, filled).
const IMG_BANDS: [&str; 4] = ["red", "green", "blue", "nir"];

fn img_section(im: &ImgCell, loss: &Loss) -> vsf::VsfSection {
    let steps = pyr::Steps::tapered(loss.img);
    let mut s = vsf::VsfSection::new("img");
    s.add_field("loss", pyr::steps_vsf(&steps));
    for (name, band) in IMG_BANDS.iter().zip(im.bands()) {
        let masked = true;
        let mem: Vec<f32> = band.iter().map(|&v| if masked && v == 0 { f32::NAN } else { v as f32 }).collect();
        let (plane, mask) = quantise_plane(&mem, 0.0, 1.0);
        s.add_field_multi(*name, pyr::encode(&plane, &steps));
        if let Some(m) = mask {
            s.add_field(format!("{name}mask"), pyr::mask_vsf(&m));
        }
    }
    s
}

fn decode_img(fields: &HashMap<String, Vec<VsfType>>) -> Option<ImgCell> {
    let steps = pyr::steps_from_vsf(fields.get("loss").and_then(|v| v.first()));
    let mut im = ImgCell::new();
    let mut any = false;
    for (name, out) in IMG_BANDS.iter().zip(im.bands_mut()) {
        let Some(values) = fields.get(*name) else { continue };
        let plane = pyr::decode(values, &steps)?;
        let mask = match fields.get(&format!("{name}mask")).and_then(|v| v.first()) {
            Some(m) => Some(pyr::mask_from_vsf(m)?),
            None => None,
        };
        let masked = true;
        *out = plane
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if mask.as_ref().is_some_and(|m| !m[i]) {
                    0
                } else if masked {
                    v.clamp(1, 255) as u8
                } else {
                    v.clamp(0, 255) as u8
                }
            })
            .collect();
        any = true;
    }
    any.then_some(im)
}

impl DemPlanes {
    /// Pack for the renderer: `[elev_q u16 | nx i16 | ny i16 | nz i16]` per texel (0.25 m steps from -500 m, 0xFFFF no data). Normals come from central differences over the half-plane (apron at the edges, one- sided next to no-data) mapped through the cell's UV→east/north Jacobian — snorm16, not u8, because u8 bands on gentle slopes.
    pub fn pack_texels(&self, key: CellKey) -> Vec<u64> {
        // Jacobian: metres east/north per texel step in u and in v.
        let (u0, v0, size) = key.uv_rect();
        let d = key.diamond();
        let step = size / TEX as f64;
        let (uc, vc) = (u0 + size * 0.5, v0 + size * 0.5);
        let (lat0, lon0) = uv_to_lat_lon(d, uc, vc);
        let (lat_u, lon_u) = uv_to_lat_lon(d, uc + step, vc);
        let (lat_v, lon_v) = uv_to_lat_lon(d, uc, vc + step);
        let m_lon = 111_320.0 * lat0.to_radians().cos();
        let (eu, nu) = ((lon_u - lon0) * m_lon, (lat_u - lat0) * 111_320.0);
        let (ev, nv) = ((lon_v - lon0) * m_lon, (lat_v - lat0) * 111_320.0);
        // dE/du = ge*eu + gn*nu ; dE/dv = ge*ev + gn*nv  -> solve for (ge, gn).
        let det = eu * nv - ev * nu;
        let inv = if det.abs() > 1e-9 { 1.0 / det } else { 0.0 };
        let at = |tx: i64, ty: i64, half: usize| -> f32 {
            if tx < 0 {
                self.apron[apron_idx(0, half, ty.clamp(0, TEX as i64 - 1) as usize)]
            } else if tx >= TEX as i64 {
                self.apron[apron_idx(1, half, ty.clamp(0, TEX as i64 - 1) as usize)]
            } else if ty < 0 {
                self.apron[apron_idx(2, half, tx as usize)]
            } else if ty >= TEX as i64 {
                self.apron[apron_idx(3, half, tx as usize)]
            } else {
                self.elev[tri_idx(tx as usize, ty as usize, half)]
            }
        };
        let diff = |m: f32, p: f32, c: f32| -> f32 {
            match (m.is_nan(), p.is_nan()) {
                (false, false) => (p - m) * 0.5,
                (true, false) => p - c,
                (false, true) => c - m,
                (true, true) => 0.0,
            }
        };
        let mut out = vec![0u64; TRI];
        for ty in 0..TEX {
            for tx in 0..TEX {
                for half in 0..2 {
                    let i = tri_idx(tx, ty, half);
                    let e = self.elev[i];
                    if e.is_nan() {
                        out[i] = ELEV_NODATA as u64 | (32767u64 << 48);
                        continue;
                    }
                    let (tx_, ty_) = (tx as i64, ty as i64);
                    let du = diff(at(tx_ - 1, ty_, half), at(tx_ + 1, ty_, half), e) as f64;
                    let dv = diff(at(tx_, ty_ - 1, half), at(tx_, ty_ + 1, half), e) as f64;
                    let ge = (du * nv - dv * nu) * inv;
                    let gn = (eu * dv - ev * du) * inv;
                    let s = 1.0 / (1.0 + ge * ge + gn * gn).sqrt();
                    let nx = (-ge * s * 32767.0) as i16;
                    let ny = (-gn * s * 32767.0) as i16;
                    let nz = (s * 32767.0) as i16;
                    out[i] = quantize_elev(e) as u64
                        | ((nx as u16 as u64) << 16)
                        | ((ny as u16 as u64) << 32)
                        | ((nz as u16 as u64) << 48);
                }
            }
        }
        out
    }
}

impl Cell {
    pub fn quantize(&self) -> CellPlanes {
        CellPlanes {
            dem: self.dem.as_ref().map(|d| d.planes()),
            line: self.line.clone(),
            land: self.land.clone(),
            water: self.water.clone(),
            img: self.img.clone(),
        }
    }
}

impl CellPlanes {
    /// Overlay `self` (a new bake) onto `old`. A bake's footprint is where it has elevation: inside it the new vector planes win, outside the old ones stay; without a dem the new planes replace wholesale.
    pub fn merge_over(self, old: CellPlanes) -> CellPlanes {
        let mut out = self;
        let CellPlanes { dem: old_dem, line: old_line, land: old_land, water: old_water, img: old_img } = old;
        let (mut old_line, mut old_land, mut old_water) = (old_line, old_land, old_water);
        match (&mut out.img, old_img) {
            (Some(n), Some(o)) => {
                for (np, op) in n.bands_mut().into_iter().zip(o.bands()) {
                    for i in 0..TRI {
                        if np[i] == 0 {
                            np[i] = op[i];
                        }
                    }
                }
            }
            (n @ None, Some(o)) => *n = Some(o),
            _ => {}
        }
        // The new bake's footprint, if it has one.
        let mask: Option<Vec<bool>> = out.dem.as_ref().map(|d| d.elev.iter().map(|e| !e.is_nan()).collect());
        match (&mut out.dem, old_dem) {
            (Some(new), Some(old_dem)) => {
                let mask = mask.as_ref().unwrap();
                for i in 0..TRI {
                    if !mask[i] {
                        new.elev[i] = old_dem.elev[i];
                    }
                }
                for i in 0..APRON {
                    if new.apron[i].is_nan() {
                        new.apron[i] = old_dem.apron[i];
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
                        let attrs = n.mag.len() == TRI && o.mag.len() == TRI;
                        for i in 0..TRI {
                            if !mask[i] {
                                n.class[i] = o.class[i];
                                n.cov[i] = o.cov[i];
                                if attrs {
                                    n.mag[i] = o.mag[i];
                                    n.uses[i] = o.uses[i];
                                }
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

    pub fn encode(&self, loss: &Loss) -> Result<Vec<u8>, String> {
        let mut b = VsfBuilder::new();
        if let Some(d) = &self.dem {
            b = b.add_section_direct(dem_section(d, loss));
        }
        for (name, planes) in [("line", &self.line), ("land", &self.land)] {
            if let Some(p) = planes {
                let mut fields = vec![
                    ("class".to_string(), VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&p.class)))),
                    ("cov".to_string(), VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&p.cov)))),
                ];
                if p.mag.len() == TRI {
                    fields.push(("mag".to_string(), VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&p.mag)))));
                }
                if p.uses.len() == TRI {
                    fields.push(("use".to_string(), VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&p.uses)))));
                }
                b = b.add_section(name, fields);
            }
        }
        if let Some(w) = &self.water {
            b = b.add_section(
                "water",
                vec![("cov".to_string(), VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], mem_to_disk(&w.cov))))],
            );
        }
        if let Some(im) = &self.img {
            b = b.add_section_direct(img_section(im, loss));
        }
        b.build().map_err(|e| format!("cell build: {e:?}"))
    }
}

/// Write a cell, merged over whatever is already on disk under that key — so regional bakes compose instead of clobbering each other.
pub fn write_cell(out: &Path, key: CellKey, cell: &Cell, loss: &Loss) -> Result<(), String> {
    let path = out.join(key.path());
    let mut planes = cell.quantize();
    if let Ok(existing) = std::fs::read(&path) {
        if let Ok(old) = decode_cell(&existing) {
            planes = planes.merge_over(old);
        }
    }
    let bytes = planes.encode(loss)?;
    write_file(&path, &bytes)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    // Whole-file zstd: VSF internals untouched, the bucket keeps the .vsf.zst names, and empty planes shrink to nothing.
    let z = zstd::encode_all(bytes, 3).map_err(|e| e.to_string())?;
    std::fs::write(path, z).map_err(|e| format!("{}: {e}", path.display()))
}

/// Decode a cell from raw file bytes (zstd or plain VSF) — the loader thread's entry point; no filesystem coupling. Width-agnostic reads.
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
        let fields: HashMap<String, Vec<VsfType>> = s.fields.into_iter().map(|f| (f.name, f.values)).collect();
        let first = |name: &str| fields.get(name).and_then(|v| v.first());
        match s.name.as_str() {
            "dem" => out.dem = decode_dem(&fields),
            "line" | "land" => {
                if let (Some(class), Some(cov)) = (first("class").and_then(plane_u8_mem), first("cov").and_then(plane_u8_mem)) {
                    let mag = first("mag").and_then(plane_u8_mem).unwrap_or_default();
                    let uses = first("use").and_then(plane_u8_mem).unwrap_or_default();
                    let planes = Some(ClassCell { class, cov, mag, uses });
                    if s.name == "line" {
                        out.line = planes;
                    } else {
                        out.land = planes;
                    }
                }
            }
            "water" => {
                if let Some(cov) = first("cov").and_then(plane_u8_mem) {
                    out.water = Some(CovCell { cov });
                }
            }
            "img" => {
                out.img = decode_img(&fields)
            }
            _ => {}
        }
    }
    Ok(out)
}

fn scalar_f32(v: &VsfType) -> Option<f32> {
    match v {
        VsfType::f5(x) => Some(*x),
        VsfType::f6(x) => Some(*x as f32),
        _ => None,
    }
}

fn raw_i16(v: &VsfType) -> Option<Vec<i16>> {
    match v {
        VsfType::t_i4(t) => Some(t.data.clone()),
        VsfType::v_i4(t) => Some(t.data.clone()),
        _ => None,
    }
}

fn raw_u16(v: &VsfType) -> Option<Vec<u16>> {
    match v {
        VsfType::t_u4(t) => Some(t.data.clone()),
        VsfType::v_u4(t) => Some(t.data.clone()),
        VsfType::t_u3(t) => Some(t.data.iter().map(|&x| x as u16).collect()),
        VsfType::v_u3(t) => Some(t.data.iter().map(|&x| x as u16).collect()),
        _ => None,
    }
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

    /// Children partition the parent: every triangle at the fine grid is the child of exactly one triangle at the coarse grid.
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
            uses: 0,
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
        let lit = |d: u8| -> usize {
            cells.iter().filter(|(k, _)| k.depth == d).map(|(_, c)| c.cov.iter().filter(|&&x| x > 0).count()).sum()
        };
        let peak = |d: u8| -> u8 {
            cells.iter().filter(|(k, _)| k.depth == d).map(|(_, c)| *c.cov.iter().max().unwrap()).max().unwrap()
        };
        assert!(lit(12) > 0, "base stamping produced nothing");
        // Summing children: the path stays fully covered along its length at every depth (it shrinks in texel count, never in intensity).
        assert_eq!(peak(10), 255);
        let r = lit(12) as f64 / lit(10) as f64;
        assert!((2.0..=8.0).contains(&r), "line texel count should shrink several-fold over 2 levels, got {r}");
    }

    /// A motorway covers more ground than a path along the same line, in proportion to its width.
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

    /// A square lake fills its interior (and only its interior), and the pyramid conserves its area.
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
        // Expected count: the lake's area over the LOCAL triangle area (the gnomonic face mapping varies texel size across a face, so measure one texel's parallelogram on the ground at the lake's centre).
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
        // A gentle plane rising along u (0.5 m per texel) with a ripple.
        for ty in 0..TEX / 2 {
            for tx in 0..TEX {
                for half in 0..2 {
                    let i = tri_idx(tx, ty, half);
                    dem.elev[i] = 1000.0 + tx as f32 * 0.5 + ((tx * 7 + ty * 3) % 11) as f32 * 0.02;
                }
            }
        }
        dem.apron[apron_idx(0, 0, 3)] = 999.5;
        let mut line = ClassCell::new_line();
        line.class[5] = 3;
        line.cov[5] = 200;
        line.mag[5] = 77;
        line.uses[5] = 0b101;
        let cell = Cell { dem: Some(dem.clone()), line: Some(line), land: Some(ClassCell::new()), water: None, img: None };
        let bytes = cell.quantize().encode(&Loss::LOSSLESS).unwrap();
        let back = decode_cell(&bytes).unwrap();
        let d = back.dem.as_ref().unwrap();
        // Exact to the adaptive step (range ~80 m -> 5 cm floor).
        for i in 0..TRI / 2 {
            assert!((d.elev[i] - dem.elev[i]).abs() <= 0.03, "texel {i}: {} vs {}", d.elev[i], dem.elev[i]);
        }
        assert!(d.elev[TRI - 1].is_nan());
        assert!((d.apron[apron_idx(0, 0, 3)] - 999.5).abs() <= 0.03);
        assert!(d.apron[apron_idx(1, 1, 7)].is_nan());
        assert_eq!(back.line.as_ref().unwrap().cov[5], 200);
        assert_eq!(back.line.as_ref().unwrap().mag[5], 77);
        assert_eq!(back.line.as_ref().unwrap().uses[5], 0b101);
        assert!(back.water.is_none());

        // Normals: a plane tilted along +u lights as a slope, flat ground as +z.
        let key = CellKey::containing(Coord::from_lat_lon(46.2, -122.19), 11);
        let packed = d.pack_texels(key);
        let nz = (packed[tri_idx(10, 10, 0)] >> 48) as u16 as i16;
        assert!(nz > 30000, "nz {nz}");
        let nx = (packed[tri_idx(10, 10, 0)] >> 16) as u16 as i16;
        assert!(nx != 0, "a slope along u must tilt the normal");
        assert_eq!((packed[TRI - 1] & 0xFFFF) as u16, ELEV_NODATA);

        // Merge: a second bake covering the OTHER half keeps our half.
        let mut dem2 = DemCell::new();
        for i in TRI / 2..TRI {
            dem2.elev[i] = 50.0;
        }
        let mut line2 = ClassCell::new_line();
        line2.cov[5] = 1; // outside dem2's footprint: must NOT win
        line2.cov[TRI - 1] = 9;
        let cell2 = Cell { dem: Some(dem2), line: Some(line2), land: None, water: None, img: None };
        let merged = cell2.quantize().merge_over(back);
        let d = merged.dem.as_ref().unwrap();
        assert!((d.elev[3] - dem.elev[3]).abs() <= 0.03);
        assert!((d.elev[TRI - 1] - 50.0).abs() < 1e-3);
        let l = merged.line.as_ref().unwrap();
        assert_eq!(l.cov[5], 200);
        assert_eq!(l.cov[TRI - 1], 9);
        assert!(merged.land.is_some(), "a layer absent from the new bake survives from the old");
    }

    /// Aprons: a cell's west apron is its west neighbour's east column.
    #[test]
    fn aprons_come_from_neighbours() {
        let k = CellKey::containing(Coord::from_lat_lon(46.2, -122.19), 10);
        let (cu, cv) = k.grid();
        let west = CellKey::from_grid(k.diamond(), 10, cu - 1, cv);
        let mut a = DemCell::new();
        let mut b = DemCell::new();
        for i in 0..TRI {
            a.elev[i] = 100.0;
            b.elev[i] = 200.0;
        }
        b.elev[tri_idx(TEX - 1, 7, 1)] = 250.0;
        let out = fill_aprons(vec![(k, a), (west, b)]);
        let (_, a) = out.iter().find(|(kk, _)| *kk == k).unwrap();
        assert_eq!(a.apron[apron_idx(0, 1, 7)], 250.0);
        assert_eq!(a.apron[apron_idx(0, 0, 7)], 200.0);
        assert!(a.apron[apron_idx(1, 0, 7)].is_nan(), "no east neighbour");
    }
}
