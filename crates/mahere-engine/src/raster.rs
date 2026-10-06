//! The hot loop: the #pagetable compositor. A frame is fetches — 32 px block grid with exact corners, fixed-point UV stepping inside, per-BLOCK cell resolution through the page table, nearest-fetch compose of the layers through style LUTs. No vectors, no re-rasterization, no allocation; rayon over row bands.
//!
//! Two depth selections: the dem's and the vector layers' (line, land, water share a base and live in the same cell). Each block resolves one dem ref and one vector ref; a pixel composes dem → land tint → shade → water → line, each gated by the client's layer mask.

use mahere_coord::Coord;
use mahere_tiles::{CellKey, ClassCell, CovCell, ELEV_NODATA, ImgCell};
use rayon::prelude::*;
use rustc_hash::FxHashSet;

use crate::Camera;
use crate::residency::{DemPacked, Entry, Pool};

pub const BLOCK: usize = 32;
/// Deepest depths any bake produces; regions baked shallower simply fall back to their parents through the page table.
pub const VEC_BASE_DEPTH: u8 = 14;
pub const DEM_BASE_DEPTH: u8 = 14;
pub const MIN_DEPTH: u8 = 6;

/// texel/pixel ratio constant: diamond edge 7054 km, 111320 m/deg, 256 texels/cell. ratio r(d) = ppd * K / 2^d; pick d so r ∈ (0.5, 1].
const K: f64 = 7_054_000.0 / (111_320.0 * 256.0);

pub fn select_depth(ppd: f64, last: u8, base: u8) -> u8 {
    let ideal = (ppd * K).log2();
    // Hysteresis: keep the current depth while r stays within [0.45, 1.05].
    let r_last = (ppd * K) / f64::powi(2.0, last as i32);
    if (0.45..=1.05).contains(&r_last) {
        return last;
    }
    (ideal.ceil() as i32).clamp(MIN_DEPTH as i32, base as i32) as u8
}

/// Which layers the client wants drawn — the filter. Styling is the LUTs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerMask {
    pub dem: bool,
    pub land: bool,
    pub water: bool,
    pub line: bool,
    /// Residency debug: tint pixels whose dem came from a parent of the wanted depth (one level amber, two or more red) and blocks with no dem at all magenta — so streaming and eviction are visible.
    pub debug: bool,
    /// Imagery instead of terrain: the false-colour composite 1064 nm → R, NIR → G, red → B, in place of dem/land/water (lines still obey `line`).
    pub imagery: bool,
}

impl Default for LayerMask {
    fn default() -> Self {
        LayerMask { dem: true, land: true, water: true, line: true, debug: false, imagery: false }
    }
}

/// Hypsometric tint LUT indexed by elev_q >> 4 (4 m buckets). Water and no-data are LUT rows, keeping the pixel loop branch-free on terrain type.
pub fn build_hypso_lut() -> Box<[[u8; 3]; 4096]> {
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.0, [72.0, 96.0, 60.0]),
        (500.0, [110.0, 112.0, 70.0]),
        (1200.0, [138.0, 116.0, 84.0]),
        (2200.0, [160.0, 152.0, 146.0]),
        (3000.0, [238.0, 240.0, 245.0]),
    ];
    let mut lut = Box::new([[0u8; 3]; 4096]);
    for (i, out) in lut.iter_mut().enumerate() {
        let elev = (i as f32 * 16.0) / 4.0 - 500.0; // bucket -> meters
        if elev < 0.5 {
            *out = WATER_RGB; // the sea
            continue;
        }
        let mut tint = STOPS[STOPS.len() - 1].1;
        for w in STOPS.windows(2) {
            let (e0, c0) = w[0];
            let (e1, c1) = w[1];
            if elev < e1 {
                let t = ((elev - e0) / (e1 - e0)).clamp(0.0, 1.0);
                tint = [
                    c0[0] + (c1[0] - c0[0]) * t,
                    c0[1] + (c1[1] - c0[1]) * t,
                    c0[2] + (c1[2] - c0[2]) * t,
                ];
                break;
            }
        }
        *out = [tint[0] as u8, tint[1] as u8, tint[2] as u8];
    }
    lut[4095] = BG_RGB8; // no-data = background
    lut
}

pub const BG_RGB8: [u8; 3] = [18, 20, 26];
pub const WATER_RGB: [u8; 3] = [26, 58, 82];
/// Terrain tint when the dem layer is off or absent: a neutral ground.
const FLAT_RGB: [u8; 3] = [96, 100, 96];

/// Line class id (1-based, 0 = empty) -> visible RGB. Index 12 = Waterway.
pub const CLASS_LUT: [[u8; 3]; 13] = [
    [0, 0, 0],
    [245, 150, 60],
    [238, 175, 62],
    [240, 208, 84],
    [212, 212, 168],
    [182, 192, 182],
    [142, 147, 158],
    [112, 117, 128],
    [152, 120, 88],
    [80, 230, 120],
    [125, 122, 128],
    [148, 136, 160],
    [84, 150, 210],
];

/// Land cover class id (1-based = AreaClass + 1) -> tint. Order follows mahere_osm::AreaClass: Grass, Farmland, Orchard, Scrub, Forest, Wetland, Sand, Rock, Glacier, Quarry, Industrial, Urban, Water.
pub const LAND_LUT: [[u8; 3]; 14] = [
    [0, 0, 0],
    [122, 162, 90],
    [178, 170, 108],
    [138, 160, 88],
    [118, 138, 88],
    [66, 108, 66],
    [92, 140, 128],
    [214, 200, 160],
    [150, 146, 140],
    [226, 232, 240],
    [130, 120, 110],
    [140, 132, 142],
    [152, 140, 138],
    [26, 58, 82],
];

/// One resolved layer reference for a block: planes plus the shift that maps Q30.16 UV to this entry's texel grid (depends on the entry's ACTUAL depth — a parent fallback is just a different shift).
#[derive(Clone, Copy)]
enum DemRef<'a> {
    Cell { planes: &'a DemPacked, shift: u32, prefix_shift: u32, prefix: u64 },
    None,
}

#[derive(Clone, Copy)]
enum VecRef<'a> {
    Cell {
        line: Option<&'a ClassCell>,
        land: Option<&'a ClassCell>,
        water: Option<&'a CovCell>,
        img: Option<&'a ImgCell>,
        shift: u32,
        prefix_shift: u32,
        prefix: u64,
    },
    None,
}

/// Probe the pool at `depth`, climbing parents to MIN_DEPTH until an entry satisfies `has`. Absent cells climb too (for dem a parent is coarser truth; for vector layers the parent IS the box-filtered truth).
fn probe<'a>(pool: &'a Pool, mut depth: u8, mut prefix: u64, has: impl Fn(&Entry) -> bool) -> Option<(&'a Entry, u8, u64)> {
    loop {
        if let Some(e) = pool.map.get(&CellKey { depth, prefix }) {
            if has(e) {
                return Some((e, depth, prefix));
            }
        }
        if depth == MIN_DEPTH {
            return None;
        }
        depth -= 1;
        prefix >>= 2;
    }
}

pub struct FrameLuts {
    pub hypso: Box<[[u8; 3]; 4096]>,
    pub sun: [f32; 3],
    pub mask: LayerMask,
    /// The depth the frame asked the dem for (debug tint reference).
    pub dem_depth: u8,
}

/// Corner lattice entry: diamond + Q30.16 UV (i64).
#[derive(Clone, Copy)]
struct CornerPt {
    diamond: u8,
    u: i64,
    v: i64,
}

fn corner(cam: &Camera, px: f64, py: f64, w: usize, h: usize) -> CornerPt {
    let (lat, lon) = cam.screen_to_geo(px, py, w, h);
    let c = Coord::from_lat_lon(lat, lon);
    let (iu, iv) = c.uv();
    CornerPt { diamond: c.diamond(), u: (iu as i64) << 16, v: (iv as i64) << 16 }
}

pub struct FrameStats {
    pub blocks: usize,
    pub straddle_blocks: usize,
}

/// Render one frame into `canvas` (0xRRGGBB). Returns per-frame stats and the desired cell set (for residency) derived from the corner lattice.
#[allow(clippy::too_many_arguments)]
pub fn render_frame(
    canvas: &mut [u32],
    w: usize,
    h: usize,
    cam: &Camera,
    pool: &Pool,
    luts: &FrameLuts,
    dem_depth: u8,
    vec_depth: u8,
) -> (FrameStats, Vec<CellKey>) {
    let bw = w.div_ceil(BLOCK);
    let bh = h.div_ceil(BLOCK);
    // Corner lattice (exact math), rayon'd.
    let corners: Vec<CornerPt> = (0..(bw + 1) * (bh + 1))
        .into_par_iter()
        .map(|i| {
            let (cx, cy) = (i % (bw + 1), i / (bw + 1));
            corner(cam, (cx * BLOCK) as f64, (cy * BLOCK) as f64, w, h)
        })
        .collect();

    // Desired set from the lattice: the active depths and the whole parent chain down to the base — a region may be baked several depths shallower than the view wants, and the fallback must find it rather than skip to the base. Parents are shared by many corners, so the chain adds only a handful of small cells.
    let mut desired: FxHashSet<CellKey> = FxHashSet::default();
    for c in &corners {
        let raw = raw_of(c.diamond, c.u, c.v);
        for d in MIN_DEPTH..=dem_depth.max(vec_depth) {
            desired.insert(CellKey { depth: d, prefix: raw >> (60 - 2 * d as u32) });
        }
    }
    let want: Vec<CellKey> = desired.into_iter().collect();

    let straddle_count = std::sync::atomic::AtomicUsize::new(0);

    canvas.par_chunks_mut(w * BLOCK).enumerate().for_each(|(by, band)| {
        let band_h = band.len() / w;
        for bx in 0..bw {
            let c00 = corners[by * (bw + 1) + bx];
            let c10 = corners[by * (bw + 1) + bx + 1];
            let c01 = corners[(by + 1) * (bw + 1) + bx];
            let c11 = corners[(by + 1) * (bw + 1) + bx + 1];
            let x0 = bx * BLOCK;
            let bweff = (w - x0).min(BLOCK);
            if c00.diamond != c10.diamond || c00.diamond != c01.diamond || c00.diamond != c11.diamond {
                straddle_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                render_block_exact(
                    band, w, x0, bweff, band_h, by, cam, pool, luts, dem_depth, vec_depth, h,
                    &[c00.diamond, c10.diamond, c01.diamond, c11.diamond],
                );
                continue;
            }
            render_block_interp(band, w, x0, bweff, band_h, c00, c10, c01, c11, pool, luts, dem_depth, vec_depth);
        }
    });

    (FrameStats { blocks: bw * bh, straddle_blocks: straddle_count.into_inner() }, want)
}

/// Resolve a dem ref for a full-res raw prefix base. Shift maps Q30.16 u to texel at the found depth: tx = (uQ >> (16 + 22 - d)) & 255.
fn resolve_dem<'a>(pool: &'a Pool, depth: u8, raw: u64) -> DemRef<'a> {
    let prefix = raw >> (60 - 2 * depth as u32);
    match probe(pool, depth, prefix, |e| e.dem.is_some()) {
        Some((e, d, pfx)) => DemRef::Cell {
            planes: e.dem.as_ref().unwrap(),
            shift: 16 + (22 - d as u32),
            prefix_shift: 60 - 2 * d as u32,
            prefix: pfx,
        },
        None => DemRef::None,
    }
}

fn resolve_vec<'a>(pool: &'a Pool, depth: u8, raw: u64) -> VecRef<'a> {
    let prefix = raw >> (60 - 2 * depth as u32);
    match probe(pool, depth, prefix, |e| e.has_vec()) {
        Some((e, d, pfx)) => VecRef::Cell {
            line: e.line.as_ref(),
            land: e.land.as_ref(),
            water: e.water.as_ref(),
            img: e.img.as_ref(),
            shift: 16 + (22 - d as u32),
            prefix_shift: 60 - 2 * d as u32,
            prefix: pfx,
        },
        None => VecRef::None,
    }
}

#[inline(always)]
fn raw_of(diamond: u8, uq: i64, vq: i64) -> u64 {
    ((diamond as u64) << 60)
        | (mahere_coord::morton_spread((uq >> 16) as u64) << 1)
        | mahere_coord::morton_spread((vq >> 16) as u64)
}

/// Triangle texel index for Q30.16 UV against an entry whose texel grid is `shift` bits below the UV: the UV square, then which side of `u+v = k`
/// — the carry of the two fractional parts.
#[inline(always)]
pub fn tri_index(uq: i64, vq: i64, shift: u32) -> usize {
    let tx = ((uq >> shift) & 255) as usize;
    let ty = ((vq >> shift) & 255) as usize;
    let m = (1i64 << shift) - 1;
    let half = ((((uq & m) + (vq & m)) >> shift) & 1) as usize;
    (((ty << 8) | tx) << 1) | half
}

#[inline(always)]
fn lerp3(a: [f32; 3], b: [u8; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] as f32 - a[0]) * t,
        a[1] + (b[1] as f32 - a[1]) * t,
        a[2] + (b[2] as f32 - a[2]) * t,
    ]
}

/// A dem ref's depth (from its prefix shift).
#[inline(always)]
fn dem_ref_depth(r: &DemRef) -> Option<u8> {
    match r {
        DemRef::Cell { prefix_shift, .. } => Some(((60 - prefix_shift) / 2) as u8),
        DemRef::None => None,
    }
}

/// The dem texel for a pixel: the block's ref, and on a NODATA texel the parents below it — a merged cell can carry elevation only inside the newer bake's footprint while an older, coarser bake covers the rest.
#[inline(always)]
fn dem_texel(dem: &DemRef, pool: &Pool, diamond: u8, uq: i64, vq: i64) -> Option<u64> {
    let mut r = *dem;
    loop {
        let DemRef::Cell { planes, shift, .. } = r else { return None };
        let t = planes.texel[tri_index(uq, vq, shift)];
        if (t & 0xFFFF) as u16 != ELEV_NODATA {
            return Some(t);
        }
        let d = dem_ref_depth(&r)?;
        if d <= MIN_DEPTH {
            return None;
        }
        r = resolve_dem(pool, d - 1, raw_of(diamond, uq, vq));
    }
}

#[inline(always)]
fn compose(dem: &DemRef, vec: &VecRef, pool: &Pool, diamond: u8, uq: i64, vq: i64, luts: &FrameLuts) -> u32 {
    let rgb = compose_rgb(dem, vec, pool, diamond, uq, vq, luts);
    if !luts.mask.debug {
        return rgb;
    }
    // Debug tint by fallback distance of the dem ref.
    let tint: Option<[u32; 3]> = match dem_ref_depth(dem) {
        None => Some([255, 0, 255]),
        Some(d) if d + 2 <= luts.dem_depth => Some([255, 40, 40]),
        Some(d) if d + 1 == luts.dem_depth => Some([255, 170, 0]),
        _ => None,
    };
    match tint {
        None => rgb,
        Some(t) => {
            let (r, g, b) = ((rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255);
            (((r + t[0]) / 2) << 16) | (((g + t[1]) / 2) << 8) | ((b + t[2]) / 2)
        }
    }
}

#[inline(always)]
fn compose_rgb(dem: &DemRef, vec: &VecRef, pool: &Pool, diamond: u8, uq: i64, vq: i64, luts: &FrameLuts) -> u32 {
    let mask = luts.mask;
    // Terrain: tint from elevation, shade from the normal.
    let mut tint = [FLAT_RGB[0] as f32, FLAT_RGB[1] as f32, FLAT_RGB[2] as f32];
    let mut diffuse = 1.0f32;
    let mut have_ground = false;
    if mask.dem {
        if let Some(t) = dem_texel(dem, pool, diamond, uq, vq) {
            let eq = (t & 0xFFFF) as u16;
            {
                let nx = (t >> 16) as u16 as i16 as f32;
                let ny = (t >> 32) as u16 as i16 as f32;
                let nz = (t >> 48) as u16 as i16 as f32;
                diffuse = ((nx * luts.sun[0] + ny * luts.sun[1] + nz * luts.sun[2]) / 32767.0).max(0.0);
                let c = luts.hypso[(eq >> 4) as usize];
                tint = [c[0] as f32, c[1] as f32, c[2] as f32];
                have_ground = true;
            }
        } else {
            tint = [BG_RGB8[0] as f32, BG_RGB8[1] as f32, BG_RGB8[2] as f32];
        }
    }
    let mut rgb = tint;
    if let VecRef::Cell { line, land, water, img, shift, .. } = vec {
        let i = tri_index(uq, vq, *shift);
        if mask.imagery {
            // False colour: 1064 nm → R, NIR → G, red → B; no-data stays background.
            if let Some(im) = img {
                if im.red[i] != 0 || im.nir[i] != 0 || im.i1064[i] != 0 {
                    rgb = [im.i1064[i] as f32, im.nir[i] as f32, im.red[i] as f32];
                }
            }
            if mask.line {
                if let Some(line) = line {
                    let cov = line.cov[i];
                    if cov != 0 {
                        rgb = lerp3(rgb, CLASS_LUT[(line.class[i] as usize).min(12)], cov as f32 / 255.0);
                    }
                }
            }
            return ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | rgb[2] as u32;
        }
        if mask.land {
            if let Some(land) = land {
                let lc = land.cov[i];
                if lc != 0 {
                    rgb = lerp3(rgb, LAND_LUT[(land.class[i] as usize).min(13)], lc as f32 / 255.0 * 0.85);
                }
            }
        }
        let shade = if have_ground { 0.30 + 0.70 * diffuse } else { 1.0 };
        rgb = [rgb[0] * shade, rgb[1] * shade, rgb[2] * shade];
        if mask.water {
            if let Some(water) = water {
                let wc = water.cov[i];
                if wc != 0 {
                    let s = 0.85 + 0.15 * diffuse;
                    let wr = [WATER_RGB[0] as f32 * s, WATER_RGB[1] as f32 * s, WATER_RGB[2] as f32 * s];
                    let t = wc as f32 / 255.0;
                    rgb = [rgb[0] + (wr[0] - rgb[0]) * t, rgb[1] + (wr[1] - rgb[1]) * t, rgb[2] + (wr[2] - rgb[2]) * t];
                }
            }
        }
        if mask.line {
            if let Some(line) = line {
                let cov = line.cov[i];
                if cov != 0 {
                    rgb = lerp3(rgb, CLASS_LUT[(line.class[i] as usize).min(12)], cov as f32 / 255.0);
                }
            }
        }
    } else {
        let shade = if have_ground { 0.30 + 0.70 * diffuse } else { 1.0 };
        rgb = [rgb[0] * shade, rgb[1] * shade, rgb[2] * shade];
    }
    ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | rgb[2] as u32
}

#[allow(clippy::too_many_arguments)]
fn render_block_interp(
    band: &mut [u32],
    w: usize,
    x0: usize,
    bweff: usize,
    band_h: usize,
    c00: CornerPt,
    c10: CornerPt,
    c01: CornerPt,
    c11: CornerPt,
    pool: &Pool,
    luts: &FrameLuts,
    dem_depth: u8,
    vec_depth: u8,
) {
    let d = c00.diamond;
    // Per-block cell resolve from corner prefixes (1-4 per layer).
    let mut dem_refs: [(u64, DemRef); 4] = [(u64::MAX, DemRef::None); 4];
    let mut vec_refs: [(u64, VecRef); 4] = [(u64::MAX, VecRef::None); 4];
    let mut n_dem = 0usize;
    let mut n_vec = 0usize;
    for c in [c00, c10, c01, c11] {
        let raw = raw_of(d, c.u, c.v);
        let dp = raw >> (60 - 2 * dem_depth as u32);
        if !dem_refs[..n_dem].iter().any(|&(p, _)| p == dp) {
            dem_refs[n_dem] = (dp, resolve_dem(pool, dem_depth, raw));
            n_dem += 1;
        }
        let vp = raw >> (60 - 2 * vec_depth as u32);
        if !vec_refs[..n_vec].iter().any(|&(p, _)| p == vp) {
            vec_refs[n_vec] = (vp, resolve_vec(pool, vec_depth, raw));
            n_vec += 1;
        }
    }
    // Finest ref first: a parent-fallback ref covers its resident fine siblings' footprints too, so first-match must try the fine cell before the parent or the block paints coarse where sharp data is resident.
    dem_refs[..n_dem].sort_by_key(|(_, r)| match r {
        DemRef::Cell { prefix_shift, .. } => *prefix_shift,
        DemRef::None => u32::MAX,
    });
    vec_refs[..n_vec].sort_by_key(|(_, r)| match r {
        VecRef::Cell { prefix_shift, .. } => *prefix_shift,
        VecRef::None => u32::MAX,
    });
    let one_cell = n_dem == 1 && n_vec == 1;

    // Fixed-point steps across the block (divide by BLOCK).
    let du_dx = (c10.u - c00.u) / BLOCK as i64;
    let dv_dx = (c10.v - c00.v) / BLOCK as i64;
    let du_dy = (c01.u - c00.u) / BLOCK as i64;
    let dv_dy = (c01.v - c00.v) / BLOCK as i64;
    // Bilinear twist term (c11 vs parallelogram) distributed per row.
    let twist_u = ((c11.u - c10.u) - (c01.u - c00.u)) / (BLOCK * BLOCK) as i64;
    let twist_v = ((c11.v - c10.v) - (c01.v - c00.v)) / (BLOCK * BLOCK) as i64;

    for py in 0..band_h {
        let mut uq = c00.u + du_dy * py as i64;
        let mut vq = c00.v + dv_dy * py as i64;
        let dux = du_dx + twist_u * py as i64;
        let dvx = dv_dx + twist_v * py as i64;
        let row = &mut band[py * w + x0..py * w + x0 + bweff];
        if one_cell {
            let dref = &dem_refs[0].1;
            let vref = &vec_refs[0].1;
            for px in row.iter_mut() {
                *px = compose(dref, vref, pool, d, uq, vq, luts);
                uq += dux;
                vq += dvx;
            }
        } else {
            for px in row.iter_mut() {
                let raw = raw_of(d, uq, vq);
                // Match against each ref's RESOLVED (prefix, shift): a ref that fell back to a parent covers many nominal prefixes.
                let mut dref = DemRef::None;
                for (_, r) in &dem_refs[..n_dem] {
                    if let DemRef::Cell { prefix, prefix_shift, .. } = r {
                        if raw >> prefix_shift == *prefix {
                            dref = *r;
                            break;
                        }
                    }
                }
                if matches!(dref, DemRef::None) {
                    // Boundary sliver: the pixel's cell wasn't sampled by any corner. Rare (sub-texel band along cell edges) — a full probe here is cheap and makes coverage exact.
                    dref = resolve_dem(pool, dem_depth, raw);
                }
                let mut vref = VecRef::None;
                for (_, r) in &vec_refs[..n_vec] {
                    if let VecRef::Cell { prefix, prefix_shift, .. } = r {
                        if raw >> prefix_shift == *prefix {
                            vref = *r;
                            break;
                        }
                    }
                }
                if matches!(vref, VecRef::None) {
                    vref = resolve_vec(pool, vec_depth, raw);
                }
                *px = compose(&dref, &vref, pool, d, uq, vq, luts);
                uq += dux;
                vq += dvx;
            }
        }
    }
}

/// Diamond-straddle fallback: exact per-pixel encode restricted to the corner diamonds. Rare (a few blocks per screen at most, usually zero).
#[allow(clippy::too_many_arguments)]
fn render_block_exact(
    band: &mut [u32],
    w: usize,
    x0: usize,
    bweff: usize,
    band_h: usize,
    by: usize,
    cam: &Camera,
    pool: &Pool,
    luts: &FrameLuts,
    dem_depth: u8,
    vec_depth: u8,
    h: usize,
    diamonds: &[u8; 4],
) {
    let mut ds: Vec<u8> = diamonds.to_vec();
    ds.sort_unstable();
    ds.dedup();
    for py in 0..band_h {
        let gy = by * BLOCK + py;
        let row = &mut band[py * w + x0..py * w + x0 + bweff];
        for (i, px) in row.iter_mut().enumerate() {
            let (lat, lon) = cam.screen_to_geo((x0 + i) as f64 + 0.5, gy as f64 + 0.5, w, h);
            let c = Coord::from_lat_lon_in_diamonds(lat, lon, &ds);
            let (iu, iv) = c.uv();
            let (uq, vq) = ((iu as i64) << 16, (iv as i64) << 16);
            let raw = c.raw();
            let dref = resolve_dem(pool, dem_depth, raw);
            let vref = resolve_vec(pool, vec_depth, raw);
            *px = compose(&dref, &vref, pool, c.diamond(), uq, vq, luts);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency::{DemPacked, Entry, Pool};

    #[test]
    fn depth_selection_and_hysteresis() {
        // ppd 6000 -> ideal ceil(log2(1484.9)) = 11
        assert_eq!(select_depth(6000.0, 0, 13), 11);
        // Within the hysteresis band the current depth sticks.
        assert_eq!(select_depth(6000.0, 11, 13), 11);
        assert_eq!(select_depth(6300.0, 11, 13), 11);
        // Far outside the band it snaps.
        assert_eq!(select_depth(24_000.0, 11, 13), 13);
        assert_eq!(select_depth(100.0, 11, 13), MIN_DEPTH);
    }

    /// One synthetic flat cell at depth 6; a rendered frame must light it exactly as the compose math says, through the whole block/page-table path (including parent fallback from the nominal depth). With a forest over half of it, the land tint shows exactly there.
    #[test]
    fn frame_matches_compose_reference() {
        let mut pool = Pool::default();
        // Flat terrain at 1000 m: eq = (1000+500)*4 = 6000, normal = +z.
        let eq = 6000u64;
        let nz = 32767u64;
        let texel = vec![eq | (nz << 48); mahere_tiles::TRI];
        let cam = crate::Camera { lat: 46.2, lon: -121.5, ppd: 6000.0, bearing: 0.0 };
        let c = mahere_coord::Coord::from_lat_lon(cam.lat, cam.lon);
        let key = CellKey { depth: 6, prefix: c.raw() >> (60 - 2 * 6) };
        let mut land = ClassCell::new();
        for i in 0..mahere_tiles::TRI {
            land.class[i] = 5; // Forest
            land.cov[i] = 255;
        }
        pool.map.insert(
            key,
            Entry {
                dem: Some(DemPacked { texel: texel.into_boxed_slice() }),
                line: Some(ClassCell::new()),
                land: Some(land),
                water: Some(CovCell::new()),
                img: None,
            },
        );

        let luts = FrameLuts { hypso: build_hypso_lut(), sun: [0.0, 0.0, 1.0], mask: LayerMask::default(), dem_depth: 12 };
        let (w, h) = (64usize, 64usize);
        let mut canvas = vec![0u32; w * h];
        let (stats, _want) = render_frame(&mut canvas, w, h, &cam, &pool, &luts, 12, 13);
        assert_eq!(stats.straddle_blocks, 0);
        // Expected: diffuse = 1, shade = 1.0, tint = hypso[375] lerped 85% to forest.
        let t = luts.hypso[375];
        let f = LAND_LUT[5];
        let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * 0.85) as u32;
        let expect = (mix(t[0], f[0]) << 16) | (mix(t[1], f[1]) << 8) | mix(t[2], f[2]);
        let center = canvas[(h / 2) * w + w / 2];
        assert_eq!(center, expect, "center {center:#08x} vs expected {expect:#08x}");
        assert!(canvas.iter().all(|&p| p == expect), "unresolved pixels in frame");

        // Mask off land: pure hypso.
        let luts = FrameLuts { mask: LayerMask { land: false, ..LayerMask::default() }, ..luts };
        render_frame(&mut canvas, w, h, &cam, &pool, &luts, 12, 13);
        let expect = ((t[0] as u32) << 16) | ((t[1] as u32) << 8) | t[2] as u32;
        assert_eq!(canvas[(h / 2) * w + w / 2], expect);
    }
}
