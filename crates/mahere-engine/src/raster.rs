//! The hot loop: the #pagetable compositor. A frame is fetches — 32 px
//! block grid with exact corners, fixed-point UV stepping inside, per-BLOCK
//! cell resolution through the page table, nearest-fetch compose of the dem
//! and line layers through style LUTs. No vectors, no re-rasterization, no
//! allocation; rayon over row bands.

use mahere_coord::Coord;
use mahere_tiles::CellKey;
use rayon::prelude::*;
use rustc_hash::FxHashSet;

use crate::Camera;
use crate::residency::{DemPacked, ELEV_NODATA, Entry, Layer, LinePlanes, Pool};

pub const BLOCK: usize = 32;
pub const LINE_BASE_DEPTH: u8 = 13;
pub const DEM_BASE_DEPTH: u8 = 12;
pub const MIN_DEPTH: u8 = 6;

/// texel/pixel ratio constant: diamond edge 7054 km, 111320 m/deg, 256
/// texels/cell. ratio r(d) = ppd * K / 2^d; pick d so r ∈ (0.5, 1].
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

/// Hypsometric tint LUT indexed by elev_q >> 4 (4 m buckets). Water and
/// no-data are LUT rows, keeping the pixel loop branch-free on terrain type.
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
            *out = [26, 58, 82]; // water
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
    lut[4095] = [18, 20, 26]; // no-data = background
    lut
}

/// class id (1-based, 0 = empty) -> visible RGB. Index 12 = Waterway.
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

/// One resolved layer reference for a block: planes plus the shift that maps
/// Q30.16 UV to this entry's texel grid (depends on the entry's ACTUAL
/// depth — a parent fallback is just a different shift).
#[derive(Clone, Copy)]
enum DemRef<'a> {
    Cell { planes: &'a DemPacked, shift: u32, prefix_shift: u32, prefix: u64 },
    None,
}

#[derive(Clone, Copy)]
enum LineRef<'a> {
    Cell { planes: &'a LinePlanes, shift: u32, prefix_shift: u32, prefix: u64 },
    None,
}

/// Probe the pool at `depth`, climbing parents to MIN_DEPTH. Returns the
/// entry and the depth it was found at. `Absent` entries climb too (for dem
/// a parent is coarser truth; for lines the parent IS the box-filtered
/// truth).
fn probe<'a>(pool: &'a Pool, mut depth: u8, mut prefix: u64) -> Option<(&'a Entry, u8, u64)> {
    loop {
        if let Some(e) = pool.map.get(&(depth, prefix)) {
            if !matches!(e, Entry::Absent) {
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

/// Render one frame into `canvas` (0xRRGGBB). Returns per-frame stats and
/// the desired cell set (for residency) derived from the corner lattice.
#[allow(clippy::too_many_arguments)]
pub fn render_frame(
    canvas: &mut [u32],
    w: usize,
    h: usize,
    cam: &Camera,
    dem_pool: &Pool,
    line_pool: &Pool,
    luts: &FrameLuts,
    dem_depth: u8,
    line_depth: u8,
) -> (FrameStats, Vec<(Layer, CellKey)>) {
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

    // Desired set from the lattice: active depths + 2 parent rings + base.
    let mut desired: FxHashSet<(Layer, u8, u64)> = FxHashSet::default();
    for c in &corners {
        let raw = ((c.diamond as u64) << 60)
            | (mahere_coord::morton_spread((c.u >> 16) as u64) << 1)
            | mahere_coord::morton_spread((c.v >> 16) as u64);
        for (layer, d0) in [(Layer::Dem, dem_depth), (Layer::Line, line_depth)] {
            let mut d = d0;
            loop {
                desired.insert((layer, d, raw >> (60 - 2 * d as u32)));
                if d <= d0.saturating_sub(2) || d == MIN_DEPTH {
                    break;
                }
                d -= 1;
            }
            desired.insert((layer, MIN_DEPTH, raw >> (60 - 2 * MIN_DEPTH as u32)));
        }
    }
    let want: Vec<(Layer, CellKey)> = desired
        .iter()
        .map(|&(l, depth, prefix)| (l, CellKey { depth, prefix }))
        .collect();

    let straddles: Vec<usize> = Vec::new();
    let straddle_count = std::sync::atomic::AtomicUsize::new(0);
    let _ = straddles;

    canvas
        .par_chunks_mut(w * BLOCK)
        .enumerate()
        .for_each(|(by, band)| {
            let band_h = band.len() / w;
            for bx in 0..bw {
                let c00 = corners[by * (bw + 1) + bx];
                let c10 = corners[by * (bw + 1) + bx + 1];
                let c01 = corners[(by + 1) * (bw + 1) + bx];
                let c11 = corners[(by + 1) * (bw + 1) + bx + 1];
                let x0 = bx * BLOCK;
                let bweff = (w - x0).min(BLOCK);
                if c00.diamond != c10.diamond
                    || c00.diamond != c01.diamond
                    || c00.diamond != c11.diamond
                {
                    straddle_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    render_block_exact(
                        band, w, x0, bweff, band_h, by, cam, dem_pool, line_pool, luts,
                        dem_depth, line_depth, h,
                        &[c00.diamond, c10.diamond, c01.diamond, c11.diamond],
                    );
                    continue;
                }
                render_block_interp(
                    band, w, x0, bweff, band_h, c00, c10, c01, c11, dem_pool, line_pool,
                    luts, dem_depth, line_depth,
                );
            }
        });

    (
        FrameStats {
            blocks: bw * bh,
            straddle_blocks: straddle_count.into_inner(),
        },
        want,
    )
}

/// Resolve a layer ref for a given full-res raw prefix base. Shift maps
/// Q30.16 u to texel at the found depth: tx = (uQ >> (16 + 22 - d)) & 255.
fn resolve_dem<'a>(pool: &'a Pool, depth: u8, raw_prefix_base: u64) -> DemRef<'a> {
    let prefix = raw_prefix_base >> (60 - 2 * depth as u32);
    match probe(pool, depth, prefix) {
        Some((Entry::Dem(p), d, pfx)) => DemRef::Cell {
            planes: p,
            shift: 16 + (22 - d as u32),
            prefix_shift: 60 - 2 * d as u32,
            prefix: pfx,
        },
        _ => DemRef::None,
    }
}

fn resolve_line<'a>(pool: &'a Pool, depth: u8, raw_prefix_base: u64) -> LineRef<'a> {
    let prefix = raw_prefix_base >> (60 - 2 * depth as u32);
    match probe(pool, depth, prefix) {
        Some((Entry::Line(p), d, pfx)) => LineRef::Cell {
            planes: p,
            shift: 16 + (22 - d as u32),
            prefix_shift: 60 - 2 * d as u32,
            prefix: pfx,
        },
        _ => LineRef::None,
    }
}

#[inline(always)]
fn raw_of(diamond: u8, uq: i64, vq: i64) -> u64 {
    ((diamond as u64) << 60)
        | (mahere_coord::morton_spread((uq >> 16) as u64) << 1)
        | mahere_coord::morton_spread((vq >> 16) as u64)
}

#[inline(always)]
fn compose(
    dem: &DemRef,
    line: &LineRef,
    uq: i64,
    vq: i64,
    luts: &FrameLuts,
) -> u32 {
    let (mut r, mut g, mut b) = (18u32, 20u32, 26u32); // background
    if let DemRef::Cell { planes, shift, .. } = dem {
        let tx = ((uq >> shift) & 255) as usize;
        let ty = ((vq >> shift) & 255) as usize;
        let t = planes.texel[(ty << 8) | tx];
        let eq = (t & 0xFFFF) as u16;
        if eq != ELEV_NODATA {
            let nx = (t >> 16) as u16 as i16 as f32;
            let ny = (t >> 32) as u16 as i16 as f32;
            let nz = (t >> 48) as u16 as i16 as f32;
            let diffuse =
                ((nx * luts.sun[0] + ny * luts.sun[1] + nz * luts.sun[2]) / 32767.0).max(0.0);
            let shade = 0.30 + 0.70 * diffuse;
            let tint = luts.hypso[(eq >> 4) as usize];
            r = (tint[0] as f32 * shade) as u32;
            g = (tint[1] as f32 * shade) as u32;
            b = (tint[2] as f32 * shade) as u32;
        } else {
            let tint = luts.hypso[4095];
            (r, g, b) = (tint[0] as u32, tint[1] as u32, tint[2] as u32);
        }
    }
    if let LineRef::Cell { planes, shift, .. } = line {
        let tx = ((uq >> shift) & 255) as usize;
        let ty = ((vq >> shift) & 255) as usize;
        let i = (ty << 8) | tx;
        let cov = planes.cov[i] as u32;
        if cov != 0 {
            let c = CLASS_LUT[(planes.class[i] as usize).min(12)];
            r = (r * (255 - cov) + c[0] as u32 * cov) / 255;
            g = (g * (255 - cov) + c[1] as u32 * cov) / 255;
            b = (b * (255 - cov) + c[2] as u32 * cov) / 255;
        }
    }
    (r << 16) | (g << 8) | b
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
    dem_pool: &Pool,
    line_pool: &Pool,
    luts: &FrameLuts,
    dem_depth: u8,
    line_depth: u8,
) {
    let d = c00.diamond;
    // Per-block cell resolve from corner prefixes (1-4 per layer).
    let mut dem_refs: [(u64, DemRef); 4] = [(u64::MAX, DemRef::None); 4];
    let mut line_refs: [(u64, LineRef); 4] = [(u64::MAX, LineRef::None); 4];
    let mut n_dem = 0usize;
    let mut n_line = 0usize;
    for c in [c00, c10, c01, c11] {
        let raw = raw_of(d, c.u, c.v);
        let dp = raw >> (60 - 2 * dem_depth as u32);
        if !dem_refs[..n_dem].iter().any(|&(p, _)| p == dp) {
            dem_refs[n_dem] = (dp, resolve_dem(dem_pool, dem_depth, raw));
            n_dem += 1;
        }
        let lp = raw >> (60 - 2 * line_depth as u32);
        if !line_refs[..n_line].iter().any(|&(p, _)| p == lp) {
            line_refs[n_line] = (lp, resolve_line(line_pool, line_depth, raw));
            n_line += 1;
        }
    }
    let one_cell = n_dem == 1 && n_line == 1;

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
            let lref = &line_refs[0].1;
            for px in row.iter_mut() {
                *px = compose(dref, lref, uq, vq, luts);
                uq += dux;
                vq += dvx;
            }
        } else {
            for px in row.iter_mut() {
                let raw = raw_of(d, uq, vq);
                // Match against each ref's RESOLVED (prefix, shift): a ref
                // that fell back to a parent covers many nominal prefixes.
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
                    // Boundary sliver: the pixel's cell wasn't sampled by any
                    // corner. Rare (sub-texel band along cell edges) — a full
                    // probe here is cheap and makes coverage exact.
                    dref = resolve_dem(dem_pool, dem_depth, raw);
                }
                let mut lref = LineRef::None;
                for (_, r) in &line_refs[..n_line] {
                    if let LineRef::Cell { prefix, prefix_shift, .. } = r {
                        if raw >> prefix_shift == *prefix {
                            lref = *r;
                            break;
                        }
                    }
                }
                if matches!(lref, LineRef::None) {
                    lref = resolve_line(line_pool, line_depth, raw);
                }
                *px = compose(&dref, &lref, uq, vq, luts);
                uq += dux;
                vq += dvx;
            }
        }
    }
}

/// Diamond-straddle fallback: exact per-pixel encode restricted to the
/// corner diamonds. Rare (a few blocks per screen at most, usually zero).
#[allow(clippy::too_many_arguments)]
fn render_block_exact(
    band: &mut [u32],
    w: usize,
    x0: usize,
    bweff: usize,
    band_h: usize,
    by: usize,
    cam: &Camera,
    dem_pool: &Pool,
    line_pool: &Pool,
    luts: &FrameLuts,
    dem_depth: u8,
    line_depth: u8,
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
            let dref = resolve_dem(dem_pool, dem_depth, raw);
            let lref = resolve_line(line_pool, line_depth, raw);
            *px = compose(&dref, &lref, uq, vq, luts);
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

    /// One synthetic flat cell at depth 6; a rendered frame must light it
    /// exactly as the compose math says, through the whole block/page-table
    /// path (including parent fallback from the nominal depth).
    #[test]
    fn frame_matches_compose_reference() {
        let mut dem = Pool::new_for_tests();
        let line = Pool::new_for_tests();
        // Flat terrain at 1000 m: eq = (1000+500)*4 = 6000, normal = +z.
        let eq = 6000u64;
        let nz = 32767u64;
        let texel = vec![eq | (nz << 48); mahere_tiles::TEX * mahere_tiles::TEX];
        let cam = crate::Camera { lat: 46.2, lon: -121.5, ppd: 6000.0, bearing: 0.0 };
        let c = mahere_coord::Coord::from_lat_lon(cam.lat, cam.lon);
        let prefix = c.raw() >> (60 - 2 * 6);
        dem.map.insert((6, prefix), Entry::Dem(DemPacked { texel: texel.into_boxed_slice() }));

        let luts = FrameLuts { hypso: build_hypso_lut(), sun: [0.0, 0.0, 1.0] };
        let (w, h) = (64usize, 64usize);
        let mut canvas = vec![0u32; w * h];
        let (stats, _want) =
            render_frame(&mut canvas, w, h, &cam, &dem, &line, &luts, 12, 13);
        assert_eq!(stats.straddle_blocks, 0);
        // Expected: diffuse = 1 (sun straight up, flat normal), shade = 1.0,
        // tint = hypso[6000>>4 = 375].
        let tint = luts.hypso[375];
        let expect = ((tint[0] as u32) << 16) | ((tint[1] as u32) << 8) | tint[2] as u32;
        let center = canvas[(h / 2) * w + w / 2];
        assert_eq!(center, expect, "center {center:#08x} vs expected {expect:#08x}");
        // Every pixel resolved (no background) — the whole 64x64 view sits
        // inside one depth-6 cell.
        assert!(canvas.iter().all(|&p| p == expect), "unresolved pixels in frame");
    }
}
