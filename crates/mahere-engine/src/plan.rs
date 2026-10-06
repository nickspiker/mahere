//! The frame plan: what a GPU needs to draw a frame, computed on the CPU without touching a pixel. The 32 px block lattice with exact dymaxion corners (the same lattice the CPU raster steps inside), every resident cell as a reference with the planes it carries, and a hash page table over those references keyed by (diamond, depth, cell) — so a shader resolves any pixel's cell by lookup, climbing parents exactly as the CPU's probe does, with no per-block reference lists to miss a sliver.
//!
//! Diamond-straddling blocks are subdivided into 4 px sub-blocks with exact corners restricted to the corner diamonds; a sub-block whose corners still disagree takes its first corner's diamond for all four (a seam error under four pixels wide).

use mahere_coord::{Coord, uv_to_lat_lon};
use mahere_tiles::{CellKey, ELEV_NODATA, TEX};
use rayon::prelude::*;


use crate::Camera;
use crate::raster::{BLOCK, CornerPt, ElevRange, MIN_DEPTH, corner, raw_of};
use crate::residency::Pool;

pub const NONE: u32 = u32::MAX;

/// Side of a straddle sub-block, pixels.
pub const SUB: usize = 4;

/// Page table slots; a power of two, several times the largest resident set.
pub const TABLE_N: usize = 4096;

/// Depths at which the view's neighbouring cells are fetched ahead, so a zoom out is already covered.
pub const PREFETCH_DEPTH: u8 = 10;

pub const FLAG_DEM: u32 = 1;
pub const FLAG_LINE: u32 = 2;
pub const FLAG_LAND: u32 = 4;
pub const FLAG_WATER: u32 = 8;
pub const FLAG_IMG: u32 = 16;

/// A resident cell as the shader sees it: where it is, what it carries, and the metres-per-texel Jacobian a normal needs.
#[derive(Clone, Copy, Debug)]
pub struct PlanRef {
    pub key: CellKey,
    pub diamond: u8,
    pub cu: u32,
    pub cv: u32,
    pub flags: u32,
    /// Metres east and north per texel step along u, then along v, and the inverse determinant.
    pub jac: [f32; 5],
}

/// A screen block with its corner UVs in depth-30 units and the per-pixel steps the shader interpolates with.
#[derive(Clone, Copy, Debug)]
pub struct PlanBlock {
    pub x: u32,
    pub y: u32,
    pub size: u32,
    pub diamond: u32,
    pub u0: u32,
    pub v0: u32,
    pub du_dx: f32,
    pub dv_dx: f32,
    pub du_dy: f32,
    pub dv_dy: f32,
    pub twist_u: f32,
    pub twist_v: f32,
}

/// One page table slot: `tag` = diamond | depth << 8, cell grid coordinates, reference index (NONE = empty).
#[derive(Clone, Copy, Debug)]
pub struct TableSlot {
    pub tag: u32,
    pub cu: u32,
    pub cv: u32,
    pub index: u32,
}

pub struct FramePlan {
    pub blocks: Vec<PlanBlock>,
    pub refs: Vec<PlanRef>,
    pub table: Vec<TableSlot>,
    pub want: Vec<CellKey>,
    pub elev: ElevRange,
    pub straddle_blocks: usize,
    pub dem_depth: u8,
    pub vec_depth: u8,
    /// The dem's texels are wider than about a pixel and a half: the shader interpolates elevation between texels so contours stay smooth past the base depth.
    pub magnified: bool,
    /// The largest magnitude per line class among the cells in view: every line is drawn on a linear scale up to its class's, so a view of creeks still has a brightest creek and a view of lanes a boldest lane.
    pub line_mag_hi: [u8; 16],
}

/// The hash every lookup agrees on, CPU and shader.
#[inline]
pub fn table_hash(diamond: u32, depth: u32, cu: u32, cv: u32) -> u32 {
    let h = cu.wrapping_mul(0x9E37_79B1) ^ cv.wrapping_mul(0x85EB_CA77) ^ depth.wrapping_mul(0xC2B2_AE3D) ^ diamond.wrapping_mul(0x27D4_EB2F);
    (h ^ (h >> 15)) & (TABLE_N as u32 - 1)
}

fn jacobian(key: CellKey) -> [f32; 5] {
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
    let det = eu * nv - ev * nu;
    let inv = if det.abs() > 1e-9 { 1.0 / det } else { 0.0 };
    [eu as f32, nu as f32, ev as f32, nv as f32, inv as f32]
}

fn block_of(x: usize, y: usize, size: usize, c00: CornerPt, c10: CornerPt, c01: CornerPt, c11: CornerPt) -> PlanBlock {
    let f = |q: i64| q as f64 / 65536.0;
    let n = size as f64;
    PlanBlock {
        x: x as u32,
        y: y as u32,
        size: size as u32,
        diamond: c00.diamond as u32,
        u0: (c00.u >> 16) as u32,
        v0: (c00.v >> 16) as u32,
        du_dx: ((f(c10.u) - f(c00.u)) / n) as f32,
        dv_dx: ((f(c10.v) - f(c00.v)) / n) as f32,
        du_dy: ((f(c01.u) - f(c00.u)) / n) as f32,
        dv_dy: ((f(c01.v) - f(c00.v)) / n) as f32,
        twist_u: (((f(c11.u) - f(c10.u)) - (f(c01.u) - f(c00.u))) / (n * n)) as f32,
        twist_v: (((f(c11.v) - f(c10.v)) - (f(c01.v) - f(c00.v))) / (n * n)) as f32,
    }
}

/// Plan a frame: the lattice, the references, the table, the desired set and the lattice's elevation range.
pub fn plan_frame(w: usize, h: usize, cam: &Camera, pool: &Pool, dem_depth: u8, vec_depth: u8) -> FramePlan {
    let bw = w.div_ceil(BLOCK);
    let bh = h.div_ceil(BLOCK);
    let corners: Vec<CornerPt> = (0..(bw + 1) * (bh + 1))
        .into_par_iter()
        .map(|i| {
            let (cx, cy) = (i % (bw + 1), i / (bw + 1));
            corner(cam, (cx * BLOCK) as f64, (cy * BLOCK) as f64, w, h)
        })
        .collect();

    // Desired set and the elevation range, from the lattice. The corners dedupe to their finest cells first (a few hundred from a few thousand corners), then each walks its parents; at the coarse depths the ring of neighbours comes too, so a zoom out lands on cells already resident instead of a blank screen (Nick 2026-10-06), and they are small.
    let fine = dem_depth.max(vec_depth);
    let mut fine_cells: rustc_hash::FxHashSet<u64> = rustc_hash::FxHashSet::default();
    for c in &corners {
        let raw = raw_of(c.diamond, c.u, c.v);
        fine_cells.insert(raw >> (60 - 2 * fine as u32));
    }
    let mut desired: rustc_hash::FxHashSet<CellKey> = rustc_hash::FxHashSet::default();
    for &fp in &fine_cells {
        for d in MIN_DEPTH..=fine {
            let key = CellKey { depth: d, prefix: fp >> (2 * (fine - d) as u32) };
            desired.insert(key);
            if d <= PREFETCH_DEPTH {
                let (cu, cv) = key.grid();
                let n = 1u64 << d;
                for (du, dv) in [(-1i64, -1i64), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                    let (nu, nv) = (cu as i64 + du, cv as i64 + dv);
                    if nu >= 0 && nv >= 0 && (nu as u64) < n && (nv as u64) < n {
                        desired.insert(CellKey::from_grid(key.diamond(), d, nu as u64, nv as u64));
                    }
                }
            }
        }
    }

    // Blocks: whole where the corners share a diamond, 4 px sub-blocks across a seam.
    let mut blocks = Vec::with_capacity(bw * bh);
    let mut straddle_blocks = 0;
    for by in 0..bh {
        for bx in 0..bw {
            let c00 = corners[by * (bw + 1) + bx];
            let c10 = corners[by * (bw + 1) + bx + 1];
            let c01 = corners[(by + 1) * (bw + 1) + bx];
            let c11 = corners[(by + 1) * (bw + 1) + bx + 1];
            let (x0, y0) = (bx * BLOCK, by * BLOCK);
            if c00.diamond == c10.diamond && c00.diamond == c01.diamond && c00.diamond == c11.diamond {
                blocks.push(block_of(x0, y0, BLOCK, c00, c10, c01, c11));
                continue;
            }
            straddle_blocks += 1;
            let mut ds = vec![c00.diamond, c10.diamond, c01.diamond, c11.diamond];
            ds.sort_unstable();
            ds.dedup();
            let n = BLOCK / SUB;
            let sub_corner = |px: f64, py: f64, only: Option<u8>| -> CornerPt {
                let (lat, lon) = cam.screen_to_geo(px, py, w, h);
                let c = match only {
                    Some(d) => Coord::from_lat_lon_in_diamonds(lat, lon, &[d]),
                    None => Coord::from_lat_lon_in_diamonds(lat, lon, &ds),
                };
                let (iu, iv) = c.uv();
                CornerPt { diamond: c.diamond(), u: (iu as i64) << 16, v: (iv as i64) << 16 }
            };
            let grid: Vec<CornerPt> = (0..(n + 1) * (n + 1))
                .map(|i| sub_corner((x0 + (i % (n + 1)) * SUB) as f64, (y0 + (i / (n + 1)) * SUB) as f64, None))
                .collect();
            for sy in 0..n {
                for sx in 0..n {
                    let mut s00 = grid[sy * (n + 1) + sx];
                    let mut s10 = grid[sy * (n + 1) + sx + 1];
                    let mut s01 = grid[(sy + 1) * (n + 1) + sx];
                    let mut s11 = grid[(sy + 1) * (n + 1) + sx + 1];
                    let d = s00.diamond;
                    if s10.diamond != d || s01.diamond != d || s11.diamond != d {
                        let (px, py) = ((x0 + sx * SUB) as f64, (y0 + sy * SUB) as f64);
                        s00 = sub_corner(px, py, Some(d));
                        s10 = sub_corner(px + SUB as f64, py, Some(d));
                        s01 = sub_corner(px, py + SUB as f64, Some(d));
                        s11 = sub_corner(px + SUB as f64, py + SUB as f64, Some(d));
                    }
                    blocks.push(block_of(x0 + sx * SUB, y0 + sy * SUB, SUB, s00, s10, s01, s11));
                }
            }
        }
    }

    // References: every resident cell with a plane, and the table over them.
    let mut refs: Vec<PlanRef> = Vec::with_capacity(pool.map.len());
    let mut table = vec![TableSlot { tag: 0, cu: 0, cv: 0, index: NONE }; TABLE_N];
    let mut keys: Vec<&CellKey> = pool.map.keys().collect();
    keys.sort();
    for key in keys {
        let e = &pool.map[key];
        // From the presence bits, not the planes: a GPU host may have released the CPU copies.
        let p = e.present;
        let mut flags = 0;
        if p & crate::residency::PRESENT_DEM != 0 {
            flags |= FLAG_DEM;
        }
        if p & crate::residency::PRESENT_LINE != 0 {
            flags |= FLAG_LINE;
        }
        if p & crate::residency::PRESENT_LAND != 0 {
            flags |= FLAG_LAND;
        }
        if p & crate::residency::PRESENT_WATER != 0 {
            flags |= FLAG_WATER;
        }
        if p & crate::residency::PRESENT_IMG != 0 {
            flags |= FLAG_IMG;
        }
        if flags == 0 {
            continue;
        }
        let (cu, cv) = key.grid();
        let diamond = key.diamond();
        let index = refs.len() as u32;
        refs.push(PlanRef { key: *key, diamond, cu: cu as u32, cv: cv as u32, flags, jac: if flags & FLAG_DEM != 0 { jacobian(*key) } else { [0.0; 5] } });
        let mut h = table_hash(diamond as u32, key.depth as u32, cu as u32, cv as u32) as usize;
        loop {
            if table[h].index == NONE {
                table[h] = TableSlot { tag: diamond as u32 | ((key.depth as u32) << 8), cu: cu as u32, cv: cv as u32, index };
                break;
            }
            h = (h + 1) & (TABLE_N - 1);
        }
    }

    // The elevation span of the terrain on screen: the union of the cells' spans at the terrain depth (or the nearest coarser depth that has any), each recorded once at load — no sampling, no corner that misses a summit.
    let mut elev = ElevRange::EMPTY;
    let mut d = dem_depth;
    loop {
        for r in &refs {
            if r.key.depth == d {
                let e = &pool.map[&r.key];
                if e.elev_lo != ELEV_NODATA {
                    elev.lo = elev.lo.min(e.elev_lo);
                    elev.hi = elev.hi.max(e.elev_hi);
                }
            }
        }
        if !elev.is_empty() || d <= MIN_DEPTH {
            break;
        }
        d -= 1;
    }
    let magnified = crate::raster::texel_px(cam.ppd, dem_depth) >= 1.5;
    let mut line_mag_hi = [0u8; 16];
    for r in refs.iter().filter(|r| r.key.depth == vec_depth || r.key.depth + 1 == vec_depth) {
        for (hi, &m) in line_mag_hi.iter_mut().zip(&pool.map[&r.key].line_mag_max) {
            *hi = (*hi).max(m);
        }
    }
    FramePlan { blocks, refs, table, want: desired.into_iter().collect(), elev, straddle_blocks, dem_depth, vec_depth, magnified, line_mag_hi }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_finds_every_reference() {
        let mut pool = Pool::default();
        let c = Coord::from_lat_lon(46.2, -122.19);
        for depth in MIN_DEPTH..=12 {
            let key = CellKey { depth, prefix: c.raw() >> (60 - 2 * depth as u32) };
            pool.map.insert(key, crate::residency::Entry { present: crate::residency::PRESENT_WATER, water: Some(mahere_tiles::CovCell::new()), ..Default::default() });
        }
        let cam = Camera { lat: 46.2, lon: -122.19, ppd: 6000.0, bearing: 0.3 };
        let plan = plan_frame(256, 128, &cam, &pool, 12, 12);
        assert_eq!(plan.refs.len(), 7);
        for (i, r) in plan.refs.iter().enumerate() {
            let mut h = table_hash(r.diamond as u32, r.key.depth as u32, r.cu, r.cv) as usize;
            let mut found = false;
            for _ in 0..8 {
                let s = plan.table[h];
                if s.index == i as u32 {
                    assert_eq!(s.tag, r.diamond as u32 | ((r.key.depth as u32) << 8));
                    found = true;
                    break;
                }
                h = (h + 1) & (TABLE_N - 1);
            }
            assert!(found, "reference {i} not in the table within eight probes");
        }
        assert_eq!(plan.blocks.len() >= 8 * 4, true);
        assert!(plan.want.len() >= 7);
    }
}
