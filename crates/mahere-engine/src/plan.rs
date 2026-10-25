//! The frame plan: what a GPU needs to draw a frame, computed on the CPU without touching a pixel. The 32 px block lattice with exact dymaxion corners (the same lattice the CPU raster steps inside), every resident cell as a reference with the planes it carries, and a hash page table over those references keyed by (diamond, depth, cell) — so a shader resolves any pixel's cell by lookup, climbing parents exactly as the CPU's probe does, with no per-block reference lists to miss a sliver.
//!
//! Diamond-straddling blocks are subdivided into 4 px sub-blocks with exact corners restricted to the corner diamonds; a sub-block whose corners still disagree takes its first corner's diamond for all four (a seam error under four pixels wide).

use mahere_coord::{Coord, uv_to_lat_lon};
use mahere_tiles::{CellKey, TEX};
use rayon::prelude::*;


use crate::Camera;
use crate::raster::{BLOCK, CornerPt, MIN_DEPTH, corner, raw_of};
use crate::residency::Pool;

pub const NONE: u32 = u32::MAX;
/// A table slot for a cell the loader confirmed does not exist, at the frame's terrain depth: the renderer tells "not there" from "not here yet" and draws the loading lattice only for the latter.
pub const ABSENT: u32 = u32::MAX - 1;

/// Side of a straddle sub-block, pixels.
pub const SUB: usize = 4;

/// Page table slots; a power of two, several times the largest resident set.
pub const TABLE_N: usize = 16384;

/// Depths at which the view's neighbouring cells are fetched ahead, so a zoom out is already covered.
pub const PREFETCH_DEPTH: u8 = 10;

pub const FLAG_DEM: u32 = 1;
pub const FLAG_LINE: u32 = 2;
pub const FLAG_LAND: u32 = 4;
pub const FLAG_WATER: u32 = 8;
pub const FLAG_IMG: u32 = 16;
/// A user cell (ink and selection), its own plane; tagged in the table at its depth plus 64 so it never collides with the map's cell at the same key.
pub const FLAG_INK: u32 = 32;
/// The overlays: land ownership and motor vehicle use, one GPU plane between them.
pub const FLAG_OWN: u32 = 64;
pub const FLAG_MVUM: u32 = 128;
pub const INK_TAG_DEPTH: u32 = 64;

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
    /// The cell is one this frame's view wants (on screen, or a parent or neighbour of one), not merely resident.
    pub wanted: bool,
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
    /// The references this block's pixels draw from, resolved here once instead of searched for at every pixel (2026-10-21: a Pixel 3a's GPU took a second a frame, each pixel probing the table for its terrain, its lines and its imagery): for the terrain, the vector planes and the imagery, the cells its corners lie in at the wanted depth, or the nearest coarser ones carrying the plane, up to four each, [`NONE`] after the last. A pixel matching none of them (a block touching more cells) searches as before.
    pub cand: [[u32; 4]; 3],
    /// The user's cells (ink and selection) its corners lie in, finest first, [`NONE`] after the last: a block with none skips the user's layer outright (2026-10-21: every pixel walked a dozen depths probing for ink that was not there).
    pub ink: [u32; 4],
}

/// Which planes each kind of candidate is for, and the depth it is wanted at: the terrain at the terrain's depth, the vectors and the imagery at the vectors'.
pub const FLAG_VEC: u32 = FLAG_LINE | FLAG_LAND | FLAG_WATER | FLAG_OWN | FLAG_MVUM;

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
    pub straddle_blocks: usize,
    pub dem_depth: u8,
    pub vec_depth: u8,
    /// The dem's texels are wider than about a pixel and a half: the shader interpolates elevation between texels so contours stay smooth past the base depth.
    pub magnified: bool,
    /// The largest magnitude per line class among the cells in view: every line is drawn on a linear scale up to its class's, so a view of creeks still has a brightest creek and a view of lanes a boldest lane.
    pub line_mag_hi: [u8; 32],
    /// The globe's disk on the screen: centre x, y and radius in screen pixels, and whether the limb is on the screen at all (else the shader skips the test).
    pub globe: [f32; 4],
    /// The world's pole axis in view coordinates, for lighting every fragment from where it stands on the ball.
    pub pole: [f32; 4],
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
        cand: [[NONE; 4]; 3],
        ink: [NONE; 4],
    }
}

/// Plan a frame: the lattice, the references, the table and the desired set. The elevation range in view comes from the frame drawn (the GPU's gather, the CPU's compose), not from here.
pub fn plan_frame(w: usize, h: usize, cam: &Camera, pool: &Pool, dem_depth: u8, vec_depth: u8, user: &crate::ink::UserLayer) -> FramePlan {
    let bw = w.div_ceil(BLOCK);
    let bh = h.div_ceil(BLOCK);
    let corners: Vec<CornerPt> = (0..(bw + 1) * (bh + 1))
        .into_par_iter()
        .map(|i| {
            let (cx, cy) = (i % (bw + 1), i / (bw + 1));
            corner(cam, (cx * BLOCK) as f64, (cy * BLOCK) as f64, w, h)
        })
        .collect();

    // Desired set and the elevation range, from the lattice. The corners dedupe to their finest cells first (a few hundred from a few thousand corners), then each walks its parents; at the coarse depths below the view's the ring of neighbours comes too, so a zoom out lands on cells already resident instead of a blank screen (Nick 2026-10-06), and they are small.
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
            // Not at the view's own depth, the most cells by far: a zoom out lands on the coarser rings, and a pan's next cells are prefetched into the store (2026-10-24: at the south pole, where five diamonds meet, the ring at the view's depth doubled the resident cells and a Pixel 3a's GPU arrays with them, and it was killed for memory).
            if d <= PREFETCH_DEPTH && d < fine {
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

    // References: every resident cell with a plane, and the table over them. The table is a quarter full at most: a resident set past that (a far zoom over the globe) refers the cells this frame wants first and leaves the rest, so the probes stay short and the insert can never spin.
    let mut refs: Vec<PlanRef> = Vec::with_capacity(pool.map.len().min(TABLE_N / 4));
    let mut table = vec![TableSlot { tag: 0, cu: 0, cv: 0, index: NONE }; TABLE_N];
    let mut keys: Vec<&CellKey> = pool.map.keys().collect();
    keys.sort();
    if keys.len() > TABLE_N / 4 {
        keys.sort_by_key(|k| !desired.contains(k));
        keys.truncate(TABLE_N / 4);
    }
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
        if p & crate::residency::PRESENT_OWN != 0 {
            flags |= FLAG_OWN;
        }
        if p & crate::residency::PRESENT_MVUM != 0 {
            flags |= FLAG_MVUM;
        }
        let (cu, cv) = key.grid();
        if flags == 0 {
            // Confirmed absent at the wanted depth: a marker in the table and nothing to refer to.
            if e.is_absent() && key.depth == dem_depth {
                let mut h = table_hash(key.diamond() as u32, key.depth as u32, cu as u32, cv as u32) as usize;
                for _ in 0..TABLE_N {
                    if table[h].index == NONE {
                        table[h] = TableSlot { tag: key.diamond() as u32 | ((key.depth as u32) << 8), cu: cu as u32, cv: cv as u32, index: ABSENT };
                        break;
                    }
                    h = (h + 1) & (TABLE_N - 1);
                }
            }
            continue;
        }
        let diamond = key.diamond();
        let index = refs.len() as u32;
        refs.push(PlanRef { key: *key, diamond, cu: cu as u32, cv: cv as u32, flags, jac: if flags & FLAG_DEM != 0 { jacobian(*key) } else { [0.0; 5] }, wanted: desired.contains(key) });
        let mut h = table_hash(diamond as u32, key.depth as u32, cu as u32, cv as u32) as usize;
        for _ in 0..TABLE_N {
            if table[h].index == NONE {
                table[h] = TableSlot { tag: diamond as u32 | ((key.depth as u32) << 8), cu: cu as u32, cv: cv as u32, index };
                break;
            }
            h = (h + 1) & (TABLE_N - 1);
        }
    }

    // Each block's candidates, from its four corners: the cell each lies in at the wanted depth, else the nearest coarser one carrying the plane (the shader's own search, done once a block). A block is 32 px and a cell hundreds, so its corners meet at most four cells a depth.
    let index: rustc_hash::FxHashMap<CellKey, u32> = refs.iter().enumerate().map(|(i, r)| (r.key, i as u32)).collect();
    let resolve = |raw: u64, want: u8, need: u32| -> u32 {
        let mut d = want;
        loop {
            let key = CellKey { depth: d, prefix: raw >> (60 - 2 * d as u32) };
            if let Some(&i) = index.get(&key) {
                if refs[i as usize].flags & need != 0 {
                    return i;
                }
            }
            if d <= MIN_DEPTH {
                return NONE;
            }
            d -= 1;
        }
    };
    let roles = [(dem_depth, FLAG_DEM), (vec_depth, FLAG_VEC), (vec_depth, FLAG_IMG)];
    let top = ((1u64 << 30) - 1) as f64;
    for b in &mut blocks {
        let n = b.size as f64;
        // The corners for every role; the edges' midpoints and the centre too for the vector planes, whose cell a block drawn from composites must hold (2026-10-21: the cell grid lies skewed on the screen, and a cell poking in along a block's edge between two corners was missed).
        let h = n / 2.0;
        for (i, (x, y)) in [(0.0, 0.0), (n, 0.0), (0.0, n), (n, n), (h, 0.0), (0.0, h), (n, h), (h, n), (h, h)].into_iter().enumerate() {
            let u = (b.u0 as f64 + b.du_dx as f64 * x + b.du_dy as f64 * y + b.twist_u as f64 * x * y).clamp(0.0, top);
            let v = (b.v0 as f64 + b.dv_dx as f64 * x + b.dv_dy as f64 * y + b.twist_v as f64 * x * y).clamp(0.0, top);
            let raw = raw_of(b.diamond as u8, (u as i64) << 16, (v as i64) << 16);
            for (k, &(want, need)) in roles.iter().enumerate() {
                if i >= 4 && k != 1 {
                    continue;
                }
                let r = resolve(raw, want, need);
                let c = &mut b.cand[k];
                if r != NONE && !c.contains(&r) {
                    if let Some(slot) = c.iter_mut().find(|s| **s == NONE) {
                        *slot = r;
                    }
                }
            }
        }
        // Finest first, as the search finds them: a coarse cell standing in for one corner's missing cell also covers its neighbours, and a pixel takes the first candidate holding it (2026-10-21: listed first, the coarse cell drew over the finer ones loaded beside it, a patchwork of stale tiles while cells arrived).
        for c in &mut b.cand {
            c.sort_by_key(|&r| if r == NONE { u32::MAX } else { 64 - refs[r as usize].key.depth as u32 });
        }
    }

    let magnified = crate::raster::texel_px(cam.ppd, dem_depth) >= 1.5;
    let mut line_mag_hi = [0u8; 32];
    for r in refs.iter().filter(|r| r.key.depth == vec_depth || r.key.depth + 1 == vec_depth) {
        for (hi, &m) in line_mag_hi.iter_mut().zip(&pool.map[&r.key].line_mag_max) {
            *hi = (*hi).max(m);
        }
    }
    // The user's cells, every one (they are few), as references of their own under the ink tag.
    for key in user.cells.keys() {
        if refs.len() >= TABLE_N / 4 {
            break;
        }
        let (cu, cv) = key.grid();
        let diamond = key.diamond();
        let index = refs.len() as u32;
        refs.push(PlanRef { key: *key, diamond, cu: cu as u32, cv: cv as u32, flags: FLAG_INK, jac: [0.0; 5], wanted: false });
        let mut h = table_hash(diamond as u32, key.depth as u32 + INK_TAG_DEPTH, cu as u32, cv as u32) as usize;
        for _ in 0..TABLE_N {
            if table[h].index == NONE {
                table[h] = TableSlot { tag: diamond as u32 | ((key.depth as u32 + INK_TAG_DEPTH) << 8), cu: cu as u32, cv: cv as u32, index };
                break;
            }
            h = (h + 1) & (TABLE_N - 1);
        }
    }
    // Each block's user cells, from its corners: the finest user cell each lies in from the terrain's depth up, as the shader's own walk finds it.
    if !user.cells.is_empty() {
        let ink_index: rustc_hash::FxHashMap<CellKey, u32> = refs.iter().enumerate().filter(|(_, r)| r.flags & FLAG_INK != 0).map(|(i, r)| (r.key, i as u32)).collect();
        let top = ((1u64 << 30) - 1) as f64;
        for b in &mut blocks {
            let n = b.size as f64;
            for (x, y) in [(0.0, 0.0), (n, 0.0), (0.0, n), (n, n)] {
                let u = (b.u0 as f64 + b.du_dx as f64 * x + b.du_dy as f64 * y + b.twist_u as f64 * x * y).clamp(0.0, top);
                let v = (b.v0 as f64 + b.dv_dx as f64 * x + b.dv_dy as f64 * y + b.twist_v as f64 * x * y).clamp(0.0, top);
                let raw = raw_of(b.diamond as u8, (u as i64) << 16, (v as i64) << 16);
                let found = (0..=dem_depth).rev().find_map(|d| ink_index.get(&CellKey { depth: d, prefix: raw >> (60 - 2 * d as u32) }).copied());
                if let Some(r) = found {
                    if !b.ink.contains(&r) {
                        if let Some(slot) = b.ink.iter_mut().find(|s| **s == NONE) {
                            *slot = r;
                        }
                    }
                }
            }
            b.ink.sort_by_key(|&r| if r == NONE { u32::MAX } else { 64 - refs[r as usize].key.depth as u32 });
        }
    }
    let globe = [w as f32 * 0.5, h as f32 * 0.5, cam.radius() as f32, cam.limb_visible(w, h) as u32 as f32];
    let b = cam.basis();
    let pole = [b[0][2] as f32, b[1][2] as f32, b[2][2] as f32, 0.0];
    FramePlan { blocks, refs, table, want: desired.into_iter().collect(), straddle_blocks, dem_depth, vec_depth, magnified, line_mag_hi, globe, pole }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blocks_candidate_is_the_cell_the_search_finds() {
        // The ancestors everywhere, and at the view's depth only the cell under the centre: around it the coarser cell stands in, in the same blocks as the fine one.
        let mut pool = Pool::default();
        let c = Coord::from_lat_lon(46.2, -122.19);
        let water = || crate::residency::Entry { present: crate::residency::PRESENT_WATER, water: Some(mahere_tiles::CovCell::new()), ..Default::default() };
        for depth in MIN_DEPTH..=12 {
            let key = CellKey { depth, prefix: c.raw() >> (60 - 2 * depth as u32) };
            pool.map.insert(key, water());
        }
        let cam = Camera::new(46.2, -122.19, 30000.0, 0.3);
        let (w, h) = (640, 480);
        let plan = plan_frame(w, h, &cam, &pool, 12, 12, &crate::ink::UserLayer::default());
        let index: rustc_hash::FxHashMap<CellKey, usize> = plan.refs.iter().enumerate().map(|(i, r)| (r.key, i)).collect();
        let search = |raw: u64| -> u32 {
            for d in (MIN_DEPTH..=12).rev() {
                if let Some(&i) = index.get(&CellKey { depth: d, prefix: raw >> (60 - 2 * d as u32) }) {
                    if plan.refs[i].flags & FLAG_VEC != 0 {
                        return i as u32;
                    }
                }
            }
            NONE
        };
        let mut mixed = 0;
        for b in &plan.blocks {
            let depths: Vec<u8> = b.cand[1].iter().filter(|&&r| r != NONE).map(|&r| plan.refs[r as usize].key.depth).collect();
            if depths.windows(2).any(|p| p[0] != p[1]) {
                mixed += 1;
            }
            assert!(depths.windows(2).all(|p| p[0] >= p[1]), "finest first: {depths:?}");
            let n = b.size as f64;
            for (x, y) in [(0.5, 0.5), (n - 0.5, 0.5), (0.5, n - 0.5), (n - 0.5, n - 0.5), (n * 0.5, n * 0.5)] {
                let u = (b.u0 as f64 + b.du_dx as f64 * x + b.du_dy as f64 * y + b.twist_u as f64 * x * y) as u64;
                let v = (b.v0 as f64 + b.dv_dx as f64 * x + b.dv_dy as f64 * y + b.twist_v as f64 * x * y) as u64;
                let raw = raw_of(b.diamond as u8, (u as i64) << 16, (v as i64) << 16);
                let picked = b.cand[1].iter().copied().take_while(|&r| r != NONE).find(|&r| {
                    let k = plan.refs[r as usize].key;
                    k == CellKey { depth: k.depth, prefix: raw >> (60 - 2 * k.depth as u32) }
                });
                if let Some(p) = picked {
                    assert_eq!(p, search(raw), "the block's pick is the search's cell");
                }
            }
        }
        assert!(mixed > 0, "the view has blocks where a coarse cell stands in beside a fine one");
    }

    #[test]
    fn table_finds_every_reference() {
        let mut pool = Pool::default();
        let c = Coord::from_lat_lon(46.2, -122.19);
        for depth in MIN_DEPTH..=12 {
            let key = CellKey { depth, prefix: c.raw() >> (60 - 2 * depth as u32) };
            pool.map.insert(key, crate::residency::Entry { present: crate::residency::PRESENT_WATER, water: Some(mahere_tiles::CovCell::new()), ..Default::default() });
        }
        let cam = Camera::new(46.2, -122.19, 6000.0, 0.3);
        let plan = plan_frame(256, 128, &cam, &pool, 12, 12, &crate::ink::UserLayer::default());
        assert_eq!(plan.refs.len(), (12 - MIN_DEPTH + 1) as usize);
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
