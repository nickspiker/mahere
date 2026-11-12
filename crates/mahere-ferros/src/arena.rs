//! What the map kernel reads, laid out in the layer arena (`kernel.rs` for the layout's offsets): the cells' planes in slots, the tiles and blocks of the frame's plan, the line, land and hypsometric tables, and the frame's constants. Plain words over a slice, so the host tests build the same arena the device does and run the kernel's words over it in valhall's emulator.

use crate::kernel::{self, Consts};
use mahere_engine::colour::lin;
use mahere_engine::plan::{FramePlan, NONE};
use mahere_engine::raster::{FrameLuts, alpha, is_boundary, line_min_depth, loading_noise, rgb3};
use mahere_engine::residency::Pool;
use mahere_tiles::{CellKey, TRI};

/// A cell plane's slot: 131072 texels of 8 bytes.
pub const SLOT: usize = 1 << 20;
/// The most words a kernel may have (ferros payloads/gpu: its code region, 32 pages, less its 16 words of padding).
pub const CODE_WORDS: usize = (32 << 12) / 8 - 16;

/// Where things are for a `w` × `h` frame in an arena of `bytes`: the overlay (at `kernel::OVERLAY`) a screen's words, with a workgroup's slack, and the frame as many when it is the arena's (`Out::Argb8`; the surface is the frame otherwise); the code; then the cells' slots.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub w: usize,
    pub h: usize,
    pub out: kernel::Out,
    pub frame_at: usize,
    pub code_at: usize,
    pub slots_at: usize,
    pub slots: usize,
}

impl Layout {
    pub fn new(w: usize, h: usize, bytes: usize, out: kernel::Out) -> Layout {
        let px = (w * h).next_multiple_of(16);
        let round = |b: usize| b.next_multiple_of(1 << 20);
        let frame_at = round(kernel::OVERLAY + 4 * px);
        let code_at = if out == kernel::Out::Argb8 { round(frame_at + 4 * px) } else { frame_at };
        let slots_at = round(code_at + 8 * (CODE_WORDS + 16));
        // WHY: the frame's size comes from the display; a screen whose overlay and frame leave no room for cells is one this path cannot draw.
        assert!(slots_at + 4 * SLOT <= bytes, "mahere: a {w} x {h} frame leaves the layer arena no room for cells");
        Layout { w, h, out, frame_at, code_at, slots_at, slots: (bytes - slots_at) / SLOT }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Plane {
    Dem,
    Vec,
}

/// What a slot holds: the cell, its plane, the pool's version when it was written (a cell loaded again is written again) and the last frame that drew it.
#[derive(Clone, Copy)]
struct Held {
    key: CellKey,
    plane: Plane,
    version: u64,
    frame: u64,
}

/// The slots' contents, and what the frames so far wrote and could not place.
pub struct Slots {
    held: Vec<Option<Held>>,
    pub frame: u64,
    pub written: u64,
    pub starved: u64,
}

impl Slots {
    pub fn new(n: usize) -> Slots {
        Slots { held: vec![None; n], frame: 0, written: 0, starved: 0 }
    }

    /// The slot holding `key`'s `plane`, written now if it is not there (taking a free slot, else the one drawn longest ago); None when every slot is this frame's.
    pub fn slot_for(&mut self, pool: &Pool, key: CellKey, plane: Plane, version: u64, lay: &Layout, arena: &mut [u32]) -> Option<usize> {
        let frame = self.frame;
        if let Some(i) = self.held.iter().position(|s| matches!(s, Some(h) if h.key == key && h.plane == plane)) {
            let h = self.held[i].as_mut().unwrap();
            h.frame = frame;
            let stale = h.version != version;
            h.version = version;
            if stale {
                self.write(pool, i, key, plane, lay, arena);
            }
            return Some(i);
        }
        let i = match self.held.iter().position(|s| s.is_none()) {
            Some(i) => i,
            None => {
                let (i, h) = self.held.iter().enumerate().map(|(i, s)| (i, s.unwrap())).min_by_key(|(_, h)| h.frame)?;
                if h.frame == frame {
                    return None;
                }
                i
            }
        };
        self.held[i] = Some(Held { key, plane, version, frame });
        self.write(pool, i, key, plane, lay, arena);
        Some(i)
    }

    /// A cell's plane into slot `i`: the terrain's packed texels as they are (elevation, then the normal's east, north and up, 16 bits each), or the vector planes packed two words a texel (line class, coverage, magnitude, the water's coverage; land class and coverage).
    fn write(&mut self, pool: &Pool, i: usize, key: CellKey, plane: Plane, lay: &Layout, arena: &mut [u32]) {
        self.written += 1;
        let at = (lay.slots_at + i * SLOT) / 4;
        let out = &mut arena[at..at + 2 * TRI];
        let e = &pool.map[&key];
        match plane {
            Plane::Dem => match &e.dem {
                Some(d) => {
                    for (k, t) in d.texel.iter().enumerate() {
                        out[2 * k] = *t as u32;
                        out[2 * k + 1] = (*t >> 32) as u32;
                    }
                }
                // No packed texels: no data everywhere.
                None => out.iter_mut().step_by(2).for_each(|w| *w = 0xFFFF),
            },
            Plane::Vec => {
                let z = |p: Option<&Vec<u8>>, k: usize| p.and_then(|v| v.get(k)).copied().unwrap_or(0) as u32;
                let (line, land, water) = (e.line.as_ref(), e.land.as_ref(), e.water.as_ref());
                for k in 0..TRI {
                    out[2 * k] = z(line.map(|c| &c.class), k) | z(line.map(|c| &c.cov), k) << 8 | z(line.map(|c| &c.mag), k) << 16 | z(water.map(|c| &c.cov), k) << 24;
                    out[2 * k + 1] = z(land.map(|c| &c.class), k) | z(land.map(|c| &c.cov), k) << 8;
                }
            }
        }
    }
}

/// Each block's 2 × 2 cells for each role (the terrain, the vector planes), their planes slot-resident; and the elevation range at the blocks' first corners, for the next frame's contour interval.
/// Per block, the two roles' (cu0, cv0, four texel bases, four texel shifts), as `kernel::B_DEM` and `B_VEC` lay them out.
pub fn cells(slots: &mut Slots, pool: &Pool, plan: &FramePlan, version: u64, lay: &Layout, arena: &mut [u32]) -> (Vec<[[u32; 10]; 2]>, Option<(u16, u16)>) {
    slots.frame += 1;
    let depths = [plan.dem_depth as u32, plan.vec_depth as u32];
    let planes = [Plane::Dem, Plane::Vec];
    let top = ((1u64 << 30) - 1) as f64;
    let mut out = Vec::with_capacity(plan.blocks.len());
    let (mut lo, mut hi) = (u16::MAX, 0u16);
    // The refs this frame placed (or could not), so each is looked at once a frame.
    let mut placed: Vec<(u32, u8, Option<usize>)> = Vec::new();
    for b in &plan.blocks {
        let n = b.size as f64;
        let mut corners = [(0u32, 0u32); 4];
        for (j, (x, y)) in [(0.0, 0.0), (n, 0.0), (0.0, n), (n, n)].into_iter().enumerate() {
            let u = (b.u0 as f64 + b.du_dx as f64 * x + b.du_dy as f64 * y + b.twist_u as f64 * x * y).clamp(0.0, top);
            let v = (b.v0 as f64 + b.dv_dx as f64 * x + b.dv_dy as f64 * y + b.twist_v as f64 * x * y).clamp(0.0, top);
            corners[j] = (u as u32, v as u32);
        }
        let mut rec = [[0u32; 10]; 2];
        for role in 0..2 {
            let s = 30 - depths[role];
            let cu0 = corners.iter().map(|c| c.0 >> s).min().unwrap();
            let cv0 = corners.iter().map(|c| c.1 >> s).min().unwrap();
            rec[role][0] = cu0;
            rec[role][1] = cv0;
            for k in 0..4u32 {
                let (cu, cv) = (cu0 + (k & 1), cv0 + (k >> 1));
                // The first (finest) candidate whose cell holds this cell of the wanted depth.
                let found = b.cand[role].iter().copied().take_while(|&r| r != NONE).find(|&r| {
                    let rf = &plan.refs[r as usize];
                    let up = depths[role] - rf.key.depth as u32;
                    rf.diamond as u32 == b.diamond && cu >> up == rf.cu && cv >> up == rf.cv
                });
                let Some(r) = found else { continue };
                let slot = match placed.iter().find(|e| e.0 == r && e.1 == role as u8) {
                    Some(e) => e.2,
                    None => {
                        let slot = slots.slot_for(pool, plan.refs[r as usize].key, planes[role], version, lay, arena);
                        if slot.is_none() {
                            slots.starved += 1;
                        }
                        placed.push((r, role as u8, slot));
                        slot
                    }
                };
                if let Some(i) = slot {
                    rec[role][2 + k as usize] = ((lay.slots_at + i * SLOT) / 4) as u32;
                    rec[role][6 + k as usize] = 22 - plan.refs[r as usize].key.depth as u32;
                }
            }
        }
        // The terrain's elevation at the block's first corner, from its finest candidate.
        if let Some(&r) = b.cand[0].first().filter(|&&r| r != NONE) {
            let rf = &plan.refs[r as usize];
            let ts = 22 - rf.key.depth as u32;
            let (u, v) = corners[0];
            let m = (1u32 << ts) - 1;
            let ti = ((((v >> ts) & 255) << 8 | ((u >> ts) & 255)) << 1 | (((u & m) + (v & m)) >> ts) & 1) as usize;
            if let Some(q) = pool.map.get(&rf.key).and_then(|e| e.elev_q_at(ti)) {
                if q != mahere_tiles::ELEV_NODATA {
                    lo = lo.min(q);
                    hi = hi.max(q);
                }
            }
        }
        out.push(rec);
    }
    (out, (lo <= hi).then_some((lo, hi)))
}

/// The tiles' words (one row past the screen's, block 0: the last workgroup's invocations past the last pixel read there) and the blocks' records.
pub fn blocks(plan: &FramePlan, cand: &[[[u32; 10]; 2]], lay: &Layout, arena: &mut [u32]) -> Result<(), String> {
    if plan.blocks.len() > kernel::MAX_BLOCKS {
        return Err(format!("{} blocks, more than the arena's {}", plan.blocks.len(), kernel::MAX_BLOCKS));
    }
    let bw = lay.w.div_ceil(32);
    let bh = lay.h.div_ceil(32);
    let tiles = &mut arena[kernel::TILES / 4..kernel::TILES / 4 + bw * (bh + 1)];
    tiles.fill(0);
    let mut i = 0;
    while i < plan.blocks.len() {
        let b = &plan.blocks[i];
        let t = (b.y as usize / 32) * bw + b.x as usize / 32;
        if b.size == 32 {
            tiles[t] = i as u32;
            i += 1;
        } else {
            // A straddling tile: its 64 sub-blocks of 4 px, in rows.
            tiles[t] = i as u32 | kernel::TILE_STRADDLE;
            i += 64;
        }
    }
    let at = kernel::BLOCKS / 4;
    for (k, (b, c)) in plan.blocks.iter().zip(cand).enumerate() {
        let r = &mut arena[at + k * kernel::BLOCK_WORDS..at + (k + 1) * kernel::BLOCK_WORDS];
        r[kernel::B_XY] = b.x | b.y << 16;
        r[kernel::B_DIAMOND] = b.diamond;
        r[kernel::B_U0] = b.u0;
        r[kernel::B_U0 + 1] = b.v0;
        for (j, f) in [b.du_dx, b.dv_dx, b.du_dy, b.dv_dy, b.twist_u, b.twist_v].into_iter().enumerate() {
            r[kernel::B_STEPS + j] = f.to_bits();
        }
        r[kernel::B_DEM..kernel::B_DEM + 10].copy_from_slice(&c[0]);
        r[kernel::B_VEC..kernel::B_VEC + 10].copy_from_slice(&c[1]);
        // The loading mesh's colours for the terrain's 2 × 2 cells at the wanted depth, each triangle (raster.rs `loading_noise`).
        for k in 0..4u32 {
            for half in 0..2u32 {
                let n = loading_noise(b.diamond, plan.dem_depth as u32, c[0][0] + (k & 1), c[0][1] + (k >> 1), half);
                r[kernel::B_NOISE + (2 * k + half) as usize] = n[0] as u32 | (n[1] as u32) << 8 | (n[2] as u32) << 16;
            }
        }
    }
    Ok(())
}

/// The line and land tables (the line gates follow the depth and the mask, so every frame), and the hypsometric ramp when `ramp`.
pub fn tables(luts: &FrameLuts, plan: &FramePlan, ramp: bool, arena: &mut [u32]) {
    let st = &luts.style;
    let mask = luts.mask;
    let at = kernel::LINE_TAB / 4;
    for cls in 0..22 {
        let c = st.line[cls];
        let l = lin(rgb3(c));
        // The class's opacity where it draws (raster.rs `compose_rgb`'s gate: lines on, boundaries on or not one, the depth past the class's least; a waterway's magnitude the kernel's to gate), a boundary's at 0.55.
        let shown = mask.line && (mask.boundaries || !is_boundary(cls)) && (cls == 12 || plan.dem_depth >= line_min_depth(cls, 0));
        let a = if shown { alpha(c) * if is_boundary(cls) { 0.55 } else { 1.0 } } else { 0.0 };
        arena[at + 4 * cls..at + 4 * cls + 4].copy_from_slice(&[l[0].to_bits(), l[1].to_bits(), l[2].to_bits(), a.to_bits()]);
    }
    let at = kernel::LAND_TAB / 4;
    for k in 0..14 {
        let c = st.land[k];
        let l = lin(rgb3(c));
        arena[at + 4 * k..at + 4 * k + 4].copy_from_slice(&[l[0].to_bits(), l[1].to_bits(), l[2].to_bits(), alpha(c).to_bits()]);
    }
    if ramp {
        // The g2t codes' working values, fluor's table for its 8-bit pixels (host/panel.rs `q14_from_g2t8`): the overlay's codes as the surface takes them.
        let at = kernel::CODE_TAB / 4;
        for c in 0..256 {
            arena[at + c] = ((16384.0 * (c as f32 / 256.0).atanh() + 0.5) as u32 as f32).to_bits();
        }
        let at = kernel::HYPSO_TAB / 4;
        for (k, c) in luts.hypso.iter().enumerate() {
            let l = lin(*c);
            arena[at + 4 * k..at + 4 * k + 3].copy_from_slice(&[l[0].to_bits(), l[1].to_bits(), l[2].to_bits()]);
        }
    }
}

/// The frame's constants for the kernel.
pub fn consts(luts: &FrameLuts, plan: &FramePlan, lay: &Layout, overlay: bool) -> Consts {
    let st = &luts.style;
    let l3 = |c: [u8; 4]| lin(rgb3(c));
    let l4 = |c: [u8; 4]| {
        let l = lin(rgb3(c));
        [l[0], l[1], l[2], alpha(c)]
    };
    let d = plan.dem_depth;
    Consts {
        w: lay.w as u32,
        bw: lay.w.div_ceil(32) as u32,
        dem_depth: plan.dem_depth as u32,
        vec_depth: plan.vec_depth as u32,
        mask: luts.mask.bits(),
        light: luts.light.k,
        sun: luts.sun,
        pole: [plan.pole[0], plan.pole[1], plan.pole[2]],
        disk: [plan.globe[0], plan.globe[1], plan.globe[2]],
        limb: plan.globe[3] > 0.5,
        no_dem: l3(st.no_dem),
        bg: l3(st.bg),
        sea: l3(st.sea),
        flat: l3(st.flat),
        water: l4(st.water),
        contour: l4(st.contour),
        contour_index: l4(st.contour_index),
        interval: luts.contours.interval,
        m_per_px: luts.contours.m_per_px,
        index_every: luts.contours.index_every,
        // raster.rs `line_min_depth`'s waterway rows: the least magnitude a waterway draws at, at this depth.
        waterway_mag: if d >= 8 {
            0
        } else if d >= 6 {
            64
        } else if d >= 3 {
            128
        } else {
            200
        },
        overlay,
        overlay_at: (kernel::OVERLAY / 4) as u32,
        out: lay.out,
    }
}
