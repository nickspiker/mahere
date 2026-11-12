//! What the map kernel reads, laid out in the layer arena (`kernel.rs` for the layout's offsets): the cells' planes in slots, the faces in view and the page table of the frame's plan, the line, land and hypsometric tables, and the frame's constants. Plain words over a slice, so the host tests build the same arena the device does and run the kernel's words over it in valhall's emulator.

use crate::kernel::{self, Consts};
use mahere_engine::colour::lin;
use mahere_engine::plan::NONE;
use mahere_engine::viewplan::ViewPlan;
use mahere_engine::raster::{FrameLuts, alpha, is_boundary, line_min_depth, rgb3};
use mahere_engine::residency::Pool;
use mahere_engine::PixelBox;
use mahere_coord::{FACE_COUNT, FaceFold, face_fold, face_map};
use mahere_tiles::{CellKey, TRI};

/// A cell plane's slot: 131072 texels of 8 bytes.
pub const SLOT: usize = 1 << 20;
/// The most words a kernel may have (ferros payloads/gpu: its code region, 32 pages, less its 16 words of padding).
pub const CODE_WORDS: usize = (32 << 12) / 8 - 16;

/// Where things are for a `w` × `h` frame in an arena of `bytes`: two frame regions (`regions`, the frames alternating between them, so the host writes the next frame's tables, faces, squares, overlay and code while the GPU runs this one's), or one when two leave no room for cells (both `regions` 0, the frames then not pipelined); each its tables, faces and squares (from 0), the overlay (at `kernel::OVERLAY`) a screen's words with a workgroup's slack, the frame as many when it is the arena's (`Out::Argb8`; the surface is the frame otherwise) and the code; then the cells' slots, which the regions share. `frame_at` and `code_at` are within a region.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub w: usize,
    pub h: usize,
    pub out: kernel::Out,
    pub frame_at: usize,
    pub code_at: usize,
    pub slots_at: usize,
    pub slots: usize,
    /// Each frame region's start, bytes: the same twice when there is one.
    pub regions: [usize; 2],
}

/// The fewest cell slots a layout is made with: two cells, each its terrain's slot and its vector planes'.
pub const MIN_SLOTS: usize = 1 << 2;

impl Layout {
    /// Two frame regions when they leave `bytes` room for `MIN_SLOTS` cell slots, else one (the frames not pipelined: ferros gives a kist the largest block free, which can be 32 MiB while the system's clients hold theirs, and at the 8a's 1080 x 2400 a region is 19 MiB); None when even one leaves no room: the block is too small for this screen, and the CPU raster draws.
    pub fn new(w: usize, h: usize, bytes: usize, out: kernel::Out) -> Option<Layout> {
        let px = (w * h).next_multiple_of(16);
        let round = |b: usize| b.next_multiple_of(1 << 20);
        let frame_at = round(kernel::OVERLAY + 4 * px);
        let code_at = if out == kernel::Out::Argb8 { round(frame_at + 4 * px) } else { frame_at };
        let region = round(code_at + 8 * (CODE_WORDS + 16));
        // WHY: the frame's size comes from the display and the block's from the kernel; a screen whose regions leave no room for cells is one this layout cannot draw.
        let (slots_at, regions) = if 2 * region + MIN_SLOTS * SLOT <= bytes {
            (2 * region, [0, region])
        } else if region + MIN_SLOTS * SLOT <= bytes {
            (region, [0, 0])
        } else {
            return None;
        };
        Some(Layout { w, h, out, frame_at, code_at, slots_at, slots: (bytes - slots_at) / SLOT, regions })
    }

    /// Two frame regions: a frame can be planned while the GPU draws the last.
    pub fn two(&self) -> bool {
        self.regions[1] != self.regions[0]
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
                // Neither this frame's nor the last's: the GPU may still be running the last frame's kernel over its cells while this one is planned.
                if h.frame + 1 >= frame {
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

/// The page table in frame region `region`: each role's squares, each the slot of the cell standing in for it (written into its slot when it is not there yet), and the faces' rows when there is more than one face in view.
pub fn view(slots: &mut Slots, pool: &Pool, plan: &ViewPlan, version: u64, lay: &Layout, region: usize, arena: &mut [u32]) -> Result<(), String> {
    slots.frame += 1;
    let total = plan.roles[0].squares.len() + plan.roles[1].squares.len();
    if total > kernel::MAX_SQUARES {
        return Err(format!("{total} squares in the page table, more than the arena's {}", kernel::MAX_SQUARES));
    }
    let base = lay.regions[region] / 4;
    let mut at = base + kernel::SQUARES / 4;
    for (t, plane) in plan.roles.iter().zip([Plane::Dem, Plane::Vec]) {
        // Each stand-in's square word, the cells holding a point of the screen first: a full arena starves a box's margin before what shows.
        let mut word = vec![0u32; t.refs.len()];
        for shown in [true, false] {
            for (i, r) in t.refs.iter().enumerate().filter(|(_, r)| r.on_screen == shown) {
                match slots.slot_for(pool, r.key, plane, version, lay, arena) {
                    Some(s) => word[i] = ((lay.slots_at + s * SLOT) / 4) as u32 | (22 - r.key.depth as u32) << kernel::SQUARE_SHIFT,
                    None => slots.starved += 1,
                }
            }
        }
        for (k, &sq) in t.squares.iter().enumerate() {
            arena[at + k] = if sq == NONE { 0 } else { word[sq as usize] };
        }
        at += t.squares.len();
    }
    let (faces, n, _, fold) = faces(plan, lay, region);
    let at = base + kernel::FACES / 4;
    match fold {
        // By fold: a row for each octant and centre, its face's.
        Some((_, rows)) => {
            for (e, &j) in rows.iter().enumerate() {
                arena[at + e * kernel::FACE_WORDS..at + (e + 1) * kernel::FACE_WORDS].copy_from_slice(&faces[j as usize].row);
            }
        }
        None if n > 1 => {
            for (j, f) in faces[..n].iter().enumerate() {
                arena[at + j * kernel::FACE_WORDS..at + (j + 1) * kernel::FACE_WORDS].copy_from_slice(&f.row);
            }
        }
        None => {}
    }
    Ok(())
}

/// The cosine of the farthest a face's centre may be from the camera point for the near form (50 degrees): there the camera point's (u, v) on the face lies within about half a diamond of it, so its whole part and every pixel's offset from it fit an i32.
const NEAR_COS: f64 = 0.64;

/// The faces in view as the kernel takes them, how many, and whether in the near form (`kernel::Consts::near`): each face's centroid and score offset, its (u, v) coefficients, its loading-mesh seed and its boxes, for frame region `region`.
/// A face whose diamond has no box (a sliver of it at the screen's edge holding none of the plan's points) is left out: its pixels take a neighbour's face, held inside that diamond.
/// Far from the camera point with more than `FOLD_FROM` faces, the fold too: its constants and each of its rows' face (an index into the faces).
pub fn faces(plan: &ViewPlan, lay: &Layout, region: usize) -> ([kernel::FaceConsts; kernel::MAX_FACES], usize, bool, Option<(kernel::Fold, [u8; kernel::FOLD_ROWS])>) {
    let units = (1u64 << 30) as f64;
    let kept: Vec<_> = plan.faces.iter().filter(|f| plan.roles[0].boxes[f.diamond as usize].is_some()).collect();
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    // The face plane's normal points in or out by the face's winding: its angle to the camera point either way.
    let near = kept.iter().all(|f| f.n[2].abs() / len(f.n) >= NEAR_COS);
    let top = kept.iter().map(|f| f.centroid[2]).fold(f64::MIN, f64::max);
    let squares = lay.regions[region] / 4 + kernel::SQUARES / 4;
    let starts = [squares, squares + plan.roles[0].squares.len()];
    let mut out = [kernel::FaceConsts { score: [0.; 4], row: [0; kernel::FACE_WORDS] }; kernel::MAX_FACES];
    let bits = |x: f64| (x as f32).to_bits();
    for (f, o) in kept.iter().zip(out.iter_mut()) {
        let c = f.centroid;
        if near {
            // Measured from the camera point C = (0, 0, 1): u(C + δ) − u(C) = (a · δ) / (1 + m · δ), a = (g (n · C) − n (g · C)) / (n · C)², m = n / (n · C); the score less the largest centroid's z, so the faces near C compare small numbers.
            let d = f.n[2];
            for (k, (at, g)) in [(kernel::F_U, f.gu), (kernel::F_V, f.gv)].into_iter().enumerate() {
                let gc = g[2];
                for j in 0..3 {
                    o.row[at + j] = bits((g[j] * d - f.n[j] * gc) / (d * d) * units);
                }
                let r = gc / d * units;
                let whole = r.floor();
                o.row[kernel::F_FRAC + k] = bits(r - whole);
                o.row[kernel::F_REF + k] = whole as i64 as i32 as u32;
            }
            for j in 0..3 {
                o.row[kernel::F_DEN + j] = bits(f.n[j] / d);
            }
            o.score = [c[0] as f32, c[1] as f32, c[2] as f32, (c[2] - top) as f32];
        } else {
            for j in 0..3 {
                o.row[kernel::F_U + j] = bits(f.gu[j] * units);
                o.row[kernel::F_V + j] = bits(f.gv[j] * units);
                o.row[kernel::F_DEN + j] = bits(f.n[j]);
            }
            o.score = [c[0] as f32, c[1] as f32, c[2] as f32, 0.];
        }
        o.row[kernel::F_SEED] = (f.diamond as u32).wrapping_mul(0x27D4_EB2F) ^ (plan.dem_depth as u32).wrapping_mul(0xC2B2_AE3D);
        for (role, t) in plan.roles.iter().enumerate() {
            let at = kernel::F_BOX + role * kernel::F_BOX_WORDS;
            // Both roles' boxes come from the same points, so a diamond with one has both; none, all its pixels outside.
            let b = t.boxes[f.diamond as usize].unwrap_or_default();
            o.row[at..at + kernel::F_BOX_WORDS].copy_from_slice(&[b.cu0, b.cv0, b.w, b.h, (starts[role] + b.at as usize) as u32]);
        }
    }
    let fold = (!near && kept.len() > FOLD_FROM).then(|| {
        // The fold's frame on view coordinates: view = basis · world, so world = basisᵀ · view, and the fold's turn after it.
        // The fold and the faces' centres in the world, made once (each frame's twice was 115 us of the 8a's planning at the globe).
        static FOLD: std::sync::OnceLock<(FaceFold, [[f64; 3]; FACE_COUNT])> = std::sync::OnceLock::new();
        let (fd, world) = FOLD.get_or_init(|| (face_fold(), core::array::from_fn(|f| face_map(f).centroid)));
        let b = plan.basis;
        let turn = core::array::from_fn(|i| core::array::from_fn(|j| (0..3).map(|k| fd.turn[i][k] * b[j][k]).sum::<f64>() as f32));
        let centres = fd.centres.map(|c| c.map(|x| x as f32));
        // Each row's face, or for a face out of view (or a sliver with no box) the face in view whose centre is nearest its centre: a neighbour across the edge its pixels lie along.
        let rows = fd.face.map(|f| {
            let c = world[f as usize];
            let dot = |j: usize| {
                let k = world[kept[j].face as usize];
                c[0] * k[0] + c[1] * k[1] + c[2] * k[2]
            };
            (0..kept.len()).max_by(|&a, &b| dot(a).total_cmp(&dot(b))).unwrap() as u8
        });
        (kernel::Fold { turn, centres }, rows)
    });
    (out, kept.len(), near, fold)
}

/// More faces in view than this, far from the camera point: the faces are found by fold (`kernel::Fold`: about 35 instructions a pixel, the search about six a face).
const FOLD_FROM: usize = 1 << 2;

/// The line and land tables (the line gates follow the depth and the mask, so every frame), and the hypsometric ramp when `ramp`, in frame region `region`.
pub fn tables(luts: &FrameLuts, plan: &ViewPlan, ramp: bool, lay: &Layout, region: usize, arena: &mut [u32]) {
    let base = lay.regions[region] / 4;
    let st = &luts.style;
    let mask = luts.mask;
    // A row in f16 pairs, as the kernel's colours are (`kernel::C`): (red, green), then (blue, the opacity over 255, which a coverage byte multiplies).
    let row = |l: [f32; 3], a: f32| [h2(l[0], l[1]), h2(l[2], a / 255.)];
    let at = base + kernel::LINE_TAB / 4;
    for cls in 0..22 {
        let c = st.line[cls];
        let l = lin(rgb3(c));
        // The class's opacity where it draws (raster.rs `compose_rgb`'s gate: lines on, boundaries on or not one, the depth past the class's least; a waterway's magnitude the kernel's to gate), a boundary's at 0.55.
        let shown = mask.line && (mask.boundaries || !is_boundary(cls)) && (cls == 12 || plan.dem_depth >= line_min_depth(cls, 0));
        let a = if shown { alpha(c) * if is_boundary(cls) { 0.55 } else { 1.0 } } else { 0.0 };
        arena[at + kernel::ROW * cls..at + kernel::ROW * (cls + 1)].copy_from_slice(&row(l, a));
    }
    let at = base + kernel::LAND_TAB / 4;
    for k in 0..14 {
        let c = st.land[k];
        arena[at + kernel::ROW * k..at + kernel::ROW * (k + 1)].copy_from_slice(&row(lin(rgb3(c)), alpha(c)));
    }
    if ramp {
        // The g2t codes' working values, fluor's table for its 8-bit pixels (host/panel.rs `q14_from_g2t8`): the overlay's codes as the surface takes them.
        let at = base + kernel::CODE_TAB / 4;
        for c in 0..256 {
            arena[at + c] = ((16384.0 * (c as f32 / 256.0).atanh() + 0.5) as u32 as f32).to_bits();
        }
        // The ramp's rows (red, green), (blue, blue): the kernel's colours as they are, no swizzle.
        let at = base + kernel::HYPSO_TAB / 4;
        for (k, c) in luts.hypso.iter().enumerate() {
            let l = lin(*c);
            arena[at + kernel::ROW * k..at + kernel::ROW * (k + 1)].copy_from_slice(&[h2(l[0], l[1]), h2(l[2], l[2])]);
        }
    }
}

/// Two f16 in a word, `lo` in its low half: a v2f16 the kernel reads as it is.
fn h2(lo: f32, hi: f32) -> u32 {
    valhall::f32_to_f16(lo) as u32 | (valhall::f32_to_f16(hi) as u32) << 16
}

/// The frame's constants for the kernel, its tables and overlay in frame region `region`.
pub fn consts(luts: &FrameLuts, plan: &ViewPlan, lay: &Layout, region: usize, overlay: bool, ring: Option<kernel::Ring>) -> Consts {
    let st = &luts.style;
    let l3 = |c: [u8; 4]| lin(rgb3(c));
    let l4 = |c: [u8; 4]| {
        let l = lin(rgb3(c));
        [l[0], l[1], l[2], alpha(c)]
    };
    let d = plan.dem_depth;
    let (faces, face_count, near, fold) = faces(plan, lay, region);
    let fold = fold.map(|f| f.0);
    let k = 1. / plan.radius;
    Consts {
        w: lay.w as u32,
        view: [k as f32, (-plan.centre[0] * k) as f32, (plan.centre[1] * k) as f32],
        near,
        faces,
        face_count,
        faces_at: ((lay.regions[region] + kernel::FACES) / 4) as u32,
        fold,
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
        ring,
        overlay,
        overlay_at: ((lay.regions[region] + kernel::OVERLAY) / 4) as u32,
        region: (lay.regions[region] / 4) as u32,
        out: lay.out,
        // The display's bake is the frame loop's to set (gpu.rs `prepare`).
        bake: None,
    }
}

/// The overlay's words over box `b` of a `w`-wide screen, the rest of `out` (a screen's words) untouched: the panel's pixels (fluor's; None when it has none) in front of the marks (0xRRGGBB over black, a mark's brightest channel its opacity), premultiplied RGBA bytes, the box's rows in bands on every core.
pub fn overlay_sum(marks: &[u32], panel: Option<&[u32]>, w: usize, b: PixelBox, out: &mut [u32]) {
    use rayon::prelude::*;
    if b.is_empty() {
        return;
    }
    out[b.y0 * w..b.y1 * w].par_chunks_mut(w << 4).enumerate().for_each(|(band, rows)| {
        for (k, row) in rows.chunks_mut(w).enumerate() {
            let at = (b.y0 + (band << 4) + k) * w;
            for x in b.x0..b.x1 {
                let i = at + x;
                let (m, p) = (marks[i], panel.map_or(0, |p| p[i]));
                if m == 0 && p >> 24 == 0 {
                    row[x] = 0;
                    continue;
                }
                let (mr, mg, mb) = ((m >> 16) & 255, (m >> 8) & 255, m & 255);
                let ma = mr.max(mg).max(mb);
                let (r, g, bl, a) = if p >> 24 == 0 { (0, 0, 0, 0) } else { mahere_panel::Panel::premul_pub(p) };
                let keep = 255 - a as u32;
                let ch = |c: u8, under: u32| c as u32 + under * keep / 255;
                row[x] = ch(r, mr) | ch(g, mg) << 8 | ch(bl, mb) << 16 | ch(a, ma) << 24;
            }
        }
    });
}

/// The box of the panel's pixels that show anything (opacity above zero), rows in bands on every core: what a repainted panel can have changed, with the box it had before.
pub fn panel_box(panel: &[u32], w: usize) -> PixelBox {
    use rayon::prelude::*;
    panel
        .par_chunks(w << 4)
        .enumerate()
        .map(|(band, rows)| {
            let mut b = PixelBox::EMPTY;
            for (k, row) in rows.chunks(w).enumerate() {
                // Most rows show nothing: the row's words or'd together (vectorised) say so before any search.
                if row.iter().fold(0, |a, p| a | p) >> 24 == 0 {
                    continue;
                }
                let x0 = row.iter().position(|p| p >> 24 != 0).unwrap();
                let x1 = row.iter().rposition(|p| p >> 24 != 0).unwrap() + 1;
                let y = (band << 4) + k;
                b = b.union(PixelBox { x0, y0: y, x1, y1: y + 1 });
            }
            b
        })
        .reduce(|| PixelBox::EMPTY, PixelBox::union)
}
