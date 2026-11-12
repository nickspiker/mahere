//! The map's compose as a ferros GPU kernel: one invocation a screen pixel, recorded from Rust through valhall's SSA recorder and compiled to Mali-G715 words on the device (ferros GPU-KISTS.md: compute only, no triangle raster; the owner, 2026-11-11: "it'd be a 1:1 per px", "no tilting or 3Dness").
//!
//! It computes what the CPU raster's `compose_rgb` computes (mahere-engine raster.rs), one sample a pixel at the same place, from the same cells: the terrain's light from its packed normal in the pixel's own frame on the ball, the elevation tint (sea, seabed, hypsometric or flat), land cover, the waterways under the water, the water, the other lines over it, the contours, the dark loading mesh where no cell stands in, the background off the globe; then the panel and the screen's marks in front of it (`Panel::overlay_rgba`, premultiplied), composed front to back.
//! Not yet here (the CPU path still has them): imagery, land ownership and motor vehicle use, the user's ink and selection, slope bands, raw elevation, the detail tint, and a terrain texel with no data climbing to its parent (it draws the background).
//!
//! What it writes, either way the light unconverted: the display's own working values straight into the app's surface (`Out::Surface`: gamma 2, white at 2^14, opaque; where the GPU's fence admits the apps' surfaces, ferros since 2026-11-11), the frame then on the panel with no copy on the CPU; or fluor's 8-bit pixels (`Out::Argb8`, α and darkness) in the arena, the light as a VSF `g2t` code (256 · tanh √light), which fluor's ferros host turns into working values exactly as it turns its own.
//! The CPU path's encode (the BT.2020 matrix, exposure 2/3 into Opsin's rail, then √) is the Android and Linux surfaces' contract; ferros's panel is VSF RGB in its root space and rolls its own highlights, so the light goes to it unconverted.
//!
//! Everything a frame holds still (the light, the sun, the ball's pole and disk, the contour interval, the inks, the layer mask, the depths) is a constant of the recording: the kernel is recorded again for every frame (a few hundred nodes, well under a millisecond), so it carries no loads and no code for what the frame leaves off.
//! What varies by pixel comes from the layer arena: the 32 px tiles' block index, the blocks (the plan's lattice with each role's 2 × 2 cells), the cells' texels, the tables, the overlay.
//! The tables, tiles, blocks and overlay are offsets within a frame region (`Consts::region`); the cells' slots are shared by both regions.

use valhall::ssa::{self, F, M, U};

/// Uniform slots: 0 the frame the kernel writes, 1 the arena's start (every table, block, tile, cell and the overlay are word offsets from it).
pub const SLOT_FRAME: u8 = 0;
pub const SLOT_ARENA: u8 = 1;

/// The arena's layout, bytes from its start. The tables first: each line class (22) and land class (14) as four f32 (linear red, green, blue, opacity), then the hypsometric ramp (4096 rows of linear red, green, blue and a pad).
pub const LINE_TAB: usize = 0;
pub const LAND_TAB: usize = 22 * 16;
pub const HYPSO_TAB: usize = 1 << 10;
/// The 32 px tiles' word (`TILE_STRADDLE` | the first block's index), row by row.
pub const TILES: usize = 128 << 10;
/// The blocks, `BLOCK_WORDS` each.
pub const BLOCKS: usize = 256 << 10;
pub const BLOCK_WORDS: usize = 48;
/// The g2t codes' working values (256 words, 2^14 · atanh(c / 256), fluor's own table for its 8-bit pixels): the overlay's codes into the surface's values.
pub const CODE_TAB: usize = 120 << 10;
/// The overlay (premultiplied RGBA bytes, a word a pixel), then (`Out::Argb8`) the frame, a screen's words each; then the code and the cells' slots, placed by the host from the screen's size.
pub const OVERLAY: usize = 8 << 20;
pub const MAX_BLOCKS: usize = (OVERLAY - BLOCKS) / (4 * BLOCK_WORDS);

/// A tile of straddle sub-blocks (4 px each, 8 × 8 of them in order) rather than one block.
pub const TILE_STRADDLE: u32 = 1 << 31;
/// The block's words: x | y << 16, the diamond, u0, v0 (depth-30 units), du/dx, dv/dx, du/dy, dv/dy, twist u, twist v (f32 bits); then each role (the terrain at `B_DEM`, the vector planes at `B_VEC`): the cell grid's corner (cu0, cv0) at the role's wanted depth, and its 2 × 2 cells (k = Δu + 2 Δv) as four texel bases (the slot's word offset, 0 for none) and four texel shifts (22 − the depth of the cell standing in).
pub const B_XY: usize = 0;
pub const B_DIAMOND: usize = 1;
pub const B_U0: usize = 2;
pub const B_STEPS: usize = 4;
pub const B_DEM: usize = 10;
pub const B_VEC: usize = 20;
/// The dark loading mesh's colours for the terrain's 2 × 2 cells at the wanted depth, each cell's two triangles (word 2k + half): gamma-2 bytes, red | green << 8 | blue << 16 (raster.rs `loading_noise`, worked out once a block on the CPU).
pub const B_NOISE: usize = 30;
pub const ROLE_BASE: usize = 2;
pub const ROLE_SHIFT: usize = 6;

/// A frame's constants: the screen, what is drawn and how it is lit.
#[derive(Clone, Copy, Debug)]
pub struct Consts {
    pub w: u32,
    /// Tiles across (the screen's width in 32 px blocks).
    pub bw: u32,
    pub dem_depth: u32,
    pub vec_depth: u32,
    /// The effective layer mask (`LayerMask::bits`).
    pub mask: u32,
    /// The lighting environment's quadratic form per channel (`sh::Quad`), the sun's direction (on normals of 32767 length), the world's pole in view coordinates, the disk (centre x, y, radius in pixels), whether its limb is on the screen.
    pub light: [[f32; 10]; 3],
    pub sun: [f32; 3],
    pub pole: [f32; 3],
    pub disk: [f32; 3],
    pub limb: bool,
    /// Linear inks: no terrain, the background, the open sea, flat ground, the water and its opacity, the contours and the index contours with theirs.
    pub no_dem: [f32; 3],
    pub bg: [f32; 3],
    pub sea: [f32; 3],
    pub flat: [f32; 3],
    pub water: [f32; 4],
    pub contour: [f32; 4],
    pub contour_index: [f32; 4],
    /// Contour interval (metres, 0 for none), ground metres a pixel, every n-th an index contour.
    pub interval: f32,
    pub m_per_px: f32,
    pub index_every: u32,
    /// The smallest magnitude a waterway draws at, at this terrain depth (`line_min_depth`'s class 12 rows); 256 when none does.
    pub waterway_mag: u32,
    /// An overlay to compose in front (the panel or a mark is up somewhere).
    pub overlay: bool,
    /// The overlay's word offset in the arena.
    pub overlay_at: u32,
    /// The frame region's word offset in the arena (`arena::Layout::regions`): its tables, tiles and blocks are there, each frame's in the other region from the last's, so the host writes the next frame's while the GPU reads this one's.
    pub region: u32,
    /// What the kernel writes, and where (uniform 0).
    pub out: Out,
}

/// The kernel's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    /// The display's working values (R, G, B 16 bits each at bits 0, 16, 32, gamma 2, white 2^14; bit 63 clear: opaque) into the app's surface, a panel-wide row a screen row.
    Surface,
    /// fluor's `Argb8` into a frame in the arena.
    Argb8,
}

/// The mask's bits (`LayerMask::bits`).
const DEM: u32 = 1;
const LAND: u32 = 2;
const WATER: u32 = 4;
const CONTOURS: u32 = 64;
const SLOPE: u32 = 128;
const HYPSO: u32 = 512;

/// Elevation: half-metre steps from −11000 m, G#FFFF none (mahere-tiles ELEV_STEP, ELEV_OFFSET, ELEV_NODATA).
const ELEV_STEP: f32 = 0.5;
const ELEV_OFFSET: f32 = 11000.;
const NODATA: u32 = 0xFFFF;
const WATERWAY: u32 = 12;
const CLASS_MAX: u32 = 21;
const LAND_MAX: u32 = 13;
/// raster.rs: the seabed's darkening and glaze, the least slope a contour draws on.
const SEABED_FULL_DEPTH: f32 = 6000.;
const SEABED_DARK: f32 = 0.64;
const SEABED_GLAZE: f32 = 0.35;
const CONTOUR_MIN_SLOPE: f32 = 1e-4;
/// A normal's unit length in the packed texel (snorm16).
const N1: f32 = 32767.;

fn sel(m: M, a: F, b: F) -> F {
    F::select(m, a, b)
}
fn sel3(m: M, a: [F; 3], b: [F; 3]) -> [F; 3] {
    [sel(m, a[0], b[0]), sel(m, a[1], b[1]), sel(m, a[2], b[2])]
}
fn splat3(c: [f32; 3]) -> [F; 3] {
    [F::splat(c[0]), F::splat(c[1]), F::splat(c[2])]
}
fn lerp3(a: [F; 3], b: [F; 3], t: F) -> [F; 3] {
    [(b[0] - a[0]).mul_add(t, a[0]), (b[1] - a[1]).mul_add(t, a[1]), (b[2] - a[2]).mul_add(t, a[2])]
}
/// A word of the arena at word index `w`.
fn word(w: U) -> U {
    ssa::load(SLOT_ARENA, w, 2, 0)
}
fn wordf(w: U) -> F {
    F::from_bits(word(w))
}
/// A gamma-2 byte's light, (b / 256)².
fn dec(b: U) -> F {
    let x = F::from_u32(b) * (1. / 256.);
    x * x
}
/// x clamped to [0, 1], as the CPU's `clamp(0.0, 1.0)`.
fn unit(x: F) -> F {
    x.max(F::splat(0.)).min(F::splat(1.))
}
fn round(x: F) -> F {
    (x + 0.5).floor()
}

/// Column and row of pixel `i` of a `w`-wide frame, as floats (exact below 2^24): the wallpaper's own (ferros wallpaper/src/gpu.rs `column_row`), the row from (i + ½) / w by a reciprocal, put right by the column it leaves.
fn column_row(i: F, w: u32) -> (F, F) {
    let wf = w as f32;
    let y = ((i + 0.5) * (1. / wf)).floor();
    let x = i - y * wf;
    let one = F::splat(1.);
    let zero = F::splat(0.);
    let y = y + sel(x.ge(F::splat(wf)), one, zero) - sel(x.lt(zero), one, zero);
    (i - y * wf, y)
}

/// One role's cell for the pixel: the block's 2 × 2 cells at the wanted depth (cell grid shift `s`), the pixel's (u, v); its texel's word offset, whether there is a cell at all, and which of the four it is.
fn role_texel(b: U, at: usize, s: u32, u: U, v: U, texel_words: u32) -> (U, M, U) {
    let cu0 = word(b + U::splat(at as u32));
    let cv0 = word(b + U::splat(at as u32 + 1));
    // WHY the mask: a pixel's (u, v) is rounded from the block's interpolation and can land one unit past the hull of its exact corners, from which the host took the 2 × 2 corner (cu0, cv0); a cell index of −1 there would read a word gigabytes off and fault the GPU's MMU, failing the whole frame. Masked, it reads a neighbouring entry: at worst a one-pixel speck.
    let k = ((u >> s) - cu0 + ((v >> s) - cv0 << 1)) & U::splat(3);
    let base = word(b + U::splat((at + ROLE_BASE) as u32) + k);
    let ts = word(b + U::splat((at + ROLE_SHIFT) as u32) + k);
    let one = U::splat(1);
    let m = one.shl_by(ts) - one;
    let tx = u.shr_by(ts) & U::splat(255);
    let ty = v.shr_by(ts) & U::splat(255);
    let half = ((u & m) + (v & m)).shr_by(ts) & one;
    let ti = (ty << 9) | (tx << 1) | half;
    (base + ti.wrapping_mul(U::splat(texel_words)), base.ne(U::splat(0)), k)
}

/// The pixel's own frame on the ball (raster.rs `view_normal`): the normal (east, north, up) into view coordinates.
fn view_normal(c: &Consts, sx: F, sy: F, n: [F; 3]) -> [F; 3] {
    let [cx, cy, r] = c.disk;
    let x = (sx - cx) * (1. / r);
    let y = (F::splat(cy) - sy) * (1. / r);
    let up = [x, y, (F::splat(1.) - x * x - y * y).max(F::splat(0.)).sqrt()];
    let p = c.pole;
    let e = [up[2] * p[1] - up[1] * p[2], up[0] * p[2] - up[2] * p[0], up[1] * p[0] - up[0] * p[1]];
    let len2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    // Where the pole is the pixel's up (len below 1e-6): east is the screen's right.
    let tiny = len2.lt(F::splat(1e-12));
    let inv = len2.max(F::splat(1e-24)).rsq_estimate();
    let east = [sel(tiny, F::splat(1.), e[0] * inv), sel(tiny, F::splat(0.), e[1] * inv), sel(tiny, F::splat(0.), e[2] * inv)];
    let north = [up[1] * east[2] - up[2] * east[1], up[2] * east[0] - up[0] * east[2], up[0] * east[1] - up[1] * east[0]];
    [
        n[0] * east[0] + n[1] * north[0] + n[2] * up[0],
        n[0] * east[1] + n[1] * north[1] + n[2] * up[1],
        n[0] * east[2] + n[1] * north[2] + n[2] * up[2],
    ]
}

/// The light as a VSF `g2t` code, 256 · tanh √light, toward zero and at most 255: tanh s = 1 − 2 / (1 + e^2s), e^2s as 2^(2s · log₂ e) on the special function unit.
fn g2t(l: F) -> U {
    let s = l.max(F::splat(0.)).sqrt();
    let e = (s * (2. * core::f32::consts::LOG2_E)).exp2_hw();
    let t = F::splat(1.) - F::splat(2.) * (e + 1.).rcp();
    (t * 256.).min(F::splat(255.)).to_u32()
}

/// Record the kernel for a frame with these constants: invocation g draws pixel g.
pub fn record(c: &Consts) -> ssa::Compiled {
    ssa::begin();
    ssa::reserve(2);
    ssa::uniform_slots(64);
    let i = ssa::gid();
    let (x, y) = column_row(F::from_u32(i), c.w);
    let (xi, yi) = (x.to_u32(), y.to_u32());
    let tile = word(U::splat(c.region + (TILES / 4) as u32) + (yi >> 5).wrapping_mul(U::splat(c.bw)) + (xi >> 5));
    let straddle = (tile & U::splat(TILE_STRADDLE)).ne(U::splat(0));
    let sub = (((yi & U::splat(31)) >> 2) << 3) + ((xi & U::splat(31)) >> 2);
    let block = (tile & U::splat(!TILE_STRADDLE)) + U::select(straddle, sub, U::splat(0));
    let b = U::splat(c.region + (BLOCKS / 4) as u32) + block.wrapping_mul(U::splat(BLOCK_WORDS as u32));
    let xy = word(b + U::splat(B_XY as u32));
    let fx = x - F::from_u32(xy & U::splat(0xFFFF));
    let fy = y - F::from_u32(xy >> 16);
    let st = |k: usize| wordf(b + U::splat((B_STEPS + k) as u32));
    let ou = st(0) * fx + st(2) * fy + st(4) * fx * fy;
    let ov = st(1) * fx + st(3) * fy + st(5) * fx * fy;
    let u = word(b + U::splat(B_U0 as u32)) + round(ou).to_i32();
    let v = word(b + U::splat(B_U0 as u32 + 1)) + round(ov).to_i32();
    let (sx, sy) = (x + 0.5, y + 0.5);
    let mask = c.mask;
    let zero = F::splat(0.);
    let one = F::splat(1.);
    let no = M::splat(false);

    // The terrain texel: elevation and the snorm16 normal (east, north, up).
    let wants_dem = mask & (DEM | CONTOURS | SLOPE) != 0;
    let (dem_at, have_dem_cell, dem_k) = if wants_dem { role_texel(b, B_DEM, 30 - c.dem_depth, u, v, 2) } else { (U::splat(0), no, U::splat(0)) };
    // The vector planes, packed a texel in two words: line class, coverage, magnitude and the water's coverage; land class and coverage.
    let (vec_at, have_vec, _) = role_texel(b, B_VEC, 30 - c.vec_depth, u, v, 2);
    let (w0, w1) = (word(vec_at), word(vec_at + U::splat(1)));
    let byte = |w: U, k: u32| (w >> (8 * k)) & U::splat(255);
    let (lcls, lcov, lmag, wcov) = (byte(w0, 0), byte(w0, 1), byte(w0, 2), w0 >> 24);
    let (acls, acov) = (byte(w1, 0), byte(w1, 1));
    let wet = U::select(have_vec, wcov, U::splat(0)).ne(U::splat(0));

    let mut tint = splat3(c.no_dem);
    let mut diffuse = one;
    let mut light = [one; 3];
    let mut ground = no;
    let mut seabed = no;
    let mut contour = zero;
    let mut contour_index = no;
    if wants_dem {
        let lo = word(dem_at);
        let hi = word(dem_at + U::splat(1));
        let eq = lo & U::splat(0xFFFF);
        let texel = have_dem_cell & eq.ne(U::splat(NODATA));
        // snorm16 halves: the high half read as the top of a signed word, the low one shifted there first.
        let k16 = 1. / 65536.;
        let n = [F::from_i32(lo & U::splat(0xFFFF_0000)) * k16, F::from_i32(hi << 16) * k16, F::from_i32(hi & U::splat(0xFFFF_0000)) * k16];
        let elev = F::from_u32(eq).mul_add(F::splat(ELEV_STEP), F::splat(-ELEV_OFFSET));
        if mask & DEM != 0 {
            let vn = view_normal(c, sx, sy, n);
            let d = (vn[0] * c.sun[0] + vn[1] * c.sun[1] + vn[2] * c.sun[2]) * (1. / N1);
            diffuse = sel(texel, d.max(zero), one);
            let q = [vn[0] * (1. / N1), vn[1] * (1. / N1), vn[2] * (1. / N1)];
            let m = [q[0] * q[0], q[1] * q[1], q[2] * q[2], q[0] * q[1], q[0] * q[2], q[1] * q[2], q[0], q[1], q[2]];
            for (ch, l) in light.iter_mut().enumerate() {
                let k = &c.light[ch];
                let mut e = F::splat(k[9]);
                for (t, &mt) in m.iter().enumerate() {
                    e = mt.mul_add(F::splat(k[t]), e);
                }
                *l = sel(texel, e.max(zero), one);
            }
            let sea = elev.abs().lt(F::splat(0.75)) & n[2].gt(F::splat(0.9995 * N1));
            let bed = !sea & elev.lt(zero) & wet;
            seabed = texel & bed;
            let t = unit(elev * (-1. / SEABED_FULL_DEPTH));
            let kbed = F::splat(1.) - t * SEABED_DARK;
            let bed_rgb = [F::splat(c.sea[0]) * kbed, F::splat(c.sea[1]) * kbed, F::splat(c.sea[2]) * kbed];
            let land_rgb = if mask & HYPSO != 0 {
                let row = U::splat(c.region + (HYPSO_TAB / 4) as u32) + ((eq >> 4) << 2);
                [wordf(row), wordf(row + U::splat(1)), wordf(row + U::splat(2))]
            } else {
                splat3(c.flat)
            };
            let ground_rgb = sel3(sea, splat3(c.sea), sel3(bed, bed_rgb, land_rgb));
            tint = sel3(texel, ground_rgb, splat3(c.bg));
            ground = texel;
        }
        if mask & CONTOURS != 0 && c.interval > 0. {
            let nzn = (n[2] * (1. / N1)).max(F::splat(1e-4));
            let slope = (one - nzn * nzn).max(zero).sqrt() * nzn.rcp();
            let level = round(elev * (1. / c.interval));
            let d_m = (elev - level * c.interval).abs();
            let d_px = d_m * (slope * c.m_per_px).rcp();
            let ie = c.index_every as f32;
            // level is a whole number: it is a multiple of n exactly when ⌊(level + ½) / n⌋ · n is level (the half keeps the reciprocal's rounding off the floor's edge).
            let index = ((level + 0.5) * (1. / ie)).floor().mul_add(F::splat(ie), -level).abs().lt(F::splat(0.5));
            let w = sel(index, F::splat(0.9), F::splat(0.55));
            contour = sel(texel & slope.ge(F::splat(CONTOUR_MIN_SLOPE)), unit(w + 0.5 - d_px), zero);
            contour_index = index;
        }
    }
    let ink = sel3(contour_index, [F::splat(c.contour_index[0]), F::splat(c.contour_index[1]), F::splat(c.contour_index[2])], [F::splat(c.contour[0]), F::splat(c.contour[1]), F::splat(c.contour[2])]);
    let ink_a = contour * sel(contour_index, F::splat(c.contour_index[3]), F::splat(c.contour[3]));

    let lit = |rgb: [F; 3]| -> [F; 3] {
        let l = [rgb[0] * light[0], rgb[1] * light[1], rgb[2] * light[2]];
        sel3(ground, l, rgb)
    };
    // Without vector planes: the light, then the contours.
    let plain = lerp3(lit(tint), ink, ink_a);
    // With them: land cover, the light, the contours, the waterways, the water, the other lines (raster.rs `compose_rgb`).
    let mut rgb = tint;
    if mask & LAND != 0 {
        let k = U::select(acls.lt(U::splat(LAND_MAX + 1)), acls, U::splat(LAND_MAX));
        let row = U::splat(c.region + (LAND_TAB / 4) as u32) + (k << 2);
        let col = [wordf(row), wordf(row + U::splat(1)), wordf(row + U::splat(2))];
        rgb = lerp3(rgb, col, F::from_u32(acov) * (1. / 255.) * wordf(row + U::splat(3)));
    }
    rgb = lerp3(lit(rgb), ink, ink_a);
    let k = U::select(lcls.lt(U::splat(CLASS_MAX + 1)), lcls, U::splat(CLASS_MAX));
    let row = U::splat(c.region + (LINE_TAB / 4) as u32) + (k << 2);
    let waterway = k.eq(U::splat(WATERWAY));
    let dim = sel(waterway, F::from_u32(lmag).mul_add(F::splat(0.6 / 255.), F::splat(0.4)), one);
    let lcol = [wordf(row) * dim, wordf(row + U::splat(1)) * dim, wordf(row + U::splat(2)) * dim];
    // The class's opacity (the host folds in whether the class draws at this depth and the boundaries' 0.55), a waterway's also gated by its magnitude.
    let gate = !waterway | lmag.ge(U::splat(c.waterway_mag));
    let la = sel(gate, F::from_u32(lcov) * (1. / 255.) * wordf(row + U::splat(3)), zero);
    rgb = lerp3(rgb, lcol, sel(waterway, la, zero));
    if mask & WATER != 0 {
        let s = diffuse.mul_add(F::splat(0.15), F::splat(0.85));
        let wr = [F::splat(c.water[0]) * s, F::splat(c.water[1]) * s, F::splat(c.water[2]) * s];
        let t = F::from_u32(wcov) * (c.water[3] / 255.) * sel(seabed, F::splat(SEABED_GLAZE), one);
        rgb = lerp3(rgb, wr, t);
    }
    rgb = lerp3(rgb, lcol, sel(waterway, zero, la));
    let mut rgb = sel3(have_vec, rgb, plain);

    // No cell at any depth for the terrain: each triangle of the wanted depth's cell grid its own dark colour (raster.rs `loading_noise`, the block's colours).
    if wants_dem {
        let s = 30 - c.dem_depth;
        let m = U::splat((1u32 << s) - 1);
        let half = ((u & m) + (v & m)) >> s & U::splat(1);
        let n = word(b + U::splat(B_NOISE as u32) + (dem_k << 1) + half);
        let noise = [dec(n & U::splat(255)), dec((n >> 8) & U::splat(255)), dec(n >> 16)];
        rgb = sel3(have_dem_cell, rgb, noise);
    }
    // Off the globe, where its limb is on the screen: the background.
    if c.limb {
        let (dx, dy) = (sx - c.disk[0], sy - c.disk[1]);
        let off = (dx * dx + dy * dy).gt(F::splat(c.disk[2] * c.disk[2]));
        rgb = sel3(off, splat3(c.bg), rgb);
    }

    if c.out == Out::Surface {
        // The working value, 2^14 · √light. WHY the min: a channel past 65535 would carry into the next one's bits (into green, or past blue into the see-through bit) where the light runs past 16 times white, which the SH light's sky can reach at a glint; the display takes up to four times white, so nothing that shows is cut.
        let wv = |l: F| (l.max(F::splat(0.)).sqrt() * 16384.).min(F::splat(65535.));
        let mut v = [wv(rgb[0]), wv(rgb[1]), wv(rgb[2])];
        if c.overlay {
            // The overlay in front, its premultiplied g2t codes as working values (fluor's table), what it leaves of the map under it: front to back, `under`.
            let o = word(U::splat(c.overlay_at) + i);
            let keep = F::from_u32(U::splat(255) - (o >> 24)) * (1. / 255.);
            let tab = |k: u32| wordf(U::splat(c.region + (CODE_TAB / 4) as u32) + byte(o, k));
            for (k, x) in v.iter_mut().enumerate() {
                *x = x.mul_add(keep, tab(k as u32)).min(F::splat(65535.));
            }
        }
        let q = |x: F| x.to_u32();
        ssa::store_at(SLOT_FRAME, i, 3, 0, q(v[0]) | q(v[1]) << 16);
        ssa::store_at(SLOT_FRAME, i, 3, 4, q(v[2]));
        return ssa::compile();
    }
    let code = [g2t(rgb[0]), g2t(rgb[1]), g2t(rgb[2])];
    // The overlay in front (premultiplied, its codes as the CPU's `composite_rgb` mixes them): what it leaves of the map comes through under it.
    let code = if c.overlay {
        let o = word(U::splat(c.overlay_at) + i);
        let keep = F::from_u32(U::splat(255) - (o >> 24)) * (1. / 255.);
        let mix = |k: u32, m: U| (F::from_u32(m).mul_add(keep, F::from_u32(byte(o, k)))).min(F::splat(255.)).to_u32();
        [mix(0, code[0]), mix(1, code[1]), mix(2, code[2])]
    } else {
        code
    };
    // fluor's Argb8: α 255, each channel's darkness 255 − its code.
    let dark = |c: U| U::splat(255) - c;
    let px = U::splat(0xFF00_0000) | dark(code[0]) << 16 | dark(code[1]) << 8 | dark(code[2]);
    ssa::store_at(SLOT_FRAME, i, 2, 0, px);
    ssa::compile()
}
