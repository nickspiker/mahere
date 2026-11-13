//! The map's compose as a ferros GPU kernel: one invocation a screen pixel, recorded from Rust through valhall's SSA recorder and compiled to Mali-G715 words on the device (ferros GPU-KISTS.md: compute only, no triangle raster; the owner, 2026-11-11: "it'd be a 1:1 per px", "no tilting or 3Dness").
//!
//! It computes what the CPU raster's `compose_rgb` computes (mahere-engine raster.rs), one sample a pixel at the same place, from the same cells: the terrain's light from its packed normal in the pixel's own frame on the ball, the elevation tint (sea, seabed, hypsometric or flat), land cover, the waterways under the water, the water, the other lines over it, the contours, the dark loading mesh where no cell stands in, the background off the globe; then the globe's loading ring from its numbers (`Ring`) and the panel and the screen's marks in front of it all (a resident layer in the arena, `Panel::overlay_rgba`'s premultiplied sum, updated by the box that changed), composed front to back.
//! Not yet here (the CPU path still has them): imagery, land ownership and motor vehicle use, the user's ink and selection, slope bands, raw elevation, the detail tint, and a terrain texel with no data climbing to its parent (it draws the background).
//!
//! What it writes: the display's own working values straight into the app's surface (`Out::Surface`: gamma 2, white at 2^14, opaque; where the GPU's fence admits the apps' surfaces, ferros since 2026-11-11), the frame then on the panel with no copy on the CPU, and with the display's bake (`Bake`, ferros COMPOSITOR.md "Baked", since 2026-11-12) those values made the panel's own codes as the last step, so the display only dithers them; or fluor's 8-bit pixels (`Out::Argb8`, α and darkness) in the arena, the light as a VSF `g2t` code (256 · tanh √light), which fluor's ferros host turns into working values exactly as it turns its own.
//! The CPU path's encode (the BT.2020 matrix, exposure 2/3 into Opsin's rail, then √) is the Android and Linux surfaces' contract; ferros's panel is VSF RGB in its root space and rolls its own highlights, so the light goes to it unconverted.
//!
//! What a frame chooses (the layer mask, one face or several, the fold, the ring, the overlay, the output) is the recording's structure: it carries no code for what the frame leaves off. What a frame holds still (the light, the sun, the ball's pole and disk, the camera, the face's row, the contour interval, the inks, the depths, where its tables are) is a param of the recording (valhall `ssa::param`): placed as a constant (a uniform word, or an immediate), never folded into the words, so two frames of one structure have the same words and gpu.rs compiles each structure once and binds each frame's values into it (`record` and `recording`; ferros GPU-KISTS.md, after the fourth pass: recording and compiling every frame was 0.33 to 0.47 ms of Mahere's CPU).
//! Where a warp has no use for a stretch of the work, it branches over it (valhall `ssa::when`): off the globe and under an opaque overlay the whole map, where no cell stands in the terrain's shading (and where one does the loading mesh), a pixel without vector planes their tables, beside the ring its coverage, under no overlay its codes.
//! Each pixel finds its own place on the ball from the camera (mahere-engine viewplan.rs): its direction in view coordinates, the face in view whose centroid is nearest (the encoder's choice, among the frame's faces), and its depth-30 (u, v) on that face's diamond from the face's projective form, measured from the camera point when every face in view is near it (so a deep zoom keeps its precision in f32), from the origin at the globe, where the face comes from a fold instead of the search (`Fold`: the octant of the direction in the icosahedron's standard pose and the nearest of four centres, about 80 words fewer than comparing 15 faces).
//! Its cell then comes from a page table: each role's box of squares for the diamond, each square the slot of the resident cell standing in for it, or none (the loading mesh, its colours from the encoder's hash on the GPU).
//! No lattice, no blocks, no seam sub-blocks: what the CPU planned for every 32 px tile and every 4 px across a seam (6.5 ms a frame at a town's scale on the Pro, 15.7 at the globe, 2026-11-12) is the kernel's.
//! What varies by pixel comes from the layer arena: the faces' rows (more than one face in view), the squares, the cells' texels, the tables, the overlay.
//! The tables, faces, squares and overlay are offsets within a frame region (`Consts::region`); the cells' slots are shared by both regions.

use valhall::ssa::{self, F, H, M, U};

/// Uniform slots: 0 the frame the kernel writes, 1 the arena's start (every table, block, tile, cell and the overlay are word offsets from it).
pub const SLOT_FRAME: u8 = 0;
pub const SLOT_ARENA: u8 = 1;

/// The arena's layout, bytes from its start. The tables first: each line class (22) and land class (14) a row of `ROW` words, f16 pairs (linear red and green, then blue and the opacity over 255), then the hypsometric ramp (4096 rows: red and green, blue twice).
pub const LINE_TAB: usize = 0;
pub const ROW: usize = 2;
pub const LAND_TAB: usize = 22 * 16;
pub const HYPSO_TAB: usize = 1 << 10;
/// The faces in view, a row of `FACE_WORDS` each (`F_*` below), read by face when there is more than one.
pub const FACES: usize = 128 << 10;
pub const FACE_SHIFT: u32 = 5;
pub const FACE_WORDS: usize = 1 << FACE_SHIFT;
/// The page table's squares, a word each: the terrain's boxes, then the vector planes'; each the texel base of the cell standing in (its slot's word offset, below 2^27) | its texel shift (22 − its depth) << 27, or 0 for none.
pub const SQUARES: usize = 256 << 10;
pub const SQUARE_SHIFT: u32 = 27;
/// The g2t codes' working values (256 words, 2^14 · atanh(c / 256), fluor's own table for its 8-bit pixels): the overlay's codes into the surface's values.
pub const CODE_TAB: usize = 120 << 10;
/// The overlay (premultiplied RGBA bytes, a word a pixel), then (`Out::Argb8`) the frame, a screen's words each; then the code and the cells' slots, placed by the host from the screen's size.
/// At 1 MiB the squares have 192 Ki words: a box at the role's depth is a cell 115 pixels or more across (`select_depth`), so a 1344 x 2992 screen turned any way meets about 32 x 32 squares a diamond, five diamonds at an icosahedron's corner, two roles: about 10 Ki.
/// It was 8 MiB until 2026-11-12, 7 of it squares no plan names, in each of the two frame regions: on the 8a a 32 MiB block of the arena then held one region and 13 cell slots, the globe's ten cells of depth 0 needing 20.
pub const OVERLAY: usize = 1 << 20;
/// The lettering's tiles, glyphs and lists (`letters.rs`), its last half MiB before the overlay: the squares keep 64 Ki words, six times the most a screen names (2026-11-13, the labels on the GPU; 192 Ki before).
pub const LETTERS: usize = 1 << 19;
pub const MAX_SQUARES: usize = (LETTERS - SQUARES) / 4;

/// A face's row: its (u, v) numerators' and its denominator's coefficients on the pixel's direction (f32 bits, (u, v) in depth-30 units), the fractions and whole parts of the camera point's (u, v) when measured from it (else 0), the loading mesh's hash seed for its diamond (raster.rs `loading_noise`'s diamond and depth terms), then each role's box (the terrain's at `F_BOX`, the vector planes' `F_BOX_WORDS` after): its cell grid corner (cu0, cv0), width, height, and the word offset of its squares in the arena.
pub const F_U: usize = 0;
pub const F_V: usize = 3;
pub const F_DEN: usize = 6;
pub const F_FRAC: usize = 9;
pub const F_REF: usize = 11;
pub const F_SEED: usize = 13;
pub const F_BOX: usize = 14;
pub const F_BOX_WORDS: usize = 5;
pub const F_USED: usize = F_BOX + 2 * F_BOX_WORDS;

/// The faces a frame may have in view: every one.
pub const MAX_FACES: usize = 20;

/// A face as the kernel takes it: its centroid in view coordinates and the score's offset (the argmax over the frame's faces, `score · (x, y, Z) + offset`), and its row.
#[derive(Clone, Copy, Debug)]
pub struct FaceConsts {
    pub score: [f32; 4],
    pub row: [u32; FACE_WORDS],
}

/// The faces found by folding rather than searched (mahere-coord `FaceFold`), at the globe, where the search compared up to 15 faces a pixel: the view's axes into the fold's frame (`turn`, rows, on the pixel's (x, y, z)), the four face centres in its positive octant; the faces' rows at `faces_at` by `octant << 2 | k` (`FOLD_ROWS` of them), a face out of view there as the nearest face in view.
#[derive(Clone, Copy, Debug)]
pub struct Fold {
    pub turn: [[f32; 3]; 3],
    pub centres: [[f32; 3]; 4],
}
pub const FOLD_ROWS: usize = 32;

/// A frame's constants: the screen, what is drawn and how it is lit.
#[derive(Clone, Copy, Debug)]
pub struct Consts {
    pub w: u32,
    /// Screen to view coordinates: x = sx · k + kx, y = ky − sy · k, for k = 1 / the globe's radius in pixels (`[k, kx, ky]`).
    pub view: [f32; 3],
    /// Every face in view is measured from the camera point (`viewplan`'s near form: Z = z − 1, the denominator 1 + …), else from the origin (Z = z).
    pub near: bool,
    /// The faces in view, the first `face_count`; their rows at word offset `faces_at` in the arena.
    pub faces: [FaceConsts; MAX_FACES],
    pub face_count: usize,
    pub faces_at: u32,
    /// The faces' rows by fold, not by search (far from the camera point, more than `FOLD_FROM` faces in view).
    pub fold: Option<Fold>,
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
    /// The globe's loading ring, drawn here from its numbers under the overlay and over the map: a ring that follows every pinch of the globe costs no pixels sent (it was the whole overlay painted and summed every frame of a pan at the globe, 7.3 of Mahere's 9.3 ms on the 8a, 2026-11-12).
    pub ring: Option<Ring>,
    /// An overlay to compose in front (the panel or a mark is up somewhere).
    pub overlay: bool,
    /// The overlay's word offset in the arena.
    pub overlay_at: u32,
    /// The frame region's word offset in the arena (`arena::Layout::regions`): its tables, tiles and blocks are there, each frame's in the other region from the last's, so the host writes the next frame's while the GPU reads this one's.
    pub region: u32,
    /// What the kernel writes, and where (uniform 0).
    pub out: Out,
    /// Into the surface (`Out::Surface`), the display's bake applied last, so the window is shown baked; None, the working values as before (a display that does not bake).
    pub bake: Option<Bake>,
    /// The place names (`letters.rs`), under the ring and the overlay and over the map; None when nothing is lettered.
    pub letters: Option<Letters>,
}

/// Where the frame's lettering is (`letters.rs`), word offsets in the arena: the tiles (a word each, `cols` a row), the letters' records, their lists and the glyph atlas; the halo's colour (in the kernel's output, as the letters' are) and its opacity.
#[derive(Clone, Copy, Debug)]
pub struct Letters {
    pub tiles_at: u32,
    pub cols: u32,
    pub glyphs_at: u32,
    pub list_at: u32,
    pub atlas_at: u32,
    pub halo: [f32; 4],
}

/// The globe's loading ring (mahere-engine `GlobeRing`): its centre, radius and half width in screen pixels, the end of its sweep from the top clockwise as (sin, cos), whether the sweep passes half a turn, its colour (gamma-2 bytes, as the marks').
#[derive(Clone, Copy, Debug)]
pub struct Ring {
    pub centre: [f32; 2],
    pub radius: f32,
    pub half_w: f32,
    pub end: [f32; 2],
    pub long: bool,
    pub rgb: [u8; 3],
}

impl Ring {
    pub fn from_globe(g: &mahere_engine::GlobeRing) -> Ring {
        Ring { centre: g.centre, radius: g.radius, half_w: g.half_w, end: [g.sweep.sin(), g.sweep.cos()], long: g.sweep > core::f32::consts::PI, rgb: g.rgb }
    }

    /// Its opacity at full coverage: the marks' sum takes a mark's brightest channel as its opacity.
    fn alpha(&self) -> f32 {
        self.rgb[0].max(self.rgb[1]).max(self.rgb[2]) as f32 / 255.
    }
}

/// The ferros display's bake (its op 65 thru fluor's ferros host, `fluor::host::ferros::bake`; ferros COMPOSITOR.md "Baked"): with it the kernel's last step makes the surface's working values the panel's own codes, so the display only dithers the frame (owner, 2026-11-12: Mahere's GPU kernel "ends with the matrix + tanh"). Per channel of a working value v (white 1): l = tanh²(k·v); the panel's light p = m·l (rows its red, green, blue); its code p^g × 65472 in Q16.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bake {
    pub k: f32,
    pub m: [[f32; 3]; 3],
    pub g: f32,
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
/// A frame's value: a param of the recording (valhall `ssa::param`), so the words do not depend on it and one compiled kernel serves every frame of this structure (gpu.rs). Made once and its handle used wherever the value is.
fn pf(v: f32) -> F {
    F::from_bits(ssa::param(v.to_bits()))
}
fn pf3(c: [f32; 3]) -> [F; 3] {
    c.map(pf)
}
fn pu(v: u32) -> U {
    ssa::param(v)
}
/// A word of the arena at word index `w`.
fn word(w: U) -> U {
    ssa::load(SLOT_ARENA, w, 2, 0)
}
fn wordf(w: U) -> F {
    F::from_bits(word(w))
}
/// x clamped to [0, 1], as the CPU's `clamp(0.0, 1.0)` (valhall folds it into the instruction that makes x, Valhall's destination clamp, where nothing else reads x).
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

/// One role's square for the pixel: its cell at the role's depth (the pixel's cell-grid coordinates `cu`, `cv`, its (u, v) shifted by the role's cell shift) within the box (`bx`: cu0, cv0, width, height, the squares' word offset), the square's word; 0 outside the box.
fn square(bx: &[U], cu: U, cv: U) -> U {
    let du = cu - bx[0];
    let dv = cv - bx[1];
    // WHY: a pixel's cell can lie outside its diamond's box (off the globe, or a sliver of a diamond at the screen's edge holding no point of the plan's), and its index would then name a word past the squares, up to gigabytes off: the GPU's MMU would fault the whole frame. Outside, it reads the box's first square and takes no cell.
    let inside = du.lt(bx[2]) & dv.lt(bx[3]);
    let at = bx[4] + U::select(inside, du + dv.wrapping_mul(bx[2]), U::splat(0));
    U::select(inside, word(at), U::splat(0))
}

/// A square's cell for the pixel at (u, v): its texel's word offset, and the cell's base (0: no cell).
fn role_texel(sq: U, u: U, v: U, texel_words: u32) -> (U, U) {
    let base = sq & U::splat((1 << SQUARE_SHIFT) - 1);
    let ts = sq >> SQUARE_SHIFT;
    let one = U::splat(1);
    let m = one.shl_by(ts) - one;
    let tx = u.shr_by(ts) & U::splat(255);
    let ty = v.shr_by(ts) & U::splat(255);
    let half = ((u & m) + (v & m)).shr_by(ts) & one;
    let ti = (ty << 9) | (tx << 1) | half;
    (base + ti.wrapping_mul(U::splat(texel_words)), base)
}

/// The pixel's own frame on the ball (raster.rs `view_normal`): the normal (east, north, up) into view coordinates. `disk`: the disk's centre (x, y) and 1 / its radius; `pole` the world's pole in view coordinates.
fn view_normal(disk: [F; 3], pole: [F; 3], sx: F, sy: F, n: [F; 3]) -> [F; 3] {
    let x = (sx - disk[0]) * disk[2];
    let y = (disk[1] - sy) * disk[2];
    let up = [x, y, (F::splat(1.) - x * x - y * y).max(F::splat(0.)).sqrt()];
    let p = pole;
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

/// The kernel for a frame with these constants, compiled: invocation g draws pixel g.
pub fn record(c: &Consts) -> ssa::Compiled {
    recording(c).compile()
}

/// The kernel for a frame with these constants, recorded and not compiled: its key says its structure, its params this frame's values (gpu.rs compiles each structure once).
pub fn recording(c: &Consts) -> ssa::Recording {
    ssa::begin();
    ssa::reserve(2);
    ssa::uniform_slots(64);
    let i = ssa::gid();
    let (x, y) = column_row(F::from_u32(i), c.w);
    let (sx, sy) = (x + 0.5, y + 0.5);
    let bg = C::param(c.bg);
    let disk = [pf(c.disk[0]), pf(c.disk[1]), pf(1. / c.disk[2])];
    // The overlay's word, read first: where it is opaque (the panel) nothing behind it is read.
    let o = c.overlay.then(|| word(pu(c.overlay_at) + i));
    // Where the map shows: on the disk where its limb is on the screen, and not under an opaque overlay. Elsewhere the whole map is branched over (off the globe it is the background).
    let on = c.limb.then(|| {
        let (dx, dy) = (sx - disk[0], sy - disk[1]);
        pf(c.disk[2] * c.disk[2]).ge(dx.mul_add(dx, dy * dy))
    });
    let seen = o.map(|o| (o >> 24).ne(U::splat(255)));
    let need = match (on, seen) {
        (Some(a), Some(b)) => a & b,
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => M::splat(true),
    };
    let rgb = ssa::when(need, bg, || map(c, x, y, sx, sy, disk, bg)).f32();
    // The place names wherever the overlay leaves anything to see (a name by the limb runs off the disk, as on the CPU path), as a layer of their own: the map goes under it below, in the output's own values.
    let lettered = c.letters.as_ref().map(|l| letters(l, x, y, sx, sy, seen.unwrap_or(M::splat(true))));
    let under = |front: (F, [F; 3]), back: [F; 3]| {
        let keep = F::splat(1.) - front.0;
        core::array::from_fn::<F, 3, _>(|k| back[k].mul_add(keep, front.1[k]))
    };

    if c.out == Out::Surface {
        // The working value, 2^14 · √light, as x · 1/√x with x held off 0 (2^-40, whose working value 2^-6 truncates to 0 as the light 0's does). WHY the min: a channel past 65535 would carry into the next one's bits (into green, or past blue into the see-through bit) where the light runs past 16 times white, which the SH light's sky can reach at a glint; the display takes up to four times white, so nothing that shows is cut.
        let wv = |l: F| {
            let x = l.max(F::splat(1. / (1u64 << 40) as f32));
            (x * x.rsq_estimate() * 16384.).min(F::splat(65535.))
        };
        let mut v = [wv(rgb[0]), wv(rgb[1]), wv(rgb[2])];
        if let Some(l) = lettered {
            v = under(l, v);
        }
        if let Some(r) = &c.ring {
            // The ring under the overlay, as a mark summed into it was: its code premultiplied by its coverage, then that code's working value (fluor's table's, 2^14 · atanh(c / 256), here 2^13 ln 2 · log₂((1 + c) / (1 − c)) on the special function unit, once for each distinct channel), and what it leaves of the map under it.
            v = ring(r, sx, sy, v, |cov, v| {
                let keep = F::splat(1.) - cov * pf(r.alpha());
                let mut done: Vec<(u8, F)> = Vec::with_capacity(3);
                let mut out = v;
                for (k, x) in out.iter_mut().enumerate() {
                    let b = r.rgb[k];
                    let wv = match done.iter().find(|d| d.0 == b) {
                        Some(d) => d.1,
                        None => {
                            let o = cov * pf(b as f32 / 256.);
                            let wv = ((o + 1.) * (F::splat(1.) - o).rcp()).log2_hw() * (8192. * core::f32::consts::LN_2);
                            done.push((b, wv));
                            wv
                        }
                    };
                    *x = x.mul_add(keep, wv);
                }
                out
            });
        }
        if let Some(o) = o {
            // The overlay in front, its premultiplied g2t codes as working values (fluor's table), what it leaves of the map under it: front to back, `under`. Where it is clear (its codes 0, premultiplied) the map is all there is, and the table is not read.
            let code_tab = pu(c.region + (CODE_TAB / 4) as u32);
            v = ssa::when((o >> 24).ne(U::splat(0)), v, || {
                let keep = F::from_u32(U::splat(255) - (o >> 24)) * (1. / 255.);
                let tab = |k: u32| wordf(code_tab + ((o >> (8 * k)) & U::splat(255)));
                let mut out = v;
                for (k, x) in out.iter_mut().enumerate() {
                    *x = x.mul_add(keep, tab(k as u32)).min(F::splat(65535.));
                }
                out
            });
        }
        if let Some(b) = &c.bake {
            // The bake, last (ferros COMPOSITOR.md "Baked"): the light rolled off as the display's, tanh²(k·v) with tanh s = 1 − 2 / (1 + e^2s) on the special function unit (as `g2t`), the panel's matrix, its code p^g = 2^(g · log₂ p), Q16 of the top code 65472. WHY the max: a channel's light at or under 0 (a colour past the panel's primaries on that channel) has no logarithm, and the special function unit's answer for it is not a promise; at 2^-60 its code is 0 all the same. WHY the min: a code past 65535 would carry into the next channel's bits (a light past the panel's peak, VSF red on the panel's red), the top as the display's own bake holds it.
            let k = pf(b.k * (1. / 16384.));
            let l = v.map(|x| {
                let e = (x * k * (2. * core::f32::consts::LOG2_E)).exp2_hw();
                let t = F::splat(1.) - F::splat(2.) * (e + 1.).rcp();
                t * t
            });
            let (m, g) = (b.m.map(pf3), pf(b.g));
            v = core::array::from_fn(|r| {
                let p = l[0].mul_add(m[r][0], l[1].mul_add(m[r][1], l[2] * m[r][2]));
                ((p.max(F::splat(1. / (1u64 << 60) as f32)).log2_hw() * g).exp2_hw() * 65472.).min(F::splat(65535.))
            });
        }
        let q = |x: F| x.to_u32();
        ssa::store_at(SLOT_FRAME, i, 3, 0, q(v[0]) | q(v[1]) << 16);
        ssa::store_at(SLOT_FRAME, i, 3, 4, q(v[2]));
        return ssa::finish();
    }
    let code = [g2t(rgb[0]), g2t(rgb[1]), g2t(rgb[2])];
    // The place names over the map, their colours as codes (letters.rs): the codes under them.
    let code = match lettered {
        Some(l) => {
            let v = under(l, code.map(F::from_u32));
            v.map(F::to_u32)
        }
        None => code,
    };
    // The ring under the overlay, its codes premultiplied by its coverage.
    let code = match &c.ring {
        Some(r) => {
            let codes = [F::from_u32(code[0]), F::from_u32(code[1]), F::from_u32(code[2])];
            let mixed = ring(r, sx, sy, codes, |cov, codes| {
                let keep = F::splat(1.) - cov * pf(r.alpha());
                core::array::from_fn(|k| codes[k].mul_add(keep, cov * pf(r.rgb[k] as f32)))
            });
            mixed.map(|m| m.to_u32())
        }
        None => code,
    };
    // The overlay in front (premultiplied, its codes as the CPU's `composite_rgb` mixes them): what it leaves of the map comes through under it.
    let code = match o {
        Some(o) => {
            let keep = F::from_u32(U::splat(255) - (o >> 24)) * (1. / 255.);
            let mix = |k: u32, m: U| (F::from_u32(m).mul_add(keep, F::from_u32((o >> (8 * k)) & U::splat(255)))).min(F::splat(255.)).to_u32();
            [mix(0, code[0]), mix(1, code[1]), mix(2, code[2])]
        }
        None => code,
    };
    // fluor's Argb8: α 255, each channel's darkness 255 − its code.
    let dark = |c: U| U::splat(255) - c;
    let px = U::splat(0xFF00_0000) | dark(code[0]) << 16 | dark(code[1]) << 8 | dark(code[2]);
    ssa::store_at(SLOT_FRAME, i, 2, 0, px);
    ssa::finish()
}

/// The place names at the pixel (corner x, y; centre sx, sy), where `m` is set (the map shows there): the lettering's layer there, its opacity and its colour premultiplied by it, in the kernel's output; nothing (all 0) elsewhere, and where the pixel's tile holds no letter (most of the screen), every warp beside the letters branching over all of it.
/// Front to back, as fluor's `under`: each letter under the letters before it in the tile's list (what a letter adds is its coverage times its opacity, of what the ones before it left), then the halo under all of them (the most any letter's halo covers the pixel, so two letters' halos where they meet are one halo, not one over the other), the map under the whole layer after this (`recording`).
fn letters(l: &Letters, x: F, y: F, sx: F, sy: F, m: M) -> (F, [F; 3]) {
    use crate::letters::{ATLAS_STRIDE, TILE_MOST, TILE_SHIFT};
    let zero = F::splat(0.);
    let one = F::splat(1.);
    let tile = (y.to_u32() >> TILE_SHIFT).wrapping_mul(pu(l.cols)) + (x.to_u32() >> TILE_SHIFT);
    let t = word(pu(l.tiles_at) + tile);
    let count = t & U::splat(255);
    ssa::when(m & count.ne(U::splat(0)), (zero, [zero; 3]), || {
        let list = pu(l.list_at) + (t >> 8);
        let most = F::from_u32(count);
        let (a, v, halo) = ssa::iterate(
            TILE_MOST,
            (zero, [zero; 3], zero),
            |n, (a, v, halo)| {
                // WHY the select: a pass past the tile's count reads the next tile's entries or the words past the lists, a letter's number nothing says is one; held to the first letter, its record is a record (the 8a, 2026-11-13: a GPU fault at G#0C962BC0 with the lettering on, its lanes' addresses not all inside the arena).
                let g = U::select(n.lt(most), word(list + n.to_u32()), U::splat(0));
                let at = pu(l.glyphs_at) + (g << crate::letters::GLYPH_SHIFT);
                let r = |k: u32| word(at + U::splat(k));
                let (cx, cy, ax, ay) = (F::from_bits(r(0)), F::from_bits(r(1)), F::from_bits(r(2)), F::from_bits(r(3)));
                let (turn, reach, rg, ba) = (H::from_bits(r(4)), H::from_bits(r(5)), H::from_bits(r(6)), H::from_bits(r(7)));
                // The pixel's centre in the letter's own frame: from its centre, turned back by its turn.
                let (dx, dy) = (sx - cx, sy - cy);
                let (c, s) = (turn.lo_f(), turn.hi_f());
                let lx = dx.mul_add(c, dy * s);
                let ly = dy.mul_add(c, -(dx * s));
                // WHY: a sample past the letter's reach would read its neighbour's texels in the atlas (or past the atlas's last row, outside Mahere's memory: the GPU's MMU faults the frame); within it, only its own texels and its empty border (letters.rs `place`).
                let inside = lx.abs().lt(reach.lo_f()) & ly.abs().lt(reach.hi_f());
                // A letter at rest is level on whole pixels (letters.rs), every pixel a texel's centre: that texel alone, a quarter of the turned letter's reads and none of its blend; the whole warp takes one way or the other (it is the letter's).
                // The sine squared, not its magnitude: valhall's `abs` of a half widened to f32 takes the whole word as an f32 and drops the widen (found here 2026-11-13: every turned letter was taken as level).
                let level = (s * s).lt(F::splat(1. / (1u64 << 32) as f32));
                // WHY the selects: every lane forms its read's address whether or not its block runs it (as `square` does for the cells): one outside the letter's reach reads the glyph's centre, so no address leaves the atlas (one far from the letter named texels gigabytes off and the GPU's MMU faulted the frame, the 8a, 2026-11-13).
                let (fx, fy) = (F::select(inside, ax + lx - 0.5, ax), F::select(inside, ay + ly - 0.5, ay));
                let at = ssa::when(inside & level, (zero, zero), || texel(pu(l.atlas_at), ATLAS_STRIDE as u32, fx, fy));
                let (cover, hl) = ssa::when(inside & !level, at, || sample(pu(l.atlas_at), ATLAS_STRIDE as u32, fx, fy));
                let take = (one - a) * (cover * ba.hi_f());
                let v = [rg.lo_f().mul_add(take, v[0]), rg.hi_f().mul_add(take, v[1]), ba.lo_f().mul_add(take, v[2])];
                ((a + take, v, halo.max(hl)), (n + 1.).lt(most))
            },
            |_, s| s,
        );
        let take = (one - a) * (halo * pf(l.halo[3]));
        (a + take, [pf(l.halo[0]).mul_add(take, v[0]), pf(l.halo[1]).mul_add(take, v[1]), pf(l.halo[2]).mul_add(take, v[2])])
    })
}

/// The glyph atlas's texel nearest (fx, fy) (in texels less half a texel: its centre is a whole number there, and a level letter's pixel lands on one within rounding): the letter's coverage and its halo's, 0 to 1.
fn texel(atlas: U, stride: u32, fx: F, fy: F) -> (F, F) {
    let t = round(fy).to_u32().wrapping_mul(U::splat(stride)) + round(fx).to_u32();
    let v = word(atlas + (t >> 1)).shr_by((t & U::splat(1)) << 4);
    let k = 1. / 255.;
    (F::from_u32(v & U::splat(255)) * k, F::from_u32((v >> 8) & U::splat(255)) * k)
}

/// The glyph atlas at (fx, fy) in texels less half a texel (a texel's centre is a whole number there), between the four texels about it: the letter's coverage and its halo's, 0 to 1. Two texels a word (the letter's byte, then its halo's), `stride` texels a row.
fn sample(atlas: U, stride: u32, fx: F, fy: F) -> (F, F) {
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let t0 = y0.to_u32().wrapping_mul(U::splat(stride)) + x0.to_u32();
    // Every read at one address (the word of the first texel, the next word, and the two below them, by their byte offsets): a row's two texels are that word's from the first texel's half on, run on into the next word's, the second shift past 16 leaving nothing of the next word where the first texel starts its word.
    let w = atlas + (t0 >> 1);
    let s = (t0 & U::splat(1)) << 4;
    let back = U::splat(16) - s;
    let below = stride >> 1;
    let pair = |a: U, b: U| word(a).shr_by(s) | (word(b).shl_by(back) << 16);
    let (top, bottom) = (pair(w, w + U::splat(1)), pair(w + U::splat(below), w + U::splat(below + 1)));
    let byte = |v: U, k: u32| F::from_u32((v >> (8 * k)) & U::splat(255));
    let lerp = |a: F, b: F, k: F| (b - a).mul_add(k, a);
    let k = 1. / 255.;
    let at = |c: u32| lerp(lerp(byte(top, c), byte(top, c + 2), tx), lerp(byte(bottom, c), byte(bottom, c + 2), tx), ty) * k;
    (at(0), at(1))
}

/// The ring over `v` at the pixel centred at (sx, sy): `mix(coverage, v)` where the pixel is within the ring's reach of its circle, `v` elsewhere, the work branched over by every warp beside it (most of the screen).
fn ring(r: &Ring, sx: F, sy: F, v: [F; 3], mix: impl FnOnce(F, [F; 3]) -> [F; 3]) -> [F; 3] {
    // (x, y) from the centre, y up: the arc's point at turn a is radius · (sin a, cos a).
    let x = sx - pf(r.centre[0]);
    let y = pf(r.centre[1]) - sy;
    let d = x.mul_add(x, y * y).sqrt();
    let radius = pf(r.radius);
    let reach = pf(r.half_w + 0.5);
    // Further than its reach from the circle a pixel is further from the arc and its ends too (each end is on the circle): no coverage.
    let near = (d - radius).abs().lt(reach);
    ssa::when(near, v, || {
        // Within the sweep: past the start (x ≥ 0 covers the first half turn), not past the end (sin(a − sweep) ≤ 0).
        let [s, c] = r.end;
        let first = x.ge(F::splat(0.));
        let before_end = (y * pf(s)).ge(x * pf(c));
        let inside = if r.long { first | before_end } else { first & before_end };
        // Outside it, the nearer end: the start (0, radius) or the end radius · (sin, cos).
        let ys = y - radius;
        let (ex, ey) = (x - pf(r.radius * s), y - pf(r.radius * c));
        let cap = x.mul_add(x, ys * ys).min(ex.mul_add(ex, ey * ey)).sqrt();
        let dist = F::select(inside, (d - radius).abs(), cap);
        let t = unit(reach - dist);
        mix(t * t * (F::splat(3.) - t * 2.), v)
    })
}

/// A colour in light, f16 (valhall's v2f16: one instruction for two channels, ferros GPU-KISTS.md, the owner on f16: "most of the mahere composite work can either be done in integer or f16"): red and green in one register's halves, blue in both halves of the other. 11 bits of mantissa: a light of 1 is within 2^-11 of its f32, a fifth of an 8-bit code.
#[derive(Clone, Copy)]
struct C {
    rg: H,
    b: H,
}

impl C {
    fn param(c: [f32; 3]) -> C {
        C { rg: H::param2(c[0], c[1]), b: H::param2(c[2], c[2]) }
    }
    /// A table row (`ROW` words from `row`): its colour, and its second word's high half (a line's or a land class's opacity over 255).
    fn row(row: U) -> (C, H) {
        let b = H::from_bits(word(row + U::splat(1)));
        (C { rg: H::from_bits(word(row)), b: b.lo() }, b.hi())
    }
    fn sel(m: M, a: C, b: C) -> C {
        C { rg: H::select(m, a.rg, b.rg), b: H::select(m, a.b, b.b) }
    }
    fn mul(self, o: C) -> C {
        C { rg: self.rg * o.rg, b: self.b * o.b }
    }
    fn scale(self, s: H) -> C {
        C { rg: self.rg * s, b: self.b * s }
    }
    fn lerp(self, o: C, t: H) -> C {
        C { rg: (o.rg - self.rg).mul_add(t, self.rg), b: (o.b - self.b).mul_add(t, self.b) }
    }
    /// As f32, each channel widened by the instruction that reads it.
    fn f32(self) -> [F; 3] {
        [self.rg.lo_f(), self.rg.hi_f(), self.b.lo_f()]
    }
}

/// A byte's value as an f16 in both lanes (exact: bytes are whole numbers below 2^11).
fn byte_h(b: U) -> H {
    H::splat_f(F::from_u32(b))
}

/// The map's light at the pixel (sx, sy), corner (x, y): raster.rs `compose_rgb`, in f16 (`C`) from the terrain's light on, its stretches branched over where a warp has none of them.
fn map(c: &Consts, x: F, y: F, sx: F, sy: F, disk: [F; 3], bg: C) -> C {
    let mask = c.mask;
    let zero = F::splat(0.);
    let one = F::splat(1.);
    let (zero_h, one_h) = (H::splat(0.), H::splat(1.));
    let no = M::splat(false);
    // The cell and texel at the pixel's corner, where the CPU raster samples them (its lattice steps from a block's corner a whole pixel at a time); the light and the disk at its centre, as there.
    let (u, v, face) = locate(c, x, y);
    let bx = |role: usize| &face[F_BOX + role * F_BOX_WORDS..F_BOX + (role + 1) * F_BOX_WORDS];
    // Each role's cell-grid coordinates, the shift a param (the depth changes with the zoom); one shift where both roles are at one depth.
    let grid = |depth: u32| {
        let s = pu(30 - depth);
        (s, u.shr_by(s), v.shr_by(s))
    };
    let dem_grid = grid(c.dem_depth);
    let vec_grid = if c.vec_depth == c.dem_depth { dem_grid } else { grid(c.vec_depth) };

    // The terrain texel: elevation and the snorm16 normal (east, north, up).
    let wants_dem = mask & (DEM | CONTOURS | SLOPE) != 0;
    let (dem_at, dem_base) = if wants_dem { role_texel(square(bx(0), dem_grid.1, dem_grid.2), u, v, 2) } else { (U::splat(0), U::splat(0)) };
    let have_dem_cell = if wants_dem { dem_base.ne(U::splat(0)) } else { no };
    // The vector planes, packed a texel in two words: line class, coverage, magnitude and the water's coverage; land class and coverage.
    let (vec_at, vec_base) = role_texel(square(bx(1), vec_grid.1, vec_grid.2), u, v, 2);
    let have_vec = vec_base.ne(U::splat(0));
    let (w0, w1) = (word(vec_at), word(vec_at + U::splat(1)));
    let byte = |w: U, k: u32| (w >> (8 * k)) & U::splat(255);
    let (lcls, lcov, lmag, wcov) = (byte(w0, 0), byte(w0, 1), byte(w0, 2), w0 >> 24);
    let (acls, acov) = (byte(w1, 0), byte(w1, 1));
    let wet = U::select(have_vec, wcov, U::splat(0)).ne(U::splat(0));

    // The terrain's shading, where a cell stands in: its light ((red, green) and (blue, the sun's diffuse)), its tint, the contours. Without a cell, what a pixel takes is what the CPU's does where no texel is (the background's tint, unlit), and the loading mesh goes over it below.
    let tint0 = if mask & DEM != 0 { bg } else { C::param(c.no_dem) };
    let shaded0 = (tint0, one_h, one_h, no, no, zero, no);
    let (tint, light_rg, light_bd, ground, seabed, contour, contour_index) = if wants_dem {
        ssa::when(have_dem_cell, shaded0, || {
            let lo = word(dem_at);
            let hi = word(dem_at + U::splat(1));
            let eq = lo & U::splat(0xFFFF);
            let texel = eq.ne(U::splat(NODATA));
            // snorm16 halves as the unit normal: the high half read as the top of a signed word, the low one shifted there first.
            let k16 = 1. / (65536. * N1);
            let n = [F::from_i32(lo & U::splat(0xFFFF_0000)) * k16, F::from_i32(hi << 16) * k16, F::from_i32(hi & U::splat(0xFFFF_0000)) * k16];
            let elev = F::from_u32(eq).mul_add(F::splat(ELEV_STEP), F::splat(-ELEV_OFFSET));
            let (mut tint, mut light_rg, mut light_bd, mut ground, mut seabed, mut contour, mut contour_index) = shaded0;
            if mask & DEM != 0 {
                let q = view_normal(disk, pf3(c.pole), sx, sy, n);
                // The lighting environment's quadratic form (`sh::Quad`, k₉ + Σ kₜ mₜ over q², the cross terms and q), red and green in one f16 pair, blue in the other with the sun's diffuse (q · sun) beside it: 18 paired FMAs for the CPU's 30 and its 3-term dot. Each term a lane of three products.
                let a = H::pack(q[0], q[1]);
                let b = H::pack(q[2], q[0] * q[1]);
                let sq = a * a;
                let xz = a * b.lo();
                let zz = b * b.lo();
                let terms = [sq.lo(), sq.hi(), zz.lo(), b.hi(), xz.lo(), xz.hi(), a.lo(), a.hi(), b.lo()];
                let k = &c.light;
                let mut e_rg = H::param2(k[0][9], k[1][9]);
                let mut e_bd = H::param2(k[2][9], 0.);
                for (t, &m) in terms.iter().enumerate() {
                    e_rg = m.mul_add(H::param2(k[0][t], k[1][t]), e_rg);
                    let sun = if t >= 6 { c.sun[t - 6] } else { 0. };
                    e_bd = m.mul_add(H::param2(k[2][t], sun), e_bd);
                }
                light_rg = H::select(texel, e_rg.max(zero_h), one_h);
                light_bd = H::select(texel, e_bd.max(zero_h), one_h);
                let sea = elev.abs().lt(F::splat(0.75)) & n[2].gt(F::splat(0.9995));
                let bed = !sea & elev.lt(zero) & wet;
                seabed = texel & bed;
                let t = unit(elev * (-1. / SEABED_FULL_DEPTH));
                let kbed = F::splat(1.) - t * SEABED_DARK;
                let sea_rgb = C::param(c.sea);
                let bed_rgb = sea_rgb.scale(H::splat_f(kbed));
                let land_rgb = if mask & HYPSO != 0 {
                    let row = pu(c.region + (HYPSO_TAB / 4) as u32) + ((eq >> 4) << 1);
                    C { rg: H::from_bits(word(row)), b: H::from_bits(word(row + U::splat(1))) }
                } else {
                    C::param(c.flat)
                };
                let ground_rgb = C::sel(sea, sea_rgb, C::sel(bed, bed_rgb, land_rgb));
                tint = C::sel(texel, ground_rgb, bg);
                ground = texel;
            }
            if mask & CONTOURS != 0 && c.interval > 0. {
                let nzn = n[2].max(F::splat(1e-4));
                let slope = (one - nzn * nzn).max(zero).sqrt() * nzn.rcp();
                let level = round(elev * pf(1. / c.interval));
                let d_m = (elev - level * pf(c.interval)).abs();
                let d_px = d_m * (slope * pf(c.m_per_px)).rcp();
                let ie = c.index_every as f32;
                // level is a whole number: it is a multiple of n exactly when ⌊(level + ½) / n⌋ · n is level (the half keeps the reciprocal's rounding off the floor's edge).
                let index = ((level + 0.5) * pf(1. / ie)).floor().mul_add(pf(ie), -level).abs().lt(F::splat(0.5));
                let w = sel(index, F::splat(0.9), F::splat(0.55));
                contour = sel(texel & slope.ge(F::splat(CONTOUR_MIN_SLOPE)), unit(w + 0.5 - d_px), zero);
                contour_index = index;
            }
            (tint, light_rg, light_bd, ground, seabed, contour, contour_index)
        })
    } else {
        shaded0
    };
    let light = C { rg: light_rg, b: light_bd.lo() };
    let diffuse = light_bd.hi();
    let ink = C::sel(contour_index, C::param([c.contour_index[0], c.contour_index[1], c.contour_index[2]]), C::param([c.contour[0], c.contour[1], c.contour[2]]));
    let ink_a = H::splat_f(contour) * H::select(contour_index, H::param2(c.contour_index[3], c.contour_index[3]), H::param2(c.contour[3], c.contour[3]));

    let lit = |rgb: C| C::sel(ground, rgb.mul(light), rgb);
    // Without vector planes: the light, then the contours.
    let plain = lit(tint).lerp(ink, ink_a);
    // With them: land cover, the light, the contours, the waterways, the water, the other lines (raster.rs `compose_rgb`).
    let water = mask & WATER != 0;
    let water_rgb = C::param([c.water[0], c.water[1], c.water[2]]);
    let water_a = H::param2(c.water[3] / 255., c.water[3] / 255.);
    let land_tab = pu(c.region + (LAND_TAB / 4) as u32);
    let line_tab = pu(c.region + (LINE_TAB / 4) as u32);
    let waterway_mag = pu(c.waterway_mag);
    let rgb = ssa::when(have_vec, plain, || {
        let mut rgb = tint;
        if mask & LAND != 0 {
            // Land cover where there is any: the lerp by 0 is the tint itself, so a warp with none (the sea) skips its table.
            rgb = ssa::when(acov.ne(U::splat(0)), rgb, || {
                let k = U::select(acls.lt(U::splat(LAND_MAX + 1)), acls, U::splat(LAND_MAX));
                let (col, a) = C::row(land_tab + (k << 1));
                rgb.lerp(col, byte_h(acov) * a)
            });
        }
        rgb = lit(rgb).lerp(ink, ink_a);
        // The water over what is under it, where there is any.
        let over_water = |rgb: C| -> C {
            if !water {
                return rgb;
            }
            ssa::when(wcov.ne(U::splat(0)), rgb, || {
                let s = diffuse.mul_add(H::splat(0.15), H::splat(0.85));
                let t = byte_h(wcov) * water_a * H::select(seabed, H::splat(SEABED_GLAZE), one_h);
                rgb.lerp(water_rgb.scale(s), t)
            })
        };
        // A line, where there is one that draws: a waterway under the water, any other over it; elsewhere the water alone.
        let k = U::select(lcls.lt(U::splat(CLASS_MAX + 1)), lcls, U::splat(CLASS_MAX));
        let waterway = k.eq(U::splat(WATERWAY));
        // The class's opacity (the host folds in whether the class draws at this depth and the boundaries' 0.55), a waterway's also gated by its magnitude.
        let gate = !waterway | lmag.ge(waterway_mag);
        let line = gate & lcov.ne(U::splat(0));
        ssa::when(line, over_water(rgb), || {
            let (col, a) = C::row(line_tab + (k << 1));
            let dim = sel(waterway, F::from_u32(lmag).mul_add(F::splat(0.6 / 255.), F::splat(0.4)), one);
            let lcol = col.scale(H::splat_f(dim));
            let la = byte_h(lcov) * a;
            let rgb = rgb.lerp(lcol, H::select(waterway, la, zero_h));
            let rgb = over_water(rgb);
            rgb.lerp(lcol, H::select(waterway, zero_h, la))
        })
    });

    // No cell at any depth for the terrain: each triangle of the wanted depth's cell grid its own dark colour, raster.rs `loading_noise`'s hash here (its diamond and depth terms the face's seed).
    if !wants_dem {
        return rgb;
    }
    ssa::when(dem_base.eq(U::splat(0)), rgb, || {
        let (s, cu, cv) = dem_grid;
        let one = U::splat(1);
        let m = one.shl_by(s) - one;
        let half = ((u & m) + (v & m)).shr_by(s) & one;
        let h = face[F_SEED] ^ cu.wrapping_mul(U::splat(0x9E37_79B1)) ^ cv.wrapping_mul(U::splat(0x85EB_CA77)) ^ half.wrapping_mul(U::splat(0x1656_67B1));
        let h = h ^ (h >> 15);
        let h = h.wrapping_mul(U::splat(0x2C1B_3C6D));
        let h = h ^ (h >> 12);
        // Each channel a gamma-2 byte's light, (b / 256)²: the byte over 256 in f16, squared there.
        let ch = |k: u32| F::from_u32(((h >> (8 * k)) & U::splat(255)) >> 2) * (1. / 256.);
        let g = C { rg: H::pack(ch(0), ch(1)), b: H::splat_f(ch(2)) };
        g.mul(g)
    })
}

/// The pixel at screen point (sx, sy): its depth-30 (u, v) on its face's diamond, and its face's row.
fn locate(c: &Consts, sx: F, sy: F) -> (U, U, [U; F_USED]) {
    let zero = F::splat(0.);
    let one = F::splat(1.);
    // The pixel's direction in view coordinates: (x, y) on the screen's plane in the globe's radii, z toward the viewer (0 past the limb, where the background draws), and Z the third coordinate the faces take, z − 1 measured from the camera point (from 1 − x² − y² as −q / (1 + z), with no cancellation at a deep zoom), else z.
    let [k, kx, ky] = c.view;
    let k = pf(k);
    let px = sx.mul_add(k, pf(kx));
    let py = sy.mul_add(-k, pf(ky));
    let q = px.mul_add(px, py * py);
    let z = (one - q).max(zero).sqrt();
    let pz = if c.near { -(q * (one + z).rcp()) } else { z };
    // The face: the frame's one, or the nearest centroid of its faces (the first of equals, as the encoder takes it), its row read by the face.
    let faces = &c.faces[..c.face_count];
    let face: [U; F_USED] = if let Some(fold) = &c.fold {
        // Into the fold's frame, the nearest of its four centres to |q| (the first of equals), and q's octant from its signs: the face's row, `octant << 2 | k`, a row of 2^FACE_SHIFT words.
        let q: [F; 3] = core::array::from_fn(|i| pf(fold.turn[i][0]).mul_add(px, pf(fold.turn[i][1]).mul_add(py, pz * pf(fold.turn[i][2]))));
        let a = q.map(F::abs);
        // The four centres are the fold's own, the same every frame (constants of the recording, their zeros its structure).
        let score = |k: usize| {
            let terms: Vec<usize> = (0..3).filter(|&i| fold.centres[k][i] != 0.).collect();
            terms[1..].iter().fold(a[terms[0]] * fold.centres[k][terms[0]], |s, &i| F::splat(fold.centres[k][i]).mul_add(a[i], s))
        };
        let mut best = score(0);
        let mut n = U::splat(0);
        for k in 1..4 {
            let s = score(k);
            let more = s.gt(best);
            best = F::select(more, s, best);
            n = U::select(more, U::splat(k as u32), n);
        }
        // Each sign bit (31) to its octant bit's place in the row's offset: x's at FACE_SHIFT + 2, y's at + 3, z's at + 4.
        let sign = |i: usize| (q[i].to_bits() >> (31 - FACE_SHIFT - 2 - i as u32)) & U::splat(1 << (FACE_SHIFT + 2 + i as u32));
        let at = pu(c.faces_at) + (sign(0) | sign(1) | sign(2) | n << FACE_SHIFT);
        core::array::from_fn(|w| word(at + U::splat(w as u32)))
    } else if faces.len() == 1 {
        core::array::from_fn(|w| pu(faces[0].row[w]))
    } else {
        let score = |f: &FaceConsts| pf(f.score[0]).mul_add(px, pf(f.score[1]).mul_add(py, pf(f.score[2]).mul_add(pz, pf(f.score[3]))));
        let mut best = score(&faces[0]);
        let mut n = U::splat(0);
        for (j, f) in faces.iter().enumerate().skip(1) {
            let s = score(f);
            let more = s.gt(best);
            best = F::select(more, s, best);
            n = U::select(more, U::splat(j as u32), n);
        }
        let at = pu(c.faces_at) + (n << FACE_SHIFT);
        core::array::from_fn(|w| word(at + U::splat(w as u32)))
    };
    let rf = |w: usize| F::from_bits(face[w]);
    let lin = |o: usize| rf(o).mul_add(px, rf(o + 1).mul_add(py, rf(o + 2) * pz));
    let den = if c.near { lin(F_DEN) + 1. } else { lin(F_DEN) };
    let rd = den.rcp();
    // Depth-30 (u, v): the camera point's whole part plus the floor of the rest. WHY the hold: on a seam the face the f32 search chose can disagree with the exact one by less than a unit, and a coordinate of −1 or 2^30 would name a cell outside the diamond (no square: a speck of the loading mesh along the seam); held inside, it is the edge's own cell, as the encoder holds it.
    let coord = |o: usize, k: usize| {
        let t = lin(o).mul_add(rd, rf(F_FRAC + k)).floor().to_i32() + face[F_REF + k];
        let below = t.ge(U::splat(1 << 31));
        let over = t.ge(U::splat(1 << 30));
        U::select(over, U::select(below, U::splat(0), U::splat((1 << 30) - 1)), t)
    };
    (coord(F_U, 0), coord(F_V, 1), face)
}

/// A kernel that stores what each pixel finds and nothing else, eight words a pixel from uniform 0: its depth-30 (u, v), its terrain square and texel address, its face's seed (which face it took). The projection and the page table checked against the host's.
#[cfg(test)]
pub fn record_locate(c: &Consts) -> ssa::Compiled {
    ssa::begin();
    ssa::reserve(2);
    ssa::uniform_slots(64);
    let i = ssa::gid();
    let (x, y) = column_row(F::from_u32(i), c.w);
    let (u, v, face) = locate(c, x, y);
    let s = pu(30 - c.dem_depth);
    let sq = square(&face[F_BOX..F_BOX + F_BOX_WORDS], u.shr_by(s), v.shr_by(s));
    let (at, _) = role_texel(sq, u, v, 2);
    for (k, w) in [u, v, sq, at, face[F_SEED]].into_iter().enumerate() {
        ssa::store_at(SLOT_FRAME, i, 5, 4 * k as i16, w);
    }
    ssa::compile()
}
