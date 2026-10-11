//! Notes on the map and areas of it: the user's own layer of cells beside the bucket's (Nick 2026-10-10).
//!
//! A stroke of the pen or a drawn selection is never kept as a vector. It is stamped into cells at whatever depth the view is at when it is drawn, one cell wherever the ink touches, each cell an `ink` plane (premultiplied VSF RGB at gamma 2 and an alpha, four bytes a texel) and a `sel` plane (the selection's coverage, the shape a water plane has). Above the drawn depth the parents are the mean of their children in linear light, as imagery is, so a mark goes light as the map zooms out and never tramples what is under it; the selection's parents are its mean coverage.
//!
//! A user cell is encoded as an ordinary cell whose imagery plane is the ink (lossless) and whose water plane is the selection, so the vault, the codec and the renderers' planes already know it.

use mahere_coord::Coord;
use mahere_tiles::{CellKey, CovCell, ImgCell, TEX, TRI, child_cell_texels, tri_idx};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::Camera;
use crate::raster::tri_index;

/// The pens, VSF RGB at gamma 2: red, orange, yellow, green, cyan, blue, magenta, black. A stroke keeps its own colour; the list is only what the panel offers.
pub const PENS: [[u8; 3]; 8] = [[230, 40, 40], [240, 140, 30], [240, 220, 40], [60, 200, 70], [40, 210, 220], [50, 110, 240], [220, 60, 220], [16, 16, 16]];

/// The selection's wash, linear, and how much of it.
pub const SEL_TINT: [f32; 3] = [0.35, 0.6, 1.0];
pub const SEL_ALPHA: f32 = 0.35;

/// How far above a drawn depth the parents are kept.
pub const USER_MIN_DEPTH: u8 = 0;

/// What a single touch on the map does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tool {
    #[default]
    None,
    Pen,
    Select,
}

/// Where the user's cells live between runs: the phone's vault, a directory on a desktop.
pub trait UserStore: Send + Sync {
    fn put(&self, key: CellKey, bytes: &[u8]);
    fn delete(&self, key: CellKey);
    fn load_all(&self) -> Vec<(CellKey, Vec<u8>)>;
}

/// The ink plane: per triangle texel, premultiplied VSF RGB at gamma 2 (the light already times the alpha, then the square root) and the alpha, four bytes.
#[derive(Clone)]
pub struct InkCell {
    pub rgba: Vec<u8>,
}

impl InkCell {
    pub fn new() -> InkCell {
        InkCell { rgba: vec![0; TRI * 4] }
    }

    #[inline]
    pub fn alpha(&self, i: usize) -> u8 {
        self.rgba[4 * i + 3]
    }

    #[inline]
    pub fn px(&self, i: usize) -> [u8; 4] {
        let p = &self.rgba[4 * i..4 * i + 4];
        [p[0], p[1], p[2], p[3]]
    }

    #[inline]
    pub fn set(&mut self, i: usize, p: [u8; 4]) {
        self.rgba[4 * i..4 * i + 4].copy_from_slice(&p);
    }

    pub fn is_empty(&self) -> bool {
        self.rgba.chunks_exact(4).all(|p| p[3] == 0)
    }
}

impl Default for InkCell {
    fn default() -> InkCell {
        InkCell::new()
    }
}

/// A gamma-2 byte's light, and back by truncation, as everywhere in the map.
#[inline]
fn lin_b(b: u8) -> f32 {
    let x = b as f32 / 256.0;
    x * x
}

#[inline]
fn enc_b(x: f32) -> u8 {
    (x.clamp(0.0, 1.0).sqrt() * 256.0).min(255.0) as u8
}

/// A stored texel as linear light: premultiplied r, g, b and the alpha as a fraction.
#[inline]
fn unpack(p: [u8; 4]) -> [f32; 4] {
    [lin_b(p[0]), lin_b(p[1]), lin_b(p[2]), p[3] as f32 / 255.0]
}

#[inline]
fn pack(l: [f32; 4]) -> [u8; 4] {
    [enc_b(l[0]), enc_b(l[1]), enc_b(l[2]), (l[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8]
}

#[derive(Clone, Default)]
pub struct UserCell {
    pub ink: Option<InkCell>,
    pub sel: Option<CovCell>,
    /// Bumped whenever the cell's planes change, so a renderer uploads only what changed.
    pub version: u64,
}

impl UserCell {
    pub fn is_empty(&self) -> bool {
        self.ink.as_ref().is_none_or(|c| c.is_empty()) && self.sel.as_ref().is_none_or(|c| c.is_empty())
    }
}

/// Every user cell, drawn or built above a drawn one, at every depth, with what has changed since the last save.
#[derive(Clone, Default)]
pub struct UserLayer {
    pub cells: FxHashMap<CellKey, UserCell>,
    /// The cells drawn into (not built), so the levels above are rebuilt from the right places.
    drawn: FxHashMap<CellKey, ()>,
    dirty: FxHashSet<CellKey>,
    /// Bumped on every change, for a renderer to know its planes are stale.
    pub version: u64,
    /// The cells a stroke in progress has touched, as they were before it (None: not there), for undoing it; and whether the levels above still wait to be rebuilt (deferred to the stroke's end, since mid-stroke the view reads the drawn cells themselves).
    undo: Option<FxHashMap<CellKey, Option<UserCell>>>,
    stale_above: bool,
    /// The cells changed since the levels above them were last rebuilt: only their ancestor chains are rebuilt, not every level of every cell.
    touched_since: FxHashSet<CellKey>,
}

/// A screen point's place on the diamond grid at full precision: the diamond and (u, v) in Q30.16, or nothing off the globe.
fn grid_at(cam: &Camera, px: f64, py: f64, w: usize, h: usize) -> Option<(u8, i64, i64)> {
    if !cam.on_globe(px, py, w, h) {
        return None;
    }
    let (lat, lon) = cam.screen_to_geo(px, py, w, h);
    let c = Coord::from_lat_lon(lat, lon);
    let (iu, iv) = c.uv();
    Some((c.diamond(), (iu as i64) << 16, (iv as i64) << 16))
}

/// The cell and triangle index of a Q30.16 grid position at `depth`.
fn texel_of(diamond: u8, uq: i64, vq: i64, depth: u8) -> (CellKey, usize) {
    let s = 30 - depth as u32;
    let key = CellKey::from_grid(diamond, depth, (uq >> 16) as u64 >> s, (vq >> 16) as u64 >> s);
    (key, tri_index(uq, vq, 16 + 22 - depth as u32))
}

/// The texel a screen point falls in at `depth`: the cell and the triangle index, or nothing off the globe.
fn texel_at(cam: &Camera, depth: u8, px: f64, py: f64, w: usize, h: usize) -> Option<(CellKey, usize)> {
    let (d, uq, vq) = grid_at(cam, px, py, w, h)?;
    Some(texel_of(d, uq, vq, depth))
}

/// The map from a finer cell's texels to those of its ancestor `anc`: the ancestor texel by shifting, its lower or upper half by where the fine triangle's centroid falls against that texel's diagonal (the lower triangle's centroid at a third, the upper's at two thirds; times three to stay in integers).
fn map_to_ancestor(key: CellKey, anc: CellKey) -> Vec<usize> {
    let levels = (key.depth - anc.depth) as u32;
    let (cu, cv) = key.grid();
    let (pu, pv) = anc.grid();
    let (ou, ov) = ((cu - (pu << levels)) as usize, (cv - (pv << levels)) as usize);
    let mut map = vec![0usize; TRI];
    let p = 1usize << levels;
    for ty in 0..TEX {
        for tx in 0..TEX {
            let (gx, gy) = ((ou << 8) + tx, (ov << 8) + ty);
            let (sx, sy) = (gx >> levels, gy >> levels);
            let (fx, fy) = (gx & (p - 1), gy & (p - 1));
            for half in 0..2 {
                let upper = 3 * (fx + fy) + 2 + 2 * half >= 3 * p;
                map[((ty << 8) | tx) << 1 | half] = ((sy << 8) | sx) << 1 | upper as usize;
            }
        }
    }
    map
}

/// A parent ink cell from its four children (missing ones transparent): each texel the mean of its four child texels in linear light, as imagery is averaged, so a mark goes light as the map zooms out.
fn ink_parent(kids: [Option<&InkCell>; 4]) -> InkCell {
    let mut out = InkCell::new();
    for ty in 0..TEX {
        for tx in 0..TEX {
            for half in 0..2 {
                let mut sum = [0.0f32; 4];
                for (q, i) in child_cell_texels(tx, ty, half) {
                    if let Some(k) = kids[q as usize] {
                        let l = unpack(k.px(i));
                        for c in 0..4 {
                            sum[c] += l[c];
                        }
                    }
                }
                if sum[3] > 0.0 {
                    out.set(tri_idx(tx, ty, half), pack([sum[0] * 0.25, sum[1] * 0.25, sum[2] * 0.25, sum[3] * 0.25]));
                }
            }
        }
    }
    out
}

/// A parent selection cell from its four children: the mean coverage.
fn sel_parent(kids: [Option<&CovCell>; 4]) -> CovCell {
    let mut out = CovCell::new();
    for ty in 0..TEX {
        for tx in 0..TEX {
            for half in 0..2 {
                let mut sum = 0u32;
                for (q, i) in child_cell_texels(tx, ty, half) {
                    if let Some(k) = kids[q as usize] {
                        sum += k.cov[i] as u32;
                    }
                }
                out.cov[tri_idx(tx, ty, half)] = (sum / 4) as u8;
            }
        }
    }
    out
}

impl UserLayer {
    /// A stroke of the pen along screen points, each with its width in pixels (the pen's pressure), in colour `colour` (VSF RGB, gamma 2) at opacity `alpha`: every pixel within half the width of the path, stamped into the cells at `depth`, the brush soft over its last pixel.
    pub fn stroke(&mut self, cam: &Camera, w: usize, h: usize, depth: u8, pts: &[(f64, f64, f64)], colour: [u8; 3], alpha: f32) {
        let mut touched: FxHashSet<CellKey> = FxHashSet::default();
        let lin = [lin_b(colour[0]), lin_b(colour[1]), lin_b(colour[2])];
        // Sampled every half pixel: a texel at the drawing depth is one to two pixels wide, so every texel under the brush is hit and the renderer, reading one texel a pixel, finds no gaps. The grid position is encoded only at the stamp's four corners and interpolated inside (the footprint is a few pixels, over which the grid is linear to well under a texel), as the renderer does with its blocks; a stamp whose corners straddle two diamonds is encoded sample by sample.
        let mut stamp = |x: f64, y: f64, width: f64| {
            let r = (width * 0.5).max(0.5);
            let (x0, x1) = (((x - r) * 2.0).floor() as i64, ((x + r) * 2.0).ceil() as i64);
            let (y0, y1) = (((y - r) * 2.0).floor() as i64, ((y + r) * 2.0).ceil() as i64);
            let (sx0, sy0) = (x0 as f64 * 0.5 + 0.25, y0 as f64 * 0.5 + 0.25);
            let (sx1, sy1) = (x1 as f64 * 0.5 + 0.25, y1 as f64 * 0.5 + 0.25);
            let corners = [grid_at(cam, sx0, sy0, w, h), grid_at(cam, sx1, sy0, w, h), grid_at(cam, sx0, sy1, w, h), grid_at(cam, sx1, sy1, w, h)];
            let linear = match corners {
                [Some(a), Some(b), Some(c), Some(d)] if a.0 == b.0 && a.0 == c.0 && a.0 == d.0 => Some((a, b, c, d)),
                _ => None,
            };
            let (nx, ny) = ((x1 - x0).max(1) as f64, (y1 - y0).max(1) as f64);
            for py in y0..=y1 {
                for px in x0..=x1 {
                    let (cx, cy) = (px as f64 * 0.5 + 0.25, py as f64 * 0.5 + 0.25);
                    let dist = ((cx - x) * (cx - x) + (cy - y) * (cy - y)).sqrt();
                    if dist > r {
                        continue;
                    }
                    // Soft over the last pixel of the radius.
                    let edge = (r - dist).clamp(0.0, 1.0) as f32;
                    let a = alpha * edge;
                    if a <= 0.0 {
                        continue;
                    }
                    let texel = match linear {
                        Some((a0, b0, c0, d0)) => {
                            let (fx, fy) = ((px - x0) as f64 / nx, (py - y0) as f64 / ny);
                            let lerp = |p: i64, q: i64, t: f64| p as f64 + (q - p) as f64 * t;
                            let u = lerp(a0.1, b0.1, fx) + (lerp(c0.1, d0.1, fx) - lerp(a0.1, b0.1, fx)) * fy;
                            let v = lerp(a0.2, b0.2, fx) + (lerp(c0.2, d0.2, fx) - lerp(a0.2, b0.2, fx)) * fy;
                            Some(texel_of(a0.0, u.round() as i64, v.round() as i64, depth))
                        }
                        None => texel_at(cam, depth, cx, cy, w, h),
                    };
                    if let Some((key, i)) = texel {
                        let ink = self.ink_cell(key);
                        // A stroke's stamps overlap: the brush only ever raises a texel toward its own colour at its alpha, so the overlaps do not darken, and earlier ink of another colour shows through by what the stroke leaves.
                        let have = ink.alpha(i) as f32 / 255.0;
                        if a > have {
                            let d = unpack(ink.px(i));
                            let k = 1.0 - a;
                            let same = d[3] > 0.0 && (d[0] - lin[0] * d[3]).abs() + (d[1] - lin[1] * d[3]).abs() + (d[2] - lin[2] * d[3]).abs() < 0.02;
                            let mixed = if same || d[3] == 0.0 { [lin[0] * a, lin[1] * a, lin[2] * a, a] } else { [lin[0] * a + d[0] * k, lin[1] * a + d[1] * k, lin[2] * a + d[2] * k, a + d[3] * k] };
                            ink.set(i, pack(mixed));
                        }
                        touched.insert(key);
                    }
                }
            }
        };
        match pts {
            [] => {}
            [p] => stamp(p.0, p.1, p.2),
            _ => {
                for s in pts.windows(2) {
                    let (ax, ay, bx, by) = (s[0].0, s[0].1, s[1].0, s[1].1);
                    let n = ((bx - ax).hypot(by - ay) * 2.0).ceil().max(1.0) as usize;
                    for k in 0..=n {
                        let t = k as f64 / n as f64;
                        stamp(ax + (bx - ax) * t, ay + (by - ay) * t, s[0].2 + (s[1].2 - s[0].2) * t);
                    }
                }
            }
        }
        for k in touched.iter().copied().collect::<Vec<_>>() {
            self.push_down(k);
        }
        self.after_change(depth, touched);
    }

    /// Whether the selection covers a screen point: the finest cell with a selection plane there decides.
    pub fn selected_at(&self, cam: &Camera, w: usize, h: usize, depth: u8, px: f64, py: f64) -> bool {
        let Some((d, uq, vq)) = grid_at(cam, px, py, w, h) else { return false };
        let (key, _) = texel_of(d, uq, vq, depth);
        let mut k = key;
        loop {
            if let Some(sel) = self.cells.get(&k).and_then(|c| c.sel.as_ref()) {
                let shift = 16 + 22 - k.depth as u32;
                return sel.cov[tri_index(uq, vq, shift)] >= 128;
            }
            if k.depth == USER_MIN_DEPTH {
                return false;
            }
            k = k.parent();
        }
    }

    /// A closed shape drawn on the screen, added to the selection or taken from it: every pixel inside the polygon (even-odd), stamped into the cells at `depth`.
    pub fn select(&mut self, cam: &Camera, w: usize, h: usize, depth: u8, poly: &[(f64, f64)], add: bool) {
        if poly.len() < 3 {
            return;
        }
        let (mut y0, mut y1) = (f64::MAX, f64::MIN);
        for p in poly {
            y0 = y0.min(p.1);
            y1 = y1.max(p.1);
        }
        let y0 = y0.floor().max(0.0) as i64;
        let y1 = y1.ceil().min(h as f64 - 1.0) as i64;
        let mut touched: FxHashSet<CellKey> = FxHashSet::default();
        let mut xs: Vec<f64> = Vec::new();
        // Every half pixel, as a stroke is stamped, so no texel inside the shape is missed.
        for py2 in (y0 * 2)..=(y1 * 2 + 1) {
            let yc = py2 as f64 * 0.5 + 0.25;
            xs.clear();
            for s in 0..poly.len() {
                let (a, b) = (poly[s], poly[(s + 1) % poly.len()]);
                if (a.1 <= yc) != (b.1 <= yc) {
                    xs.push(a.0 + (yc - a.1) / (b.1 - a.1) * (b.0 - a.0));
                }
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for pair in xs.chunks(2) {
                if pair.len() < 2 {
                    break;
                }
                let (x0, x1) = ((pair[0] * 2.0).round().max(0.0) as i64, (pair[1] * 2.0).round().min(w as f64 * 2.0) as i64);
                for px2 in x0..x1 {
                    if let Some((key, i)) = texel_at(cam, depth, px2 as f64 * 0.5 + 0.25, yc, w, h) {
                        let sel = self.sel_cell(key);
                        sel.cov[i] = if add { 255 } else { 0 };
                        touched.insert(key);
                    }
                }
            }
        }
        self.after_change(depth, touched);
    }

    /// The nearest ancestor of `key` that `has` a plane, drawn or built, with the texel map into it.
    fn ancestor_with(&self, key: CellKey, has: impl Fn(&UserCell) -> bool) -> Option<(CellKey, Vec<usize>)> {
        let mut k = key;
        while k.depth > USER_MIN_DEPTH {
            k = k.parent();
            if self.cells.get(&k).is_some_and(&has) {
                return Some((k, map_to_ancestor(key, k)));
            }
        }
        None
    }

    /// The nearest drawn ancestor that `has` a plane (a built one may be stale mid-rebuild), with the texel map into it.
    fn drawn_ancestor_with(&self, key: CellKey, has: impl Fn(&UserCell) -> bool) -> Option<(CellKey, Vec<usize>)> {
        let mut k = key;
        while k.depth > USER_MIN_DEPTH {
            k = k.parent();
            if self.drawn.contains_key(&k) && self.cells.get(&k).is_some_and(&has) {
                return Some((k, map_to_ancestor(key, k)));
            }
        }
        None
    }

    fn resample_ink(&self, from: CellKey, map: &[usize]) -> InkCell {
        let ink = self.cells[&from].ink.as_ref().unwrap();
        let mut out = InkCell::new();
        for (i, &j) in map.iter().enumerate() {
            out.set(i, ink.px(j));
        }
        out
    }

    fn resample_sel(&self, from: CellKey, map: &[usize]) -> CovCell {
        let sel = self.cells[&from].sel.as_ref().unwrap();
        let mut out = CovCell::new();
        for (i, &j) in map.iter().enumerate() {
            out.cov[i] = sel.cov[j];
        }
        out
    }

    /// What the nearest ancestor with an ink plane says about this cell's texels, resampled into it: a cell drawn at a new depth starts as what was already there, so a stroke at one zoom never clears a mark made at another.
    fn ink_from_parents(&self, key: CellKey) -> Option<InkCell> {
        let (k, map) = self.ancestor_with(key, |c| c.ink.is_some())?;
        Some(self.resample_ink(k, &map))
    }

    fn sel_from_parents(&self, key: CellKey) -> Option<CovCell> {
        let (k, map) = self.ancestor_with(key, |c| c.sel.is_some())?;
        Some(self.resample_sel(k, &map))
    }

    /// A cell about to change: remembered as it was, once, for the stroke's undo.
    fn remember(&mut self, key: CellKey) {
        if let Some(undo) = &mut self.undo {
            if !undo.contains_key(&key) {
                undo.insert(key, self.cells.get(&key).cloned());
            }
        }
    }

    /// The cell at `key` ready for the pen: made if missing, its ink plane seeded from the nearest ancestor's if it had none.
    fn ink_cell(&mut self, key: CellKey) -> &mut InkCell {
        self.remember(key);
        if self.cells.get(&key).is_none_or(|c| c.ink.is_none()) {
            let seed = self.ink_from_parents(key).unwrap_or_default();
            self.cells.entry(key).or_default().ink = Some(seed);
        }
        let cell = self.cells.get_mut(&key).unwrap();
        cell.version += 1;
        cell.ink.as_mut().unwrap()
    }

    /// The cell at `key` ready for a selection: made if missing, its selection plane seeded from the nearest ancestor's if it had none.
    fn sel_cell(&mut self, key: CellKey) -> &mut CovCell {
        self.remember(key);
        if self.cells.get(&key).is_none_or(|c| c.sel.is_none()) {
            let seed = self.sel_from_parents(key).unwrap_or_else(CovCell::new);
            self.cells.entry(key).or_default().sel = Some(seed);
        }
        let cell = self.cells.get_mut(&key).unwrap();
        cell.version += 1;
        cell.sel.as_mut().unwrap()
    }

    /// A coarse cell's ink pushed down into every finer cell beneath it: where the fine texel is clear and the coarse one inked, the fine takes it. So a stroke made zoomed out shows at once through finer cells drawn earlier, the finest cell still deciding.
    fn push_down(&mut self, key: CellKey) {
        let Some(ink) = self.cells.get(&key).and_then(|c| c.ink.clone()) else { return };
        let below: Vec<CellKey> = self.cells.keys().filter(|k| k.depth > key.depth && k.ancestor(key.depth) == key).copied().collect();
        for k in below {
            let map = map_to_ancestor(k, key);
            self.remember(k);
            let cell = self.cells.get_mut(&k).unwrap();
            let fine = cell.ink.get_or_insert_with(InkCell::new);
            let mut changed = false;
            for (i, &j) in map.iter().enumerate() {
                if fine.alpha(i) == 0 && ink.alpha(j) != 0 {
                    fine.set(i, ink.px(j));
                    changed = true;
                }
            }
            if changed {
                cell.version += 1;
                self.dirty.insert(k);
                self.touched_since.insert(k);
            }
        }
    }

    /// A stroke begins: from here every touched cell is remembered, and the levels above wait until it ends.
    pub fn begin_undo(&mut self) {
        self.undo = Some(FxHashMap::default());
    }

    /// The stroke stands: the undo is forgotten and the levels above are rebuilt now.
    pub fn commit_undo(&mut self) {
        self.undo = None;
        if self.stale_above {
            self.rebuild_chains();
            self.stale_above = false;
            self.version += 1;
        }
    }

    /// The stroke is undone: every cell it touched goes back to what it was.
    pub fn rollback_undo(&mut self) {
        if let Some(undo) = self.undo.take() {
            for (k, before) in undo {
                match before {
                    Some(mut c) => {
                        c.version += 1_000_000;
                        self.cells.insert(k, c);
                    }
                    None => {
                        self.cells.remove(&k);
                        self.drawn.remove(&k);
                    }
                }
            }
            self.stale_above = false;
            self.touched_since.clear();
            self.version += 1;
        }
    }

    /// Clear the whole selection.
    pub fn clear_selection(&mut self) {
        let mut touched = FxHashSet::default();
        for (k, c) in self.cells.iter_mut() {
            if c.sel.take().is_some() {
                c.version += 1;
                touched.insert(*k);
            }
        }
        self.after_change(0, touched);
    }

    /// Clear all the ink.
    pub fn clear_ink(&mut self) {
        let mut touched = FxHashSet::default();
        for (k, c) in self.cells.iter_mut() {
            if c.ink.take().is_some() {
                c.version += 1;
                touched.insert(*k);
            }
        }
        self.after_change(0, touched);
    }

    /// After cells changed at `depth`: they count as drawn, the levels above are rebuilt (or, mid-stroke, left for the stroke's end), the version bumped.
    fn after_change(&mut self, depth: u8, touched: FxHashSet<CellKey>) {
        for k in &touched {
            if k.depth == depth {
                self.drawn.insert(*k, ());
            }
        }
        self.touched_since.extend(touched.iter().copied());
        if self.undo.is_some() {
            self.stale_above = true;
        } else {
            self.rebuild_chains();
        }
        for k in touched {
            self.dirty.insert(k);
        }
        self.version += 1;
    }

    /// One level above a set of cells: the parent from its children (ink the mean in linear light, selection the mean coverage), completed from the nearest drawn ancestor where the children give nothing; a drawn parent keeps its own marks, taking from below only where below is louder.
    fn build_parent(&mut self, key: CellKey, built_version: u64) {
        let kids_ink: [Option<InkCell>; 4] = std::array::from_fn(|q| self.cells.get(&key.child(q as u64)).and_then(|c| c.ink.clone()));
        let kids_sel: [Option<CovCell>; 4] = std::array::from_fn(|q| self.cells.get(&key.child(q as u64)).and_then(|c| c.sel.clone()));
        let any_ink = kids_ink.iter().any(|k| k.is_some());
        let any_sel = kids_sel.iter().any(|k| k.is_some());
        let mut new_ink = any_ink.then(|| ink_parent(std::array::from_fn(|q| kids_ink[q].as_ref())));
        let mut new_sel = any_sel.then(|| sel_parent(std::array::from_fn(|q| kids_sel[q].as_ref())));
        if let Some((k, map)) = self.drawn_ancestor_with(key, |c| c.ink.is_some()) {
            let seed = self.resample_ink(k, &map);
            let c = new_ink.get_or_insert_with(InkCell::new);
            for i in 0..TRI {
                if c.alpha(i) == 0 && seed.alpha(i) != 0 {
                    c.set(i, seed.px(i));
                }
            }
        }
        if let Some((k, map)) = self.drawn_ancestor_with(key, |c| c.sel.is_some()) {
            let seed = self.resample_sel(k, &map);
            let c = new_sel.get_or_insert_with(CovCell::new);
            for i in 0..TRI {
                if c.cov[i] == 0 {
                    c.cov[i] = seed.cov[i];
                }
            }
        }
        let drawn = self.drawn.contains_key(&key);
        let entry = self.cells.entry(key).or_default();
        if drawn {
            if let Some(c) = new_ink {
                let mine = entry.ink.get_or_insert_with(InkCell::new);
                for i in 0..TRI {
                    if c.alpha(i) > mine.alpha(i) {
                        mine.set(i, c.px(i));
                    }
                }
            }
            if let Some(c) = new_sel {
                let mine = entry.sel.get_or_insert_with(CovCell::new);
                for i in 0..TRI {
                    mine.cov[i] = mine.cov[i].max(c.cov[i]);
                }
            }
        } else {
            entry.ink = new_ink;
            entry.sel = new_sel;
        }
        entry.version = built_version;
        if entry.is_empty() {
            self.cells.remove(&key);
            self.drawn.remove(&key);
        }
    }

    /// The levels above the cells changed since the last rebuild, and only those, coarsest last so each level sees children already current. Empties are dropped on the way.
    fn rebuild_chains(&mut self) {
        let touched: Vec<CellKey> = self.touched_since.drain().collect();
        let mut todo: std::collections::BTreeMap<u8, FxHashSet<CellKey>> = std::collections::BTreeMap::new();
        for k in &touched {
            let mut a = *k;
            while a.depth > USER_MIN_DEPTH {
                a = a.parent();
                todo.entry(a.depth).or_default().insert(a);
            }
        }
        for k in &touched {
            if self.cells.get(k).is_some_and(|c| c.is_empty()) {
                self.cells.remove(k);
                self.drawn.remove(k);
            }
        }
        let built_version = self.version + 1;
        for (_, keys) in todo.iter().rev() {
            for &key in keys {
                self.build_parent(key, built_version);
            }
        }
    }

    /// Every level above every drawn cell, from nothing: for a layer just loaded.
    fn rebuild_above(&mut self) {
        let drawn = self.drawn.clone();
        self.cells.retain(|k, _| drawn.contains_key(k));
        self.touched_since = drawn.keys().copied().collect();
        self.rebuild_chains();
    }

    /// The ink at a raw coordinate, from the finest cell at or above `depth` that has an ink plane there: premultiplied VSF RGB at gamma 2 and the alpha, or nothing where the alpha is zero. The finest cell decides, which holds because every cell carries what coarser cells have there: seeded when made, pushed down when a coarser stroke lands, completed when built.
    pub fn ink_at(&self, raw: u64, depth: u8) -> Option<[u8; 4]> {
        let (iu, iv) = Coord::from_raw(raw).uv();
        let (uq, vq) = ((iu as i64) << 16, (iv as i64) << 16);
        let mut d = depth;
        loop {
            let key = CellKey { depth: d, prefix: raw >> (60 - 2 * d as u32) };
            if let Some(ink) = self.cells.get(&key).and_then(|c| c.ink.as_ref()) {
                let i = tri_index(uq, vq, 16 + 22 - d as u32);
                let p = ink.px(i);
                return (p[3] != 0).then_some(p);
            }
            if d == USER_MIN_DEPTH {
                return None;
            }
            d -= 1;
        }
    }

    /// The selection's coverage at a raw coordinate, from the finest cell at or above `depth` with a selection plane.
    pub fn sel_at(&self, raw: u64, depth: u8) -> u8 {
        let (iu, iv) = Coord::from_raw(raw).uv();
        let (uq, vq) = ((iu as i64) << 16, (iv as i64) << 16);
        let mut d = depth;
        loop {
            let key = CellKey { depth: d, prefix: raw >> (60 - 2 * d as u32) };
            if let Some(sel) = self.cells.get(&key).and_then(|c| c.sel.as_ref()) {
                return sel.cov[tri_index(uq, vq, 16 + 22 - d as u32)];
            }
            if d == USER_MIN_DEPTH {
                return 0;
            }
            d -= 1;
        }
    }

    /// The selection as drawn: every drawn cell's selection plane (an area snapshots these).
    pub fn drawn_selection(&self) -> Vec<(CellKey, CovCell)> {
        self.drawn.keys().filter_map(|k| self.cells.get(k).and_then(|c| c.sel.clone()).filter(|s| !s.is_empty()).map(|s| (*k, s))).collect()
    }

    pub fn has_selection(&self) -> bool {
        self.cells.values().any(|c| c.sel.as_ref().is_some_and(|s| !s.is_empty()))
    }

    /// The selection replaced by planes (an area shown again): each lands in its cell, and the drawn cells beneath take it resampled, as a new cell is seeded, so the finest cell still decides.
    pub fn load_selection(&mut self, planes: Vec<(CellKey, CovCell)>) {
        self.clear_selection();
        let mut by_depth: FxHashMap<u8, FxHashSet<CellKey>> = FxHashMap::default();
        for (key, cov) in planes {
            let below: Vec<CellKey> = self.drawn.keys().filter(|k| k.depth > key.depth && k.ancestor(key.depth) == key).copied().collect();
            for k in below {
                let map = map_to_ancestor(k, key);
                let mut s = CovCell::new();
                for (i, &j) in map.iter().enumerate() {
                    s.cov[i] = cov.cov[j];
                }
                let cell = self.cells.entry(k).or_default();
                cell.sel = Some(s);
                cell.version += 1;
                by_depth.entry(k.depth).or_default().insert(k);
            }
            let cell = self.cells.entry(key).or_default();
            cell.sel = Some(cov);
            cell.version += 1;
            by_depth.entry(key.depth).or_default().insert(key);
        }
        for (d, touched) in by_depth {
            self.after_change(d, touched);
        }
    }

    /// The cells changed since the last call, cloned for saving elsewhere, None where one is gone.
    pub fn take_dirty_cells(&mut self) -> Vec<(CellKey, Option<UserCell>)> {
        let keys: Vec<CellKey> = self.dirty.drain().collect();
        keys.into_iter().map(|k| (k, self.cells.get(&k).filter(|c| !c.is_empty()).cloned())).collect()
    }

    /// A cell loaded from the vault: drawn, as every saved cell is one the pyramid can be rebuilt from.
    pub fn load(&mut self, key: CellKey, bytes: &[u8]) {
        if let Some(c) = decode_user_cell(bytes) {
            self.cells.insert(key, c);
            self.drawn.insert(key, ());
            self.version += 1;
        }
    }

    /// After loading: the levels above rebuilt once.
    pub fn loaded(&mut self) {
        self.rebuild_above();
        self.version += 1;
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// A user cell as an ordinary cell's bytes: the ink its imagery plane (red, green, blue premultiplied, the alpha in the fourth band; lossless), the selection its water plane.
pub fn encode_user_cell(c: &UserCell) -> Vec<u8> {
    let img = c.ink.as_ref().map(|ink| {
        let mut im = ImgCell::new();
        for i in 0..TRI {
            let p = ink.px(i);
            im.red[i] = p[0];
            im.green[i] = p[1];
            im.blue[i] = p[2];
            im.nir[i] = p[3];
        }
        im.tone = 0;
        im
    });
    let cell = mahere_tiles::Cell { img, water: c.sel.clone(), ..Default::default() };
    cell.quantize().encode(&mahere_tiles::Loss::LOSSLESS).unwrap_or_default()
}

pub fn decode_user_cell(bytes: &[u8]) -> Option<UserCell> {
    let p = mahere_tiles::decode_cell(bytes).ok()?;
    let ink = p.img.map(|im| {
        let mut ink = InkCell::new();
        for i in 0..TRI {
            ink.set(i, [im.red[i], im.green[i], im.blue[i], im.nir[i]]);
        }
        ink
    });
    Some(UserCell { ink, sel: p.water, version: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Camera {
        Camera::new(46.2, -122.19, 24000.0, 0.3)
    }

    fn pen(u: &UserLayer, px: f64, py: f64, depth: u8) -> Option<[u8; 4]> {
        let (lat, lon) = cam().screen_to_geo(px, py, 800, 600);
        u.ink_at(Coord::from_lat_lon(lat, lon).raw(), depth)
    }

    #[test]
    fn a_stroke_lands_where_it_was_drawn_and_above() {
        let mut u = UserLayer::default();
        let (w, h) = (800, 600);
        u.stroke(&cam(), w, h, 12, &[(100.0, 100.0, 6.0), (300.0, 250.0, 6.0)], PENS[2], 1.0);
        assert!(u.cells.keys().any(|k| k.depth == 12));
        let on = pen(&u, 200.0, 175.0, 12).expect("on the stroke");
        assert_eq!(on[3], 255);
        assert!(on[0] > 200 && on[2] < 80, "yellow: {on:?}");
        let above = pen(&u, 200.0, 175.0, 9).expect("the parents carry it");
        assert!(above[3] < 255 && above[3] > 0, "lighter above, as an area: alpha {}", above[3]);
        assert!(pen(&u, 600.0, 500.0, 12).is_none(), "off the stroke");
        // Averaged as an area, a thin stroke fades with each level above and is gone long before the top: the levels it has vanished from are not kept.
        assert!(u.cells.keys().all(|k| k.depth >= 3), "faded out above");
    }

    #[test]
    fn a_stroke_at_a_new_depth_keeps_what_was_there() {
        let mut u = UserLayer::default();
        let (w, h) = (800, 600);
        u.stroke(&cam(), w, h, 10, &[(100.0, 300.0, 6.0), (700.0, 300.0, 6.0)], PENS[0], 1.0);
        u.stroke(&cam(), w, h, 12, &[(400.0, 100.0, 6.0), (400.0, 500.0, 6.0)], PENS[3], 1.0);
        let first = pen(&u, 200.0, 300.0, 13).expect("the first stroke, seen from the finer depth, where the second cut across its cells");
        assert!(first[0] > first[1], "red: {first:?}");
        let second = pen(&u, 400.0, 200.0, 13).expect("the second stroke");
        assert!(second[1] > second[0], "green: {second:?}");
    }

    #[test]
    fn a_coarse_stroke_shows_through_finer_cells_at_once() {
        let mut u = UserLayer::default();
        let (w, h) = (800, 600);
        u.stroke(&cam(), w, h, 12, &[(100.0, 100.0, 6.0), (700.0, 100.0, 6.0)], PENS[3], 1.0);
        u.begin_undo();
        u.stroke(&cam(), w, h, 9, &[(400.0, 50.0, 6.0), (400.0, 550.0, 6.0)], PENS[0], 1.0);
        let mid = pen(&u, 400.0, 300.0, 13).expect("the coarse stroke through the fine cells, mid-stroke");
        assert!(mid[0] > mid[1]);
        u.commit_undo();
        assert!(pen(&u, 400.0, 300.0, 13).is_some(), "and after the lift");
    }

    /// Seeding a fine cell from a coarse one: every fine triangle takes the coarse triangle it lies in, so a coarse texel inked in its lower half only inks the fine triangles under that diagonal.
    #[test]
    fn a_seed_follows_the_ancestor_diagonal() {
        let mut u = UserLayer::default();
        let parent = CellKey::from_grid(2, 10, 5, 7);
        let mut ink = InkCell::new();
        for ty in 0..TEX {
            for tx in 0..TEX {
                ink.set(((ty << 8) | tx) << 1, [200, 0, 0, 255]);
            }
        }
        u.cells.insert(parent, UserCell { ink: Some(ink), sel: None, version: 0 });
        let child = parent.child(0).child(0);
        let seed = u.ink_from_parents(child).unwrap();
        let at = |tx: usize, ty: usize, half: usize| seed.alpha(((ty << 8) | tx) << 1 | half) != 0;
        assert!(at(0, 0, 0) && at(0, 0, 1) && at(3, 0, 0) && at(0, 3, 0), "under the diagonal");
        assert!(!at(3, 3, 0) && !at(3, 3, 1) && !at(3, 0, 1) && !at(2, 2, 1), "over it");
    }

    #[test]
    fn a_selection_adds_from_outside_and_takes_from_inside() {
        let mut u = UserLayer::default();
        let (w, h) = (800, 600);
        let square = [(100.0, 100.0), (400.0, 100.0), (400.0, 400.0), (100.0, 400.0)];
        assert!(!u.selected_at(&cam(), w, h, 12, 250.0, 250.0));
        u.select(&cam(), w, h, 12, &square, true);
        assert!(u.selected_at(&cam(), w, h, 12, 250.0, 250.0), "inside the square");
        assert!(!u.selected_at(&cam(), w, h, 12, 500.0, 500.0), "outside it");
        let bite = [(200.0, 200.0), (300.0, 200.0), (300.0, 300.0), (200.0, 300.0)];
        u.select(&cam(), w, h, 12, &bite, false);
        assert!(!u.selected_at(&cam(), w, h, 12, 250.0, 250.0), "taken out");
        assert!(u.selected_at(&cam(), w, h, 12, 150.0, 150.0), "the rest stays");
        let dirty = u.take_dirty_cells();
        assert!(!dirty.is_empty());
        let (k, cell) = dirty.iter().find(|(_, c)| c.is_some()).unwrap().clone();
        let bytes = encode_user_cell(cell.as_ref().unwrap());
        let mut again = UserLayer::default();
        again.load(k, &bytes);
        again.loaded();
        assert!(again.cells.contains_key(&k));
    }

    /// Ink survives the codec exactly: every byte of every texel.
    #[test]
    fn ink_round_trips_losslessly() {
        let mut u = UserLayer::default();
        u.stroke(&cam(), 800, 600, 12, &[(100.0, 100.0, 9.0), (300.0, 250.0, 3.0)], PENS[5], 0.7);
        let key = *u.cells.keys().find(|k| k.depth == 12).unwrap();
        let cell = u.cells[&key].clone();
        let back = decode_user_cell(&encode_user_cell(&cell)).unwrap();
        assert_eq!(back.ink.unwrap().rgba, cell.ink.unwrap().rgba);
    }
}
