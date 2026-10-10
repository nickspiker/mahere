//! Notes on the map and areas of it: the user's own layer of cells beside the bucket's (Nick 2026-10-10).
//!
//! A stroke of the pen or a drawn selection is never kept as a vector. It is stamped into cells at whatever depth the view is at when it is drawn, one cell wherever the ink touches, each cell an `ink` plane (the pen's colour index and coverage, the shape a line plane has) and a `sel` plane (the selection's coverage, the shape a water plane has). Above the drawn depth the parents are rebuilt by the same box filters the map's lines and water use, so zooming out keeps the marks; zooming in magnifies the finest cell there is, as the map does.
//!
//! A user cell is encoded as an ordinary cell whose line plane is the ink and whose water plane is the selection, so the vault, the codec and the renderers' planes already know it.

use mahere_coord::Coord;
use mahere_tiles::{CellKey, ClassCell, ClassMerge, CovCell, TRI};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::Camera;
use crate::raster::tri_index;

/// The pens, VSF RGB at gamma 2: red, orange, yellow, green, cyan, blue, magenta, black. A pen's index in the ink plane is one more than its index here (0 is no ink).
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

#[derive(Clone, Default)]
pub struct UserCell {
    pub ink: Option<ClassCell>,
    pub sel: Option<CovCell>,
    /// Bumped whenever the cell's planes change, so a renderer uploads only what changed.
    pub version: u64,
}

impl UserCell {
    pub fn is_empty(&self) -> bool {
        self.ink.as_ref().is_none_or(|c| c.cov.iter().all(|&v| v == 0)) && self.sel.as_ref().is_none_or(|c| c.is_empty())
    }
}

/// Every user cell, drawn or built above a drawn one, at every depth, with what has changed since the last save.
#[derive(Clone, Default)]
pub struct UserLayer {
    pub cells: FxHashMap<CellKey, UserCell>,
    /// The depths cells were drawn at (not built), so the pyramid above them is rebuilt from the right places.
    drawn: FxHashMap<CellKey, ()>,
    dirty: FxHashSet<CellKey>,
    /// Bumped on every change, for a renderer to know its planes are stale.
    pub version: u64,
    /// The cells a stroke in progress has touched, as they were before it (None: not there), for undoing it; and whether the levels above still wait to be rebuilt (deferred to the stroke's end, since mid-stroke the view reads the drawn cells themselves).
    undo: Option<FxHashMap<CellKey, Option<UserCell>>>,
    stale_above: bool,
}

/// The texel a screen point falls in at `depth`: the cell and the triangle index, or nothing off the globe.
fn texel_at(cam: &Camera, depth: u8, px: f64, py: f64, w: usize, h: usize) -> Option<(CellKey, usize)> {
    if !cam.on_globe(px, py, w, h) {
        return None;
    }
    let (lat, lon) = cam.screen_to_geo(px, py, w, h);
    let c = Coord::from_lat_lon(lat, lon);
    let (iu, iv) = c.uv();
    let (uq, vq) = ((iu as i64) << 16, (iv as i64) << 16);
    let shift = 16 + 22 - depth as u32;
    Some((CellKey::containing(c, depth), tri_index(uq, vq, shift)))
}

impl UserLayer {
    /// A stroke of the pen along screen points, each with its width in pixels (the pen's pressure), in pen `pen` (an index into [`PENS`]): every pixel within half the width of the path, stamped into the cells at `depth`.
    pub fn stroke(&mut self, cam: &Camera, w: usize, h: usize, depth: u8, pts: &[(f64, f64, f64)], pen: u8) {
        let mut touched: FxHashSet<CellKey> = FxHashSet::default();
        // Sampled every half pixel: a texel at the drawing depth is one to two pixels wide, so every texel under the brush is hit and the renderer, reading one texel a pixel, finds no gaps.
        let mut stamp = |x: f64, y: f64, width: f64| {
            let r = (width * 0.5).max(0.5);
            let (x0, x1) = (((x - r) * 2.0).floor() as i64, ((x + r) * 2.0).ceil() as i64);
            let (y0, y1) = (((y - r) * 2.0).floor() as i64, ((y + r) * 2.0).ceil() as i64);
            for py in y0..=y1 {
                for px in x0..=x1 {
                    let (cx, cy) = (px as f64 * 0.5 + 0.25, py as f64 * 0.5 + 0.25);
                    if (cx - x) * (cx - x) + (cy - y) * (cy - y) > r * r {
                        continue;
                    }
                    if let Some((key, i)) = texel_at(cam, depth, cx, cy, w, h) {
                        let ink = self.ink_cell(key);
                        ink.class[i] = pen + 1;
                        ink.cov[i] = 255;
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
        let Some((key, _)) = texel_at(cam, depth, px, py, w, h) else { return false };
        let (lat, lon) = cam.screen_to_geo(px, py, w, h);
        let c = Coord::from_lat_lon(lat, lon);
        let (iu, iv) = c.uv();
        let (uq, vq) = ((iu as i64) << 16, (iv as i64) << 16);
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

    /// The nearest ancestor of `key` that `has` a plane, with the map from this cell's texel index to the ancestor's: a cell drawn at a new depth starts as what was already there, resampled (nearest texel), so a stroke at one zoom never clears a mark made at another (Nick 2026-10-10).
    fn ancestor_with(&self, key: CellKey, has: impl Fn(&UserCell) -> bool) -> Option<(CellKey, Vec<usize>)> {
        let mut k = key;
        let mut levels = 0u32;
        while k.depth > USER_MIN_DEPTH {
            k = k.parent();
            levels += 1;
            if self.cells.get(&k).is_some_and(&has) {
                let _ = levels;
                return Some((k, Self::map_to_ancestor(key, k)));
            }
        }
        None
    }

    /// What the nearest ancestor with a selection plane says about this cell's texels, resampled into it.
    fn sel_from_parents(&self, key: CellKey) -> Option<CovCell> {
        let (k, map) = self.ancestor_with(key, |c| c.sel.is_some())?;
        let sel = self.cells[&k].sel.as_ref().unwrap();
        let mut out = CovCell::new();
        for (i, &j) in map.iter().enumerate() {
            out.cov[i] = sel.cov[j];
        }
        Some(out)
    }

    /// What the nearest ancestor with an ink plane says about this cell's texels, resampled into it.
    fn ink_from_parents(&self, key: CellKey) -> Option<ClassCell> {
        let (k, map) = self.ancestor_with(key, |c| c.ink.is_some())?;
        let ink = self.cells[&k].ink.as_ref().unwrap();
        let mut out = ClassCell::new();
        for (i, &j) in map.iter().enumerate() {
            out.cov[i] = ink.cov[j];
            out.class[i] = ink.class[j];
        }
        Some(out)
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
    fn ink_cell(&mut self, key: CellKey) -> &mut ClassCell {
        self.remember(key);
        if self.cells.get(&key).is_none_or(|c| c.ink.is_none()) {
            let seed = self.ink_from_parents(key).unwrap_or_else(ClassCell::new);
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

    /// A stroke begins: from here every touched cell is remembered, and the levels above wait until it ends.
    pub fn begin_undo(&mut self) {
        self.undo = Some(FxHashMap::default());
    }

    /// The stroke stands: the undo is forgotten and the levels above are rebuilt now.
    pub fn commit_undo(&mut self) {
        self.undo = None;
        if self.stale_above {
            self.rebuild_above();
            self.stale_above = false;
            self.version += 1;
        }
    }

    /// The stroke is undone: every cell it touched goes back to what it was.
    pub fn rollback_undo(&mut self) {
        if let Some(undo) = self.undo.take() {
            for (k, before) in undo {
                match before {
                    Some(c) => {
                        let mut c = c;
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
            self.version += 1;
        }
    }

    /// Clear the whole selection.
    pub fn clear_selection(&mut self) {
        let mut touched = FxHashSet::default();
        for (k, c) in self.cells.iter_mut() {
            if c.sel.take().is_some() {
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
                touched.insert(*k);
            }
        }
        self.after_change(0, touched);
    }

    /// After cells changed at `depth`: they count as drawn, the levels above are rebuilt (or, mid-stroke, left for the stroke's end), the empties dropped, the version bumped.
    fn after_change(&mut self, depth: u8, touched: FxHashSet<CellKey>) {
        for k in &touched {
            if k.depth == depth {
                self.drawn.insert(*k, ());
            }
        }
        if self.undo.is_some() {
            self.stale_above = true;
        } else {
            self.rebuild_above();
        }
        self.cells.retain(|k, c| {
            let keep = !c.is_empty();
            if !keep {
                self.drawn.remove(k);
            }
            keep
        });
        for k in touched {
            self.dirty.insert(k);
        }
        self.version += 1;
    }

    /// The levels above every drawn depth, from the deepest up: at each level the box filter of the level below, with cells drawn at that level laid over it where they have ink or selection.
    fn rebuild_above(&mut self) {
        let Some(deepest) = self.drawn.keys().map(|k| k.depth).max() else { return };
        // Drop every built cell; only drawn ones stay. The cells built below carry this rebuild's version, so a renderer sees them as new.
        let drawn = self.drawn.clone();
        self.cells.retain(|k, _| drawn.contains_key(k));
        let built_version = self.version + 1;
        let mut depth = deepest;
        while depth > USER_MIN_DEPTH {
            let mut ink: std::collections::HashMap<CellKey, ClassCell> = std::collections::HashMap::new();
            let mut sel: std::collections::HashMap<CellKey, CovCell> = std::collections::HashMap::new();
            for (k, c) in self.cells.iter().filter(|(k, _)| k.depth == depth) {
                if let Some(i) = &c.ink {
                    ink.insert(*k, i.clone());
                }
                if let Some(s) = &c.sel {
                    sel.insert(*k, s.clone());
                }
            }
            // Ink above the drawn depth: the mean of the children doubled, so a stroke halves in texels each level (its area) but a line one texel wide, two children of four, stays full and never fades or swells; the map's own lines sum, which would swell a stroke. The selection is an area: the plain mean.
            let mut ink_up = mahere_tiles::pyramid_class(ink, depth, depth - 1, ClassMerge::Dominant);
            for (k, c) in ink_up.iter_mut() {
                if k.depth == depth - 1 {
                    for v in c.cov.iter_mut() {
                        *v = (*v as u32 * 2).min(255) as u8;
                    }
                }
            }
            let sel_up = mahere_tiles::pyramid_cov(sel, depth, depth - 1);
            // A built level is complete: where the pyramid from below has nothing, the nearest drawn ancestor's mark stands (resampled), so a mark made at a coarser zoom is not hidden by a level built from a finer stroke beside it; a drawn cell at this level keeps its own, the louder of the two.
            for (k, mut c) in ink_up.into_iter().filter(|(k, _)| k.depth == depth - 1) {
                if let Some(seed) = self.ink_from_parents(k) {
                    for i in 0..TRI {
                        if c.cov[i] == 0 && seed.cov[i] != 0 {
                            c.cov[i] = seed.cov[i];
                            c.class[i] = seed.class[i];
                        }
                    }
                }
                let cell = self.cells.entry(k).or_default();
                cell.version = built_version;
                match &mut cell.ink {
                    Some(mine) => {
                        for i in 0..TRI {
                            if c.cov[i] > mine.cov[i] {
                                mine.cov[i] = c.cov[i];
                                mine.class[i] = c.class[i];
                            }
                        }
                    }
                    None => cell.ink = Some(c),
                }
            }
            for (k, mut c) in sel_up.into_iter().filter(|(k, _)| k.depth == depth - 1) {
                if let Some(seed) = self.sel_from_parents(k) {
                    for i in 0..TRI {
                        if c.cov[i] == 0 {
                            c.cov[i] = seed.cov[i];
                        }
                    }
                }
                let cell = self.cells.entry(k).or_default();
                cell.version = built_version;
                match &mut cell.sel {
                    Some(mine) => {
                        for i in 0..TRI {
                            mine.cov[i] = mine.cov[i].max(c.cov[i]);
                        }
                    }
                    None => cell.sel = Some(c),
                }
            }
            depth -= 1;
        }
    }

    /// Whether the cell at `key` was drawn (as against built above drawn ones).
    pub fn is_drawn(&self, key: CellKey) -> bool {
        self.drawn.contains_key(&key)
    }

    /// The ink at a raw coordinate, from the finest cell at or above `depth` that has an ink plane there: the pen (an index into [`PENS`]) and the coverage. The finest cell decides, which holds because every cell carries what coarser cells have there: seeded when made, pushed down when a coarser stroke lands, completed when built.
    pub fn ink_at(&self, raw: u64, depth: u8) -> Option<(u8, u8)> {
        let (iu, iv) = Coord::from_raw(raw).uv();
        let (uq, vq) = ((iu as i64) << 16, (iv as i64) << 16);
        let mut d = depth;
        loop {
            let key = CellKey { depth: d, prefix: raw >> (60 - 2 * d as u32) };
            if let Some(ink) = self.cells.get(&key).and_then(|c| c.ink.as_ref()) {
                let i = tri_index(uq, vq, 16 + 22 - d as u32);
                if ink.cov[i] != 0 {
                    return Some((ink.class[i] - 1, ink.cov[i]));
                }
                return None;
            }
            if d == USER_MIN_DEPTH {
                return None;
            }
            d -= 1;
        }
    }

    /// The map from a finer cell's texels to those of its ancestor `anc`: the ancestor texel by shifting, its lower or upper half by where the fine triangle's centroid falls against that texel's diagonal (the lower triangle's centroid at a third, the upper's at two thirds; times three to stay in integers).
    fn map_to_ancestor(key: CellKey, anc: CellKey) -> Vec<usize> {
        let levels = (key.depth - anc.depth) as u32;
        let (cu, cv) = key.grid();
        let (pu, pv) = anc.grid();
        let (ou, ov) = ((cu - (pu << levels)) as usize, (cv - (pv << levels)) as usize);
        let mut map = vec![0usize; TRI];
        let p = 1usize << levels;
        for ty in 0..256 {
            for tx in 0..256 {
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

    /// A coarse cell's ink pushed down into every finer cell beneath it: where the fine texel is empty and the coarse one inked, the fine takes it. So a stroke made zoomed out shows at once through finer cells drawn earlier, the finest cell still deciding (Nick 2026-10-10).
    fn push_down(&mut self, key: CellKey) {
        let Some(ink) = self.cells.get(&key).and_then(|c| c.ink.clone()) else { return };
        let below: Vec<CellKey> = self.cells.keys().filter(|k| k.depth > key.depth && k.ancestor(key.depth) == key).copied().collect();
        for k in below {
            let map = Self::map_to_ancestor(k, key);
            self.remember(k);
            let cell = self.cells.get_mut(&k).unwrap();
            let fine = cell.ink.get_or_insert_with(ClassCell::new);
            let mut changed = false;
            for (i, &j) in map.iter().enumerate() {
                if fine.cov[i] == 0 && ink.cov[j] != 0 {
                    fine.cov[i] = ink.cov[j];
                    fine.class[i] = ink.class[j];
                    changed = true;
                }
            }
            if changed {
                cell.version += 1;
                self.dirty.insert(k);
            }
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

    /// The cells changed since the last call, for saving, and whether each still exists.
    pub fn take_dirty(&mut self) -> Vec<(CellKey, Option<Vec<u8>>)> {
        let keys: Vec<CellKey> = self.dirty.drain().collect();
        keys.into_iter()
            .map(|k| {
                let bytes = self.cells.get(&k).filter(|c| !c.is_empty()).map(|c| encode_user_cell(c));
                (k, bytes)
            })
            .collect()
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

/// A user cell as an ordinary cell's bytes: the ink its line plane, the selection its water plane.
pub fn encode_user_cell(c: &UserCell) -> Vec<u8> {
    let cell = mahere_tiles::Cell { line: c.ink.clone(), water: c.sel.clone(), ..Default::default() };
    cell.quantize().encode(&mahere_tiles::Loss { dem_m: 0.0, img: 0 }).unwrap_or_default()
}

pub fn decode_user_cell(bytes: &[u8]) -> Option<UserCell> {
    let p = mahere_tiles::decode_cell(bytes).ok()?;
    Some(UserCell { ink: p.line, sel: p.water, version: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Camera {
        Camera::new(46.2, -122.19, 24000.0, 0.3)
    }

    #[test]
    fn a_stroke_lands_where_it_was_drawn_and_above() {
        let mut u = UserLayer::default();
        let (w, h) = (800, 600);
        u.stroke(&cam(), w, h, 12, &[(100.0, 100.0, 6.0), (300.0, 250.0, 6.0)], 2);
        let drawn: Vec<&CellKey> = u.cells.keys().filter(|k| k.depth == 12).collect();
        assert!(!drawn.is_empty());
        let (lat, lon) = cam().screen_to_geo(200.0, 175.0, w, h);
        let raw = Coord::from_lat_lon(lat, lon).raw();
        assert_eq!(u.ink_at(raw, 12), Some((2, 255)), "on the stroke");
        assert!(u.ink_at(raw, 9).is_some(), "the parents carry it");
        let (lat, lon) = cam().screen_to_geo(600.0, 500.0, w, h);
        assert_eq!(u.ink_at(Coord::from_lat_lon(lat, lon).raw(), 12), None, "off the stroke");
        assert!(u.cells.keys().any(|k| k.depth == USER_MIN_DEPTH), "built to the top");
    }

    #[test]
    fn a_stroke_at_a_new_depth_keeps_what_was_there() {
        let mut u = UserLayer::default();
        let (w, h) = (800, 600);
        u.stroke(&cam(), w, h, 10, &[(100.0, 300.0, 6.0), (700.0, 300.0, 6.0)], 0);
        u.stroke(&cam(), w, h, 12, &[(400.0, 100.0, 6.0), (400.0, 500.0, 6.0)], 3);
        let (lat, lon) = cam().screen_to_geo(200.0, 300.0, w, h);
        let raw = Coord::from_lat_lon(lat, lon).raw();
        assert_eq!(u.ink_at(raw, 13).map(|(p, _)| p), Some(0), "the first stroke, seen from the finer depth, where the second cut across its cells");
        let (lat, lon) = cam().screen_to_geo(400.0, 200.0, w, h);
        let raw = Coord::from_lat_lon(lat, lon).raw();
        assert_eq!(u.ink_at(raw, 13).map(|(p, _)| p), Some(3), "the second stroke");
    }

    /// Seeding a fine cell from a coarse one: every fine triangle takes the coarse triangle it lies in, so a coarse texel inked in its lower half only inks the fine triangles under that diagonal.
    #[test]
    fn a_seed_follows_the_ancestor_diagonal() {
        let mut u = UserLayer::default();
        let parent = CellKey::from_grid(2, 10, 5, 7);
        let mut ink = ClassCell::new();
        for ty in 0..256 {
            for tx in 0..256 {
                ink.class[((ty << 8) | tx) << 1] = 1;
                ink.cov[((ty << 8) | tx) << 1] = 255;
            }
        }
        u.cells.insert(parent, UserCell { ink: Some(ink), sel: None, version: 0 });
        let child = parent.child(0).child(0);
        let seed = u.ink_from_parents(child).unwrap();
        // Four fine texels to a coarse one each way: the fine triangles whose centroids lie below the coarse diagonal are inked.
        let at = |tx: usize, ty: usize, half: usize| seed.cov[((ty << 8) | tx) << 1 | half] != 0;
        assert!(at(0, 0, 0) && at(0, 0, 1) && at(3, 0, 0) && at(0, 3, 0), "under the diagonal");
        assert!(!at(3, 3, 0) && !at(3, 3, 1) && !at(3, 0, 1) && !at(2, 2, 1), "over it");
    }

    /// A stroke made zoomed out, over an area with finer strokes, shows through the finer cells' empty texels the moment it is drawn, before any level is rebuilt.
    #[test]
    fn a_coarse_stroke_shows_through_finer_cells_at_once() {
        let mut u = UserLayer::default();
        let (w, h) = (800, 600);
        u.stroke(&cam(), w, h, 12, &[(100.0, 100.0, 6.0), (700.0, 100.0, 6.0)], 3);
        u.begin_undo();
        u.stroke(&cam(), w, h, 9, &[(400.0, 50.0, 6.0), (400.0, 550.0, 6.0)], 0);
        let (lat, lon) = cam().screen_to_geo(400.0, 300.0, w, h);
        let raw = Coord::from_lat_lon(lat, lon).raw();
        assert_eq!(u.ink_at(raw, 13).map(|(p, _)| p), Some(0), "the coarse stroke through the fine cells, mid-stroke");
        u.commit_undo();
        assert_eq!(u.ink_at(raw, 13).map(|(p, _)| p), Some(0), "and after the lift");
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
        let dirty = u.take_dirty();
        assert!(!dirty.is_empty());
        let (k, bytes) = dirty.iter().find(|(_, b)| b.is_some()).unwrap().clone();
        let mut again = UserLayer::default();
        again.load(k, bytes.as_ref().unwrap());
        again.loaded();
        assert!(again.cells.contains_key(&k));
    }
}
