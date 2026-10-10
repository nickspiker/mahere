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

#[derive(Clone, Default)]
pub struct UserCell {
    pub ink: Option<ClassCell>,
    pub sel: Option<CovCell>,
}

impl UserCell {
    pub fn is_empty(&self) -> bool {
        self.ink.as_ref().is_none_or(|c| c.cov.iter().all(|&v| v == 0)) && self.sel.as_ref().is_none_or(|c| c.is_empty())
    }
}

/// Every user cell, drawn or built above a drawn one, at every depth, with what has changed since the last save.
#[derive(Default)]
pub struct UserLayer {
    pub cells: FxHashMap<CellKey, UserCell>,
    /// The depths cells were drawn at (not built), so the pyramid above them is rebuilt from the right places.
    drawn: FxHashMap<CellKey, ()>,
    dirty: FxHashSet<CellKey>,
    /// Bumped on every change, for a renderer to know its planes are stale.
    pub version: u64,
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
    /// A stroke of the pen along screen points, `width` pixels wide, in pen `pen` (an index into [`PENS`]): every pixel within half the width of the path, stamped into the cells at `depth`.
    pub fn stroke(&mut self, cam: &Camera, w: usize, h: usize, depth: u8, pts: &[(f64, f64)], width: f64, pen: u8) {
        let r = (width * 0.5).max(0.5);
        let mut touched: FxHashSet<CellKey> = FxHashSet::default();
        let mut stamp = |x: f64, y: f64| {
            let (x0, x1) = ((x - r).floor() as i64, (x + r).ceil() as i64);
            let (y0, y1) = ((y - r).floor() as i64, (y + r).ceil() as i64);
            for py in y0..=y1 {
                for px in x0..=x1 {
                    let (cx, cy) = (px as f64 + 0.5, py as f64 + 0.5);
                    if (cx - x) * (cx - x) + (cy - y) * (cy - y) > r * r {
                        continue;
                    }
                    if let Some((key, i)) = texel_at(cam, depth, cx, cy, w, h) {
                        let cell = self.cells.entry(key).or_default();
                        let ink = cell.ink.get_or_insert_with(ClassCell::new);
                        ink.class[i] = pen + 1;
                        ink.cov[i] = 255;
                        touched.insert(key);
                    }
                }
            }
        };
        match pts {
            [] => {}
            [p] => stamp(p.0, p.1),
            _ => {
                for s in pts.windows(2) {
                    let (ax, ay, bx, by) = (s[0].0, s[0].1, s[1].0, s[1].1);
                    let n = ((bx - ax).hypot(by - ay) * 2.0).ceil().max(1.0) as usize;
                    for k in 0..=n {
                        let t = k as f64 / n as f64;
                        stamp(ax + (bx - ax) * t, ay + (by - ay) * t);
                    }
                }
            }
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
        for py in y0..=y1 {
            let yc = py as f64 + 0.5;
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
                let (x0, x1) = (pair[0].round().max(0.0) as i64, pair[1].round().min(w as f64) as i64);
                for px in x0..x1 {
                    if let Some((key, i)) = texel_at(cam, depth, px as f64 + 0.5, yc, w, h) {
                        let cell = self.cells.entry(key).or_default();
                        let sel = cell.sel.get_or_insert_with(CovCell::new);
                        sel.cov[i] = if add { 255 } else { 0 };
                        touched.insert(key);
                    }
                }
            }
        }
        // Taking from a selection the parents hold: the finest cell now says so where it was silent.
        if !add {
            for key in &touched {
                let covered_above = self.sel_from_parents(*key);
                if let Some(cell) = self.cells.get_mut(key) {
                    if let (Some(sel), Some(above)) = (cell.sel.as_mut(), covered_above) {
                        for i in 0..TRI {
                            if sel.cov[i] == 0 && above.cov[i] != 0 && !self.drawn.contains_key(key) {
                                sel.cov[i] = 0;
                            }
                        }
                    }
                }
            }
        }
        self.after_change(depth, touched);
    }

    /// What the nearest parent with a selection plane says about this cell's texels, resampled (nearest) into it.
    fn sel_from_parents(&self, key: CellKey) -> Option<CovCell> {
        let mut k = key;
        let mut levels = 0u32;
        while k.depth > USER_MIN_DEPTH {
            k = k.parent();
            levels += 1;
            if let Some(sel) = self.cells.get(&k).and_then(|c| c.sel.as_ref()) {
                let mut out = CovCell::new();
                let (cu, cv) = key.grid();
                let (pu, pv) = k.grid();
                let (ou, ov) = ((cu - (pu << levels)) as usize, (cv - (pv << levels)) as usize);
                for ty in 0..256 {
                    for tx in 0..256 {
                        let (sx, sy) = (((ou << 8) + tx) >> levels, ((ov << 8) + ty) >> levels);
                        for half in 0..2 {
                            out.cov[((ty << 8) | tx) << 1 | half] = sel.cov[((sy << 8) | sx) << 1 | half];
                        }
                    }
                }
                return Some(out);
            }
        }
        None
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

    /// After cells changed at `depth`: they count as drawn, the levels above are rebuilt, the empties dropped, the version bumped.
    fn after_change(&mut self, depth: u8, touched: FxHashSet<CellKey>) {
        for k in &touched {
            if k.depth == depth {
                self.drawn.insert(*k, ());
            }
        }
        self.rebuild_above();
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
        // Drop every built cell; only drawn ones stay.
        let drawn = self.drawn.clone();
        self.cells.retain(|k, _| drawn.contains_key(k));
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
            let ink_up = mahere_tiles::pyramid_class(ink, depth, depth - 1, ClassMerge::Major);
            let sel_up = mahere_tiles::pyramid_cov(sel, depth, depth - 1);
            for (k, c) in ink_up.into_iter().filter(|(k, _)| k.depth == depth - 1) {
                let cell = self.cells.entry(k).or_default();
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
            for (k, c) in sel_up.into_iter().filter(|(k, _)| k.depth == depth - 1) {
                let cell = self.cells.entry(k).or_default();
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

    /// The ink at a raw coordinate, from the finest cell at or above `depth` that has any there: the pen (an index into [`PENS`]) and the coverage.
    pub fn ink_at(&self, raw: u64, depth: u8) -> Option<(u8, u8)> {
        let diamond = (raw >> 60) as u8;
        let (iu, iv) = Coord::from_raw(raw).uv();
        let (uq, vq) = ((iu as i64) << 16, (iv as i64) << 16);
        let _ = diamond;
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
    Some(UserCell { ink: p.line, sel: p.water })
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
        u.stroke(&cam(), w, h, 12, &[(100.0, 100.0), (300.0, 250.0)], 6.0, 2);
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
