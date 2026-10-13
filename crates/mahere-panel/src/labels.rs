//! Labels on the map (Nick 2026-10-13): laid out from what is drawn, never from stored label geometry. The engine hands over every named texel in view; here the points are lettered beside their mark, an area at the point deepest inside it (a distance fill) along its long axis when it is long, a line along the longest run of it straight enough for its name, with top always up. One priority order decides who goes first (places and summits, then lakes, highways, rivers, villages, trails and streets, other areas, the small marks), each label taking its room on a grid so none overlap. A label placed last time goes first in its tier, so labels hold still while the view settles.
//!
//! Drawn once when the camera rests into a buffer of the screen's size; while the camera moves a host carries that buffer with the map by [`Labeller::transform`] and draws it again at the next rest.
use std::collections::{HashMap, HashSet, VecDeque};

use fluor::canvas::{Canvas, Damage};
use fluor::text::{TextRenderer, TextStyle};
use fluor::theme;
use mahere_engine::labels::NamedTexels;
use mahere_tiles::Named;

const fn ink(rgb: u32, alpha: u8) -> u32 {
    (theme::dark(theme::fmt(theme::vsf(rgb))) & 0x00FF_FFFF) | ((alpha as u32) << 24)
}

/// The halo every label sits on, the map's paper.
const HALO: u32 = ink(0xF2_F0_E8, 210);
const PLACE: u32 = ink(0x1E_1E_22, 255);
const SUMMIT: u32 = ink(0x5A_36_1A, 255);
const WATER: u32 = ink(0x1A_52_86, 255);
const ROAD: u32 = ink(0x2C_2C_30, 255);
const TRAIL: u32 = ink(0x74_2A_18, 255);
const BOUNDARY: u32 = ink(0x2A_5C_2C, 255);
const GREEN: u32 = ink(0x2E_5E_30, 255);
const MARK: u32 = ink(0x46_46_4C, 255);

/// Water's names lean, as maps letter them: a shear of tan 12°.
const ITALIC: f32 = 0.2126;

/// The collision grid's cell, pixels.
const GRID: f32 = 4.0;

// The kinds in a name table (see `mahere_osm::names`): lines are their class id, areas 32 on, points 64 on.
const MOTORWAY: u8 = 1;
const PRIMARY: u8 = 3;
const SECONDARY: u8 = 4;
const TERTIARY: u8 = 5;
const RESIDENTIAL: u8 = 6;
const SERVICE: u8 = 7;
const TRACK: u8 = 8;
const PATH: u8 = 9;
const RAIL: u8 = 10;
const POWER: u8 = 11;
const WATERWAY: u8 = 12;
const PARK: u8 = 13;
const ADMIN: u8 = 17;
const AREA: u8 = 32;
const WATER_AREA: u8 = AREA + 12;
const POINT: u8 = 64;
const CITY: u8 = POINT;
const TOWN: u8 = POINT + 1;
const VILLAGE: u8 = POINT + 2;
const HAMLET: u8 = POINT + 3;
const PEAK: u8 = POINT + 4;
const VOLCANO: u8 = POINT + 5;

/// Who is lettered first: lower first. None for what is not lettered at all.
fn tier(n: &Named) -> Option<u8> {
    Some(match n.kind {
        CITY | TOWN => 0,
        PEAK | VOLCANO => 1,
        WATER_AREA => 2,
        MOTORWAY..=PRIMARY => 3,
        WATERWAY => 4,
        VILLAGE | HAMLET => 5,
        SECONDARY | TERTIARY | TRACK | PATH => 6,
        PARK..=ADMIN => 6,
        RESIDENTIAL | SERVICE | RAIL => 7,
        POWER => return None,
        k if k >= POINT => 9,
        k if k >= AREA => 8,
        _ => 7,
    })
}

/// How a kind is lettered: its size against the base font, its colour, its weight, its lean.
fn style_of(n: &Named) -> (f32, u32, u16, f32) {
    match n.kind {
        CITY => (1.25, PLACE, 700, 0.0),
        TOWN => (1.1, PLACE, 700, 0.0),
        VILLAGE => (0.95, PLACE, 400, 0.0),
        HAMLET => (0.85, PLACE, 400, 0.0),
        PEAK | VOLCANO => (0.85, SUMMIT, 400, 0.0),
        WATER_AREA => (0.95, WATER, 400, ITALIC),
        WATERWAY => (0.8, WATER, 400, ITALIC),
        MOTORWAY..=PRIMARY => (0.82, ROAD, 700, 0.0),
        TRACK | PATH => (0.78, TRAIL, 400, 0.0),
        PARK..=ADMIN => (0.78, BOUNDARY, 400, 0.0),
        k if k >= POINT => (0.75, MARK, 400, 0.0),
        k if k >= AREA => (0.8, GREEN, 400, 0.0),
        _ => (0.78, ROAD, 400, 0.0),
    }
}

/// One glyph as laid out: its character, centre and angle.
struct Glyph {
    ch: char,
    at: [f32; 2],
    angle: f32,
}

/// A label as laid out.
struct Laid {
    name: Named,
    glyphs: Vec<Glyph>,
    size: f32,
    colour: u32,
    weight: u16,
    shear: f32,
    /// A mark drawn at the point: a summit's triangle, a dot for the small marks.
    mark: Option<([f32; 2], bool)>,
}

/// What was drawn, for a tap to find: a label's name and the boxes of its glyphs, in the screen as it was when drawn.
pub struct Placed {
    pub name: Named,
    pub boxes: Vec<[f32; 4]>,
}

pub struct Labeller {
    text: TextRenderer,
    /// The lettering, the screen's size, in fluor's pixel convention.
    pub buf: Vec<u32>,
    pub w: usize,
    pub h: usize,
    pub placed: Vec<Placed>,
    /// Two screen points of the drawn view and where on the globe they were, so the buffer can follow the map until the next rest.
    anchor: Option<([f64; 2], [f64; 2], (f64, f64), (f64, f64))>,
    /// The view the buffer was drawn for, to tell when it is out of date.
    drawn_for: Option<(i64, i64, i64, i64, u8, usize)>,
    advance: HashMap<(char, u16, u32), f32>,
}

impl Default for Labeller {
    fn default() -> Self {
        Self::new()
    }
}

/// A free-cell grid over the screen: labels claim boxes and test before claiming.
struct Room {
    cols: usize,
    rows: usize,
    taken: Vec<bool>,
}

impl Room {
    fn new(w: usize, h: usize) -> Room {
        let (cols, rows) = ((w as f32 / GRID).ceil() as usize + 1, (h as f32 / GRID).ceil() as usize + 1);
        Room { cols, rows, taken: vec![false; cols * rows] }
    }

    fn cells(&self, b: [f32; 4]) -> Option<(usize, usize, usize, usize)> {
        let x0 = (b[0] / GRID).floor();
        let y0 = (b[1] / GRID).floor();
        let x1 = (b[2] / GRID).floor();
        let y1 = (b[3] / GRID).floor();
        // A box off the screen does not count against anything, nor take anything.
        if x1 < 0.0 || y1 < 0.0 || x0 >= self.cols as f32 || y0 >= self.rows as f32 {
            return None;
        }
        Some((x0.max(0.0) as usize, y0.max(0.0) as usize, (x1 as usize).min(self.cols - 1), (y1 as usize).min(self.rows - 1)))
    }

    fn free(&self, boxes: &[[f32; 4]]) -> bool {
        // A label runs off no edge: every box on the screen.
        let (w, h) = ((self.cols - 1) as f32 * GRID, (self.rows - 1) as f32 * GRID);
        boxes.iter().all(|&b| {
            if b[0] < 0.0 || b[1] < 0.0 || b[2] > w || b[3] > h {
                return false;
            }
            let Some((x0, y0, x1, y1)) = self.cells(b) else { return true };
            (y0..=y1).all(|y| (x0..=x1).all(|x| !self.taken[y * self.cols + x]))
        })
    }

    fn take(&mut self, boxes: &[[f32; 4]]) {
        for &b in boxes {
            if let Some((x0, y0, x1, y1)) = self.cells(b) {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        self.taken[y * self.cols + x] = true;
                    }
                }
            }
        }
    }
}

/// A glyph's box: a square of the font size around its centre, which holds the glyph at any angle.
fn glyph_box(g: &Glyph, size: f32, pad: f32) -> [f32; 4] {
    let r = size * 0.62 + pad;
    [g.at[0] - r, g.at[1] - r, g.at[0] + r, g.at[1] + r]
}

/// Where to ask fluor to draw a glyph turned by `angle` so it lands centred on `p`: fluor turns a glyph's position about the canvas's origin along with the glyph, so the position goes in turned back.
fn unturn(p: [f32; 2], angle: f32) -> [f32; 2] {
    let (s, c) = angle.sin_cos();
    [p[0] * c + p[1] * s, -p[0] * s + p[1] * c]
}

/// A polyline's cumulative lengths.
fn lengths(path: &[[f32; 2]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(path.len());
    let mut s = 0.0;
    for (i, p) in path.iter().enumerate() {
        if i > 0 {
            s += (p[0] - path[i - 1][0]).hypot(p[1] - path[i - 1][1]);
        }
        out.push(s);
    }
    out
}

/// The point at arc length `s` along a path, and the direction there.
fn along(path: &[[f32; 2]], len: &[f32], s: f32) -> ([f32; 2], f32) {
    let i = len.partition_point(|&l| l < s).clamp(1, path.len() - 1);
    let (a, b) = (path[i - 1], path[i]);
    let seg = (len[i] - len[i - 1]).max(1e-6);
    let t = ((s - len[i - 1]) / seg).clamp(0.0, 1.0);
    ([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t], (b[1] - a[1]).atan2(b[0] - a[0]))
}

/// The smallest turn between two directions, radians.
fn turn(a: f32, b: f32) -> f32 {
    let d = (b - a).rem_euclid(std::f32::consts::TAU);
    d.min(std::f32::consts::TAU - d)
}

/// A line's runs on the screen, each as its centreline: the texels binned on a coarse grid, the bins joined to their neighbours, and each joined piece's longest path through it (its diameter, found by two breadth-first walks), the points of each bin averaged and smoothed. The longest path follows a road through bends and switchbacks where a straight fit would cut across them.
fn centrelines(pts: &[[f32; 2]], bin: f32) -> Vec<Vec<[f32; 2]>> {
    let mut bins: HashMap<(i32, i32), (f32, f32, u32)> = HashMap::new();
    for p in pts {
        let e = bins.entry(((p[0] / bin).floor() as i32, (p[1] / bin).floor() as i32)).or_insert((0.0, 0.0, 0));
        e.0 += p[0];
        e.1 += p[1];
        e.2 += 1;
    }
    let near = |c: (i32, i32)| -> Vec<(i32, i32)> {
        let mut v = Vec::with_capacity(8);
        for dy in -1..=1 {
            for dx in -1..=1 {
                if (dx, dy) != (0, 0) {
                    v.push((c.0 + dx, c.1 + dy));
                }
            }
        }
        v
    };
    let walk = |from: (i32, i32), within: &HashSet<(i32, i32)>| -> ((i32, i32), HashMap<(i32, i32), (i32, i32)>) {
        let mut parent: HashMap<(i32, i32), (i32, i32)> = HashMap::new();
        parent.insert(from, from);
        let mut q = VecDeque::from([from]);
        let mut last = from;
        while let Some(c) = q.pop_front() {
            last = c;
            for n in near(c) {
                if within.contains(&n) && !parent.contains_key(&n) {
                    parent.insert(n, c);
                    q.push_back(n);
                }
            }
        }
        (last, parent)
    };
    let mut seen: HashSet<(i32, i32)> = HashSet::new();
    let mut out = Vec::new();
    let mut starts: Vec<(i32, i32)> = bins.keys().copied().collect();
    starts.sort_unstable();
    for s in starts {
        if seen.contains(&s) {
            continue;
        }
        // The piece this bin belongs to.
        let mut piece: HashSet<(i32, i32)> = HashSet::new();
        let mut q = VecDeque::from([s]);
        piece.insert(s);
        while let Some(c) = q.pop_front() {
            for n in near(c) {
                if bins.contains_key(&n) && piece.insert(n) {
                    q.push_back(n);
                }
            }
        }
        seen.extend(piece.iter().copied());
        if piece.len() < 3 {
            continue;
        }
        let (a, _) = walk(s, &piece);
        let (b, parent) = walk(a, &piece);
        let mut path = Vec::new();
        let mut c = b;
        loop {
            let (x, y, n) = bins[&c];
            path.push([x / n as f32, y / n as f32]);
            if c == a {
                break;
            }
            c = parent[&c];
        }
        for _ in 0..2 {
            if path.len() > 2 {
                let prev = path.clone();
                for i in 1..path.len() - 1 {
                    path[i] = [(prev[i - 1][0] + 2.0 * prev[i][0] + prev[i + 1][0]) / 4.0, (prev[i - 1][1] + 2.0 * prev[i][1] + prev[i + 1][1]) / 4.0];
                }
            }
        }
        out.push(path);
    }
    out.sort_by(|a, b| lengths(b).last().partial_cmp(&lengths(a).last()).unwrap_or(std::cmp::Ordering::Equal));
    out
}

impl Labeller {
    pub fn new() -> Labeller {
        Labeller { text: TextRenderer::new(), buf: Vec::new(), w: 0, h: 0, placed: Vec::new(), anchor: None, drawn_for: None, advance: HashMap::new() }
    }

    /// Whether the lettering is out of date for this view: the camera, the depth or the cells resident changed since it was drawn (a cell arriving, leaving or replaced by a newer copy).
    pub fn stale(&self, map: &mahere_engine::MapCore, w: usize, h: usize) -> bool {
        self.drawn_for != Some(Self::view_key(map, w, h))
    }

    fn view_key(map: &mahere_engine::MapCore, w: usize, h: usize) -> (i64, i64, i64, i64, u8, usize) {
        let c = &map.cam;
        ((c.lat() * 1e7) as i64, (c.lon() * 1e7) as i64, (c.ppd * 100.0) as i64, (c.bearing() * 1e5) as i64, map.vec_depth(), map.pool_version() as usize ^ (w << 20) ^ (h << 40))
    }

    fn advance(&mut self, ch: char, size: f32, weight: u16) -> f32 {
        let key = (ch, weight, (size * 16.0) as u32);
        if let Some(&a) = self.advance.get(&key) {
            return a;
        }
        let a = self.text.measure_text(&ch.to_string(), &TextStyle::new(size, PLACE).weight(weight));
        self.advance.insert(key, a);
        a
    }

    fn width(&mut self, s: &str, size: f32, weight: u16) -> f32 {
        s.chars().map(|c| self.advance(c, size, weight)).sum()
    }

    /// Lay out and draw the labels for the view, `font` the panel's base font size in pixels.
    pub fn update(&mut self, map: &mut mahere_engine::MapCore, w: usize, h: usize, font: f32) {
        let t = std::time::Instant::now();
        self.w = w;
        self.h = h;
        let texels = map.gather_names(w, h);
        let gathered = t.elapsed();
        let previous: HashSet<(u8, String, String)> = self.placed.iter().map(|p| (p.name.kind, p.name.name.clone(), p.name.reference.clone())).collect();
        let laid = self.layout(&texels, w, h, font, &previous);
        self.draw(&laid, w, h);
        self.placed = laid.iter().map(|l| Placed { name: l.name.clone(), boxes: l.glyphs.iter().map(|g| glyph_box(g, l.size, 0.0)).collect() }).collect();
        // Two points of this view, by where they are on the globe, so the buffer can be carried with the map.
        let (a, b) = ([w as f64 * 0.25, h as f64 * 0.5], [w as f64 * 0.75, h as f64 * 0.5]);
        self.anchor = Some((a, b, map.cam.screen_to_geo(a[0], a[1], w, h), map.cam.screen_to_geo(b[0], b[1], w, h)));
        self.drawn_for = Some(Self::view_key(map, w, h));
        if std::env::var_os("MAHERE_LABEL_LOG").is_some() {
            eprintln!("labels: {} lines, {} areas, {} points gathered in {:.1} ms; {} placed, {:.1} ms in all", texels.lines.len(), texels.areas.len(), texels.points.len(), gathered.as_secs_f64() * 1e3, laid.len(), t.elapsed().as_secs_f64() * 1e3);
        }
    }

    /// The map moved since the lettering was drawn: where each pixel of the screen now was in the drawn buffer, as `[a, b, c, d, e, f]` with the buffer's x = a·x + b·y + c and y = d·x + e·y + f. A turn, a scale and a shift, near enough while the camera moves; the next rest draws it again. None when there is nothing drawn.
    pub fn transform(&self, cam: &mahere_engine::Camera, w: usize, h: usize) -> Option<[f32; 6]> {
        let (a, b, ga, gb) = self.anchor?;
        let (ax, ay) = cam.geo_to_screen(ga.0, ga.1, w, h);
        let (bx, by) = cam.geo_to_screen(gb.0, gb.1, w, h);
        // The similarity taking the points as they are now back to where they were drawn.
        let (nx, ny) = (bx - ax, by - ay);
        let (ox, oy) = (b[0] - a[0], b[1] - a[1]);
        let n2 = nx * nx + ny * ny;
        if n2 < 1e-6 {
            return None;
        }
        let re = (ox * nx + oy * ny) / n2;
        let im = (oy * nx - ox * ny) / n2;
        let c = a[0] - (re * ax - im * ay);
        let f = a[1] - (im * ax + re * ay);
        Some([re as f32, -im as f32, c as f32, im as f32, re as f32, f as f32])
    }

    /// The label under a screen point, as the screen is now.
    pub fn hit(&self, cam: &mahere_engine::Camera, w: usize, h: usize, x: f32, y: f32) -> Option<&Named> {
        let m = self.transform(cam, w, h)?;
        let (bx, by) = (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]);
        self.placed.iter().find(|p| p.boxes.iter().any(|b| bx >= b[0] && bx <= b[2] && by >= b[1] && by <= b[3])).map(|p| &p.name)
    }

    fn layout(&mut self, texels: &NamedTexels, w: usize, h: usize, font: f32, previous: &HashSet<(u8, String, String)>) -> Vec<Laid> {
        enum Shape<'a> {
            Point([f32; 2]),
            Area(&'a [[f32; 2]]),
            Line(&'a [[f32; 2]]),
        }
        let mut todo: Vec<(u8, bool, u32, usize, &Named, Shape)> = Vec::new();
        for (n, p) in &texels.points {
            if let Some(t) = tier(n) {
                todo.push((t, !previous.contains(&(n.kind, n.name.clone(), n.reference.clone())), n.rank, 1, n, Shape::Point(*p)));
            }
        }
        for (n, pts) in &texels.areas {
            if let Some(t) = tier(n) {
                todo.push((t, !previous.contains(&(n.kind, n.name.clone(), n.reference.clone())), n.rank, pts.len(), n, Shape::Area(pts)));
            }
        }
        for (n, pts) in &texels.lines {
            if let Some(t) = tier(n) {
                todo.push((t, !previous.contains(&(n.kind, n.name.clone(), n.reference.clone())), n.rank, pts.len(), n, Shape::Line(pts)));
            }
        }
        todo.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)).then(b.2.cmp(&a.2)).then(b.3.cmp(&a.3)));
        let mut room = Room::new(w, h);
        let mut out = Vec::new();
        let pad = font * 0.12;
        for (_, _, _, _, n, shape) in todo {
            let label = n.label();
            if label.is_empty() {
                continue;
            }
            let (rel, colour, weight, shear) = style_of(n);
            let size = font * rel;
            let laid: Vec<Laid> = match shape {
                Shape::Point(p) => self.point(n, &label, p, size, colour, weight, &room, pad).into_iter().collect(),
                Shape::Area(pts) => self.area(n, &label, pts, texels.area_step_px, size, colour, weight, shear, &room, pad).into_iter().collect(),
                Shape::Line(pts) => self.line(n, &label, pts, texels.texel_px, size, colour, weight, shear, &mut room, pad),
            };
            for l in laid {
                let boxes: Vec<[f32; 4]> = l.glyphs.iter().map(|g| glyph_box(g, l.size, pad)).collect();
                room.take(&boxes);
                out.push(l);
            }
        }
        out
    }

    /// A point's label: beside its mark, right of it first, then left, above and below; a town's name centred on it.
    #[allow(clippy::too_many_arguments)]
    fn point(&mut self, n: &Named, label: &str, p: [f32; 2], size: f32, colour: u32, weight: u16, room: &Room, pad: f32) -> Option<Laid> {
        let place = n.kind <= HAMLET;
        let summit = n.kind == PEAK || n.kind == VOLCANO;
        let width = self.width(label, size, weight);
        let gap = size * 0.55;
        let spots: Vec<[f32; 2]> = if place {
            vec![[p[0] - width / 2.0, p[1]], [p[0] - width / 2.0, p[1] - size * 1.1], [p[0] - width / 2.0, p[1] + size * 1.1]]
        } else {
            vec![[p[0] + gap, p[1]], [p[0] - gap - width, p[1]], [p[0] - width / 2.0, p[1] - size * 1.05], [p[0] - width / 2.0, p[1] + size * 1.05]]
        };
        for start in spots {
            let mut x = start[0];
            let mut glyphs = Vec::new();
            for ch in label.chars() {
                let a = self.advance(ch, size, weight);
                glyphs.push(Glyph { ch, at: [x + a / 2.0, start[1]], angle: 0.0 });
                x += a;
            }
            let mut boxes: Vec<[f32; 4]> = glyphs.iter().map(|g| glyph_box(g, size, pad)).collect();
            if !place {
                boxes.push([p[0] - size * 0.4, p[1] - size * 0.4, p[0] + size * 0.4, p[1] + size * 0.4]);
            }
            if room.free(&boxes) {
                let mark = (!place).then_some((p, summit));
                return Some(Laid { name: n.clone(), glyphs, size, colour, weight, shear: 0.0, mark });
            }
        }
        None
    }

    /// An area's label: at the point deepest inside it (a distance fill over its samples), along its long axis when it is long, as large as fits inside it up to its kind's size.
    #[allow(clippy::too_many_arguments)]
    fn area(&mut self, n: &Named, label: &str, pts: &[[f32; 2]], step: f32, size: f32, colour: u32, weight: u16, shear: f32, room: &Room, pad: f32) -> Option<Laid> {
        if pts.len() < 4 {
            return None;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        let cell = |p: [f32; 2]| ((p[0] / step).round() as i32, (p[1] / step).round() as i32);
        for &p in pts {
            let (x, y) = cell(p);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        // One empty cell around, so the edge of the box is outside.
        let (gw, gh) = ((x1 - x0 + 3) as usize, (y1 - y0 + 3) as usize);
        if gw * gh > 4_000_000 {
            return None;
        }
        let mut inside = vec![false; gw * gh];
        for &p in pts {
            let (x, y) = cell(p);
            inside[(y - y0 + 1) as usize * gw + (x - x0 + 1) as usize] = true;
        }
        // Chamfer distance to the outside, two passes, in cells.
        let mut d: Vec<f32> = inside.iter().map(|&i| if i { f32::MAX } else { 0.0 }).collect();
        let (s, dg) = (1.0f32, std::f32::consts::SQRT_2);
        for y in 0..gh {
            for x in 0..gw {
                let i = y * gw + x;
                if d[i] == 0.0 {
                    continue;
                }
                let mut v = d[i];
                if x > 0 { v = v.min(d[i - 1] + s); }
                if y > 0 { v = v.min(d[i - gw] + s); }
                if x > 0 && y > 0 { v = v.min(d[i - gw - 1] + dg); }
                if x + 1 < gw && y > 0 { v = v.min(d[i - gw + 1] + dg); }
                d[i] = v;
            }
        }
        for y in (0..gh).rev() {
            for x in (0..gw).rev() {
                let i = y * gw + x;
                if d[i] == 0.0 {
                    continue;
                }
                let mut v = d[i];
                if x + 1 < gw { v = v.min(d[i + 1] + s); }
                if y + 1 < gh { v = v.min(d[i + gw] + s); }
                if x + 1 < gw && y + 1 < gh { v = v.min(d[i + gw + 1] + dg); }
                if x > 0 && y + 1 < gh { v = v.min(d[i + gw - 1] + dg); }
                d[i] = v;
            }
        }
        let (best, depth) = d.iter().enumerate().fold((0usize, 0.0f32), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        let centre = [((best % gw) as i32 + x0 - 1) as f32 * step, ((best / gw) as i32 + y0 - 1) as f32 * step];
        // The long axis from the second moments of the cells inside.
        let (mut mx, mut my, mut cnt) = (0.0f64, 0.0f64, 0.0f64);
        for (i, &v) in inside.iter().enumerate() {
            if v {
                mx += (i % gw) as f64;
                my += (i / gw) as f64;
                cnt += 1.0;
            }
        }
        let (mx, my) = (mx / cnt, my / cnt);
        let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
        for (i, &v) in inside.iter().enumerate() {
            if v {
                let (dx, dy) = ((i % gw) as f64 - mx, (i / gw) as f64 - my);
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy;
            }
        }
        let axis = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let spread = ((sxx - syy).powi(2) + 4.0 * sxy * sxy).sqrt();
        let (l1, l2) = ((sxx + syy + spread) / 2.0, ((sxx + syy - spread) / 2.0).max(1e-9));
        let mut angle = if l1 / l2 > 4.0 { axis as f32 } else { 0.0 };
        // Top up: an angle past a quarter turn reads upside down.
        if angle > std::f32::consts::FRAC_PI_2 {
            angle -= std::f32::consts::PI;
        } else if angle < -std::f32::consts::FRAC_PI_2 {
            angle += std::f32::consts::PI;
        }
        let at = |x: f32, y: f32| -> bool {
            let (cx, cy) = ((x / step).round() as i32 - x0 + 1, (y / step).round() as i32 - y0 + 1);
            cx >= 0 && cy >= 0 && (cx as usize) < gw && (cy as usize) < gh && inside[cy as usize * gw + cx as usize]
        };
        // Largest first, from the kind's size scaled by how much room there is, down to the smallest readable.
        let room_px = depth * step;
        let max = (size * (1.0 + (room_px / (size * 6.0)).min(1.0))).max(size);
        let min = size * 0.8;
        let (ca, sa) = (angle.cos(), angle.sin());
        let mut sz = max;
        while sz >= min {
            let width = self.width(label, sz, weight);
            // The text's box, sampled, must lie inside.
            let fits = (0..=8).all(|k| {
                let u = (k as f32 / 8.0 - 0.5) * width;
                [-0.4f32, 0.0, 0.4].iter().all(|&v| {
                    let (vx, vy) = (u * ca - v * sz * sa, u * sa + v * sz * ca);
                    at(centre[0] + vx, centre[1] + vy)
                })
            });
            if fits {
                let mut s = -width / 2.0;
                let mut glyphs = Vec::new();
                for ch in label.chars() {
                    let a = self.advance(ch, sz, weight);
                    let u = s + a / 2.0;
                    glyphs.push(Glyph { ch, at: [centre[0] + u * ca, centre[1] + u * sa], angle });
                    s += a;
                }
                let boxes: Vec<[f32; 4]> = glyphs.iter().map(|g| glyph_box(g, sz, pad)).collect();
                if room.free(&boxes) {
                    return Some(Laid { name: n.clone(), glyphs, size: sz, colour, weight, shear, mark: None });
                }
                return None;
            }
            sz -= (size * 0.08).max(0.5);
        }
        None
    }

    /// A line's labels: on each long enough run, the stretch straight enough for the name nearest the run's middle, and on a long run more of them at least a screen's third apart, each letter turned to the line, top up.
    #[allow(clippy::too_many_arguments)]
    fn line(&mut self, n: &Named, label: &str, pts: &[[f32; 2]], texel_px: f32, size: f32, colour: u32, weight: u16, shear: f32, room: &mut Room, pad: f32) -> Vec<Laid> {
        let advances: Vec<(char, f32)> = label.chars().map(|c| (c, self.advance(c, size, weight))).collect();
        let width: f32 = advances.iter().map(|a| a.1).sum();
        let bin = (texel_px * 3.0).max(6.0);
        let mut out = Vec::new();
        let repeat = (self.w.max(self.h) as f32 / 3.0).max(width * 2.0);
        for path in centrelines(pts, bin).into_iter().take(4) {
            let len = lengths(&path);
            let total = *len.last().unwrap_or(&0.0);
            if total < width * 1.15 + size {
                continue;
            }
            let mut taken: Vec<f32> = Vec::new();
            let step = (width / 8.0).max(4.0);
            let mut starts: Vec<f32> = Vec::new();
            let mut s0 = size * 0.5;
            while s0 + width <= total - size * 0.5 {
                starts.push(s0);
                s0 += step;
            }
            // Nearest the middle first.
            starts.sort_by(|a, b| ((a + width / 2.0) - total / 2.0).abs().partial_cmp(&((b + width / 2.0) - total / 2.0).abs()).unwrap_or(std::cmp::Ordering::Equal));
            for s in starts {
                if taken.iter().any(|&t| (t - s).abs() < repeat) {
                    continue;
                }
                // The letters along this stretch, and how much it bends from letter to letter.
                let mut at = s;
                let mut glyphs: Vec<Glyph> = Vec::new();
                let mut bends = 0.0;
                let mut worst = 0.0f32;
                for &(ch, a) in &advances {
                    let (p, dir) = along(&path, &len, at + a / 2.0);
                    if let Some(last) = glyphs.last() {
                        let t = turn(last.angle, dir);
                        bends += t;
                        worst = worst.max(t);
                    }
                    glyphs.push(Glyph { ch, at: p, angle: dir });
                    at += a;
                }
                if worst > 0.45 || bends > 1.0 {
                    continue;
                }
                // Top up: a stretch running right to left is lettered from its other end, and one running straight down too, so a vertical name reads upward as maps letter them.
                let (first, last) = (glyphs.first().unwrap().at, glyphs.last().unwrap().at);
                let (dx, dy) = (last[0] - first[0], last[1] - first[1]);
                let steep = dx.abs() <= dx.hypot(dy) * 0.05;
                if (!steep && dx < 0.0) || (steep && dy > 0.0) {
                    let mut at = s + width;
                    glyphs.clear();
                    for &(ch, a) in &advances {
                        let (p, dir) = along(&path, &len, at - a / 2.0);
                        glyphs.push(Glyph { ch, at: p, angle: dir + std::f32::consts::PI });
                        at -= a;
                    }
                }
                let boxes: Vec<[f32; 4]> = glyphs.iter().map(|g| glyph_box(g, size, pad)).collect();
                if !room.free(&boxes) {
                    continue;
                }
                room.take(&boxes);
                taken.push(s);
                out.push(Laid { name: n.clone(), glyphs, size, colour, weight, shear, mark: None });
            }
        }
        out
    }

    fn draw(&mut self, laid: &[Laid], w: usize, h: usize) {
        self.w = w;
        self.h = h;
        self.buf.clear();
        self.buf.resize(w * h, 0);
        let mut damage = Damage::new();
        let mut canvas = Canvas::new(&mut self.buf, w, h, &mut damage);
        // Fluor composites front to back: every label's letters first, then the marks, then all the halos under them, so no halo covers a neighbour's letters.
        for l in laid {
            for g in &l.glyphs {
                let style = TextStyle::new(l.size, l.colour).weight(l.weight).shear(l.shear).rotation(g.angle);
                let at = unturn(g.at, g.angle);
                self.text.draw_text_center(&mut canvas, &g.ch.to_string(), at[0], at[1], &style, None, None);
            }
        }
        for l in laid {
            if let Some((p, summit)) = l.mark {
                // A summit's triangle, point up; a dot for the small marks.
                let (glyph, fill, scale) = if summit { ("\u{25B2}", SUMMIT, 0.7) } else { ("\u{2022}", MARK, 1.0) };
                self.text.draw_text_center(&mut canvas, glyph, p[0], p[1], &TextStyle::new(l.size * scale, fill), None, None);
            }
        }
        // The halos: each letter again in the paper's colour, a little around it.
        for l in laid {
            let r = (l.size * 0.11).max(1.0);
            for g in &l.glyphs {
                let style = TextStyle::new(l.size, HALO).weight(l.weight).shear(l.shear).rotation(g.angle);
                for k in 0..8 {
                    let a = k as f32 * std::f32::consts::FRAC_PI_4;
                    let at = unturn([g.at[0] + r * a.cos(), g.at[1] + r * a.sin()], g.angle);
                    self.text.draw_text_center(&mut canvas, &g.ch.to_string(), at[0], at[1], &style, None, None);
                }
            }
        }
    }

    /// The lettering as premultiplied RGBA bytes for a GPU host.
    pub fn rgba(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.buf.len() * 4];
        for (i, &p) in self.buf.iter().enumerate() {
            let (r, g, b, a) = crate::Panel::premul_pub(p);
            out[4 * i] = r;
            out[4 * i + 1] = g;
            out[4 * i + 2] = b;
            out[4 * i + 3] = a;
        }
        out
    }

    /// For a CPU frontend: the lettering over a visible 0xRRGGBB canvas, in place.
    pub fn composite_rgb(&self, canvas: &mut [u32]) {
        for (i, &p) in self.buf.iter().enumerate() {
            if p >> 24 == 0 || i >= canvas.len() {
                continue;
            }
            let (r, g, b, a) = crate::Panel::premul_pub(p);
            let keep = 255 - a as u32;
            let c = canvas[i];
            let mix = |cv: u32, v: u8| (v as u32 + cv * keep / 255).min(255);
            canvas[i] = (mix((c >> 16) & 255, r) << 16) | (mix((c >> 8) & 255, g) << 8) | mix(c & 255, b);
        }
    }
}
