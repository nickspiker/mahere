//! Labels on the map (Nick 2026-10-13): laid out from what is drawn, never from stored label geometry. The engine hands over every named texel in view; here the points are lettered beside their mark, an area at the point deepest inside it (a distance fill) along its long axis when it is long, a line along the longest run of it straight enough for its name, with top always up. One priority order decides who goes first (places and summits, then lakes, highways, rivers, villages, trails and streets, other areas, the small marks), each label taking its room on a grid so none overlap. A label placed last time goes first in its tier, so labels hold still while the view settles.
//!
//! Laid out when the camera rests; drawn every frame letter by letter (Nick 2026-10-13: no rendering to a bitmap). Each label keeps where it is on the globe and its letters' places about that point; each letter is a glyph fluor rasterised once into an atlas, its halo beside it. A frame turns the labels into quads at their anchors' places on the screen now (a line's or an area's name turns with the map, a point's stays level) for the GPU to draw, or the CPU frontend draws the same quads itself.
use std::collections::{HashMap, HashSet, VecDeque};

use fluor::canvas::{Canvas, Damage};
use fluor::text::{TextRenderer, TextStyle};
use fluor::theme;
use mahere_engine::labels::NamedTexels;
use mahere_tiles::Named;


// The theme's label colours by index (`mahere_engine::theme::LABEL_NAMES`).
const PLACE: usize = 0;
const SUMMIT: usize = 1;
const WATER: usize = 2;
const ROAD: usize = 3;
const TRAIL: usize = 4;
const BOUNDARY: usize = 5;
const GREEN: usize = 6;
const MARK: usize = 7;
const HALO: usize = 8;
const ROUTE: usize = 9;
const ROUTE_CASING: usize = 10;

/// A theme colour (VSF RGB at gamma 2 with opacity) as display values 0..1 with its opacity, the way the screen takes it.
/// A theme colour as stored (VSF RGB at gamma 2, 0..1, opacity last), for the GPU, which encodes it for the display as it does the map's own colours, through the same highlights (Nick 2026-10-21: a halo glaring white under linear highlights).
fn stored(c: [u8; 4]) -> [f32; 4] {
    [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, c[3] as f32 / 255.0]
}

/// How strong a halo is against its theme's opacity (Nick 2026-10-21: the halos read a little bright on every theme).
const HALO_STRENGTH: f32 = 0.8;

fn display(c: [u8; 4]) -> [f32; 4] {
    let d = theme::vsf(((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32);
    [((d >> 16) & 255) as f32 / 255.0, ((d >> 8) & 255) as f32 / 255.0, (d & 255) as f32 / 255.0, c[3] as f32 / 255.0]
}

/// A rasterised glyph's identity: the character, its size, weight and lean, its halo's width (sizes in sixteenths of a pixel), and its rotation bin. As fluor's own rotation stack does, the outline is turned in font space to the bin's angle before it is rasterised, bins a pixel of arc apart at the glyph's size, so a letter at any angle has fluor's own edges and the quad turns only by what is left over (nothing, at rest).
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct GlyphKey {
    ch: char,
    /// The typeface, an index into [`mahere_engine::theme::LABEL_FONTS`].
    family: u8,
    size: u32,
    weight: u16,
    shear: u32,
    halo: u32,
    bin: i32,
}

/// The typefaces a theme letters its map in, ASCII only (assets/fonts/README.md): variable fonts drawn at any weight on their wght axis (fluor on cosmic-text 0.19), Mahere Mono in five static weights; loaded into the letterer's text renderer beside the panel's Open Sans, which every letter outside them falls back to.
const LABEL_FONT_FILES: [&[u8]; 11] = [
    include_bytes!("../assets/fonts/Cinzel.ttf"),
    include_bytes!("../assets/fonts/EBGaramond.ttf"),
    include_bytes!("../assets/fonts/Fredoka.ttf"),
    include_bytes!("../assets/fonts/MahereMono-300.ttf"),
    include_bytes!("../assets/fonts/MahereMono-400.ttf"),
    include_bytes!("../assets/fonts/MahereMono-500.ttf"),
    include_bytes!("../assets/fonts/MahereMono-600.ttf"),
    include_bytes!("../assets/fonts/MahereMono-700.ttf"),
    include_bytes!("../assets/fonts/OpenSansVariable.ttf"),
    include_bytes!("../assets/fonts/Oxanium.ttf"),
    include_bytes!("../assets/fonts/RobotoCondensed.ttf"),
];

/// Atlas pixels per screen pixel (Nick 2026-10-16: letters composite into the still frame's 3× layer and bin down with the map, for the same organic scaling as its lines): a glyph is rasterised this much larger, drawn texel for pixel into a 3× target and box-filtered into a 1× one. The GPU shader's ATLAS_DENSITY is the same number.
pub const ATLAS_DENSITY: f32 = 3.0;

/// A glyph's rotation bins, radians apart: one pixel of arc at half its size.
fn bin_step(size: f32) -> f32 {
    2.0 / size.max(4.0)
}

/// Where to ask fluor to draw a glyph turned by `angle` so it lands centred on `p`: fluor turns a glyph's position about the canvas's origin along with the glyph, so the position goes in turned back.
fn unturn(p: [f32; 2], angle: f32) -> [f32; 2] {
    let (s, c) = angle.sin_cos();
    [p[0] * c + p[1] * s, -p[0] * s + p[1] * c]
}

/// The glyphs as one texture: two bytes a pixel, the letter's coverage and its halo's, packed in shelves. Grows to 4096 square, then starts again with what the labels in view use.
pub struct Atlas {
    pub side: usize,
    pub rg: Vec<u8>,
    /// Bumped whenever the texture changed, for a host to upload it again.
    pub version: u64,
    x: usize,
    y: usize,
    row: usize,
    slots: HashMap<GlyphKey, [u16; 4]>,
}

impl Atlas {
    fn new() -> Atlas {
        Atlas { side: 2048, rg: vec![0; 2048 * 2048 * 2], version: 1, x: 0, y: 0, row: 0, slots: HashMap::new() }
    }

    fn clear(&mut self, side: usize) {
        *self = Atlas { side, rg: vec![0; side * side * 2], version: self.version + 1, x: 0, y: 0, row: 0, slots: HashMap::new() };
    }

    /// Room for a glyph of this many pixels: the next place on the shelf, or on a new shelf; None when the texture is full.
    fn place(&mut self, w: usize, h: usize) -> Option<(usize, usize)> {
        if self.x + w + 1 > self.side {
            self.x = 0;
            self.y += self.row + 1;
            self.row = 0;
        }
        if self.y + h + 1 > self.side {
            return None;
        }
        let at = (self.x, self.y);
        self.x += w + 1;
        self.row = self.row.max(h);
        Some(at)
    }
}

/// A label as kept between layouts: where it is on the globe, its letters (atlas glyph, offset from the anchor in pixels, angle as laid out) as laid out, whether it turns with the map, its colour (an index into the theme's) and whether it has a halo.
#[derive(Clone)]
struct Kept {
    name: Named,
    anchor: (f64, f64),
    letters: Vec<(GlyphKey, [f32; 2], f32)>,
    turns: bool,
    colour: usize,
    halo: bool,
}

/// One letter for a frame: centre and half extents in screen pixels, angle, the atlas rectangle (0..1), the letter's colour and its halo's (display values, opacity last). As 20 floats, the GPU's instance layout.
pub type GlyphQuad = [f32; 20];

/// One stretch of the route for a frame: its ends in screen pixels, half its width, and its colour. As 12 floats, the GPU's instance layout.
pub type Segment = [f32; 12];

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
// The details of 2026-10-16 (Nick: exit numbers, islands, bays, lookouts, hot springs "and other goodies").
const FERRY: u8 = mahere_tiles::osm::RoadClass::Ferry as u8 + 1;
const AERIALWAY: u8 = mahere_tiles::osm::RoadClass::Aerialway as u8 + 1;
const RIDGE: u8 = mahere_tiles::osm::RoadClass::Ridge as u8 + 1;
const CLIFF: u8 = mahere_tiles::osm::RoadClass::Cliff as u8 + 1;
use mahere_tiles::osm::names::PointKind as Pk;
const EXIT: u8 = POINT + Pk::Exit as u8;
const ISLAND: u8 = POINT + Pk::Island as u8;
const ISLET: u8 = POINT + Pk::Islet as u8;
const BAY: u8 = POINT + Pk::Bay as u8;
const STRAIT: u8 = POINT + Pk::Strait as u8;
const SEA: u8 = POINT + Pk::Sea as u8;
const CAPE: u8 = POINT + Pk::Cape as u8;
const NEIGHBOURHOOD: u8 = POINT + Pk::Neighbourhood as u8;
const LOCALITY: u8 = POINT + Pk::Locality as u8;
const DRINKING_WATER: u8 = POINT + Pk::DrinkingWater as u8;
const TOILETS: u8 = POINT + Pk::Toilets as u8;
const SHELTER: u8 = POINT + Pk::Shelter as u8;
const RANGER_STATION: u8 = POINT + Pk::RangerStation as u8;
const LOOKOUT: u8 = POINT + Pk::Lookout as u8;
const HOT_SPRING: u8 = POINT + Pk::HotSpring as u8;
const CAVE: u8 = POINT + Pk::Cave as u8;
const GATE: u8 = POINT + Pk::Gate as u8;
const FORD: u8 = POINT + Pk::Ford as u8;

/// A point's mark: its symbol, the theme colour it takes, its size against the label's. Every symbol is in the fonts every build carries (Open Sans, Noto Sans Symbols 2).
fn mark_of(kind: u8) -> (char, usize, f32) {
    match kind {
        PEAK | VOLCANO => ('\u{25B2}', SUMMIT, 0.7),
        LOOKOUT => ('\u{265C}', SUMMIT, 0.9),
        HOT_SPRING => ('\u{2668}', WATER, 0.95),
        DRINKING_WATER => ('\u{25C9}', WATER, 0.7),
        FORD => ('\u{2248}', WATER, 0.9),
        SHELTER => ('\u{1F6D6}', MARK, 0.9),
        RANGER_STATION => ('\u{25C6}', MARK, 0.7),
        CAVE => ('\u{25D3}', MARK, 0.8),
        GATE => ('\u{25A0}', MARK, 0.5),
        _ => ('\u{2022}', MARK, 1.0),
    }
}

/// The points whose names sit centred on them with no mark: settlements, districts, islands, capes, waters, exits.
fn centred(kind: u8) -> bool {
    kind <= HAMLET || matches!(kind, EXIT | ISLAND | ISLET | BAY | STRAIT | SEA | CAPE | NEIGHBOURHOOD | LOCALITY)
}

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
        ISLAND | BAY | STRAIT | SEA => 2,
        NEIGHBOURHOOD => 5,
        FERRY | RIDGE => 6,
        EXIT => 7,
        ISLET | CAPE | LOCALITY | AERIALWAY | CLIFF => 8,
        k if k >= POINT => 9,
        k if k >= AREA => 8,
        _ => 7,
    })
}

/// The kinds lettered by rank among their own in view: places by population, summits by height, rivers by catchment, lakes by area.
fn rank_group(n: &Named) -> Option<u8> {
    match n.kind {
        CITY..=HAMLET => Some(0),
        PEAK | VOLCANO => Some(1),
        WATERWAY => Some(2),
        WATER_AREA => Some(3),
        _ => None,
    }
}

/// A ranked kind's size before its step.
fn ranked_base(n: &Named) -> f32 {
    match n.kind {
        CITY..=HAMLET => 1.0,
        PEAK | VOLCANO => 0.85,
        WATERWAY => 0.8,
        _ => 0.95,
    }
}

/// The five steps, smallest rank in view to largest: size against the kind's base, and weight.
const RANK_SIZE: [f32; 5] = [0.85, 0.95, 1.05, 1.2, 1.4];
const RANK_WEIGHT: [u16; 5] = [300, 400, 500, 600, 700];

/// How a kind is lettered: its size against the base font, its colour (an index into the theme's), its weight, its lean.
fn style_of(n: &Named) -> (f32, usize, u16, f32) {
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
        EXIT => (0.72, PLACE, 700, 0.0),
        ISLAND => (1.0, PLACE, 600, 0.0),
        ISLET | CAPE => (0.8, PLACE, 400, 0.0),
        NEIGHBOURHOOD => (0.82, PLACE, 500, 0.0),
        LOCALITY => (0.78, GREEN, 400, ITALIC),
        BAY | STRAIT => (1.0, WATER, 400, ITALIC),
        SEA => (1.2, WATER, 400, ITALIC),
        LOOKOUT => (0.75, SUMMIT, 400, 0.0),
        HOT_SPRING | DRINKING_WATER | FORD => (0.75, WATER, 400, 0.0),
        FERRY => (0.75, WATER, 400, ITALIC),
        RIDGE => (0.8, SUMMIT, 400, ITALIC),
        AERIALWAY | CLIFF => (0.7, ROAD, 400, 0.0),
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
    colour: usize,
    weight: u16,
    shear: f32,
    /// A mark drawn at the point, by its kind ([`mark_of`]): a summit's triangle, a lookout's tower, a dot for the small marks.
    mark: Option<([f32; 2], u8)>,
}

/// The view a layout is for: the camera, the depth and the resident cells, the route, the labels layer and the theme. A different key is a stale layout.
type ViewKey = (i64, i64, i64, i64, u8, usize, u64, bool, u64);

/// A finished layout, what each frame draws from: the labels as anchors and letters, the atlas their glyphs are in, where the view was, the theme's colours, and the route.
#[derive(Clone)]
pub struct Lettering {
    kept: Vec<Kept>,
    pub atlas: std::sync::Arc<Atlas>,
    /// Two screen points of the laid-out view and where on the globe they were: how far the map has turned since, which the turning labels follow.
    reference: Option<([f64; 2], [f64; 2], (f64, f64), (f64, f64))>,
    drawn_for: Option<ViewKey>,
    /// The camera's scale when it was laid out, pixels per degree.
    ppd: f64,
    palette: [[u8; 4]; 11],
    /// The route, on the globe, and its width in pixels.
    route: Vec<(f64, f64)>,
    route_width: f32,
    route_casing: f32,
}

impl Default for Lettering {
    fn default() -> Self {
        Lettering { kept: Vec::new(), atlas: std::sync::Arc::new(Atlas::new()), reference: None, drawn_for: None, ppd: 0.0, palette: mahere_engine::theme::LABELS_LIGHT, route: Vec::new(), route_width: 4.0, route_casing: 6.0 }
    }
}

/// What the worker lays out from: the map's names job (None with labels off), the panel's font, the theme's colours and look, the route, and the key of the view.
struct LabelJob {
    names: Option<mahere_engine::labels::NameJob>,
    cam: mahere_engine::Camera,
    w: usize,
    h: usize,
    font: f32,
    palette: [[u8; 4]; 11],
    look: [f32; 4],
    amount: [f32; 2],
    family: u8,
    route: Vec<(f64, f64)>,
    key: ViewKey,
    /// The font packs fetched so far, for the worker to load before it lays out.
    fonts: Option<std::sync::Arc<crate::fonts::FontPacks>>,
}

/// The worker's side: the text renderer, the master atlas, the advances measured, the last layout's names (placed first again), and the cells' names read so far.
struct Letterer {
    text: TextRenderer,
    w: usize,
    h: usize,
    atlas: Atlas,
    /// The atlas version last handed to the drawing thread.
    sent: u64,
    shared: std::sync::Arc<Atlas>,
    advance: HashMap<(char, u8, u16, u32), f32>,
    /// The typeface of the layout being made.
    family: u8,
    look: [f32; 4],
    amount: [f32; 2],
    previous: HashSet<(u8, String, String)>,
    names: mahere_engine::labels::NamesCache,
    /// How far through the font packs' arrivals this renderer has loaded.
    font_cursor: crate::fonts::FontCursor,
    /// The regional face the view's Han draws in (its names' kana or Hangul decide).
    han: fluor::packs::Han,
}

/// Labels (Nick 2026-10-13): laid out on a worker thread, never the drawing thread (a layout once fetched cells mid-frame and hung the app); each frame draws from the latest finished layout.
pub struct Labeller {
    tx: std::sync::mpsc::Sender<LabelJob>,
    rx: std::sync::mpsc::Receiver<Lettering>,
    shown: Lettering,
    busy: bool,
    asked: Option<ViewKey>,
    last: std::time::Instant,
    /// The font packs, when the host gave them, and how many had arrived when the shown layout was asked for.
    fonts: Option<std::sync::Arc<crate::fonts::FontPacks>>,
    fonts_laid: usize,
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

/// A letter on a curve between two arc lengths (the second before the first runs it backwards, top up the other way): centred on the curve halfway, turned along the chord from where it starts to where it ends, as maps letter a bend, so the letters neither crowd on the inside nor splay on the outside.
fn letter(path: &[[f32; 2]], len: &[f32], ch: char, from: f32, to: f32) -> Glyph {
    let (a, _) = along(path, len, from);
    let (b, _) = along(path, len, to);
    let (mid, _) = along(path, len, (from + to) / 2.0);
    Glyph { ch, at: mid, angle: (b[1] - a[1]).atan2(b[0] - a[0]) }
}

/// A polyline resampled every `step` pixels along its length and smoothed by a Gaussian of `sigma` pixels, its ends held: a smooth curve through a line's texels at the scale a label reads it.
fn smooth(path: &[[f32; 2]], step: f32, sigma: f32) -> Vec<[f32; 2]> {
    let len = lengths(path);
    let total = *len.last().unwrap_or(&0.0);
    if path.len() < 3 || total < step * 3.0 {
        return path.to_vec();
    }
    let n = (total / step).ceil() as usize + 1;
    let even: Vec<[f32; 2]> = (0..n).map(|k| along(path, &len, (k as f32 * step).min(total)).0).collect();
    let r = ((sigma / step) * 2.5).ceil() as i64;
    let weights: Vec<f32> = (-r..=r).map(|k| (-0.5 * (k as f32 * step / sigma).powi(2)).exp()).collect();
    (0..n)
        .map(|i| {
            // Near an end the window is cut to what is there on both sides, so the end stays where it is.
            let reach = r.min(i as i64).min((n - 1 - i) as i64);
            let (mut x, mut y, mut wsum) = (0.0, 0.0, 0.0);
            for k in -reach..=reach {
                let w = weights[(k + r) as usize];
                let p = even[(i as i64 + k) as usize];
                x += p[0] * w;
                y += p[1] * w;
                wsum += w;
            }
            [x / wsum, y / wsum]
        })
        .collect()
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
        let (tx, jobs) = std::sync::mpsc::channel::<LabelJob>();
        let (done, rx) = std::sync::mpsc::channel::<Lettering>();
        std::thread::Builder::new()
            .name("labels".into())
            .spawn(move || {
                let mut text = TextRenderer::new();
                for data in LABEL_FONT_FILES {
                    text.load_font_data_named(data.to_vec());
                }
                let mut l = Letterer { text, family: 0, w: 0, h: 0, atlas: Atlas::new(), sent: 0, shared: std::sync::Arc::new(Atlas::new()), advance: HashMap::new(), look: mahere_engine::theme::LABEL_LOOK, amount: mahere_engine::theme::LABEL_AMOUNT, previous: HashSet::new(), names: Default::default() , font_cursor: Default::default(), han: fluor::packs::Han::Simplified };
                while let Ok(mut job) = jobs.recv() {
                    // Only the newest job matters: a camera that moved on has made the others stale.
                    while let Ok(newer) = jobs.try_recv() {
                        job = newer;
                    }
                    if done.send(l.lay_out(job)).is_err() {
                        break;
                    }
                }
            })
            .expect("labels thread");
        Labeller { tx, rx, shown: Lettering::default(), busy: false, asked: None, last: std::time::Instant::now(), fonts: None, fonts_laid: 0 }
    }

    /// The font packs to draw names in scripts the binary lacks (Nick 2026-10-16): a layout that needed one asks for it, and one arriving lays the view out again.
    pub fn set_fonts(&mut self, fonts: std::sync::Arc<crate::fonts::FontPacks>) {
        self.fonts = Some(fonts);
    }

    /// Font packs arrived since the shown layout was asked for.
    fn fonts_new(&self) -> bool {
        self.fonts.as_ref().is_some_and(|f| f.arrived() != self.fonts_laid)
    }

    /// Whether the lettering shown is out of date for this view: the camera, the depth or the cells resident changed since it was laid out (a cell arriving, leaving or replaced by a newer copy).
    pub fn stale(&self, map: &mahere_engine::MapCore, w: usize, h: usize) -> bool {
        self.shown.drawn_for != Some(Self::view_key(map, w, h))
    }

    /// Whether a layout is running or due: a host keeps drawing frames until it lands.
    pub fn pending(&self, map: &mahere_engine::MapCore, w: usize, h: usize) -> bool {
        // A pack being fetched keeps frames coming, so its arrival is seen and the view lettered again in it.
        self.busy || (Self::wanted(map) && (self.stale(map, w, h) || self.fonts_new() || self.fonts.as_ref().is_some_and(|f| f.busy())))
    }

    fn job(&self, map: &mut mahere_engine::MapCore, w: usize, h: usize, font: f32) -> LabelJob {
        let (palette, look, amount) = map.themes.get(map.theme).map_or((mahere_engine::theme::LABELS_LIGHT, mahere_engine::theme::LABEL_LOOK, mahere_engine::theme::LABEL_AMOUNT), |t| (t.label, t.label_look, t.label_amount));
        let family = map.themes.get(map.theme).and_then(|t| mahere_engine::theme::LABEL_FONTS.iter().position(|f| *f == t.label_font)).unwrap_or(0) as u8;
        let key = Self::view_key(map, w, h);
        let names = map.layers().labels.then(|| map.name_job(w, h));
        LabelJob { names, cam: map.cam, w, h, font, palette, look, amount, family, route: map.route_points().map_or(Vec::new(), |p| p.to_vec()), key, fonts: self.fonts.clone() }
    }

    /// Ask the worker for a layout when one is due: the view rests, the lettering is stale, none is running, and every cell is in or a second has passed since the last ask (so cells still arriving never hold the labels back). Never waits.
    pub fn ask(&mut self, map: &mut mahere_engine::MapCore, w: usize, h: usize, font: f32, at_rest: bool) {
        if !Self::wanted(map) {
            if !self.shown.kept.is_empty() || !self.shown.route.is_empty() {
                self.shown = Lettering { atlas: self.shown.atlas.clone(), ..Lettering::default() };
            }
            return;
        }
        let key = Self::view_key(map, w, h);
        let fonts_new = self.fonts_new();
        if self.busy || !at_rest || (!fonts_new && (self.shown.drawn_for == Some(key) || self.asked == Some(key))) {
            return;
        }
        if !map.converged() && self.last.elapsed() < std::time::Duration::from_secs(1) {
            return;
        }
        let job = self.job(map, w, h, font);
        if self.tx.send(job).is_ok() {
            self.busy = true;
            self.asked = Some(key);
            self.last = std::time::Instant::now();
            self.fonts_laid = self.fonts.as_ref().map_or(0, |f| f.arrived());
        }
    }

    /// A finished layout, if one landed: true when the frame should be drawn again with it.
    pub fn poll(&mut self) -> bool {
        let mut landed = false;
        while let Ok(l) = self.rx.try_recv() {
            self.shown = l;
            landed = true;
        }
        if landed {
            self.busy = false;
        }
        landed
    }

    /// Lay out the view now and wait for it (a check or a test).
    pub fn update(&mut self, map: &mut mahere_engine::MapCore, w: usize, h: usize, font: f32) {
        let job = self.job(map, w, h, font);
        let key = job.key;
        if self.tx.send(job).is_ok() {
            while let Ok(l) = self.rx.recv() {
                let done = l.drawn_for == Some(key);
                self.shown = l;
                if done {
                    break;
                }
            }
        }
        self.busy = false;
    }

    pub fn atlas(&self) -> &Atlas {
        &self.shown.atlas
    }

    pub fn glyphs(&self, cam: &mahere_engine::Camera, w: usize, h: usize) -> Vec<GlyphQuad> {
        self.shown.glyphs(cam, w, h)
    }

    pub fn segments(&self, cam: &mahere_engine::Camera, w: usize, h: usize) -> Vec<Segment> {
        self.shown.segments(cam, w, h)
    }

    pub fn draw_cpu(&self, canvas: &mut [u32], cam: &mahere_engine::Camera, w: usize, h: usize) {
        self.shown.draw_cpu(canvas, cam, w, h)
    }

    pub fn hit(&self, cam: &mahere_engine::Camera, w: usize, h: usize, x: f32, y: f32) -> Option<&Named> {
        self.shown.hit(cam, w, h, x, y)
    }

    fn view_key(map: &mahere_engine::MapCore, w: usize, h: usize) -> ViewKey {
        let c = &map.cam;
        ((c.lat() * 1e7) as i64, (c.lon() * 1e7) as i64, (c.ppd * 100.0) as i64, (c.bearing() * 1e5) as i64, map.vec_depth(), map.pool_version() as usize ^ (w << 20) ^ (h << 40), map.route_version, map.layers().labels, map.style_version())
    }

    /// Whether there is anything to draw: labels shown, or a route.
    pub fn wanted(map: &mahere_engine::MapCore) -> bool {
        map.layers().labels || map.route_points().is_some()
    }
}

impl Letterer {
    /// The typeface's family name.
    fn face(&self) -> &'static str {
        match mahere_engine::theme::LABEL_FONTS[(self.family as usize).min(mahere_engine::theme::LABEL_FONTS.len() - 1)] {
            // The variable Open Sans under a name of its own, so the panel's static Regular and Bold never stand in for the weights between.
            "Open Sans" => "Open Sans Variable",
            f => f,
        }
    }

    /// The regional face for a label's Han characters, from its name and where it sits ([`crate::fonts::han_at`]).
    fn use_han(&mut self, name: &str, at: (f64, f64)) {
        let han = crate::fonts::han_at(name, at.0, at.1);
        if han != self.han {
            self.han = han;
            self.text.set_han(han);
        }
    }

    /// A glyph's cache family: the theme's typeface, or for a CJK character its regional face (one character's Japanese and Chinese forms differ, and the theme does not draw either).
    fn key_family(&self, ch: char) -> u8 {
        if fluor::packs::is_cjk(ch) { 0x80 | self.han as u8 } else { self.family }
    }

    fn advance(&mut self, ch: char, size: f32, weight: u16) -> f32 {
        let key = (ch, self.key_family(ch), weight, (size * 16.0) as u32);
        if let Some(&a) = self.advance.get(&key) {
            return a;
        }
        let a = self.text.measure_text(&ch.to_string(), &TextStyle::new(size, 0xFF00_0000).weight(weight).font(self.face()));
        self.advance.insert(key, a);
        a
    }

    fn width(&mut self, s: &str, size: f32, weight: u16) -> f32 {
        s.chars().map(|c| self.advance(c, size, weight)).sum()
    }

    /// Lay out the labels for a job, rasterising into the atlas any glyph they use that it lacks: a finished layout for the drawing thread.
    fn lay_out(&mut self, job: LabelJob) -> Lettering {
        let t = std::time::Instant::now();
        let (w, h) = (job.w, job.h);
        self.w = w;
        self.h = h;
        self.look = job.look;
        self.amount = job.amount;
        self.family = job.family;
        // Font packs arrived since the last layout: loaded, and every advance and glyph measured before them measured again.
        if let Some(f) = &job.fonts {
            if crate::fonts::load_into(&mut self.text, f, &mut self.font_cursor) {
                self.advance.clear();
                let side = self.atlas.side;
                self.atlas.clear(side);
            }
        }
        let font = job.font * self.look[0];
        let texels = match &job.names {
            Some(n) => mahere_engine::labels::gather(n, &mut self.names),
            None => NamedTexels::default(),
        };
        let han = crate::fonts::han_for(texels.lines.iter().map(|(n, _)| n.name.as_str()).chain(texels.areas.iter().map(|(n, _)| n.name.as_str())).chain(texels.points.iter().map(|(n, _)| n.name.as_str())));
        self.han = han;
        self.text.set_han(han);
        let gathered = t.elapsed();
        let previous = std::mem::take(&mut self.previous);
        let laid = self.layout(&texels, &job.cam, w, h, font, &previous);
        let kept = self.keep(&laid, &job.cam, w, h);
        self.previous = kept.iter().map(|k| (k.name.kind, k.name.name.clone(), k.name.reference.clone())).collect();
        if self.atlas.version != self.sent {
            self.shared = std::sync::Arc::new(Atlas { side: self.atlas.side, rg: self.atlas.rg.clone(), version: self.atlas.version, x: self.atlas.x, y: self.atlas.y, row: self.atlas.row, slots: self.atlas.slots.clone() });
            self.sent = self.atlas.version;
        }
        // The packs this layout's names needed and lacked, fetched for the next.
        if let Some(f) = &job.fonts {
            crate::fonts::ask(&mut self.text, f, &mut self.font_cursor);
        }
        let route_width = (font * 0.32).max(3.0);
        let (a, b) = ([w as f64 * 0.25, h as f64 * 0.5], [w as f64 * 0.75, h as f64 * 0.5]);
        if std::env::var_os("MAHERE_LABEL_LOG").is_some() {
            eprintln!("labels: {} lines, {} areas, {} points gathered in {:.1} ms; {} placed, {} glyphs in the atlas, {:.1} ms in all", texels.lines.len(), texels.areas.len(), texels.points.len(), gathered.as_secs_f64() * 1e3, laid.len(), self.atlas.slots.len(), t.elapsed().as_secs_f64() * 1e3);
        }
        Lettering {
            kept,
            atlas: self.shared.clone(),
            reference: Some((a, b, job.cam.screen_to_geo(a[0], a[1], w, h), job.cam.screen_to_geo(b[0], b[1], w, h))),
            drawn_for: Some(job.key),
            ppd: job.cam.ppd,
            palette: job.palette,
            route: job.route,
            route_width,
            route_casing: route_width + (font * 0.14).max(1.5),
        }
    }

    /// The laid-out labels kept as anchors on the globe and letters about them, each letter's glyph in the atlas.
    fn keep(&mut self, laid: &[Laid], cam: &mahere_engine::Camera, w: usize, h: usize) -> Vec<Kept> {
        let mut kept = Vec::new();
        for l in laid {
            let n = l.glyphs.len().max(1) as f32;
            // A mark alone (an unnamed tap or gate) anchors at its mark.
            let centre = match (l.glyphs.is_empty(), l.mark) {
                (true, Some((p, _))) => p,
                _ => l.glyphs.iter().fold([0.0f32; 2], |c, g| [c[0] + g.at[0] / n, c[1] + g.at[1] / n]),
            };
            let anchor = cam.screen_to_geo(centre[0] as f64, centre[1] as f64, w, h);
            self.use_han(&l.name.name, anchor);
            let halo = (l.size * self.look[1] * 16.0) as u32;
            let letters = l.glyphs.iter().map(|g| (self.glyph(g.ch, l.size, l.weight, l.shear, halo, g.angle), [g.at[0] - centre[0], g.at[1] - centre[1]], g.angle)).collect();
            // A point's name stays level; a line's and an area's turn with the map.
            let turns = !l.name.is_point();
            kept.push(Kept { name: l.name.clone(), anchor, letters, turns, colour: l.colour, halo: halo > 0 });
            if let Some((p, kind)) = l.mark {
                // The kind's symbol, level, no halo.
                let (ch, colour, scale) = mark_of(kind);
                let key = self.glyph(ch, l.size * scale, 400, 0.0, 0, 0.0);
                kept.push(Kept { name: l.name.clone(), anchor: cam.screen_to_geo(p[0] as f64, p[1] as f64, w, h), letters: vec![(key, [0.0, 0.0], 0.0)], turns: false, colour, halo: false });
            }
        }
        kept
    }

    /// A glyph's place in the atlas, rasterised by fluor the first time it is asked for, its outline turned to the angle's bin: the letter's coverage, and around it the halo, a distance out from the letter's edge.
    fn glyph(&mut self, ch: char, size: f32, weight: u16, shear: f32, halo: u32, angle: f32) -> GlyphKey {
        let bin = (angle / bin_step(size)).round() as i32;
        let key = GlyphKey { ch, family: self.key_family(ch), size: (size * 16.0) as u32, weight, shear: shear.to_bits(), halo, bin };
        if self.atlas.slots.contains_key(&key) {
            return key;
        }
        let r = halo as f32 / 16.0;
        let adv = self.advance(ch, size, weight).max(size * 0.3);
        let turn = bin as f32 * bin_step(size);
        // Rasterised at the atlas's density: from here every length is atlas pixels.
        let (size, r, adv) = (size * ATLAS_DENSITY, r * ATLAS_DENSITY, adv * ATLAS_DENSITY);
        // A square that holds the glyph at any angle, its halo around it.
        // Even, so the glyph's centre is a pixel corner and every glyph of a label snaps to the grid alike (an odd one rounds half a pixel off its neighbours).
        let side = ((adv + size * 0.4).hypot(size * 1.6) + 2.0 * r + 4.0).ceil() as usize;
        let side = side + side % 2;
        let (cw, chh) = (side, side);
        let mut px = vec![0u32; cw * chh];
        {
            let mut damage = Damage::new();
            let mut canvas = Canvas::new(&mut px, cw, chh, &mut damage);
            let style = TextStyle::new(size, 0xFF00_0000).weight(weight).shear(shear).rotation(turn).font(self.face());
            let at = unturn([cw as f32 / 2.0, chh as f32 / 2.0], turn);
            self.text.draw_text_center(&mut canvas, &ch.to_string(), at[0], at[1], &style, None, None);
        }
        let cover: Vec<u8> = px.iter().map(|p| (p >> 24) as u8).collect();
        // The halo: the chamfer distance from the letter, feathered over a pixel, at least the letter itself.
        let mut d: Vec<f32> = cover.iter().map(|&c| if c >= 128 { 0.0 } else { f32::MAX }).collect();
        let (st, dg) = (1.0f32, std::f32::consts::SQRT_2);
        for y in 0..chh {
            for x in 0..cw {
                let i = y * cw + x;
                let mut v = d[i];
                if x > 0 { v = v.min(d[i - 1] + st); }
                if y > 0 { v = v.min(d[i - cw] + st); }
                if x > 0 && y > 0 { v = v.min(d[i - cw - 1] + dg); }
                if x + 1 < cw && y > 0 { v = v.min(d[i - cw + 1] + dg); }
                d[i] = v;
            }
        }
        for y in (0..chh).rev() {
            for x in (0..cw).rev() {
                let i = y * cw + x;
                let mut v = d[i];
                if x + 1 < cw { v = v.min(d[i + 1] + st); }
                if y + 1 < chh { v = v.min(d[i + cw] + st); }
                if x + 1 < cw && y + 1 < chh { v = v.min(d[i + cw + 1] + dg); }
                if x > 0 && y + 1 < chh { v = v.min(d[i + cw - 1] + dg); }
                d[i] = v;
            }
        }
        // Trimmed to what the letter and its halo cover, its centre kept: less atlas, smaller quads.
        let used = |x: usize, y: usize| cover[y * cw + x] > 0 || (r > 0.0 && d[y * cw + x] < r + 0.5);
        let (mut tx0, mut ty0, mut tx1, mut ty1) = (cw, chh, 0, 0);
        for y in 0..chh {
            for x in 0..cw {
                if used(x, y) {
                    tx0 = tx0.min(x);
                    ty0 = ty0.min(y);
                    tx1 = tx1.max(x + 1);
                    ty1 = ty1.max(y + 1);
                }
            }
        }
        if tx0 >= tx1 {
            // A space: nothing to draw, a one-pixel slot.
            (tx0, ty0, tx1, ty1) = (cw / 2, chh / 2, cw / 2 + 1, chh / 2 + 1);
        }
        // Symmetric about the centre, so the quad's centre is still the glyph's.
        let (mx, my) = ((cw / 2 - tx0.min(cw / 2)).max(tx1.max(cw / 2) - cw / 2), (chh / 2 - ty0.min(chh / 2)).max(ty1.max(chh / 2) - chh / 2));
        let (ox, oy) = (cw / 2 - mx, chh / 2 - my);
        let (cw2, ch2) = ((2 * mx).min(cw - ox), (2 * my).min(chh - oy));
        let place = match self.atlas.place(cw2, ch2) {
            Some(p) => p,
            None => {
                // Full: a bigger texture up to 4096, else start again (the labels in view ask for theirs again as they are kept).
                let side = (self.atlas.side * 2).min(4096);
                self.atlas.clear(side);
                self.atlas.place(cw2, ch2).unwrap_or((0, 0))
            }
        };
        let side = self.atlas.side;
        for y in 0..ch2.min(side - place.1) {
            for x in 0..cw2.min(side - place.0) {
                let i = (oy + y) * cw + ox + x;
                let o = ((place.1 + y) * side + place.0 + x) * 2;
                let c = cover[i];
                let hv = if r > 0.0 { ((r + 0.5 - d[i]).clamp(0.0, 1.0) * 255.0) as u8 } else { 0 };
                self.atlas.rg[o] = c;
                self.atlas.rg[o + 1] = hv.max(if r > 0.0 { c } else { 0 });
            }
        }
        self.atlas.slots.insert(key, [place.0 as u16, place.1 as u16, cw2 as u16, ch2 as u16]);
        self.atlas.version += 1;
        key
    }

    fn layout(&mut self, texels: &NamedTexels, cam: &mahere_engine::Camera, w: usize, h: usize, font: f32, previous: &HashSet<(u8, String, String)>) -> Vec<Laid> {
        enum Shape<'a> {
            Point([f32; 2]),
            Area(&'a [[f32; 2]]),
            Line(&'a [[f32; 2]]),
        }
        // The density's cut in the order of kinds: below it nothing is lettered.
        let limit = mahere_engine::theme::label_tier_limit(self.amount[0]);
        let mut todo: Vec<(u8, bool, u32, usize, &Named, Shape)> = Vec::new();
        for (n, p) in &texels.points {
            if let Some(t) = tier(n).filter(|&t| t <= limit) {
                todo.push((t, !previous.contains(&(n.kind, n.name.clone(), n.reference.clone())), n.rank, 1, n, Shape::Point(*p)));
            }
        }
        for (n, pts) in &texels.areas {
            if let Some(t) = tier(n).filter(|&t| t <= limit) {
                todo.push((t, !previous.contains(&(n.kind, n.name.clone(), n.reference.clone())), n.rank, pts.len(), n, Shape::Area(pts)));
            }
        }
        for (n, pts) in &texels.lines {
            if let Some(t) = tier(n).filter(|&t| t <= limit) {
                todo.push((t, !previous.contains(&(n.kind, n.name.clone(), n.reference.clone())), n.rank, pts.len(), n, Shape::Line(pts)));
            }
        }
        todo.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)).then(b.2.cmp(&a.2)).then(b.3.cmp(&a.3)));
        // Each ranked kind's span in view, log of rank.
        let mut ranks = [(f32::MAX, f32::MIN); 4];
        for (_, _, _, _, n, _) in &todo {
            if let Some(g) = rank_group(n) {
                let r = (n.rank as f32 + 1.0).ln();
                let e = &mut ranks[g as usize];
                *e = (e.0.min(r), e.1.max(r));
            }
        }
        let mut room = Room::new(w, h);
        let mut out = Vec::new();
        let pad = font * self.look[3];
        for (_, _, _, _, n, shape) in todo {
            let label = n.label();
            if label.is_empty() {
                continue;
            }
            let (rel, colour, weight, shear) = style_of(n);
            // A ranked kind's size and weight by where it stands among its kind in view (Nick 2026-10-14: the biggest town on screen the boldest, the smallest the lightest, on a log scale): five steps.
            let (rel, weight) = match rank_group(n).map(|g| (g, ranks[g as usize])) {
                Some((_, (lo, hi))) => {
                    let t = if hi > lo { ((n.rank as f32 + 1.0).ln() - lo) / (hi - lo) } else { 1.0 };
                    let step = (t.clamp(0.0, 1.0) * 4.0).round() as usize;
                    (ranked_base(n) * RANK_SIZE[step], RANK_WEIGHT[step])
                }
                None => (rel, weight),
            };
            let size = font * rel;
            // Measured in the face its letters will be drawn in.
            let at = match &shape {
                Shape::Point(p) => *p,
                Shape::Area(pts) | Shape::Line(pts) => pts[pts.len() / 2],
            };
            self.use_han(&n.name, cam.screen_to_geo(at[0] as f64, at[1] as f64, w, h));
            let laid: Vec<Laid> = match shape {
                Shape::Point(p) => self.point(n, &label, p, size, colour, weight, &room, pad).into_iter().collect(),
                Shape::Area(pts) => self.area(n, &label, pts, texels.area_step_px, size, colour, weight, shear, &room, pad).into_iter().collect(),
                Shape::Line(pts) => self.line(n, &label, pts, texels.texel_px, size, colour, weight, shear, &mut room, pad),
            };
            for l in laid {
                let mut boxes: Vec<[f32; 4]> = l.glyphs.iter().map(|g| glyph_box(g, l.size, pad)).collect();
                // The mark holds its ground too, so marks never pile on one another.
                if let Some((p, _)) = l.mark {
                    boxes.push([p[0] - l.size * 0.4, p[1] - l.size * 0.4, p[0] + l.size * 0.4, p[1] + l.size * 0.4]);
                }
                room.take(&boxes);
                out.push(l);
            }
        }
        out
    }

    /// A point's label: beside its mark, right of it first, then left, above and below; a town's name centred on it.
    #[allow(clippy::too_many_arguments)]
    fn point(&mut self, n: &Named, label: &str, p: [f32; 2], size: f32, colour: usize, weight: u16, room: &Room, pad: f32) -> Option<Laid> {
        let place = centred(n.kind);
        // An unnamed tap, gate or lookout is its mark alone (its label is only its kind's word, for the tap card); an unnamed toilet says WC.
        let unnamed = n.name.is_empty() && n.kind >= DRINKING_WATER;
        if unnamed && n.kind != TOILETS {
            let mark_box = [p[0] - size * 0.4, p[1] - size * 0.4, p[0] + size * 0.4, p[1] + size * 0.4];
            return room.free(&[mark_box]).then(|| Laid { name: n.clone(), glyphs: Vec::new(), size, colour, weight, shear: 0.0, mark: Some((p, n.kind)) });
        }
        let label = if unnamed { "WC" } else { label };
        let place = place || unnamed;
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
                let mark = (!place).then_some((p, n.kind));
                return Some(Laid { name: n.clone(), glyphs, size, colour, weight, shear: 0.0, mark });
            }
        }
        None
    }

    /// An area's label: at the point deepest inside it (a distance fill over its samples), along its long axis when it is long, as large as fits inside it up to its kind's size.
    #[allow(clippy::too_many_arguments)]
    fn area(&mut self, n: &Named, label: &str, pts: &[[f32; 2]], step: f32, size: f32, colour: usize, weight: u16, shear: f32, room: &Room, pad: f32) -> Option<Laid> {
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
        let at = |x: f32, y: f32| -> bool {
            let (cx, cy) = ((x / step).round() as i32 - x0 + 1, (y / step).round() as i32 - y0 + 1);
            cx >= 0 && cy >= 0 && (cx as usize) < gw && (cy as usize) < gh && inside[cy as usize * gw + cx as usize]
        };
        // Along the shape's spine first (Nick 2026-10-14: fill in from the shore until a line is left, the spine of the lake, and letter along it); straight through the deepest point when the shape has no spine long enough.
        let room_px = depth * step;
        let max = (size * (1.0 + (room_px / (size * 6.0)).min(1.0))).max(size);
        let min = size * 0.8;
        if let Some(l) = self.on_spine(n, label, &d, gw, gh, (x0, y0), step, depth, (max, min, size), colour, weight, shear, room, pad, &at) {
            return Some(l);
        }
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
        // Largest first, from the kind's size scaled by how much room there is, down to the smallest readable.
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

    /// An area's name along its spine: the ridge of the distance to the shore (what is left after filling in from the outside until a line remains), its longest path, smoothed at the letters' own scale, the name centred on it (or a little either side), every letter inside the shape and the bends gentle; None when the shape has no spine long enough, so the name goes straight instead.
    #[allow(clippy::too_many_arguments)]
    fn on_spine(&mut self, n: &Named, label: &str, d: &[f32], gw: usize, gh: usize, origin: (i32, i32), step: f32, depth: f32, sizes: (f32, f32, f32), colour: usize, weight: u16, shear: f32, room: &Room, pad: f32, inside: &dyn Fn(f32, f32) -> bool) -> Option<Laid> {
        let (max, min, size) = sizes;
        if depth < 2.0 {
            return None;
        }
        // The ridge: cells no nearer the shore than their neighbours on either side in some direction, well inside (the shallow ridges of a shoreline's bumps left out).
        let mut ridge = Vec::new();
        for y in 1..gh - 1 {
            for x in 1..gw - 1 {
                let i = y * gw + x;
                let v = d[i];
                if v < depth * 0.4 {
                    continue;
                }
                let crest = (v >= d[i - 1] && v >= d[i + 1]) || (v >= d[i - gw] && v >= d[i + gw]) || (v >= d[i - gw - 1] && v >= d[i + gw + 1]) || (v >= d[i - gw + 1] && v >= d[i + gw - 1]);
                if crest {
                    ridge.push([(x as i32 + origin.0 - 1) as f32 * step, (y as i32 + origin.1 - 1) as f32 * step]);
                }
            }
        }
        let spine = centrelines(&ridge, step * 1.6).into_iter().next()?;
        let mut sz = max;
        while sz >= min {
            // A broad curve: bends under a few letters' size are the shoreline's, not the lake's.
            let path = smooth(&spine, 2.0, sz * 2.0);
            let len = lengths(&path);
            let total = *len.last().unwrap_or(&0.0);
            let advances: Vec<(char, f32)> = label.chars().map(|c| (c, self.advance(c, sz, weight))).collect();
            let width: f32 = advances.iter().map(|a| a.1).sum();
            if total >= width * 1.05 {
                for shift in [0.0f32, -0.12, 0.12, -0.25, 0.25] {
                    let s = (total / 2.0 - width / 2.0 + shift * total).clamp(0.0, total - width);
                    let mut glyphs: Vec<Glyph> = Vec::new();
                    let (mut at, mut bends, mut worst) = (s, 0.0f32, 0.0f32);
                    for &(ch, a) in &advances {
                        let g = letter(&path, &len, ch, at, at + a);
                        if let Some(last) = glyphs.last() {
                            let t = turn(last.angle, g.angle);
                            bends += t;
                            worst = worst.max(t);
                        }
                        glyphs.push(g);
                        at += a;
                    }
                    if worst > 0.35 || bends > 0.9 {
                        continue;
                    }
                    // Top up, as a line's name.
                    let (first, last) = (glyphs.first().unwrap().at, glyphs.last().unwrap().at);
                    let (dx, dy) = (last[0] - first[0], last[1] - first[1]);
                    let steep = dx.abs() <= dx.hypot(dy) * 0.05;
                    if (!steep && dx < 0.0) || (steep && dy > 0.0) {
                        let mut at = s + width;
                        glyphs.clear();
                        for &(ch, a) in &advances {
                            glyphs.push(letter(&path, &len, ch, at, at - a));
                            at -= a;
                        }
                    }
                    // Every letter inside the shape, its top and foot too.
                    let within = glyphs.iter().zip(&advances).all(|(g, &(_, a))| {
                        let (c, s_) = (g.angle.cos(), g.angle.sin());
                        [(-0.5f32, -0.4f32), (0.5, -0.4), (-0.5, 0.4), (0.5, 0.4), (0.0, 0.0)].iter().all(|&(u, v)| {
                            let (ux, vy) = (u * a, v * sz);
                            inside(g.at[0] + ux * c - vy * s_, g.at[1] + ux * s_ + vy * c)
                        })
                    });
                    if !within {
                        continue;
                    }
                    let boxes: Vec<[f32; 4]> = glyphs.iter().map(|g| glyph_box(g, sz, pad)).collect();
                    if room.free(&boxes) {
                        return Some(Laid { name: n.clone(), glyphs, size: sz, colour, weight, shear, mark: None });
                    }
                }
            }
            sz -= (size * 0.08).max(0.5);
        }
        None
    }

    /// A line's labels: on each long enough run, the stretch straight enough for the name nearest the run's middle, and on a long run more of them at least a screen's third apart, each letter turned to the line, top up.
    #[allow(clippy::too_many_arguments)]
    fn line(&mut self, n: &Named, label: &str, pts: &[[f32; 2]], texel_px: f32, size: f32, colour: usize, weight: u16, shear: f32, room: &mut Room, pad: f32) -> Vec<Laid> {
        let advances: Vec<(char, f32)> = label.chars().map(|c| (c, self.advance(c, size, weight))).collect();
        let width: f32 = advances.iter().map(|a| a.1).sum();
        let bin = (texel_px * 3.0).max(6.0);
        let mut out = Vec::new();
        let repeat = (self.w.max(self.h) as f32 * self.amount[1]).max(width * 1.5);
        for path in centrelines(pts, bin).into_iter().take(4) {
            // The curve fit at the label's own scale: wiggles under half a letter cannot twist the name.
            let path = smooth(&path, 2.0, (size * self.look[2]).max(0.5));
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
                    let g = letter(&path, &len, ch, at, at + a);
                    if let Some(last) = glyphs.last() {
                        let t = turn(last.angle, g.angle);
                        bends += t;
                        worst = worst.max(t);
                    }
                    glyphs.push(g);
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
                        glyphs.push(letter(&path, &len, ch, at, at - a));
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
}

impl Lettering {
    /// How far the map has turned on the screen since the layout, radians.
    fn turned(&self, cam: &mahere_engine::Camera, w: usize, h: usize) -> f32 {
        let Some((a, b, ga, gb)) = self.reference else { return 0.0 };
        let (ax, ay) = cam.geo_to_screen(ga.0, ga.1, w, h);
        let (bx, by) = cam.geo_to_screen(gb.0, gb.1, w, h);
        ((by - ay).atan2(bx - ax) - (b[1] - a[1]).atan2(b[0] - a[0])) as f32
    }

    /// Whether the map is still near the scale this lettering was laid out at: within twice either way it is drawn as it is while the next layout comes; past that it is a different view's lettering, crowded or sparse, and is not drawn (Nick 2026-10-18: a layout made zoomed in over Siberia stayed on the globe, piled up in one corner, while the globe's own was still coming).
    fn near_scale(&self, cam: &mahere_engine::Camera) -> bool {
        self.ppd <= 0.0 || (0.5..=2.0).contains(&(cam.ppd / self.ppd))
    }

    /// The frame's letters, each a quad at its label's anchor as the camera has it now: a turning label's letters turned with the map about the anchor, a level one's as laid out.
    pub fn glyphs(&self, cam: &mahere_engine::Camera, w: usize, h: usize) -> Vec<GlyphQuad> {
        if !self.near_scale(cam) {
            return Vec::new();
        }
        let phi = self.turned(cam, w, h);
        let (sp, cp) = phi.sin_cos();
        let halo = { let h = stored(self.palette[HALO]); [h[0], h[1], h[2], h[3] * HALO_STRENGTH] };
        let side = self.atlas.side as f32;
        let mut out = Vec::new();
        for k in &self.kept {
            let (ax, ay) = cam.geo_to_screen(k.anchor.0, k.anchor.1, w, h);
            let (ax, ay) = (ax as f32, ay as f32);
            if ax < -(w as f32) || ay < -(h as f32) || ax > 2.0 * w as f32 || ay > 2.0 * h as f32 || !cam.on_globe(ax as f64, ay as f64, w, h) {
                continue;
            }
            let text = stored(self.palette[k.colour]);
            let halo = if k.halo { halo } else { [0.0; 4] };
            for (key, off, angle) in &k.letters {
                let Some(slot) = self.atlas.slots.get(key) else { continue };
                let (o, a) = if k.turns { ([off[0] * cp - off[1] * sp, off[0] * sp + off[1] * cp], angle + phi) } else { (*off, *angle) };
                // The quad turns only by what the glyph's bin left over; with nothing left (at rest) it sits on the pixel grid, texel for pixel.
                let residual = a - key.bin as f32 * bin_step(key.size as f32 / 16.0);
                let (sx, sy, sw, sh) = (slot[0] as f32, slot[1] as f32, slot[2] as f32, slot[3] as f32);
                let (mut cx, mut cy) = (ax + o[0], ay + o[1]);
                // At rest the glyph sits on the atlas's own grid (a third of a screen pixel), texel for pixel in the 3× layer.
                let k = ATLAS_DENSITY;
                if residual.abs() < 1e-3 {
                    cx = ((cx * k - sw / 2.0).round() + sw / 2.0) / k;
                    cy = ((cy * k - sh / 2.0).round() + sh / 2.0) / k;
                }
                out.push([cx, cy, sw / (2.0 * k), sh / (2.0 * k), residual, 0.0, 0.0, 0.0, sx / side, sy / side, (sx + sw) / side, (sy + sh) / side, text[0], text[1], text[2], text[3], halo[0], halo[1], halo[2], halo[3]]);
            }
        }
        out
    }

    /// The frame's route: each stretch between its points on the screen now, the casing's stretches first (under), then the line's.
    pub fn segments(&self, cam: &mahere_engine::Camera, w: usize, h: usize) -> Vec<Segment> {
        if self.route.len() < 2 {
            return Vec::new();
        }
        let pts: Vec<(f32, f32)> = self.route.iter().map(|&(la, lo)| {
            let (x, y) = cam.geo_to_screen(la, lo, w, h);
            (x as f32, y as f32)
        }).collect();
        let mut out = Vec::with_capacity(2 * pts.len());
        for (width, colour) in [(self.route_casing, ROUTE_CASING), (self.route_width, ROUTE)] {
            let c = display(self.palette[colour]);
            for s in pts.windows(2) {
                out.push([s[0].0, s[0].1, s[1].0, s[1].1, width / 2.0, 0.0, 0.0, 0.0, c[0], c[1], c[2], c[3]]);
            }
        }
        out
    }

    /// For a CPU frontend: the frame's route and letters drawn straight into a display 0xRRGGBB canvas.
    pub fn draw_cpu(&self, canvas: &mut [u32], cam: &mahere_engine::Camera, w: usize, h: usize) {
        let blend = |canvas: &mut [u32], i: usize, c: [f32; 3], a: f32| {
            if a <= 0.0 {
                return;
            }
            let p = canvas[i];
            let mix = |shift: u32, v: f32| -> u32 { ((((p >> shift) & 255) as f32 * (1.0 - a) + v * 255.0 * a).round() as u32).min(255) };
            canvas[i] = (mix(16, c[0]) << 16) | (mix(8, c[1]) << 8) | mix(0, c[2]);
        };
        for s in self.segments(cam, w, h) {
            let (a, b, half) = ([s[0], s[1]], [s[2], s[3]], s[4]);
            let (x0, x1) = ((a[0].min(b[0]) - half - 1.0).floor().max(0.0) as usize, ((a[0].max(b[0]) + half + 1.0).ceil().max(0.0) as usize).min(w));
            let (y0, y1) = ((a[1].min(b[1]) - half - 1.0).floor().max(0.0) as usize, ((a[1].max(b[1]) + half + 1.0).ceil().max(0.0) as usize).min(h));
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let len2 = (dx * dx + dy * dy).max(1e-6);
            for y in y0..y1 {
                for x in x0..x1 {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let t = (((px - a[0]) * dx + (py - a[1]) * dy) / len2).clamp(0.0, 1.0);
                    let d = (px - (a[0] + dx * t)).hypot(py - (a[1] + dy * t));
                    blend(canvas, y * w + x, [s[8], s[9], s[10]], (half - d + 0.5).clamp(0.0, 1.0) * s[11]);
                }
            }
        }
        let side = self.atlas.side;
        let sample = |u: f32, v: f32| -> (f32, f32) {
            let (fx, fy) = (u * side as f32 - 0.5, v * side as f32 - 0.5);
            let (x0, y0) = (fx.floor(), fy.floor());
            let (tx, ty) = (fx - x0, fy - y0);
            let at = |x: f32, y: f32| -> (f32, f32) {
                if x < 0.0 || y < 0.0 || x >= side as f32 || y >= side as f32 {
                    return (0.0, 0.0);
                }
                let o = (y as usize * side + x as usize) * 2;
                (self.atlas.rg[o] as f32 / 255.0, self.atlas.rg[o + 1] as f32 / 255.0)
            };
            let (a, b, c, d) = (at(x0, y0), at(x0 + 1.0, y0), at(x0, y0 + 1.0), at(x0 + 1.0, y0 + 1.0));
            let l = |p: f32, q: f32, t: f32| p + (q - p) * t;
            (l(l(a.0, b.0, tx), l(c.0, d.0, tx), ty), l(l(a.1, b.1, tx), l(c.1, d.1, tx), ty))
        };
        for g in self.glyphs(cam, w, h) {
            let (cx, cy, hx, hy, ang) = (g[0], g[1], g[2], g[3], g[4]);
            // The glyph carries its colours as stored; the CPU's canvas takes them as the panel's own inks are made.
            let ink = |c: &[f32]| { let d = display([(c[0] * 255.0).round() as u8, (c[1] * 255.0).round() as u8, (c[2] * 255.0).round() as u8, 255]); [d[0], d[1], d[2]] };
            let (text_rgb, halo_rgb) = (ink(&g[12..15]), ink(&g[16..19]));
            let (sa, ca) = ang.sin_cos();
            let reach = hx.hypot(hy) + 1.0;
            let (x0, x1) = ((cx - reach).floor().max(0.0) as usize, ((cx + reach).ceil().max(0.0) as usize).min(w));
            let (y0, y1) = ((cy - reach).floor().max(0.0) as usize, ((cy + reach).ceil().max(0.0) as usize).min(h));
            for y in y0..y1 {
                for x in x0..x1 {
                    // Back into the glyph's own frame: unturned, then where in its atlas rectangle.
                    let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                    let (lx, ly) = (dx * ca + dy * sa, -dx * sa + dy * ca);
                    if lx.abs() > hx || ly.abs() > hy {
                        continue;
                    }
                    let (u, v) = (g[8] + (lx / hx * 0.5 + 0.5) * (g[10] - g[8]), g[9] + (ly / hy * 0.5 + 0.5) * (g[11] - g[9]));
                    // A screen pixel spans the atlas's density in texels each way: their box average, as the GPU's 1× frame takes it.
                    let n = ATLAS_DENSITY as i32;
                    let (mut ta, mut ha) = (0.0, 0.0);
                    for j in 0..n {
                        for i in 0..n {
                            let du = ((i as f32 + 0.5) / n as f32 - 0.5) * ATLAS_DENSITY / side as f32;
                            let dv = ((j as f32 + 0.5) / n as f32 - 0.5) * ATLAS_DENSITY / side as f32;
                            let (t, hh) = sample(u + du, v + dv);
                            ta += t;
                            ha += hh;
                        }
                    }
                    let (ta, ha) = (ta / (n * n) as f32, ha / (n * n) as f32);
                    let i = y * w + x;
                    blend(canvas, i, halo_rgb, ha * g[19]);
                    blend(canvas, i, text_rgb, ta * g[15]);
                }
            }
        }
    }

    /// The label under a screen point, as the screen is now.
    pub fn hit(&self, cam: &mahere_engine::Camera, w: usize, h: usize, x: f32, y: f32) -> Option<&Named> {
        if !self.near_scale(cam) {
            return None;
        }
        let phi = self.turned(cam, w, h);
        let (sp, cp) = phi.sin_cos();
        self.kept.iter().find(|k| {
            let (ax, ay) = cam.geo_to_screen(k.anchor.0, k.anchor.1, w, h);
            k.letters.iter().any(|(key, off, _)| {
                let o = if k.turns { [off[0] * cp - off[1] * sp, off[0] * sp + off[1] * cp] } else { *off };
                let r = self.atlas.slots.get(key).map_or(8.0, |s| s[3] as f32 / (2.0 * ATLAS_DENSITY));
                (x - (ax as f32 + o[0])).hypot(y - (ay as f32 + o[1])) <= r
            })
        }).map(|k| &k.name)
    }
}
