//! The on-map controls (Nick 2026-10-06, after Lumis): one gear button in the corner shows and hides everything else, so the map can have the whole screen with only the gear showing. Open, a column holds the layer toggles and the readouts — position, world code, elevation, heading, scale, frame time.
//!
//! Painted with fluor's widgets and text into the panel's own buffer in fluor's pixel convention (α + darkness, topmost first), then composited by whichever renderer owns the screen: the GPU takes it as a premultiplied RGBA overlay over the engine's marks, the CPU frontend lays it over its canvas. Taps are routed here first; a tap the panel does not take is the map's.

use fluor::canvas::{Canvas, Damage};
use fluor::paint::{self, HitId, HIT_NONE};
use fluor::text::{TextRenderer, TextStyle};
use fluor::theme;
use fluor::widgets::{Checkbox, Slider};
use mahere_engine::theme::{Field, Theme};
use mahere_engine::{LayerMask, MeasureView};

/// The two modes the panel switches besides the layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Controls {
    pub real_sun: bool,
    /// The front camera lights the map.
    pub real_light: bool,
    pub follow_heading: bool,
    /// Measure from the fix rather than the screen centre.
    pub lock_to_fix: bool,
    /// Bytes the cell cache may hold; the slider sets it.
    pub cache_budget: u64,
    /// The theme, an index into the engine's themes.
    pub theme: usize,
    /// Highlights compressed into white (exposure 2/3 into Opsin's rail); off, linear: the stored range straight, 2.3 stops darker. The row shows it the other way up, as "Linear highlights".
    pub compressed: bool,
}

/// What the readouts show.
#[derive(Clone, Copy, Debug, Default)]
pub struct Readouts {
    pub lat: f64,
    pub lon: f64,
    pub elev: Option<f32>,
    /// Degrees the screen's up is turned from true north, clockwise positive, -180..180.
    pub heading_deg: f64,
    pub m_per_px: f64,
    pub frame_ms: f32,
    pub resident: usize,
    /// The phone's true heading from its orientation sensor, degrees clockwise from north, when it has one.
    pub phone_heading: Option<f32>,
    /// Bytes the cell cache holds now, and the most the slider may allow (what is cached plus the free space).
    pub cache_used: u64,
    pub cache_max: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layer {
    RealSun,
    RealLight,
    Linear,
    FollowHeading,
    LockToFix,
    Dem,
    Hypso,
    Boundaries,
    Land,
    Water,
    Line,
    Contours,
    Slope,
    Infrared,
    Imagery,
    Debug,
}

const LAYERS: [(Layer, &str); 16] = [
    (Layer::Dem, "Terrain"),
    (Layer::Hypso, "Elevation tint"),
    (Layer::Land, "Land cover"),
    (Layer::Water, "Water"),
    (Layer::Line, "Lines"),
    (Layer::Boundaries, "Boundaries"),
    (Layer::Contours, "Contours"),
    (Layer::Slope, "Slope bands"),
    (Layer::Imagery, "Imagery"),
    (Layer::Infrared, "Infrared"),
    (Layer::Debug, "Residency"),
    (Layer::RealSun, "Real sun"),
    (Layer::RealLight, "Real light"),
    (Layer::Linear, "Linear highlights"),
    (Layer::FollowHeading, "Follow heading"),
    (Layer::LockToFix, "Measure from me"),
];

/// A row that takes a tap but changes nothing: a layer another layer has greyed, or the one of real sun and real light the other has taken (the sun is either the almanac's or the camera's).
fn inert(mask: &LayerMask, ctl: &Controls, l: Layer) -> bool {
    match l {
        Layer::RealSun => ctl.real_light,
        Layer::RealLight => ctl.real_sun,
        Layer::Linear | Layer::FollowHeading | Layer::LockToFix => false,
        _ => get(&mask.inert(), &Controls::default(), l),
    }
}

fn get(mask: &LayerMask, ctl: &Controls, l: Layer) -> bool {
    match l {
        Layer::RealSun => ctl.real_sun,
        Layer::RealLight => ctl.real_light,
        Layer::Linear => !ctl.compressed,
        Layer::FollowHeading => ctl.follow_heading,
        Layer::LockToFix => ctl.lock_to_fix,
        Layer::Dem => mask.dem,
        Layer::Hypso => mask.hypso,
        Layer::Land => mask.land,
        Layer::Water => mask.water,
        Layer::Line => mask.line,
        Layer::Boundaries => mask.boundaries,
        Layer::Contours => mask.contours,
        Layer::Slope => mask.slope,
        Layer::Infrared => mask.infrared,
        Layer::Imagery => mask.imagery,
        Layer::Debug => mask.debug,
    }
}

fn set(mask: &mut LayerMask, ctl: &mut Controls, l: Layer, v: bool) {
    match l {
        Layer::RealSun => ctl.real_sun = v,
        Layer::RealLight => ctl.real_light = v,
        Layer::Linear => ctl.compressed = !v,
        Layer::FollowHeading => ctl.follow_heading = v,
        Layer::LockToFix => ctl.lock_to_fix = v,
        Layer::Dem => mask.dem = v,
        Layer::Hypso => mask.hypso = v,
        Layer::Land => mask.land = v,
        Layer::Water => mask.water = v,
        Layer::Line => mask.line = v,
        Layer::Boundaries => mask.boundaries = v,
        Layer::Contours => mask.contours = v,
        Layer::Slope => mask.slope = v,
        Layer::Infrared => mask.infrared = v,
        Layer::Imagery => mask.imagery = v,
        Layer::Debug => mask.debug = v,
    }
}

/// A colour authored as visible 0xRRGGBB with an opacity, in fluor's stored form on this platform.
const fn ink(rgb: u32, alpha: u8) -> u32 {
    (theme::dark(theme::fmt(theme::vsf(rgb))) & 0x00FF_FFFF) | ((alpha as u32) << 24)
}

/// Fluor's production zoom bounds, 12.5% to 300%.
pub const RU_MIN: f32 = 0.125;
pub const RU_MAX: f32 = 3.0;

/// Harmonic mean, the smooth blend of two size candidates (no kink where they cross); zero if either is.
fn hm(a: f32, b: f32) -> f32 {
    let sum = a + b;
    if sum <= 0.0 { 0.0 } else { 2.0 * a * b / sum }
}

const PANEL_BG: u32 = ink(0x0E_12_1A, 222);
const GEAR_BG: u32 = ink(0x14_18_22, 230);
const PANEL_DIM: u32 = ink(0x0E_12_1A, 170);
const STRIP_FILL: u32 = ink(0x3A_5A_7A, 200);
const SLIDER_TRACK: u32 = ink(0x2A_30_3C, 255);
const SLIDER_FILL: u32 = ink(0x1A_22_4E, 255);
const SLIDER_USED: u32 = ink(0x40_9C_FF, 255);
const SLIDER_KNOB: u32 = ink(0xE0_E0_DC, 255);
const STRIP_EDGE: u32 = ink(0xFF_C4_40, 255);
const READOUT: u32 = ink(0xC8_CC_D4, 255);
const READOUT_DIM: u32 = ink(0x80_86_92, 255);

/// The pages, stacked: opening one slides the page under it left into a rail and opens the new one beside it (Nick 2026-10-08). The rail takes a tap back.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Layers,
    Themes,
    /// The legend as the editor: every colour and number of the current theme a row.
    Edit,
    /// One field's sliders.
    Field(Field),
}

/// What the editor asks of the host, polled after every tap and drag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThemeEdit {
    /// A field changed; apply it live.
    Set(Field, [f32; 4]),
    /// The editor closed: save the current theme.
    Done,
    /// A copy of the current theme under the next number, selected.
    Duplicate,
    /// A built-in back to what shipped.
    Reset,
    /// The user's own theme removed.
    Delete,
}

/// A row of the editor: a field to open, or an action.
#[derive(Clone, Copy)]
enum Row {
    Field(Field, [f32; 4]),
    Act(ThemeEdit),
}

/// The rail's share of the column when a page is stacked over another.
const RAIL: f32 = 0.38;

pub struct Panel {
    open: bool,
    dirty: bool,
    /// The UI's scale, fluor's RU multiplier: 1 is the default size, pinched while the panel is open and kept in the settings.
    ru: f32,
    stack: Vec<Page>,
    /// The theme page's rows: the back row, then one per theme, then the edit row, as (centre y, half height) in screen pixels.
    theme_rows: Vec<(f32, f32)>,
    /// The editor's rows: (centre y, half height, what the row is).
    edit_rows: Vec<(f32, f32, Row)>,
    /// How far the editor's list is scrolled, and a press on it: where it started, the row under it, and whether it has moved (a scroll, not a choice).
    edit_scroll: f32,
    /// How far the editor's list may scroll: its height past the screen, from the last paint.
    edit_extent: f32,
    press: Option<(f32, f32, Option<Row>, bool)>,
    /// The open field's sliders and the values they hold, and which slider a finger is on.
    sliders: Vec<Slider>,
    field_vals: [f32; 4],
    held: Option<usize>,
    edits: Vec<ThemeEdit>,
    hits: HitId,
    checks: Vec<(Layer, Checkbox)>,
    text: TextRenderer,
    buf: Vec<u32>,
    w: usize,
    h: usize,
    font: f32,
    gear: (f32, f32, f32),
    panel_w: f32,
    /// The theme row above the layers: x0, centre y, half height.
    theme_row: (f32, f32, f32),
    /// The cache slider's track on screen (x0, y, width), and whether a press is riding it.
    slider: (f32, f32, f32),
    slider_held: bool,
    cache_max: u64,
    /// The ruler strip's close button (centre and radius) while a measurement is shown, and whether a tap just hit it.
    strip_close: Option<(f32, f32, f32)>,
    clear_measure: bool,
}

impl Default for Panel {
    fn default() -> Self {
        Self::new()
    }
}

impl Panel {
    pub fn new() -> Panel {
        let mut hits: HitId = HIT_NONE;
        let checks = LAYERS.iter().map(|&(l, label)| (l, Checkbox::new(&mut hits, label, 0.0, 0.0, 10.0, 10.0, 10.0, false))).collect();
        Panel { open: false, dirty: true, ru: 1.0, stack: vec![Page::Layers], theme_rows: Vec::new(), edit_rows: Vec::new(), edit_scroll: 0.0, edit_extent: 0.0, press: None, sliders: Vec::new(), field_vals: [0.0; 4], held: None, edits: Vec::new(), hits, checks, text: TextRenderer::new(), buf: Vec::new(), w: 0, h: 0, font: 16.0, gear: (0.0, 0.0, 0.0), panel_w: 0.0, theme_row: (0.0, 0.0, 0.0), slider: (0.0, 0.0, 0.0), slider_held: false, cache_max: CACHE_MIN + 1, strip_close: None, clear_measure: false }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn set_open(&mut self, open: bool) {
        self.open = open;
        self.dirty = true;
    }

    /// Open on the theme page.
    pub fn show_themes(&mut self) {
        self.open = true;
        self.stack = vec![Page::Layers, Page::Themes];
        self.dirty = true;
    }

    /// What the editor changed since last asked.
    pub fn take_theme_edits(&mut self) -> Vec<ThemeEdit> {
        std::mem::take(&mut self.edits)
    }

    fn page(&self) -> Page {
        *self.stack.last().unwrap_or(&Page::Layers)
    }

    /// Back one page; leaving the editor asks the host to save.
    fn pop(&mut self) {
        if let Some(Page::Edit) = self.stack.pop() {
            self.edits.push(ThemeEdit::Done);
        }
        if self.stack.is_empty() {
            self.stack.push(Page::Layers);
        }
        self.sliders.clear();
        self.held = None;
        self.dirty = true;
    }

    /// The rail's width when a page is stacked, else zero: the content starts after it.
    fn rail_w(&self) -> f32 {
        if self.stack.len() >= 2 { self.panel_w * RAIL } else { 0.0 }
    }

    /// Open a field's page: one slider per value, from the theme's own bytes.
    fn open_field(&mut self, f: Field, vals: [f32; 4]) {
        self.field_vals = vals;
        let n = match f {
            Field::Hypso(_) => 4,
            Field::ContourAlpha(_) => 1,
            _ => 3,
        };
        self.sliders = (0..n).map(|i| Slider::new(&mut self.hits, 0.0, 0.0, 10.0, 10.0, Self::slider_pos(f, i, self.field_vals[i]))).collect();
        self.stack.push(Page::Field(f));
        self.held = None;
        self.dirty = true;
    }

    /// A field value's place on its slider, 0..1, and back: bytes over 255, metres over 4000, an opacity as is, a light over 3.
    fn slider_pos(f: Field, i: usize, v: f32) -> f32 {
        match (f, i) {
            (Field::Hypso(_), 3) => v / 4000.0,
            (Field::ContourAlpha(_), _) => v,
            (Field::Sun | Field::Sky, _) => v / 3.0,
            _ => v / 255.0,
        }
    }

    fn slider_val(f: Field, i: usize, pos: f32) -> f32 {
        match (f, i) {
            (Field::Hypso(_), 3) => pos * 4000.0,
            (Field::ContourAlpha(_), _) => pos,
            (Field::Sun | Field::Sky, _) => pos * 3.0,
            _ => (pos * 255.0).round(),
        }
    }

    /// True once after anything the panel shows changed (opened, closed, a row flipped), for hosts that only repaint on change.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// The UI's scale (fluor's RU multiplier), clamped as fluor clamps it. Relayout follows on the next paint.
    pub fn set_ru(&mut self, ru: f32) {
        let ru = ru.clamp(RU_MIN, RU_MAX);
        if (ru - self.ru).abs() > 1e-4 {
            self.ru = ru;
            self.dirty = true;
        }
    }

    pub fn ru(&self) -> f32 {
        self.ru
    }

    pub fn close(&mut self) {
        if self.open {
            self.open = false;
            self.dirty = true;
        }
    }

    /// Sizes for a screen, every one a multiple of one unit, Photon's: the harmonic mean of the span (the screen's own harmonic mean of width and height) over 32 times the RU scale, and a thirteenth of the height, so the panel scales with the screen's shape, with the pinch, and never past what a short screen can hold. No clamps on the unit itself.
    fn layout(&mut self, w: usize, h: usize) {
        self.w = w;
        self.h = h;
        let (wf, hf) = (w as f32, h as f32);
        let span = if w + h > 0 { 2.0 * wf * hf / (wf + hf) } else { 0.0 };
        let unit = hm(span / 32.0 * self.ru, hf / 13.0);
        self.font = (unit * 0.5).max(6.0);
        let r = self.font * 1.15;
        let pad = self.font * 0.6;
        self.gear = (pad + r, pad + r, r);
        self.panel_w = (self.font * 13.0).min(w as f32 * 0.7);
        let row = self.font * 1.7;
        let x0 = self.font * 0.8;
        // The theme row sits first, then the layers.
        let theme_y = self.gear.1 + r + self.font * 0.9 + row * 0.5;
        self.theme_row = (x0, theme_y, row * 0.5);
        let top = theme_y + row * 0.8;
        let mut last = top;
        for (i, (l, cb)) in self.checks.iter_mut().enumerate() {
            // The modes sit a little apart from the layers.
            let gap = if matches!(l, Layer::RealSun | Layer::RealLight | Layer::Linear | Layer::FollowHeading | Layer::LockToFix) { row * 0.5 } else { 0.0 };
            let cy = top + row * (i as f32 + 0.5) + gap;
            cb.set_font_size(self.font);
            cb.set_rect(x0 + (self.panel_w - x0 * 1.5) * 0.5, cy, self.panel_w - x0 * 1.5, row);
            last = cy + row * 0.5;
        }
        // The cache slider below the modes: a track the width of the column.
        self.slider = (x0, last + row * 1.3, self.panel_w - x0 * 2.0);
    }

    /// A tap at screen (x, y): true if the panel took it. The gear toggles the panel; a row flips its layer in `mask`.
    pub fn tap(&mut self, x: f32, y: f32, w: usize, h: usize, mask: &mut LayerMask, ctl: &mut Controls, _themes: &[Theme]) -> bool {
        self.layout(w, h);
        if let Some((cx, cy, cr)) = self.strip_close {
            if (x - cx).powi(2) + (y - cy).powi(2) <= (cr * 1.4).powi(2) {
                self.clear_measure = true;
                self.dirty = true;
                return true;
            }
        }
        let (gx, gy, r) = self.gear;
        if (x - gx).powi(2) + (y - gy).powi(2) <= (r * 1.3).powi(2) {
            self.open = !self.open;
            self.dirty = true;
            return true;
        }
        if !self.open {
            return false;
        }
        if x >= self.panel_w {
            return false;
        }
        // The rail is the page under this one: a tap there goes back.
        if self.stack.len() >= 2 && x < self.rail_w() {
            self.pop();
            return true;
        }
        match self.page() {
            Page::Themes => {
                let n = self.theme_rows.len();
                for (i, &(cy, hh)) in self.theme_rows.iter().enumerate() {
                    if (y - cy).abs() <= hh {
                        if i == 0 {
                            self.pop();
                        } else if i + 1 == n {
                            self.stack.push(Page::Edit);
                            self.edit_scroll = 0.0;
                        } else {
                            ctl.theme = i - 1;
                        }
                        self.dirty = true;
                        return true;
                    }
                }
                return true;
            }
            Page::Edit => {
                // The row opens on release, so a drag scrolls the list instead.
                let under = self.edit_rows.iter().find(|&&(cy, hh, _)| (y - cy).abs() <= hh).map(|&(_, _, r)| r);
                self.press = Some((x, y, under, false));
                return true;
            }
            Page::Field(f) => {
                for (i, sl) in self.sliders.iter_mut().enumerate() {
                    let b = sl.bbox();
                    if y >= b.y - b.h * 0.5 && y <= b.y + b.h * 1.5 && x >= b.x - self.font && x <= b.x + b.w + self.font {
                        sl.set_value_from_x(x);
                        self.held = Some(i);
                        self.field_vals[i] = Self::slider_val(f, i, sl.value());
                        self.edits.push(ThemeEdit::Set(f, self.field_vals));
                        self.dirty = true;
                        return true;
                    }
                }
                return true;
            }
            Page::Layers => {}
        }
        // The theme row above the layers opens the theme page.
        let (tx, ty, tr) = self.theme_row;
        if (y - ty).abs() <= tr && x >= tx && x < self.panel_w {
            self.stack.push(Page::Themes);
            self.dirty = true;
            return true;
        }
        for (l, cb) in &mut self.checks {
            if cb.bbox().contains(x, y) {
                // A row a dominating layer has greyed takes the tap but changes nothing.
                if inert(mask, ctl, *l) {
                    return true;
                }
                let v = !get(mask, ctl, *l);
                set(mask, ctl, *l, v);
                cb.set_checked(v);
                self.dirty = true;
                return true;
            }
        }
        if self.slider_press(x, y, ctl) {
            return true;
        }
        // Anywhere else on the column is the panel's, not the map's.
        x < self.panel_w
    }

    /// Paint for a `w × h` screen: the gear always (or, when `orb` gives one, the light as a ball in its place), the column when open. Returns the buffer in fluor's pixel convention.
    pub fn paint(&mut self, w: usize, h: usize, mask: LayerMask, ctl: Controls, r: &Readouts, measure: Option<&MeasureView>, themes: &[Theme], orb: Option<&dyn Fn(usize) -> Option<Vec<[u8; 4]>>>) -> &[u32] {
        self.layout(w, h);
        self.buf.clear();
        self.buf.resize(w * h, 0);
        if w == 0 || h == 0 {
            return &self.buf;
        }
        let font = self.font;
        let (gx, gy, gr) = self.gear;
        let ink_col = theme::TEXTBOX_TEXT;
        // The light as a ball where the gear sits, when a mode has one.
        let ball = orb.and_then(|f| {
            let size = (gr * 2.0) as usize;
            f(size).map(|img| (size, img))
        });
        {
            let mut damage = Damage::new();
            let mut canvas = Canvas::new(&mut self.buf, w, h, &mut damage);
            if ball.is_none() {
                // Topmost first: the gear's hole, its ring and eight teeth, then the disc under them.
                paint::circle_filled(&mut canvas, gx as isize, gy as isize, (gr * 0.18) as isize, GEAR_BG, None, None);
                paint::circle_filled(&mut canvas, gx as isize, gy as isize, (gr * 0.36) as isize, ink_col, None, None);
                for k in 0..8 {
                    let a = k as f32 * core::f32::consts::FRAC_PI_4;
                    let (s, c) = a.sin_cos();
                    let (tx, ty) = (gx + c * gr * 0.58, gy + s * gr * 0.58);
                    paint::circle_filled(&mut canvas, tx as isize, ty as isize, (gr * 0.17).max(2.0) as isize, ink_col, None, None);
                }
                paint::circle_filled(&mut canvas, gx as isize, gy as isize, (gr * 0.52) as isize, GEAR_BG, None, None);
                paint::circle_filled(&mut canvas, gx as isize, gy as isize, (gr * 0.44) as isize, ink_col, None, None);
            }
            paint::circle_filled(&mut canvas, gx as isize, gy as isize, gr as isize, GEAR_BG, None, None);
        }
        // The ball's display pixels straight into the buffer over the disc.
        if let Some((size, img)) = &ball {
            let (x0, y0) = ((gx - gr) as isize, (gy - gr) as isize);
            for py in 0..*size {
                for px in 0..*size {
                    let p = img[py * size + px];
                    if p[3] == 0 {
                        continue;
                    }
                    let (x, y) = (x0 + px as isize, y0 + py as isize);
                    if x < 0 || y < 0 || x as usize >= w || y as usize >= h {
                        continue;
                    }
                    self.buf[y as usize * w + x as usize] = (theme::dark(theme::fmt(((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32)) & 0x00FF_FFFF) | 0xFF00_0000;
                }
            }
        }
        let (page, rail, depth) = (self.page(), self.rail_w(), self.stack.len());
        let under = if depth >= 2 { self.stack[depth - 2] } else { Page::Layers };
        let mut damage = Damage::new();
        let mut canvas = Canvas::new(&mut self.buf, w, h, &mut damage);
        self.strip_close = None;
        if let Some(m) = measure {
            self.strip_close = Some(Self::paint_strip(&mut canvas, &mut self.text, w, h, font, m));
        }
        if !self.open {
            return &self.buf;
        }
        if page != Page::Layers {
            // Top first: the content page, its background, then the rail (the page under it, smaller and dimmed) and the column's background.
            let cw = self.panel_w - rail;
            let current = themes.get(ctl.theme);
            match page {
                Page::Themes => self.theme_rows = Self::paint_themes(&mut canvas, &mut self.text, font, rail, cw, self.gear, h, themes, ctl.theme),
                Page::Edit => {
                    if let Some(t) = current {
                        let (rows, extent) = Self::paint_edit(&mut canvas, &mut self.text, font, rail, cw, self.gear, h, t, self.edit_scroll);
                        self.edit_rows = rows;
                        self.edit_extent = extent;
                    }
                }
                Page::Field(f) => Self::paint_field(&mut canvas, &mut self.text, font, rail, cw, self.gear, f, &mut self.sliders, self.field_vals, current),
                Page::Layers => {}
            }
            paint::fill_rect(&mut canvas, rail as isize, 0, cw as isize, h as isize, PANEL_BG, None, None);
            if depth >= 2 {
                paint::fill_rect(&mut canvas, 0, 0, rail as isize, h as isize, PANEL_DIM, None, None);
                let small = font * 0.72;
                match under {
                    Page::Themes => {
                        Self::paint_themes(&mut canvas, &mut self.text, small, 0.0, rail, self.gear, h, themes, ctl.theme);
                    }
                    Page::Edit => {
                        if let Some(t) = current {
                            let _ = Self::paint_edit(&mut canvas, &mut self.text, small, 0.0, rail, self.gear, h, t, 0.0);
                        }
                    }
                    _ => {
                        self.text.draw_text_left(&mut canvas, "\u{2039}", font * 0.9, self.gear.1 + self.gear.2 + font * 1.7, &TextStyle::new(font, theme::TEXTBOX_TEXT), None, None);
                    }
                }
            }
            paint::fill_rect(&mut canvas, 0, 0, self.panel_w as isize, h as isize, PANEL_BG, None, None);
            return &self.buf;
        }
        // The theme row: the current theme's name and a chevron.
        let (tx, ty, _) = self.theme_row;
        let current = themes.get(ctl.theme).map_or("Trail", |t| t.name.as_str());
        self.text.draw_text_left(&mut canvas, &format!("Theme   {current}  \u{203a}"), tx, ty, &TextStyle::new(font, theme::TEXTBOX_TEXT), None, None);
        for (l, cb) in &mut self.checks {
            cb.set_checked(get(&mask, &ctl, *l));
            // Greyed: a wash of the panel colour over the row, painted first so it lies on top.
            if inert(&mask, &ctl, *l) {
                let b = cb.bbox();
                paint::fill_rect(&mut canvas, b.x as isize, b.y as isize, b.w as isize, b.h as isize, PANEL_DIM, None, None);
            }
            cb.render_content_into(&mut canvas, &mut self.text, None, None);
        }
        self.cache_max = r.cache_max;
        Self::paint_slider(&mut canvas, &mut self.text, font, self.slider, &ctl, r);
        // Readouts below the slider, one line each, dim labels.
        let x = font * 0.9;
        let step = font * 1.35;
        let mut y = self.slider.1 + font * 1.6;
        let small = TextStyle::new(font * 0.9, READOUT);
        let dim = TextStyle::new(font * 0.75, READOUT_DIM);
        // The world code is seven words: wrapped to the column so none is cut.
        let words = vsf::types::WorldCoord::from_lat_lon(r.lat, r.lon).to_words();
        let max_w = self.panel_w - x * 2.0;
        let mut world_lines: Vec<String> = Vec::new();
        for word in words.split_whitespace() {
            match world_lines.last_mut() {
                Some(line) if self.text.measure_text(&format!("{line} {word}"), &small) <= max_w => {
                    line.push(' ');
                    line.push_str(word);
                }
                _ => world_lines.push(word.to_string()),
            }
        }
        let mut lines: Vec<(&str, Vec<String>)> = vec![
            ("position", vec![format!("{:.5}  {:.5}", r.lat, r.lon)]),
            ("world", world_lines),
            ("elevation", vec![r.elev.map_or("—".into(), |e| format!("{} m", e.round() as i64))]),
            ("heading", vec![format!("{:+}°", r.heading_deg.round() as i64)]),
            ("phone", vec![r.phone_heading.map_or("—".into(), |h| format!("{}°", h.round() as i64))]),
            ("scale", vec![format!("{:.2} m/px", r.m_per_px)]),
            ("frame", vec![format!("{:.1} ms  ·  {} cells", r.frame_ms, r.resident)]),
        ];
        for (label, values) in lines.iter_mut() {
            if y + step > h as f32 {
                break;
            }
            self.text.draw_text_left(&mut canvas, label, x, y, &dim, None, None);
            y += font * 0.85;
            for v in values.iter() {
                self.text.draw_text_left(&mut canvas, v, x, y, &small, None, None);
                y += font * 1.1;
            }
            y += font * 0.4;
        }
        let _ = step;
        paint::fill_rect(&mut canvas, 0, 0, self.panel_w as isize, h as isize, PANEL_BG, None, None);
        &self.buf
    }

    /// One painted pixel as premultiplied visible (r, g, b, a): fluor stores accumulated darkness with `dark ≤ α`, so the premultiplied channel is `α − dark`; Android's theme bytes are R↔B swapped.
    #[inline]
    fn premul(p: u32) -> (u8, u8, u8, u8) {
        let a = (p >> 24) as u8;
        let c0 = a - ((p >> 16) & 255).min(a as u32) as u8;
        let c1 = a - ((p >> 8) & 255).min(a as u32) as u8;
        let c2 = a - (p & 255).min(a as u32) as u8;
        if cfg!(target_os = "android") { (c2, c1, c0, a) } else { (c0, c1, c2, a) }
    }

    /// The screen overlay for the GPU as premultiplied RGBA bytes: the engine's marks (0xRRGGBB ink over black, the ink's brightness its coverage) under the panel.
    pub fn overlay_rgba(&self, marks: &[u32], w: usize, h: usize) -> Vec<u8> {
        let n = w * h;
        let mut out = vec![0u8; n * 4];
        for i in 0..n {
            let m = marks.get(i).copied().unwrap_or(0);
            let (mr, mg, mb) = ((m >> 16) as u8, (m >> 8) as u8, m as u8);
            let ma = mr.max(mg).max(mb);
            let (r, g, b, a) = if self.buf.len() == n { Self::premul(self.buf[i]) } else { (0, 0, 0, 0) };
            let keep = 255 - a as u32;
            out[4 * i] = r + ((mr as u32 * keep) / 255) as u8;
            out[4 * i + 1] = g + ((mg as u32 * keep) / 255) as u8;
            out[4 * i + 2] = b + ((mb as u32 * keep) / 255) as u8;
            out[4 * i + 3] = a + ((ma as u32 * keep) / 255) as u8;
        }
        out
    }

    /// For a CPU frontend: the panel over a visible 0xRRGGBB canvas, in place.
    pub fn composite_rgb(&self, canvas: &mut [u32], w: usize, h: usize) {
        let n = w * h;
        if self.buf.len() != n || canvas.len() < n {
            return;
        }
        for i in 0..n {
            let p = self.buf[i];
            if p >> 24 == 0 {
                continue;
            }
            let (r, g, b, a) = Self::premul(p);
            let keep = 255 - a as u32;
            let c = canvas[i];
            let cr = r as u32 + (((c >> 16) & 255) * keep) / 255;
            let cg = g as u32 + (((c >> 8) & 255) * keep) / 255;
            let cb = b as u32 + ((c & 255) * keep) / 255;
            canvas[i] = (cr.min(255) << 16) | (cg.min(255) << 8) | cb.min(255);
        }
    }

    pub fn hit_count(&self) -> HitId {
        self.hits
    }
}

impl Panel {
    /// How many profile samples a strip on a `w`-wide screen shows: one per pixel of its width.
    pub fn strip_samples(w: usize) -> usize {
        let font = (w.min(2400) as f32 / 26.0).clamp(14.0, 36.0);
        (w as f32 - font * 2.0).max(2.0) as usize
    }

    /// The measurement strip along the bottom: the elevation profile from origin to target as a filled area, with distance and bearing on the left and the elevations on the right.
    /// Returns the close button's centre and radius.
    fn paint_strip(canvas: &mut Canvas, text: &mut TextRenderer, w: usize, h: usize, font: f32, m: &MeasureView) -> (f32, f32, f32) {
        let band = font * 6.5;
        let top = h as f32 - band;
        let margin = font;
        // The close button, top right of the strip: a cross in a disc.
        let (cx, cy, cr) = (w as f32 - font * 0.9, top + font * 0.9, font * 0.55);
        text.draw_text_center(canvas, "\u{00d7}", cx, cy, &TextStyle::new(font * 1.1, READOUT), None, None);
        paint::circle_filled(canvas, cx as isize, cy as isize, cr as isize, GEAR_BG, None, None);
        let (lo, hi) = m.profile.iter().filter(|e| !e.is_nan()).fold((f32::MAX, f32::MIN), |(a, b), &e| (a.min(e), b.max(e)));
        let have = lo <= hi;
        let span = (hi - lo).max(1.0);
        // Text first (topmost), then the profile, then the band under both.
        let small = TextStyle::new(font * 0.9, READOUT);
        let dim = TextStyle::new(font * 0.75, READOUT_DIM);
        let dist = if m.distance_m < 1000.0 { format!("{} m", m.distance_m.round() as i64) } else { format!("{:.2} km", m.distance_m / 1000.0) };
        text.draw_text_left(canvas, &format!("{dist}   {}°", m.bearing_deg.round() as i64), margin, top + font * 0.9, &small, None, None);
        let e = |v: Option<f32>| v.map_or("—".to_string(), |e| format!("{} m", e.round() as i64));
        let rise = match (m.elev_origin, m.elev_target) {
            (Some(a), Some(b)) => format!("   {:+} m", (b - a).round() as i64),
            _ => String::new(),
        };
        text.draw_text_right(canvas, &format!("{} → {}{rise}", e(m.elev_origin), e(m.elev_target)), w as f32 - margin - font * 1.6, top + font * 0.9, &small, None, None);
        if have {
            text.draw_text_left(canvas, &format!("{} m", hi.round() as i64), margin, top + font * 1.9, &dim, None, None);
            text.draw_text_left(canvas, &format!("{} m", lo.round() as i64), margin, h as f32 - font * 0.6, &dim, None, None);
        }
        // The profile: one column per sample, from the band's floor up to the elevation.
        let floor_y = h as f32 - font * 0.5;
        let ceil_y = top + font * 1.8;
        let x0 = margin;
        let n = m.profile.len().max(1);
        for (i, &el) in m.profile.iter().enumerate() {
            if el.is_nan() {
                continue;
            }
            let x = x0 + i as f32 * (w as f32 - 2.0 * margin) / n as f32;
            let y = floor_y - (el - lo) / span * (floor_y - ceil_y);
            paint::fill_rect(canvas, x as isize, y as isize, 1, (floor_y - y).max(1.0) as isize, STRIP_FILL, None, None);
            paint::fill_rect(canvas, x as isize, y as isize - 1, 1, 2, STRIP_EDGE, None, None);
        }
        paint::fill_rect(canvas, 0, top as isize, w as isize, band as isize, PANEL_BG, None, None);
        (cx, cy, cr)
    }

    /// True once after a tap on the ruler's close button.
    pub fn take_clear_measure(&mut self) -> bool {
        std::mem::take(&mut self.clear_measure)
    }
}

/// Least the cache slider allows.
pub const CACHE_MIN: u64 = 64 << 20;

fn bytes_text(b: u64) -> String {
    if b >= 1 << 30 { format!("{:.1} GB", b as f64 / (1u64 << 30) as f64) } else { format!("{} MB", b >> 20) }
}

impl Panel {
    /// A press on the slider track sets the budget and arms a drag; true if the press was on it.
    fn slider_press(&mut self, x: f32, y: f32, ctl: &mut Controls) -> bool {
        let (sx, sy, sw) = self.slider;
        if !self.open || (y - sy).abs() > self.font * 1.1 || x < sx - self.font || x > sx + sw + self.font {
            return false;
        }
        self.slider_held = true;
        self.slider_set(x, ctl);
        true
    }

    fn slider_set(&mut self, x: f32, ctl: &mut Controls) {
        let (sx, _, sw) = self.slider;
        let t = ((x - sx) / sw.max(1.0)).clamp(0.0, 1.0);
        let max = self.cache_max.max(CACHE_MIN + 1);
        ctl.cache_budget = CACHE_MIN + ((max - CACHE_MIN) as f64 * t as f64) as u64;
        self.dirty = true;
    }

    /// A move while the slider is held: the budget follows the finger. True while it does.
    pub fn drag(&mut self, x: f32, y: f32, ctl: &mut Controls) -> bool {
        if let (Some((px, py, under, moved)), Page::Edit) = (self.press, self.page()) {
            let moved = moved || (y - py).abs() + (x - px).abs() > 12.0;
            if moved {
                // Never past the last row: the list's height less the screen's, known from the last paint.
                self.edit_scroll = (self.edit_scroll - (y - py)).clamp(0.0, self.edit_extent);
                self.dirty = true;
            }
            self.press = Some((x, y, under, moved));
            return true;
        }
        if let (Some(i), Page::Field(f)) = (self.held, self.page()) {
            if let Some(sl) = self.sliders.get_mut(i) {
                sl.set_value_from_x(x);
                self.field_vals[i] = Self::slider_val(f, i, sl.value());
                self.edits.push(ThemeEdit::Set(f, self.field_vals));
                self.dirty = true;
            }
            return true;
        }
        if !self.slider_held {
            return false;
        }
        self.slider_set(x, ctl);
        true
    }

    pub fn release(&mut self) {
        self.slider_held = false;
        self.held = None;
        match self.press.take() {
            Some((_, _, Some(Row::Field(f, v)), false)) => self.open_field(f, v),
            Some((_, _, Some(Row::Act(a)), false)) => {
                self.edits.push(a);
                // Deleting leaves the editor; the others stay on the theme they changed.
                if a == ThemeEdit::Delete {
                    self.pop();
                }
                self.dirty = true;
            }
            _ => {}
        }
    }

    fn paint_slider(canvas: &mut Canvas, text: &mut TextRenderer, font: f32, slider: (f32, f32, f32), ctl: &Controls, r: &Readouts) {
        let (sx, sy, sw) = slider;
        let max = r.cache_max.max(CACHE_MIN + 1);
        let t = ((ctl.cache_budget.saturating_sub(CACHE_MIN)) as f64 / (max - CACHE_MIN) as f64).clamp(0.0, 1.0) as f32;
        let used = (r.cache_used as f64 / max as f64).clamp(0.0, 1.0) as f32;
        let label = format!("cache  {} of {}  ·  holding {}", bytes_text(ctl.cache_budget), bytes_text(max), bytes_text(r.cache_used));
        text.draw_text_left(canvas, &label, sx, sy - font * 1.0, &TextStyle::new(font * 0.75, READOUT_DIM), None, None);
        // Knob, then the budget's fill, then what is held, then the track under all of them.
        let th = (font * 0.28).max(3.0);
        paint::circle_filled(canvas, (sx + sw * t) as isize, sy as isize, (font * 0.45) as isize, SLIDER_KNOB, None, None);
        paint::fill_rect(canvas, sx as isize, (sy - th * 0.5) as isize, (sw * t) as isize, th as isize, SLIDER_FILL, None, None);
        paint::fill_rect(canvas, sx as isize, (sy - th * 0.5) as isize, (sw * used) as isize, th as isize, SLIDER_USED, None, None);
        paint::fill_rect(canvas, sx as isize, (sy - th * 0.5) as isize, sw as isize, th as isize, SLIDER_TRACK, None, None);
    }
}

impl Panel {
    /// The theme page: a back row, one row per theme (the current one ticked), and under them the legend of the chosen theme — the terrain ramp, the land cover swatches, the line classes and the inks. Returns the rows as (centre y, half height), the back row first.
    #[allow(clippy::too_many_arguments)]
    fn paint_themes(canvas: &mut Canvas, text: &mut TextRenderer, font: f32, ox: f32, panel_w: f32, gear: (f32, f32, f32), h: usize, themes: &[Theme], current: usize) -> Vec<(f32, f32)> {
        let x0 = ox + font * 0.9;
        let panel_w = ox + panel_w;
        let row = font * 1.6;
        let mut y = gear.1 + gear.2 + font * 0.9 + row * 0.5;
        let mut rows = Vec::new();
        let title = TextStyle::new(font, theme::TEXTBOX_TEXT);
        text.draw_text_left(canvas, "\u{2039}  Layers", x0, y, &title, None, None);
        rows.push((y, row * 0.5));
        y += row * 1.2;
        for (i, t) in themes.iter().enumerate() {
            let mark = if i == current { "\u{25cf}" } else { "\u{25cb}" };
            text.draw_text_left(canvas, &format!("{mark}  {}", t.name), x0, y, &title, None, None);
            rows.push((y, row * 0.5));
            y += row;
        }
        // The last row opens the editor on the current theme.
        let edit_label = match themes.get(current) {
            Some(t) => format!("\u{270e}  Edit {}", t.name),
            None => "\u{270e}  Edit".into(),
        };
        text.draw_text_left(canvas, &edit_label, x0, y, &title, None, None);
        rows.push((y, row * 0.5));
        let _ = (y, panel_w, h);
        rows
    }
}

impl Panel {
    /// The editor: the back row, then every field of the theme as a row with its swatch (a number for the numbers), one to a line, scrolled by `scroll`. Returns the rows on screen for hit testing, with their values.
    #[allow(clippy::too_many_arguments)]
    fn paint_edit(canvas: &mut Canvas, text: &mut TextRenderer, font: f32, ox: f32, cw: f32, gear: (f32, f32, f32), h: usize, t: &Theme, scroll: f32) -> (Vec<(f32, f32, Row)>, f32) {
        let x0 = ox + font * 0.9;
        let row = font * 1.15;
        let top = gear.1 + gear.2 + font * 0.9 + font * 0.8;
        let title = TextStyle::new(font, theme::TEXTBOX_TEXT);
        let small = TextStyle::new(font * 0.8, READOUT);
        let dim = TextStyle::new(font * 0.72, READOUT_DIM);
        text.draw_text_left(canvas, &format!("\u{2039}  {}", t.name), x0, top, &title, None, None);
        let mut y = top + row * 1.3 - scroll;
        let mut rows = Vec::new();
        let mut section = "";
        let _ = cw;
        for f in Field::all() {
            let sec = match f {
                Field::Hypso(_) => "terrain",
                Field::Sea | Field::Flat | Field::Bg | Field::NoDem => "ground",
                Field::Land(_) => "land cover",
                Field::Line(_) => "lines",
                Field::Water | Field::Contour | Field::ContourIndex | Field::ContourAlpha(_) => "inks",
                Field::Sun | Field::Sky => "light",
            };
            if sec != section {
                section = sec;
                y += font * 0.3;
                if y > top + row && y < h as f32 {
                    text.draw_text_left(canvas, sec, x0, y, &dim, None, None);
                }
                y += font * 0.9;
            }
            if y > h as f32 + row {
                break;
            }
            let v = t.get(f);
            // Rows scrolled under the title are skipped, not drawn over it.
            if y > top + row {
                if f.is_colour() {
                    let c = [v[0] as u8, v[1] as u8, v[2] as u8];
                    let (sw, sh) = if matches!(f, Field::Line(_)) { (font * 0.8, (font * 0.16).max(2.0)) } else { (font * 0.8, font * 0.6) };
                    paint::fill_rect(canvas, x0 as isize, (y - sh * 0.5) as isize, sw as isize, sh as isize, swatch(c), None, None);
                } else {
                    let label = match f {
                        Field::ContourAlpha(_) => format!("{:.2}", v[0]),
                        _ => format!("{:.1}", 0.3 * v[0] + 0.6 * v[1] + 0.1 * v[2]),
                    };
                    text.draw_text_left(canvas, &label, x0, y, &small, None, None);
                }
                let name = match f {
                    Field::Hypso(i) => format!("{} m", t.hypso[i as usize].0.round() as i64),
                    _ => f.name(),
                };
                // A number is wider than a swatch: its name sits further along.
                let name_x = x0 + if f.is_colour() { font * 1.1 } else { font * 2.6 };
                text.draw_text_left(canvas, &name, name_x, y, &small, None, None);
                rows.push((y, row * 0.5, Row::Field(f, v)));
            }
            y += row;
        }
        // The actions: a copy for anyone, the shipped theme back for a built-in that has been changed, deletion for the user's own.
        let mut actions: Vec<(&str, ThemeEdit)> = vec![("\u{29c9}  Duplicate", ThemeEdit::Duplicate)];
        if t.is_builtin() && t.edited {
            actions.push(("\u{21ba}  Reset to shipped", ThemeEdit::Reset));
        }
        if !t.is_builtin() {
            actions.push(("\u{2715}  Delete", ThemeEdit::Delete));
        }
        y += font * 0.6;
        for (label, a) in actions {
            if y > top + row && y < h as f32 + row {
                text.draw_text_left(canvas, label, x0, y, &TextStyle::new(font * 0.9, theme::TEXTBOX_TEXT), None, None);
                rows.push((y, row * 0.5, Row::Act(a)));
            }
            y += row * 1.2;
        }
        // The list's reach past the screen, with the scroll put back: what a drag may take it to.
        let extent = (y + scroll + font - h as f32).max(0.0);
        (rows, extent)
    }

    /// One field: its name, the colour or light as a swatch through the display encode, and a slider per value with the value beside it.
    fn paint_field(canvas: &mut Canvas, text: &mut TextRenderer, font: f32, ox: f32, cw: f32, gear: (f32, f32, f32), f: Field, sliders: &mut [Slider], vals: [f32; 4], t: Option<&Theme>) {
        let _ = t;
        let x0 = ox + font * 0.9;
        let mut y = gear.1 + gear.2 + font * 0.9 + font * 0.8;
        let title = TextStyle::new(font, theme::TEXTBOX_TEXT);
        let small = TextStyle::new(font * 0.8, READOUT);
        let dim = TextStyle::new(font * 0.72, READOUT_DIM);
        text.draw_text_left(canvas, &format!("\u{2039}  {}", f.name()), x0, y, &title, None, None);
        y += font * 1.6;
        // The swatch: a colour as the map shows it unlit; a light as the colour it would make white ground.
        let sw = cw - font * 1.8;
        if f.is_colour() {
            paint::fill_rect(canvas, x0 as isize, y as isize, sw as isize, (font * 1.6) as isize, swatch([vals[0] as u8, vals[1] as u8, vals[2] as u8]), None, None);
        } else if !matches!(f, Field::ContourAlpha(_)) {
            let c = mahere_engine::colour::Display::default().encode([vals[0] * 0.5, vals[1] * 0.5, vals[2] * 0.5], true);
            paint::fill_rect(canvas, x0 as isize, y as isize, sw as isize, (font * 1.6) as isize, (theme::dark(theme::fmt(((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32)) & 0x00FF_FFFF) | 0xFF00_0000, None, None);
        }
        y += font * 2.4;
        let labels: [&str; 4] = match f {
            Field::ContourAlpha(_) => ["opacity", "", "", ""],
            Field::Hypso(_) => ["R", "G", "B", "metres"],
            _ => ["R", "G", "B", ""],
        };
        for (i, sl) in sliders.iter_mut().enumerate() {
            text.draw_text_left(canvas, labels[i], x0, y, &dim, None, None);
            let val = match (f, i) {
                (Field::Hypso(_), 3) => format!("{} m", vals[3].round() as i64),
                (Field::ContourAlpha(_), _) => format!("{:.2}", vals[0]),
                (Field::Sun | Field::Sky, _) => format!("{:.2}", vals[i]),
                _ => format!("{}", vals[i] as u8),
            };
            text.draw_text_left(canvas, &val, x0 + font * 4.5, y, &small, None, None);
            y += font * 0.9;
            sl.set_rect(x0 + sw * 0.5, y, sw, font * 1.1);
            sl.render_content_into(canvas, None, HIT_NONE);
            y += font * 1.7;
        }
    }
}

/// A legend swatch of an authored theme colour, as the map shows it unlit (through the same display encode), opaque, in fluor's stored form.
fn swatch(c: [u8; 3]) -> u32 {
    let c = mahere_engine::colour::Display::default().swatch(c);
    (theme::dark(theme::fmt(((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32)) & 0x00FF_FFFF) | 0xFF00_0000
}
