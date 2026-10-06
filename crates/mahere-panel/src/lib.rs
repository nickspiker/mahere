//! The on-map controls (Nick 2026-10-06, after Lumis): one gear button in the corner shows and hides everything else, so the map can have the whole screen with only the gear showing. Open, a column holds the layer toggles and the readouts — position, world code, elevation, heading, scale, frame time.
//!
//! Painted with fluor's widgets and text into the panel's own buffer in fluor's pixel convention (α + darkness, topmost first), then composited by whichever renderer owns the screen: the GPU takes it as a premultiplied RGBA overlay over the engine's marks, the CPU frontend lays it over its canvas. Taps are routed here first; a tap the panel does not take is the map's.

use fluor::canvas::{Canvas, Damage};
use fluor::paint::{self, HitId, HIT_NONE};
use fluor::text::{TextRenderer, TextStyle};
use fluor::theme;
use fluor::widgets::Checkbox;
use mahere_engine::{LayerMask, MeasureView};

/// The two modes the panel switches besides the layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Controls {
    pub real_sun: bool,
    pub follow_heading: bool,
    /// Measure from the fix rather than the screen centre.
    pub lock_to_fix: bool,
    /// Bytes the cell cache may hold; the slider sets it.
    pub cache_budget: u64,
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
    Canopy,
    Imagery,
    Debug,
}

const LAYERS: [(Layer, &str); 14] = [
    (Layer::Dem, "Terrain"),
    (Layer::Hypso, "Elevation tint"),
    (Layer::Land, "Land cover"),
    (Layer::Water, "Water"),
    (Layer::Line, "Lines"),
    (Layer::Boundaries, "Boundaries"),
    (Layer::Contours, "Contours"),
    (Layer::Slope, "Slope bands"),
    (Layer::Canopy, "Canopy"),
    (Layer::Imagery, "Imagery"),
    (Layer::Debug, "Residency"),
    (Layer::RealSun, "Real sun"),
    (Layer::FollowHeading, "Follow heading"),
    (Layer::LockToFix, "Measure from me"),
];

fn get(mask: &LayerMask, ctl: &Controls, l: Layer) -> bool {
    match l {
        Layer::RealSun => ctl.real_sun,
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
        Layer::Canopy => mask.canopy,
        Layer::Imagery => mask.imagery,
        Layer::Debug => mask.debug,
    }
}

fn set(mask: &mut LayerMask, ctl: &mut Controls, l: Layer, v: bool) {
    match l {
        Layer::RealSun => ctl.real_sun = v,
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
        Layer::Canopy => mask.canopy = v,
        Layer::Imagery => mask.imagery = v,
        Layer::Debug => mask.debug = v,
    }
}

/// A colour authored as visible 0xRRGGBB with an opacity, in fluor's stored form on this platform.
const fn ink(rgb: u32, alpha: u8) -> u32 {
    (theme::dark(theme::fmt(theme::vsf(rgb))) & 0x00FF_FFFF) | ((alpha as u32) << 24)
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

pub struct Panel {
    open: bool,
    dirty: bool,
    hits: HitId,
    checks: Vec<(Layer, Checkbox)>,
    text: TextRenderer,
    buf: Vec<u32>,
    w: usize,
    h: usize,
    font: f32,
    gear: (f32, f32, f32),
    panel_w: f32,
    /// The cache slider's track on screen (x0, y, width), and whether a press is riding it.
    slider: (f32, f32, f32),
    slider_held: bool,
    cache_max: u64,
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
        Panel { open: false, dirty: true, hits, checks, text: TextRenderer::new(), buf: Vec::new(), w: 0, h: 0, font: 16.0, gear: (0.0, 0.0, 0.0), panel_w: 0.0, slider: (0.0, 0.0, 0.0), slider_held: false, cache_max: CACHE_MIN + 1 }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn set_open(&mut self, open: bool) {
        self.open = open;
        self.dirty = true;
    }

    /// True once after anything the panel shows changed (opened, closed, a row flipped), for hosts that only repaint on change.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// Sizes for a screen: the font from the short side, the gear in the top-left corner, the column's width.
    fn layout(&mut self, w: usize, h: usize) {
        self.w = w;
        self.h = h;
        self.font = (w.min(h) as f32 / 26.0).clamp(14.0, 36.0);
        let r = self.font * 1.15;
        let pad = self.font * 0.6;
        self.gear = (pad + r, pad + r, r);
        self.panel_w = (self.font * 13.0).min(w as f32 * 0.7);
        let row = self.font * 1.7;
        let x0 = self.font * 0.8;
        let top = self.gear.1 + r + self.font * 0.9;
        let mut last = top;
        for (i, (l, cb)) in self.checks.iter_mut().enumerate() {
            // The modes sit a little apart from the layers.
            let gap = if matches!(l, Layer::RealSun | Layer::FollowHeading | Layer::LockToFix) { row * 0.5 } else { 0.0 };
            let cy = top + row * (i as f32 + 0.5) + gap;
            cb.set_font_size(self.font);
            cb.set_rect(x0 + (self.panel_w - x0 * 1.5) * 0.5, cy, self.panel_w - x0 * 1.5, row);
            last = cy + row * 0.5;
        }
        // The cache slider below the modes: a track the width of the column.
        self.slider = (x0, last + row * 1.3, self.panel_w - x0 * 2.0);
    }

    /// A tap at screen (x, y): true if the panel took it. The gear toggles the panel; a row flips its layer in `mask`.
    pub fn tap(&mut self, x: f32, y: f32, w: usize, h: usize, mask: &mut LayerMask, ctl: &mut Controls) -> bool {
        self.layout(w, h);
        let (gx, gy, r) = self.gear;
        if (x - gx).powi(2) + (y - gy).powi(2) <= (r * 1.3).powi(2) {
            self.open = !self.open;
            self.dirty = true;
            return true;
        }
        if !self.open {
            return false;
        }
        let inert = mask.inert();
        for (l, cb) in &mut self.checks {
            if cb.bbox().contains(x, y) {
                // A row a dominating layer has greyed takes the tap but changes nothing.
                if get(&inert, &Controls::default(), *l) {
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

    /// Paint for a `w × h` screen: the gear always, the column when open. Returns the buffer in fluor's pixel convention.
    pub fn paint(&mut self, w: usize, h: usize, mask: LayerMask, ctl: Controls, r: &Readouts, measure: Option<&MeasureView>) -> &[u32] {
        self.layout(w, h);
        self.buf.clear();
        self.buf.resize(w * h, 0);
        if w == 0 || h == 0 {
            return &self.buf;
        }
        let mut damage = Damage::new();
        let mut canvas = Canvas::new(&mut self.buf, w, h, &mut damage);
        let font = self.font;
        // Topmost first: the gear's hole, its ring and eight teeth, then the disc under them.
        let (gx, gy, gr) = self.gear;
        let ink_col = theme::TEXTBOX_TEXT;
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
        paint::circle_filled(&mut canvas, gx as isize, gy as isize, gr as isize, GEAR_BG, None, None);
        if let Some(m) = measure {
            Self::paint_strip(&mut canvas, &mut self.text, w, h, font, m);
        }
        if !self.open {
            return &self.buf;
        }
        let inert = mask.inert();
        for (l, cb) in &mut self.checks {
            cb.set_checked(get(&mask, &ctl, *l));
            // Greyed: a wash of the panel colour over the row, painted first so it lies on top.
            if get(&inert, &Controls::default(), *l) {
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
    fn paint_strip(canvas: &mut Canvas, text: &mut TextRenderer, w: usize, h: usize, font: f32, m: &MeasureView) {
        let band = font * 6.5;
        let top = h as f32 - band;
        let margin = font;
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
        text.draw_text_right(canvas, &format!("{} → {}{rise}", e(m.elev_origin), e(m.elev_target)), w as f32 - margin, top + font * 0.9, &small, None, None);
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
    pub fn drag(&mut self, x: f32, ctl: &mut Controls) -> bool {
        if !self.slider_held {
            return false;
        }
        self.slider_set(x, ctl);
        true
    }

    pub fn release(&mut self) {
        self.slider_held = false;
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
