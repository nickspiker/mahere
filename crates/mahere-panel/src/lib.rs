//! The on-map controls (Nick 2026-10-06, after Lumis): one gear button in the corner shows and hides everything else, so the map can have the whole screen with only the gear showing. Open, a column holds the layer toggles and the readouts — position, world code, elevation, heading, scale, frame time.
//!
//! Painted with fluor's widgets and text into the panel's own buffer in fluor's pixel convention (α + darkness, topmost first), then composited by whichever renderer owns the screen: the GPU takes it as a premultiplied RGBA overlay over the engine's marks, the CPU frontend lays it over its canvas. Taps are routed here first; a tap the panel does not take is the map's.

pub mod fonts;
pub mod labels;

use fluor::canvas::{Canvas, Damage};
use fluor::paint::{self, HitId, HIT_NONE};
use fluor::text::{TextRenderer, TextStyle};
use fluor::theme;
use fluor::widgets::{Checkbox, Slider};
pub use mahere_engine::areas::{AREA_MAX_DEPTH, AREA_MIN_DEPTH, AreaView, AreasView};

/// The app's name as shown beside the orb and as the window's title.
pub const APP_NAME: &str = "Mahere";

/// The app orb: the paper-craft relief map, fluor's 256 × 256 VSF orb. The orb is the settings button on every platform, as in Photon (Nick 2026-10-10): the chrome draws it on a desktop, the panel draws it on a phone.
pub const APP_ORB_VSF: &[u8] = include_bytes!("../assets/mahere_orb.vsf");
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
    /// What a touch on the map does: 0 navigates, 1 draws with the pen, 2 draws a selection. The pen's colour, an index into the engine's pens.
    pub tool: u8,
    /// The pen's colour and opacity, VSF RGB at gamma 2.
    pub pen_rgba: [u8; 4],
    /// How a tap on the map routes there: 0 it doesn't, 1 by car, 2 by bike, 3 on foot.
    pub travel: u8,
    /// A track is being recorded (Nick 2026-10-13), the screen on or off.
    pub record: bool,
    /// One-shot asks: the ink or the selection cleared.
    pub clear_ink: bool,
    pub clear_sel: bool,
    /// One-shot: the locate button was tapped, centre on the fix.
    pub locate: bool,
    /// A tap measures (the ruler) rather than only telling what is there.
    pub ruler: bool,
}

/// What the toolbox offers (Nick 2026-10-14: a popup toolbox for ruler, nav, info and the rest, so a plain tap just shows info): the tool a tap or a touch is for, and the pen's and the clears' one-shots.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Pick {
    Info,
    Ruler,
    Drive,
    Ride,
    Walk,
    Pen,
    Select,
    PenColour,
    ClearInk,
    ClearSel,
}

impl Pick {
    fn word(self) -> &'static str {
        match self {
            Pick::Info => "Info",
            Pick::Ruler => "Ruler",
            Pick::Drive => "Drive",
            Pick::Ride => "Ride",
            Pick::Walk => "Walk",
            Pick::Pen => "Pen",
            Pick::Select => "Select",
            Pick::PenColour => "Pen colour",
            Pick::ClearInk => "Clear ink",
            Pick::ClearSel => "Clear selection",
        }
    }

    /// The tool the controls have on.
    fn current(ctl: &Controls) -> Pick {
        match (ctl.tool, ctl.travel, ctl.ruler) {
            (1, _, _) => Pick::Pen,
            (2, _, _) => Pick::Select,
            (_, 1, _) => Pick::Drive,
            (_, 2, _) => Pick::Ride,
            (_, 3, _) => Pick::Walk,
            (_, _, true) => Pick::Ruler,
            _ => Pick::Info,
        }
    }

    /// The toolbox's list for these controls: the tools, then the pen's colour and the clears beside the tool they belong to.
    fn list(ctl: &Controls) -> Vec<Pick> {
        let mut v = vec![Pick::Info, Pick::Ruler, Pick::Drive, Pick::Ride, Pick::Walk, Pick::Pen, Pick::Select];
        match Pick::current(ctl) {
            Pick::Pen => v.extend([Pick::PenColour, Pick::ClearInk]),
            Pick::Select => v.push(Pick::ClearSel),
            _ => {}
        }
        v
    }
}

/// What the readouts show.
#[derive(Clone, Copy, Debug, Default)]
pub struct Readouts {
    pub lat: f64,
    pub lon: f64,
    pub elev: Option<f32>,
    /// The GPS's own height above sea level (its height less the geoid) and its vertical accuracy, when there is a fix with a height.
    pub gps_height: Option<(f32, Option<f32>)>,
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
    /// There is a GPS fix to go to.
    pub has_fix: bool,
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
    Own,
    Mvum,
    Raw,
    Land,
    Water,
    Line,
    Labels,
    Contours,
    Slope,
    Infrared,
    Imagery,
    Debug,
    /// The areas page: saved selections, downloaded and shared.
    Areas,
    /// The updates page.
    Updates,
    /// Recording a track, the screen on or off.
    Record,
}

const LAYERS: [(Layer, &str); 23] = [
    (Layer::Dem, "Terrain"),
    (Layer::Hypso, "Elevation tint"),
    (Layer::Raw, "Raw elevation"),
    (Layer::Land, "Land cover"),
    (Layer::Water, "Water"),
    (Layer::Line, "Lines"),
    (Layer::Labels, "Labels"),
    (Layer::Boundaries, "Boundaries"),
    (Layer::Own, "Land ownership"),
    (Layer::Mvum, "Motor vehicle use"),
    (Layer::Contours, "Contours"),
    (Layer::Slope, "Slope bands"),
    (Layer::Imagery, "Imagery"),
    (Layer::Infrared, "Infrared"),
    (Layer::Debug, "Detail"),
    (Layer::RealSun, "Real sun"),
    (Layer::RealLight, "Real light"),
    (Layer::Linear, "Linear highlights"),
    (Layer::FollowHeading, "Follow heading"),
    (Layer::LockToFix, "Measure from me"),
    (Layer::Record, "Record track"),
    (Layer::Areas, "Areas  \u{203a}"),
    (Layer::Updates, "Updates  \u{203a}"),
];

/// A row that takes a tap but changes nothing: a layer another layer has greyed, or the one of real sun and real light the other has taken (the sun is either the almanac's or the camera's).
fn inert(mask: &LayerMask, ctl: &Controls, l: Layer) -> bool {
    match l {
        Layer::RealSun => ctl.real_light,
        Layer::RealLight => ctl.real_sun,
        Layer::Linear | Layer::FollowHeading | Layer::LockToFix | Layer::Record | Layer::Areas | Layer::Updates => false,
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
        Layer::Labels => mask.labels,
        Layer::Boundaries => mask.boundaries,
        Layer::Own => mask.own,
        Layer::Raw => mask.raw,
        Layer::Mvum => mask.mvum,
        Layer::Contours => mask.contours,
        Layer::Slope => mask.slope,
        Layer::Infrared => mask.infrared,
        Layer::Imagery => mask.imagery,
        Layer::Debug => mask.debug,
        Layer::Record => ctl.record,
        Layer::Areas | Layer::Updates => false,
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
        Layer::Labels => mask.labels = v,
        Layer::Boundaries => mask.boundaries = v,
        Layer::Own => mask.own = v,
        Layer::Raw => mask.raw = v,
        Layer::Mvum => mask.mvum = v,
        Layer::Contours => mask.contours = v,
        Layer::Slope => mask.slope = v,
        Layer::Infrared => mask.infrared = v,
        Layer::Imagery => mask.imagery = v,
        Layer::Debug => mask.debug = v,
        Layer::Record => ctl.record = v,
        Layer::Areas | Layer::Updates => {}
    }
}

/// A colour authored as visible 0xRRGGBB with an opacity, in fluor's stored form on this platform.
const fn ink(rgb: u32, alpha: u8) -> u32 {
    (theme::dark(theme::fmt(theme::vsf(rgb))) & 0x00FF_FFFF) | ((alpha as u32) << 24)
}

/// The label look's sliders, low and high: size, glow, smoothing, spacing (`Theme::label_look`).
const LOOK_RANGE: [(f32, f32); 4] = [(0.3, 2.0), (0.0, 0.4), (0.0, 2.0), (0.0, 1.0)];

/// The label amount's sliders, low and high: density, repeat (`Theme::label_amount`).
const AMOUNT_RANGE: [(f32, f32); 2] = [(0.0, 1.0), (0.1, 1.0)];

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
/// The footer's ground: a shade apart from the column's, so the readouts read as their own panel.
const FOOTER_BG: u32 = ink(0x18_1E_2A, 236);
/// The footer's height in font sizes: the slider and its label, six readouts and the world code's label and three lines.
const FOOTER_LINES: f32 = 15.0;
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
    /// The pen's colour: a mixer of red, green, blue and opacity.
    Pen,
    /// The saved areas, and one of them.
    Areas,
    Area(usize),
    /// Updates: this build, the release and dev channels' builds, automatic updates.
    Updates,
}

/// What the Updates page shows (the host's updater, as text): this build, each channel's row (a short title, a small line under it, and whether a tap gets it), the line of what is happening, a download's progress, and whether updates are automatic.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UpdatesView {
    pub this: String,
    pub channels: [(String, String, bool); 2],
    pub status: String,
    pub progress: Option<f32>,
    pub auto: bool,
}

/// What a tap on the Updates page asks of the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateAct {
    /// The page opened: check both channels.
    Check,
    /// Get a channel's build: 0 release, 1 dev.
    Get(usize),
    Auto(bool),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum UpdateRow {
    Back,
    Get(usize),
    Auto,
}

/// A row of the areas pages.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AreaRow {
    Back,
    Save,
    Open(usize),
    Download,
    Cancel,
    Refresh,
    Share,
    Show,
    Delete,
}

/// What the areas pages ask of the host, polled after every tap and drag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AreaAct {
    /// The selection saved as a new area; the host opens it with [`Panel::show_area`].
    Save,
    SetDepth(usize, u8),
    Download(usize),
    Cancel(usize),
    /// Download again, asking for what was absent before.
    Refresh(usize),
    Share(usize),
    /// The area's shape shown as the selection.
    Show(usize),
    Delete(usize),
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
    /// The real light, as it stands, into the theme as its own light.
    CaptureLight,
    /// The theme's own light removed: back to its sun and sky.
    ClearLight,
}

/// A row of the editor: a field to open, or an action.
#[derive(Clone, Copy)]
enum Row {
    Field(Field, [f32; 4]),
    Act(ThemeEdit),
}


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
    /// The font packs, when the host gave them, and how far through their arrivals this renderer has loaded.
    fonts: Option<std::sync::Arc<fonts::FontPacks>>,
    font_cursor: usize,
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
    /// The areas pages' rows from the last paint (centre y, half height, the row), what they asked, and what was shown.
    area_rows: Vec<(f32, f32, AreaRow)>,
    area_acts: Vec<AreaAct>,
    areas: AreasView,
    update_rows: Vec<(f32, f32, UpdateRow)>,
    update_acts: Vec<UpdateAct>,
    updates: UpdatesView,
    /// Where the host draws the orb itself (a desktop's chrome, with the title beside it): the panel then lays out under it, takes a tap there as the toggle, and paints no orb of its own.
    host_orb: Option<(f32, f32, f32)>,
    /// The Layers page scrolls between the orb and the footer (Nick 2026-10-11): how far, how far it can, a press on it (where it started, whether it has moved), and the band it scrolls in (top, bottom, where the footer starts).
    layers_scroll: f32,
    layers_extent: f32,
    layers_press: Option<(f32, f32, bool)>,
    layers_band: (f32, f32),
    /// The app orb decoded once, and resampled to the last size asked, in fluor's stored form.
    orb_icon: Option<fluor::host::icon::Icon>,
    orb_scaled: (usize, Vec<u32>),
    /// The locate button as last painted (centre, radius), for its tap.
    locate_btn: Option<(f32, f32, f32)>,
    /// The toolbox: its button as last painted (left, top, width, height), whether its list is up, and the list's rows.
    tool_btn: Option<(f32, f32, f32, f32)>,
    toolbox_open: bool,
    tool_rows: Vec<((f32, f32, f32, f32), Pick)>,
}

impl Default for Panel {
    fn default() -> Self {
        Self::new()
    }
}

impl Panel {
    /// The font packs to draw text in scripts the binary lacks, shared with the labels ([`labels::Labeller::set_fonts`]).
    pub fn set_fonts(&mut self, fonts: std::sync::Arc<fonts::FontPacks>) {
        self.fonts = Some(fonts);
    }

    /// Whether a font pack is being fetched or has arrived unloaded: a host keeps a frame coming until it is painted.
    pub fn fonts_due(&self) -> bool {
        self.fonts.as_ref().is_some_and(|f| f.busy() || f.arrived() > self.font_cursor)
    }

    pub fn new() -> Panel {
        let mut hits: HitId = HIT_NONE;
        let checks = LAYERS.iter().map(|&(l, label)| (l, Checkbox::new(&mut hits, label, 0.0, 0.0, 10.0, 10.0, 10.0, false))).collect();
        Panel { fonts: None, font_cursor: 0, open: false, dirty: true, ru: 1.0, stack: vec![Page::Layers], theme_rows: Vec::new(), edit_rows: Vec::new(), edit_scroll: 0.0, edit_extent: 0.0, press: None, sliders: Vec::new(), field_vals: [0.0; 4], held: None, edits: Vec::new(), hits, checks, text: TextRenderer::new(), buf: Vec::new(), w: 0, h: 0, font: 16.0, gear: (0.0, 0.0, 0.0), panel_w: 0.0, theme_row: (0.0, 0.0, 0.0), slider: (0.0, 0.0, 0.0), slider_held: false, cache_max: CACHE_MIN + 1, strip_close: None, clear_measure: false, area_rows: Vec::new(), area_acts: Vec::new(), areas: AreasView::default(), update_rows: Vec::new(), update_acts: Vec::new(), updates: UpdatesView::default(), host_orb: None, layers_scroll: 0.0, layers_extent: 0.0, layers_press: None, layers_band: (0.0, 0.0), orb_icon: fluor::host::icon::Icon::from_vsf_bytes(APP_ORB_VSF).ok(), orb_scaled: (0, Vec::new()), locate_btn: None, tool_btn: None, toolbox_open: false, tool_rows: Vec::new() }
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

    /// Open on the pen's mixer (the preview example).
    pub fn show_pen(&mut self, rgba: [u8; 4]) {
        self.open = true;
        self.stack = vec![Page::Layers];
        self.open_pen(rgba);
    }

    /// The orb the host draws, (centre x, centre y, radius) in screen pixels, or None for the panel to draw its own.
    pub fn set_host_orb(&mut self, orb: Option<(f32, f32, f32)>) {
        if self.host_orb != orb {
            self.host_orb = orb;
            self.dirty = true;
        }
    }

    /// Back, from the phone's back button or Escape: one page up the stack, the panel closed from its first page. False when the panel was already closed, for the host to do whatever back means outside it.
    pub fn back(&mut self) -> bool {
        if !self.open {
            return false;
        }
        if self.stack.len() > 1 {
            self.pop();
        } else {
            self.open = false;
            self.dirty = true;
        }
        self.layers_press = None;
        self.press = None;
        true
    }

    /// Show or hide the panel (the host's orb was tapped).
    pub fn toggle(&mut self) {
        self.open = !self.open;
        self.dirty = true;
    }

    /// The app orb at `size` pixels across, box-filtered from the decoded icon; empty when the icon did not decode.
    fn orb_pixels(&mut self, size: usize) -> &[u32] {
        if self.orb_scaled.0 != size {
            let mut out = Vec::new();
            if let Some(icon) = &self.orb_icon {
                let (iw, ih) = (icon.width as usize, icon.height as usize);
                out = vec![0u32; size * size];
                for y in 0..size {
                    for x in 0..size {
                        let (sx0, sx1) = (x * iw / size, ((x + 1) * iw / size).max(x * iw / size + 1));
                        let (sy0, sy1) = (y * ih / size, ((y + 1) * ih / size).max(y * ih / size + 1));
                        let (mut r, mut g, mut b, mut n) = (0u32, 0u32, 0u32, 0u32);
                        for sy in sy0..sy1.min(ih) {
                            for sx in sx0..sx1.min(iw) {
                                let p = icon.pixels[sy * iw + sx];
                                r += (p >> 16) & 255;
                                g += (p >> 8) & 255;
                                b += p & 255;
                                n += 1;
                            }
                        }
                        let n = n.max(1);
                        // The icon's pixels are canonical 0xAARRGGBB; the buffer is the platform's order (red and blue swapped on Android, where the orb showed them swapped, 2026-10-14).
                        out[y * size + x] = theme::fmt(0xFF00_0000 | ((r / n) << 16) | ((g / n) << 8) | (b / n));
                    }
                }
            }
            self.orb_scaled = (size, out);
        }
        &self.orb_scaled.1
    }

    /// Open on one area (a new one saved, or one shared in).
    pub fn show_area(&mut self, i: usize, depth: u8) {
        self.open = true;
        self.stack = vec![Page::Layers, Page::Areas, Page::Area(i)];
        self.open_area_slider(depth);
    }

    /// The area page's slider: the depth over its range.
    fn open_area_slider(&mut self, depth: u8) {
        self.field_vals = [depth as f32, 0.0, 0.0, 0.0];
        self.sliders = vec![Slider::new(&mut self.hits, 0.0, 0.0, 10.0, 10.0, Self::depth_pos(depth))];
        self.held = None;
        self.dirty = true;
    }

    fn depth_pos(depth: u8) -> f32 {
        (depth.clamp(AREA_MIN_DEPTH, AREA_MAX_DEPTH) - AREA_MIN_DEPTH) as f32 / (AREA_MAX_DEPTH - AREA_MIN_DEPTH) as f32
    }

    fn depth_of(pos: f32) -> u8 {
        (AREA_MIN_DEPTH as f32 + pos.clamp(0.0, 1.0) * (AREA_MAX_DEPTH - AREA_MIN_DEPTH) as f32).round() as u8
    }

    /// Repaint on the next frame (a download's progress changed).
    pub fn invalidate(&mut self) {
        self.dirty = true;
    }

    /// Back one page (the preview example).
    pub fn pop_page(&mut self) {
        self.pop();
    }

    /// What the areas pages asked since last polled.
    pub fn take_area_acts(&mut self) -> Vec<AreaAct> {
        std::mem::take(&mut self.area_acts)
    }

    /// What the Updates page shows, from the host's updater; repaints when it changed while the page is open.
    pub fn set_updates(&mut self, v: UpdatesView) {
        if v != self.updates {
            self.updates = v;
            if self.stack.last() == Some(&Page::Updates) {
                self.dirty = true;
            }
        }
    }

    /// Open the Updates page (gpu_check's preview).
    pub fn show_updates(&mut self) {
        self.stack = vec![Page::Layers, Page::Updates];
        self.dirty = true;
    }

    pub fn take_update_acts(&mut self) -> Vec<UpdateAct> {
        std::mem::take(&mut self.update_acts)
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

    /// Where a page's content starts: every page has the whole column (the rail that showed the page under it is gone, Nick 2026-10-11); its title row goes back.
    fn rail_w(&self) -> f32 {
        0.0
    }

    /// Open the toolbox's list, as a tap on its button does (for a check that renders it).
    pub fn show_toolbox(&mut self) {
        self.toolbox_open = true;
        self.dirty = true;
    }

    /// A toolbox pick into the controls: one tool at a time (a tap's info, the ruler, a way to travel, the pen, the selection), or a one-shot.
    fn pick(&mut self, p: Pick, ctl: &mut Controls) {
        let tool = |ctl: &mut Controls, tool: u8, travel: u8, ruler: bool| {
            ctl.tool = tool;
            ctl.travel = travel;
            ctl.ruler = ruler;
        };
        match p {
            Pick::Info => tool(ctl, 0, 0, false),
            Pick::Ruler => tool(ctl, 0, 0, true),
            Pick::Drive => tool(ctl, 0, 1, false),
            Pick::Ride => tool(ctl, 0, 2, false),
            Pick::Walk => tool(ctl, 0, 3, false),
            Pick::Pen => tool(ctl, 1, 0, false),
            Pick::Select => tool(ctl, 2, 0, false),
            Pick::PenColour => {
                self.open = true;
                let rgba = ctl.pen_rgba;
                self.open_pen(rgba);
            }
            Pick::ClearInk => ctl.clear_ink = true,
            Pick::ClearSel => ctl.clear_sel = true,
        }
    }

    /// Whether a tap's height is on a page's title row ("‹  Title" goes back, Nick 2026-10-14: the pen and field pages had no way back but the phone's own).
    fn on_title(&self, y: f32) -> bool {
        let ty = self.gear.1 + self.gear.2 + self.font * 0.9 + self.font * 0.8;
        (y - ty).abs() <= self.font * 0.8
    }

    /// Open the pen's mixer: four sliders from the pen's bytes.
    fn open_pen(&mut self, rgba: [u8; 4]) {
        self.field_vals = [rgba[0] as f32, rgba[1] as f32, rgba[2] as f32, rgba[3] as f32];
        self.sliders = (0..4).map(|i| Slider::new(&mut self.hits, 0.0, 0.0, 10.0, 10.0, self.field_vals[i] / 255.0)).collect();
        self.stack.push(Page::Pen);
        self.held = None;
        self.dirty = true;
    }

    /// Open a field's page: one slider per value, from the theme's own bytes.
    fn open_field(&mut self, f: Field, vals: [f32; 4]) {
        self.field_vals = vals;
        let n = match f {
            Field::Sun | Field::Sky => 3,
            Field::LabelAmount => 2,
            Field::LabelFont => 1,
            _ => 4,
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
            (Field::Sun | Field::Sky, _) => v / 3.0,
            (Field::LabelLook, _) => (v - LOOK_RANGE[i].0) / (LOOK_RANGE[i].1 - LOOK_RANGE[i].0),
            (Field::LabelAmount, _) => (v - AMOUNT_RANGE[i].0) / (AMOUNT_RANGE[i].1 - AMOUNT_RANGE[i].0),
            (Field::LabelFont, _) => v / (mahere_engine::theme::LABEL_FONTS.len() - 1) as f32,
            _ => v / 255.0,
        }
    }

    fn slider_val(f: Field, i: usize, pos: f32) -> f32 {
        match (f, i) {
            (Field::Hypso(_), 3) => pos * 4000.0,
            (Field::Sun | Field::Sky, _) => pos * 3.0,
            (Field::LabelLook, _) => LOOK_RANGE[i].0 + pos * (LOOK_RANGE[i].1 - LOOK_RANGE[i].0),
            (Field::LabelAmount, _) => AMOUNT_RANGE[i].0 + pos * (AMOUNT_RANGE[i].1 - AMOUNT_RANGE[i].0),
            (Field::LabelFont, _) => (pos * (mahere_engine::theme::LABEL_FONTS.len() - 1) as f32).round(),
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
        self.gear = self.host_orb.unwrap_or((pad + r, pad + r, r));
        let r = self.gear.2;
        self.panel_w = (self.font * 13.0).min(w as f32 * 0.7);
        let row = self.font * 1.7;
        let x0 = self.font * 0.8;
        // The footer: the cache slider and the readouts, fixed at the bottom, the layers scrolling above it.
        let footer_top = (h as f32 - self.font * FOOTER_LINES).max(self.gear.1 + r + row * 3.0);
        let band_top = self.gear.1 + r + self.font * 0.5;
        self.layers_band = (band_top, footer_top);
        let scroll = self.layers_scroll;
        // The theme row sits first, then the layers.
        let theme_y = self.gear.1 + r + self.font * 0.9 + row * 0.5 - scroll;
        self.theme_row = (x0, theme_y, row * 0.5);
        let top = theme_y + row * 0.8;
        let mut last = top;
        for (i, (l, cb)) in self.checks.iter_mut().enumerate() {
            // The modes sit a little apart from the layers, the pages from the modes.
            let gap = match l {
                Layer::RealSun | Layer::RealLight | Layer::Linear | Layer::FollowHeading | Layer::LockToFix | Layer::Record => row * 0.5,
                Layer::Areas | Layer::Updates => row,
                _ => 0.0,
            };
            let cy = top + row * (i as f32 + 0.5) + gap;
            cb.set_font_size(self.font);
            cb.set_rect(x0 + (self.panel_w - x0 * 1.5) * 0.5, cy, self.panel_w - x0 * 1.5, row);
            last = cy + row * 0.5;
        }
        // How far the list runs past the footer, from where it would sit unscrolled.
        self.layers_extent = (last + scroll + row * 0.5 - footer_top).max(0.0);
        if self.layers_scroll > self.layers_extent {
            self.layers_scroll = self.layers_extent;
        }
        // The cache slider at the footer's top: a track the width of the column.
        self.slider = (x0, footer_top + self.font * 1.9, self.panel_w - x0 * 2.0);
    }

    /// Whether a point on the Layers page is on a row the list shows (inside the band it scrolls in).
    fn in_band(&self, y: f32) -> bool {
        y >= self.layers_band.0 && y < self.layers_band.1
    }

    /// A press released where it started on the Layers page: the row under it opens or flips. True when something the host applies changed.
    fn activate_layers_row(&mut self, x: f32, y: f32, mask: &mut LayerMask, ctl: &mut Controls) -> bool {
        if !self.in_band(y) {
            return false;
        }
        let (tx, ty, tr) = self.theme_row;
        if (y - ty).abs() <= tr && x >= tx && x < self.panel_w {
            self.stack.push(Page::Themes);
            self.dirty = true;
            return false;
        }
        for (l, cb) in &mut self.checks {
            if cb.bbox().contains(x, y) {
                if *l == Layer::Areas {
                    self.stack.push(Page::Areas);
                    self.dirty = true;
                    return false;
                }
                if *l == Layer::Updates {
                    self.stack.push(Page::Updates);
                    self.update_acts.push(UpdateAct::Check);
                    self.dirty = true;
                    return false;
                }
                // A row a dominating layer has greyed takes the tap but changes nothing.
                if inert(mask, ctl, *l) {
                    return false;
                }
                let v = !get(mask, ctl, *l);
                set(mask, ctl, *l, v);
                cb.set_checked(get(mask, ctl, *l));
                self.dirty = true;
                return true;
            }
        }
        false
    }

    /// A tap at screen (x, y): true if the panel took it. The orb toggles the panel; a row flips its layer in `mask`.
    pub fn tap(&mut self, x: f32, y: f32, w: usize, h: usize, mask: &mut LayerMask, ctl: &mut Controls, _themes: &[Theme]) -> bool {
        self.layout(w, h);
        let inside = |r: (f32, f32, f32, f32)| x >= r.0 && x <= r.0 + r.2 && y >= r.1 && y <= r.1 + r.3;
        // The toolbox's list takes the next tap: a row picks, anywhere else only closes it.
        if self.toolbox_open {
            self.toolbox_open = false;
            self.dirty = true;
            if let Some(&(_, pick)) = self.tool_rows.iter().find(|(r, _)| inside(*r)) {
                self.pick(pick, ctl);
            }
            return true;
        }
        if self.tool_btn.is_some_and(inside) {
            self.toolbox_open = true;
            self.dirty = true;
            return true;
        }
        if let Some((cx, cy, cr)) = self.strip_close {
            if (x - cx).powi(2) + (y - cy).powi(2) <= (cr * 1.4).powi(2) {
                self.clear_measure = true;
                self.dirty = true;
                return true;
            }
        }
        if let Some((bx, by, br)) = self.locate_btn {
            if (x - bx).powi(2) + (y - by).powi(2) <= (br * 1.3).powi(2) {
                ctl.locate = true;
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
                if self.on_title(y) {
                    self.pop();
                    return true;
                }
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
            Page::Updates => {
                let rows = self.update_rows.clone();
                for (cy, hh, row) in rows {
                    if (y - cy).abs() <= hh {
                        match row {
                            UpdateRow::Back => self.pop(),
                            UpdateRow::Get(i) => {
                                if self.updates.channels[i].2 {
                                    self.update_acts.push(UpdateAct::Get(i));
                                }
                            }
                            UpdateRow::Auto => {
                                self.updates.auto = !self.updates.auto;
                                self.update_acts.push(UpdateAct::Auto(self.updates.auto));
                            }
                        }
                        self.dirty = true;
                        return true;
                    }
                }
                return true;
            }
            Page::Areas => {
                let rows = self.area_rows.clone();
                for (cy, hh, row) in rows {
                    if (y - cy).abs() <= hh {
                        match row {
                            AreaRow::Back => self.pop(),
                            AreaRow::Save => {
                                if self.areas.has_selection {
                                    self.area_acts.push(AreaAct::Save);
                                }
                            }
                            AreaRow::Open(i) => {
                                let depth = self.areas.areas.get(i).map_or(AREA_MAX_DEPTH, |a| a.depth);
                                self.stack.push(Page::Area(i));
                                self.open_area_slider(depth);
                            }
                            _ => {}
                        }
                        self.dirty = true;
                        return true;
                    }
                }
                return true;
            }
            Page::Area(i) => {
                if let Some(sl) = self.sliders.first_mut() {
                    let b = sl.bbox();
                    if y >= b.y - b.h * 0.5 && y <= b.y + b.h * 1.5 && x >= b.x - self.font && x <= b.x + b.w + self.font {
                        sl.set_value_from_x(x);
                        self.held = Some(0);
                        let d = Self::depth_of(sl.value());
                        self.field_vals[0] = d as f32;
                        self.area_acts.push(AreaAct::SetDepth(i, d));
                        self.dirty = true;
                        return true;
                    }
                }
                let rows = self.area_rows.clone();
                for (cy, hh, row) in rows {
                    if (y - cy).abs() <= hh {
                        match row {
                            AreaRow::Back => self.pop(),
                            AreaRow::Download => self.area_acts.push(AreaAct::Download(i)),
                            AreaRow::Cancel => self.area_acts.push(AreaAct::Cancel(i)),
                            AreaRow::Refresh => self.area_acts.push(AreaAct::Refresh(i)),
                            AreaRow::Share => self.area_acts.push(AreaAct::Share(i)),
                            AreaRow::Show => self.area_acts.push(AreaAct::Show(i)),
                            AreaRow::Delete => {
                                self.area_acts.push(AreaAct::Delete(i));
                                self.pop();
                            }
                            _ => {}
                        }
                        self.dirty = true;
                        return true;
                    }
                }
                return true;
            }
            Page::Pen => {
                if self.on_title(y) {
                    self.pop();
                    return true;
                }
                for (i, sl) in self.sliders.iter_mut().enumerate() {
                    let b = sl.bbox();
                    if y >= b.y - b.h * 0.5 && y <= b.y + b.h * 1.5 && x >= b.x - self.font && x <= b.x + b.w + self.font {
                        sl.set_value_from_x(x);
                        self.held = Some(i);
                        self.field_vals[i] = (sl.value() * 255.0).round();
                        ctl.pen_rgba[i] = self.field_vals[i] as u8;
                        self.dirty = true;
                        return true;
                    }
                }
                return true;
            }
            Page::Layers => {}
        }
        // The list takes a press and acts on its release, so a drag scrolls it instead; the footer's slider acts at once.
        if self.in_band(y) {
            self.layers_press = Some((x, y, false));
            return true;
        }
        let _ = mask;
        if self.slider_press(x, y, ctl) {
            return true;
        }
        // Anywhere else on the column is the panel's, not the map's.
        x < self.panel_w
    }

    /// Paint for a `w × h` screen: the orb always (the app orb, or the light as a ball when `orb` gives one) with the app's name, unless the host draws its own; the column when open. Returns the buffer in fluor's pixel convention.
    #[allow(clippy::too_many_arguments)]
    pub fn paint(&mut self, w: usize, h: usize, mask: LayerMask, ctl: Controls, r: &Readouts, measure: Option<&MeasureView>, themes: &[Theme], areas: &AreasView, orb: Option<&dyn Fn(usize) -> Option<Vec<[u8; 4]>>>) -> &[u32] {
        self.layout(w, h);
        // Font packs (Nick 2026-10-16): what arrived loaded and painted; what the last paint's text lacked asked for.
        if let Some(f) = self.fonts.clone() {
            if fonts::load_into(&mut self.text, &f, &mut self.font_cursor) {
                self.dirty = true;
            }
            f.want(self.text.wanted_packs());
        }
        if self.areas != *areas {
            self.areas = areas.clone();
            self.dirty = true;
        }
        // An area gone from under its page: back to the list.
        if let Page::Area(i) = self.page() {
            if i >= self.areas.areas.len() {
                self.pop();
            }
        }
        self.buf.clear();
        self.buf.resize(w * h, 0);
        if w == 0 || h == 0 {
            return &self.buf;
        }
        let font = self.font;
        let (gx, gy, gr) = self.gear;
        let ink_col = theme::TEXTBOX_TEXT;
        // The orb where the settings live: the light as a ball when a mode has one, the app orb otherwise, the app's name beside it. A host that draws its own orb (a desktop's chrome) draws both there instead.
        if self.host_orb.is_none() {
            let size = (gr * 2.0) as usize;
            let ball: Option<Vec<u32>> = orb.and_then(|f| f(size)).map(|img| {
                img.iter().map(|p| if p[3] == 0 { 0 } else { (theme::dark(theme::fmt(((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32)) & 0x00FF_FFFF) | 0xFF00_0000 }).collect()
            });
            let px: Vec<u32> = match ball {
                Some(b) => b,
                None => self.orb_pixels(size).to_vec(),
            };
            let (x0, y0) = ((gx - gr) as isize, (gy - gr) as isize);
            let rr = gr * gr;
            if px.len() == size * size {
                for py in 0..size {
                    for qx in 0..size {
                        let (dx, dy) = (qx as f32 + 0.5 - gr, py as f32 + 0.5 - gr);
                        let p = px[py * size + qx];
                        if dx * dx + dy * dy > rr || p == 0 {
                            continue;
                        }
                        let (x, y) = (x0 + qx as isize, y0 + py as isize);
                        if x < 0 || y < 0 || x as usize >= w || y as usize >= h {
                            continue;
                        }
                        self.buf[y as usize * w + x as usize] = p;
                    }
                }
            }
            let mut damage = Damage::new();
            let mut canvas = Canvas::new(&mut self.buf, w, h, &mut damage);
            // The name, level with the orb's centre; topmost first, so the text, then a dark edge under it to read over any map.
            let style = TextStyle::new(font * 1.05, ink_col);
            let shade = TextStyle::new(font * 1.05, GEAR_BG);
            let (nx, ny) = (gx + gr + font * 0.5, gy);
            self.text.draw_text_left(&mut canvas, APP_NAME, nx, ny, &style, None, None);
            for (ox, oy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (1.0, 1.0)] {
                self.text.draw_text_left(&mut canvas, APP_NAME, nx + ox, ny + oy, &shade, None, None);
            }
            paint::circle_filled(&mut canvas, gx as isize, gy as isize, gr as isize, GEAR_BG, None, None);
        }
        let (page, rail) = (self.page(), self.rail_w());
        let mut damage = Damage::new();
        let mut canvas = Canvas::new(&mut self.buf, w, h, &mut damage);
        self.strip_close = None;
        let mut bottom = h as f32;
        if let Some(m) = measure {
            self.strip_close = Some(Self::paint_strip(&mut canvas, &mut self.text, w, h, font, m));
            bottom -= Self::strip_band(font, m);
        }
        // The locate button, bottom right above any strip; dim without a fix.
        let br = gr * 0.8;
        let (bx, by) = (w as f32 - br - font * 0.6, bottom - br - font * 0.6);
        Self::paint_locate(&mut canvas, bx, by, br, r.has_fix);
        self.locate_btn = Some((bx, by, br));
        // The toolbox, left of the locate button: the tool on, and its list above when open.
        let (tw, th) = (font * 5.2, font * 1.5);
        let btn = (bx - br - font * 0.5 - tw, by - th * 0.5, tw, th);
        self.tool_btn = Some(btn);
        self.tool_rows.clear();
        let current = Pick::current(&ctl);
        if self.toolbox_open {
            let rows = Pick::list(&ctl);
            let (rw, rh, gap) = (font * 7.0, font * 1.45, font * 0.2);
            let rx = btn.0 + btn.2 - rw;
            let mut ry = btn.1 - gap - rh;
            for p in rows.into_iter().rev() {
                let rect = (rx, ry, rw, rh);
                Self::paint_pill(&mut canvas, &mut self.text, rect, p.word(), p == current, font);
                self.tool_rows.push((rect, p));
                ry -= rh + gap;
            }
        }
        Self::paint_pill(&mut canvas, &mut self.text, btn, current.word(), current != Pick::Info, font);
        if !self.open {
            // The scale bar, bottom left; the open panel shows the scale itself.
            Self::paint_scale(&mut canvas, &mut self.text, font, font * 0.8, bottom - font * 0.8, r.m_per_px, (w as f32 * 0.3).min(font * 10.0));
            return &self.buf;
        }
        if page != Page::Layers {
            // Top first: the content page, then the column's background.
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
                Page::Pen => Self::paint_pen(&mut canvas, &mut self.text, font, rail, cw, self.gear, &mut self.sliders, self.field_vals),
                Page::Areas => self.area_rows = Self::paint_areas(&mut canvas, &mut self.text, font, rail, cw, self.gear, &self.areas),
                Page::Updates => self.update_rows = Self::paint_updates(&mut canvas, &mut self.text, font, rail, cw, self.gear, &self.updates),
                Page::Area(i) => self.area_rows = Self::paint_area(&mut canvas, &mut self.text, font, rail, cw, self.gear, &self.areas, i, &mut self.sliders, self.field_vals[0] as u8),
                Page::Layers => {}
            }
            paint::fill_rect(&mut canvas, 0, 0, self.panel_w as isize, h as isize, PANEL_BG, None, None);
            return &self.buf;
        }
        // Topmost first: the footer (its slider and readouts, then its own background), then the list clipped to its band, then the column's background.
        let (band_top, footer_top) = self.layers_band;
        self.cache_max = r.cache_max;
        Self::paint_slider(&mut canvas, &mut self.text, font, self.slider, &ctl, r);
        let x = font * 0.9;
        let right = self.panel_w - font * 0.9;
        let mut y = self.slider.1 + font * 1.4;
        let small = TextStyle::new(font * 0.82, READOUT);
        let dim = TextStyle::new(font * 0.72, READOUT_DIM);
        let rows: [(&str, String); 7] = [
            ("position", format!("{:.5}  {:.5}", r.lat, r.lon)),
            ("elevation", r.elev.map_or("\u{2014}".into(), |e| format!("{} m", e.round() as i64))),
            ("GPS height", r.gps_height.map_or("\u{2014}".into(), |(h, v)| match v {
                Some(v) => format!("{} m  \u{b1}{}", h.round() as i64, v.round() as i64),
                None => format!("{} m", h.round() as i64),
            })),
            ("heading", format!("{:+}\u{b0}", r.heading_deg.round() as i64)),
            ("phone", r.phone_heading.map_or("\u{2014}".into(), |h| format!("{}\u{b0}", h.round() as i64))),
            ("scale", format!("{:.2} m/px", r.m_per_px)),
            ("frame", format!("{:.1} ms  \u{b7}  {} cells", r.frame_ms, r.resident)),
        ];
        let line = font * 1.15;
        for (label, value) in rows.iter() {
            if y > h as f32 {
                break;
            }
            self.text.draw_text_left(&mut canvas, label, x, y, &dim, None, None);
            self.text.draw_text_right(&mut canvas, value, right, y, &small, None, None);
            y += line;
        }
        // The world code is seven words: under its label, wrapped to the column so none is cut.
        let words = vsf::types::WorldCoord::from_lat_lon(r.lat, r.lon).to_words();
        let max_w = self.panel_w - x * 2.0;
        let mut world_lines: Vec<String> = Vec::new();
        for word in words.split_whitespace() {
            match world_lines.last_mut() {
                Some(l) if self.text.measure_text(&format!("{l} {word}"), &small) <= max_w => {
                    l.push(' ');
                    l.push_str(word);
                }
                _ => world_lines.push(word.to_string()),
            }
        }
        self.text.draw_text_left(&mut canvas, "world", x, y, &dim, None, None);
        y += font * 0.9;
        for l in &world_lines {
            if y > h as f32 {
                break;
            }
            self.text.draw_text_left(&mut canvas, l, x, y, &small, None, None);
            y += font * 1.05;
        }
        // The footer's own ground and a hairline along its top, so it reads apart from the list.
        paint::fill_rect(&mut canvas, 0, footer_top as isize, self.panel_w as isize, 1.max((font / 24.0) as isize), READOUT_DIM, None, None);
        paint::fill_rect(&mut canvas, 0, footer_top as isize, self.panel_w as isize, (h as f32 - footer_top) as isize, FOOTER_BG, None, None);
        let clip = Some(paint::Clip::new(0, band_top.max(0.0) as usize, self.panel_w as usize, footer_top.max(0.0) as usize));
        // Where the list runs past the footer, a thin bar at the column's edge shows where in it the view is.
        if self.layers_extent > 0.0 {
            let span = footer_top - band_top;
            let len = span * span / (span + self.layers_extent);
            let at = band_top + (span - len) * (self.layers_scroll / self.layers_extent);
            paint::fill_rect(&mut canvas, (self.panel_w - font * 0.25) as isize, at as isize, (font * 0.12).max(2.0) as isize, len as isize, READOUT_DIM, None, None);
        }
        // The theme row: the current theme's name and a chevron.
        let (tx, ty, _) = self.theme_row;
        let current = themes.get(ctl.theme).map_or("Trail", |t| t.name.as_str());
        self.text.draw_text_left(&mut canvas, &format!("Theme   {current}  \u{203a}"), tx, ty, &TextStyle::new(font, theme::TEXTBOX_TEXT), clip, None);
        for (l, cb) in &mut self.checks {
            cb.set_checked(get(&mask, &ctl, *l));
            let b = cb.bbox();
            if b.y + b.h < band_top || b.y > footer_top {
                continue;
            }
            // Greyed: a wash of the panel colour over the row, painted first so it lies on top.
            if inert(&mask, &ctl, *l) {
                paint::fill_rect(&mut canvas, b.x as isize, b.y as isize, b.w as isize, b.h as isize, PANEL_DIM, clip, None);
            }
            cb.render_content_into(&mut canvas, &mut self.text, clip, None);
        }
        paint::fill_rect(&mut canvas, 0, 0, self.panel_w as isize, h as isize, PANEL_BG, None, None);
        &self.buf
    }

    /// One painted pixel as premultiplied visible (r, g, b, a): fluor stores accumulated darkness with `dark ≤ α`, so the premultiplied channel is `α − dark`; Android's theme bytes are R↔B swapped.
    #[inline]
    /// A pixel of fluor's (α + darkness) as premultiplied RGBA in this platform's channel order.
    pub fn premul_pub(p: u32) -> (u8, u8, u8, u8) {
        Self::premul(p)
    }

    /// The panel's base font size, pixels, for anything lettered to match it.
    pub fn font_px(&self) -> f32 {
        self.font
    }

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
    /// The measurement strip's height: what is at the point, a line each, above the distance and the profile.
    fn strip_band(font: f32, m: &MeasureView) -> f32 {
        if !m.ruler {
            // The card: what is there, a line each, and where.
            return font * 2.2 + (m.info.len() + 1) as f32 * font * 1.05;
        }
        font * 6.5 + m.info.len() as f32 * font * 1.05
    }

    /// The scale bar: the longest round distance (1, 2 or 5 of a power of ten metres) that fits in `max_px`, its length in metres or kilometres over it; text first, then the bar on a dark edge, as everything here paints top first.
    #[allow(clippy::too_many_arguments)]
    fn paint_scale(canvas: &mut Canvas, text: &mut TextRenderer, font: f32, x0: f32, y: f32, m_per_px: f64, max_px: f32) {
        if !(m_per_px > 0.0) {
            return;
        }
        let most = max_px as f64 * m_per_px;
        let mut d = 10f64.powf(most.log10().floor());
        for k in [5.0, 2.0] {
            if d * k <= most {
                d *= k;
                break;
            }
        }
        let px = (d / m_per_px) as f32;
        let label = if d >= 1000.0 { format!("{} km", d / 1000.0) } else if d >= 1.0 { format!("{} m", d) } else { format!("{} cm", d * 100.0) };
        let t = (font * 0.16).max(2.0);
        let style = TextStyle::new(font * 0.75, theme::TEXTBOX_TEXT);
        let shade = TextStyle::new(font * 0.75, GEAR_BG);
        let ty = y - font * 0.55;
        text.draw_text_left(canvas, &label, x0, ty, &style, None, None);
        for (ox, oy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
            text.draw_text_left(canvas, &label, x0 + ox, ty + oy, &shade, None, None);
        }
        // The bar and its end ticks, then the dark edge under them.
        let (xi, yi, pi, ti) = (x0 as isize, y as isize, px as isize, t as isize);
        let tick = (font * 0.35) as isize;
        paint::fill_rect(canvas, xi, yi - ti / 2, pi, ti, theme::TEXTBOX_TEXT, None, None);
        paint::fill_rect(canvas, xi, yi - tick, ti, tick + ti / 2, theme::TEXTBOX_TEXT, None, None);
        paint::fill_rect(canvas, xi + pi - ti, yi - tick, ti, tick + ti / 2, theme::TEXTBOX_TEXT, None, None);
        paint::fill_rect(canvas, xi - 2, yi - tick - 2, pi + 4, tick + ti / 2 + 4, GEAR_BG, None, None);
    }

    /// A rounded button with its word centred, lit when it is the one on; text first, then the body under it.
    fn paint_pill(canvas: &mut Canvas, text: &mut TextRenderer, r: (f32, f32, f32, f32), word: &str, lit: bool, font: f32) {
        let (x, y, w, h) = r;
        let ink = if lit { theme::TEXTBOX_TEXT } else { READOUT };
        text.draw_text_center(canvas, word, x + w * 0.5, y + h * 0.5, &TextStyle::new(font * 0.8, ink), None, None);
        let rad = (h * 0.5) as isize;
        let bg = if lit { SLIDER_USED } else { GEAR_BG };
        paint::fill_rect(canvas, (x + h * 0.5) as isize, y as isize, (w - h) as isize, h as isize, bg, None, None);
        paint::circle_filled(canvas, (x + h * 0.5) as isize, (y + h * 0.5) as isize, rad, bg, None, None);
        paint::circle_filled(canvas, (x + w - h * 0.5) as isize, (y + h * 0.5) as isize, rad, bg, None, None);
    }

    /// The locate button: a crosshair (four arms and a dot) on the gear's dark disc, dim without a fix.
    fn paint_locate(canvas: &mut Canvas, x: f32, y: f32, r: f32, live: bool) {
        let ink = if live { theme::TEXTBOX_TEXT } else { READOUT_DIM };
        let (xi, yi) = (x as isize, y as isize);
        let t = (r * 0.12).max(2.0) as isize;
        let (inner, outer) = ((r * 0.32) as isize, (r * 0.72) as isize);
        paint::circle_filled(canvas, xi, yi, (r * 0.14).max(2.0) as isize, ink, None, None);
        paint::fill_rect(canvas, xi - outer, yi - t / 2, outer - inner, t, ink, None, None);
        paint::fill_rect(canvas, xi + inner, yi - t / 2, outer - inner, t, ink, None, None);
        paint::fill_rect(canvas, xi - t / 2, yi - outer, t, outer - inner, ink, None, None);
        paint::fill_rect(canvas, xi - t / 2, yi + inner, t, outer - inner, ink, None, None);
        paint::circle_filled(canvas, xi, yi, r as isize, GEAR_BG, None, None);
    }

    fn paint_strip(canvas: &mut Canvas, text: &mut TextRenderer, w: usize, h: usize, font: f32, m: &MeasureView) -> (f32, f32, f32) {
        // What is at the point, a line each, above the distance and the profile; the strip grows to hold them.
        let info_h = m.info.len() as f32 * font * 1.05;
        let band = Self::strip_band(font, m);
        let strip_top = h as f32 - band;
        let margin = font;
        // The close button, top right of the strip: a cross in a disc.
        let (cx, cy, cr) = (w as f32 - font * 0.9, strip_top + font * 0.9, font * 0.55);
        text.draw_text_center(canvas, "\u{00d7}", cx, cy, &TextStyle::new(font * 1.1, READOUT), None, None);
        paint::circle_filled(canvas, cx as isize, cy as isize, cr as isize, GEAR_BG, None, None);
        let point = TextStyle::new(font * 0.9, theme::TEXTBOX_TEXT);
        for (k, line) in m.info.iter().enumerate() {
            text.draw_text_left(canvas, line, margin, strip_top + font * 0.9 + k as f32 * font * 1.05, &point, None, None);
        }
        if !m.ruler {
            // A tap's card (Nick 2026-10-14: a little info at the bottom per tap): the lines, then the place in degrees.
            let (lat, lon) = m.target;
            let at = format!("{:.5}\u{b0} {}  {:.5}\u{b0} {}", lat.abs(), if lat >= 0.0 { "N" } else { "S" }, lon.abs(), if lon >= 0.0 { "E" } else { "W" });
            text.draw_text_left(canvas, &at, margin, strip_top + font * 0.9 + m.info.len() as f32 * font * 1.05, &TextStyle::new(font * 0.8, READOUT), None, None);
            paint::fill_rect(canvas, 0, strip_top as isize, w as isize, band as isize, PANEL_BG, None, None);
            return (cx, cy, cr);
        }
        let top = strip_top + info_h;
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
        paint::fill_rect(canvas, 0, strip_top as isize, w as isize, band as isize, PANEL_BG, None, None);
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
        if let (Some((px, py, moved)), Page::Layers) = (self.layers_press, self.page()) {
            let moved = moved || (y - py).abs() + (x - px).abs() > 12.0;
            if moved {
                self.layers_scroll = (self.layers_scroll - (y - py)).clamp(0.0, self.layers_extent);
                self.dirty = true;
            }
            self.layers_press = Some((x, y, moved));
            return true;
        }
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
        if let (Some(0), Page::Area(i)) = (self.held, self.page()) {
            if let Some(sl) = self.sliders.first_mut() {
                sl.set_value_from_x(x);
                let d = Self::depth_of(sl.value());
                if self.field_vals[0] != d as f32 {
                    self.field_vals[0] = d as f32;
                    self.area_acts.push(AreaAct::SetDepth(i, d));
                }
                self.dirty = true;
            }
            return true;
        }
        if let (Some(i), Page::Pen) = (self.held, self.page()) {
            if let Some(sl) = self.sliders.get_mut(i) {
                sl.set_value_from_x(x);
                self.field_vals[i] = (sl.value() * 255.0).round();
                ctl.pen_rgba[i] = self.field_vals[i] as u8;
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

    /// The finger lifted. A press on the Layers list that never moved opens or flips its row there; true when that changed something the host applies from `mask` and `ctl`.
    pub fn release(&mut self, mask: &mut LayerMask, ctl: &mut Controls) -> bool {
        self.slider_held = false;
        self.held = None;
        let mut changed = false;
        if let Some((x, y, moved)) = self.layers_press.take() {
            if !moved && self.page() == Page::Layers {
                changed = self.activate_layers_row(x, y, mask, ctl);
            }
        }
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
        changed
    }

    fn paint_slider(canvas: &mut Canvas, text: &mut TextRenderer, font: f32, slider: (f32, f32, f32), ctl: &Controls, r: &Readouts) {
        let (sx, sy, sw) = slider;
        let max = r.cache_max.max(CACHE_MIN + 1);
        let t = ((ctl.cache_budget.saturating_sub(CACHE_MIN)) as f64 / (max - CACHE_MIN) as f64).clamp(0.0, 1.0) as f32;
        let used = (r.cache_used as f64 / max as f64).clamp(0.0, 1.0) as f32;
        let label = format!("cache  {}  \u{b7}  {} held", bytes_text(ctl.cache_budget), bytes_text(r.cache_used));
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
        text.draw_text_left(canvas, "\u{2039}  Themes", x0, y, &title, None, None);
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
                Field::Own(_) | Field::Wild => "land ownership",
                Field::Mvum(_) => "motor vehicle use",
                Field::Water | Field::Contour | Field::ContourIndex => "inks",
                Field::Sun | Field::Sky => "light",
                Field::Label(_) | Field::LabelLook | Field::LabelAmount | Field::LabelFont => "labels",
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
        let mut actions: Vec<(&str, ThemeEdit)> = vec![("\u{29c9}  Duplicate", ThemeEdit::Duplicate), ("\u{25cb}  Capture light", ThemeEdit::CaptureLight)];
        if t.probe.is_some() {
            actions.push(("\u{2715}  Clear light", ThemeEdit::ClearLight));
        }
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
        if matches!(f, Field::LabelLook | Field::LabelAmount | Field::LabelFont) {
            // No swatch: the look is numbers, shown beside their sliders.
        } else if f.is_colour() {
            paint::fill_rect(canvas, x0 as isize, y as isize, sw as isize, (font * 1.6) as isize, swatch([vals[0] as u8, vals[1] as u8, vals[2] as u8]), None, None);
        } else {
            let c = mahere_engine::colour::Display::default().encode([vals[0] * 0.5, vals[1] * 0.5, vals[2] * 0.5], true);
            paint::fill_rect(canvas, x0 as isize, y as isize, sw as isize, (font * 1.6) as isize, (theme::dark(theme::fmt(((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32)) & 0x00FF_FFFF) | 0xFF00_0000, None, None);
        }
        y += font * 2.4;
        let labels: [&str; 4] = match f {
            Field::Hypso(_) => ["R", "G", "B", "metres"],
            Field::LabelLook => ["Size", "Glow", "Smoothing", "Spacing"],
            Field::LabelAmount => ["Density", "Repeat", "", ""],
            Field::LabelFont => ["Typeface", "", "", ""],
            _ => ["R", "G", "B", "A"],
        };
        for (i, sl) in sliders.iter_mut().enumerate() {
            text.draw_text_left(canvas, labels[i], x0, y, &dim, None, None);
            let val = match (f, i) {
                (Field::Hypso(_), 3) => format!("{} m", vals[3].round() as i64),
                (Field::Sun | Field::Sky, _) => format!("{:.2}", vals[i]),
                (Field::LabelLook, 0) => format!("{:.2}\u{d7}", vals[i]),
                (Field::LabelLook, 2) => format!("{:.2} letters", vals[i]),
                (Field::LabelLook, _) => format!("{:.2} of a letter", vals[i]),
                (Field::LabelAmount, 0) => format!("to {}", mahere_engine::theme::LABEL_TIERS[mahere_engine::theme::label_tier_limit(vals[0]) as usize]),
                (Field::LabelAmount, _) => format!("every {:.2} screen", vals[i]),
                (Field::LabelFont, _) => mahere_engine::theme::LABEL_FONTS[(vals[0].round().max(0.0) as usize).min(mahere_engine::theme::LABEL_FONTS.len() - 1)].to_string(),
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

impl Panel {
    /// The areas list: the back row, a row to save the selection (dim without one), then every area with what is on the device.
    /// The Updates page: this build, then each channel's row (a tap gets its build, when it is not this one), the line of what is happening and a download's bar, and the automatic switch.
    fn paint_updates(canvas: &mut Canvas, text: &mut TextRenderer, font: f32, ox: f32, cw: f32, gear: (f32, f32, f32), u: &UpdatesView) -> Vec<(f32, f32, UpdateRow)> {
        let x0 = ox + font * 0.9;
        let row = font * 1.6;
        let mut y = gear.1 + gear.2 + font * 0.9 + row * 0.5;
        let title = TextStyle::new(font, theme::TEXTBOX_TEXT);
        let dim = TextStyle::new(font, READOUT_DIM);
        let small = TextStyle::new(font * 0.75, READOUT_DIM);
        let mut rows = Vec::new();
        text.draw_text_left(canvas, "\u{2039}  Updates", x0, y, &title, None, None);
        rows.push((y, row * 0.5, UpdateRow::Back));
        y += row * 1.1;
        text.draw_text_left(canvas, &format!("This build  {}", u.this), x0, y, &small, None, None);
        y += row * 1.1;
        for (i, (label, detail, live)) in u.channels.iter().enumerate() {
            text.draw_text_left(canvas, label, x0, y, if *live { &title } else { &dim }, None, None);
            text.draw_text_left(canvas, detail, x0 + font * 0.4, y + font * 0.95, &small, None, None);
            rows.push((y + font * 0.4, row * 0.75, UpdateRow::Get(i)));
            y += row * 1.5;
        }
        if !u.status.is_empty() {
            text.draw_text_left(canvas, &u.status, x0, y, &small, None, None);
            y += row * 0.8;
        }
        if let Some(p) = u.progress {
            let w = cw - font * 1.8;
            paint::fill_rect(canvas, x0 as isize, y as isize, (w * p.clamp(0.0, 1.0)) as isize, (font * 0.35) as isize, SLIDER_USED, None, None);
            paint::fill_rect(canvas, x0 as isize, y as isize, w as isize, (font * 0.35) as isize, SLIDER_TRACK, None, None);
            y += row * 0.8;
        }
        y += row * 0.4;
        text.draw_text_left(canvas, &format!("{}  Automatic updates", if u.auto { "\u{2611}" } else { "\u{2610}" }), x0, y, &title, None, None);
        text.draw_text_left(canvas, "release builds, on Wi-Fi", x0 + font * 0.4, y + font * 0.95, &small, None, None);
        rows.push((y, row * 0.5, UpdateRow::Auto));
        rows
    }

    fn paint_areas(canvas: &mut Canvas, text: &mut TextRenderer, font: f32, ox: f32, cw: f32, gear: (f32, f32, f32), areas: &AreasView) -> Vec<(f32, f32, AreaRow)> {
        let x0 = ox + font * 0.9;
        let right = ox + cw - font * 0.9;
        let row = font * 1.6;
        let mut y = gear.1 + gear.2 + font * 0.9 + row * 0.5;
        let title = TextStyle::new(font, theme::TEXTBOX_TEXT);
        let dim = TextStyle::new(font, READOUT_DIM);
        let small = TextStyle::new(font * 0.75, READOUT_DIM);
        let mut rows = Vec::new();
        text.draw_text_left(canvas, "\u{2039}  Areas", x0, y, &title, None, None);
        rows.push((y, row * 0.5, AreaRow::Back));
        y += row * 1.2;
        text.draw_text_left(canvas, "+  Save selection", x0, y, if areas.has_selection { &title } else { &dim }, None, None);
        rows.push((y, row * 0.5, AreaRow::Save));
        y += row * 1.2;
        // Each area two lines: its name, then what is on the device, small.
        for (i, a) in areas.areas.iter().enumerate() {
            text.draw_text_left(canvas, &a.name, x0, y, &title, None, None);
            let status = match (a.progress, a.fetched) {
                (Some((done, total, _)), _) if total > 0 => format!("downloading  {}%", done * 100 / total),
                (_, Some(b)) => format!("{} on this device", bytes_text(b)),
                _ => "not downloaded".to_string(),
            };
            text.draw_text_left(canvas, &status, x0 + font * 0.4, y + font * 0.95, &small, None, None);
            rows.push((y + font * 0.4, row * 0.75, AreaRow::Open(i)));
            y += row * 1.5;
        }
        let _ = right;
        rows
    }

    /// One area: its detail on a slider (texel size, cells and an estimate, or what is on the device), a download's progress, then its actions.
    #[allow(clippy::too_many_arguments)]
    fn paint_area(canvas: &mut Canvas, text: &mut TextRenderer, font: f32, ox: f32, cw: f32, gear: (f32, f32, f32), areas: &AreasView, i: usize, sliders: &mut [Slider], depth: u8) -> Vec<(f32, f32, AreaRow)> {
        let x0 = ox + font * 0.9;
        let sw = cw - font * 1.8;
        let row = font * 1.6;
        let mut y = gear.1 + gear.2 + font * 0.9 + row * 0.5;
        let title = TextStyle::new(font, theme::TEXTBOX_TEXT);
        let small = TextStyle::new(font * 0.8, READOUT);
        let dim = TextStyle::new(font * 0.72, READOUT_DIM);
        let mut rows = Vec::new();
        let Some(a) = areas.areas.get(i) else { return rows };
        text.draw_text_left(canvas, &format!("\u{2039}  {}", a.name), x0, y, &title, None, None);
        rows.push((y, row * 0.5, AreaRow::Back));
        y += row * 1.4;
        // The detail: the texel's edge at the depth, the depth itself small beside it.
        let m = mahere_tiles::texel_m(depth);
        let edge = if m >= 10.0 { format!("{m:.0} m") } else { format!("{m:.1} m") };
        text.draw_text_left(canvas, "Detail", x0, y, &dim, None, None);
        text.draw_text_left(canvas, &format!("{edge} texels"), x0 + font * 3.2, y, &small, None, None);
        y += font * 0.9;
        if let Some(sl) = sliders.first_mut() {
            sl.set_rect(x0 + sw * 0.5, y, sw, font * 1.1);
            sl.render_content_into(canvas, None, HIT_NONE);
        }
        y += font * 1.6;
        // Two short lines: the cells, then the bytes.
        let (cells, bytes) = match (a.progress, a.fetched) {
            (Some((done, total, bytes)), _) => (format!("{done} of {total} cells"), format!("{} so far", bytes_text(bytes))),
            (None, Some(b)) => (format!("{} cells  \u{b7}  depth {depth}", a.cells), format!("{} on this device", bytes_text(b))),
            (None, None) => (format!("{} cells  \u{b7}  depth {depth}", a.cells), format!("about {}", bytes_text(a.est_bytes))),
        };
        text.draw_text_left(canvas, &cells, x0, y, &small, None, None);
        y += font * 0.9;
        text.draw_text_left(canvas, &bytes, x0, y, &small, None, None);
        y += font * 0.9;
        // A download's progress as a bar.
        if let Some((done, total, _)) = a.progress {
            let th = (font * 0.28).max(3.0);
            let t = if total > 0 { done as f32 / total as f32 } else { 0.0 };
            paint::fill_rect(canvas, x0 as isize, (y - th * 0.5) as isize, (sw * t) as isize, th as isize, SLIDER_FILL, None, None);
            paint::fill_rect(canvas, x0 as isize, (y - th * 0.5) as isize, sw as isize, th as isize, SLIDER_TRACK, None, None);
            y += font * 0.9;
        }
        y += row * 0.6;
        let actions: Vec<(&str, AreaRow)> = if a.progress.is_some() {
            vec![("\u{25a0}  Cancel", AreaRow::Cancel)]
        } else if a.fetched.is_some() {
            vec![("Download again", AreaRow::Refresh)]
        } else {
            vec![("Download", AreaRow::Download)]
        };
        for (label, r) in actions.into_iter().chain([("\u{203a}  Share", AreaRow::Share), ("\u{25cb}  Show selection", AreaRow::Show), ("\u{2715}  Delete", AreaRow::Delete)]) {
            text.draw_text_left(canvas, label, x0, y, &title, None, None);
            rows.push((y, row * 0.5, r));
            y += row;
        }
        rows
    }

    /// The pen's mixer: the colour as a swatch, then red, green, blue and opacity sliders.
    fn paint_pen(canvas: &mut Canvas, text: &mut TextRenderer, font: f32, ox: f32, cw: f32, gear: (f32, f32, f32), sliders: &mut [Slider], vals: [f32; 4]) {
        let x0 = ox + font * 0.9;
        let mut y = gear.1 + gear.2 + font * 0.9 + font * 0.8;
        let title = TextStyle::new(font, theme::TEXTBOX_TEXT);
        let small = TextStyle::new(font * 0.8, READOUT);
        let dim = TextStyle::new(font * 0.72, READOUT_DIM);
        text.draw_text_left(canvas, "\u{2039}  Pen colour", x0, y, &title, None, None);
        y += font * 1.6;
        // The swatch: the colour at its opacity over the panel, as a stroke would sit over a dark ground.
        let sw = cw - font * 1.8;
        let blended = pen_swatch([vals[0] as u8, vals[1] as u8, vals[2] as u8, vals[3] as u8]);
        paint::fill_rect(canvas, x0 as isize, y as isize, sw as isize, (font * 1.6) as isize, blended, None, None);
        y += font * 2.4;
        let labels = ["R", "G", "B", "A"];
        for (i, sl) in sliders.iter_mut().enumerate() {
            text.draw_text_left(canvas, labels[i], x0, y, &dim, None, None);
            text.draw_text_left(canvas, &format!("{}", vals[i] as u8), x0 + font * 4.5, y, &small, None, None);
            y += font * 0.9;
            sl.set_rect(x0 + sw * 0.5, y, sw, font * 1.1);
            sl.render_content_into(canvas, None, HIT_NONE);
            y += font * 1.7;
        }
    }
}

/// The pen's colour as a swatch: the opaque swatch mixed over the panel by the pen's opacity, so a faint pen shows faint.
fn pen_swatch(rgba: [u8; 4]) -> u32 {
    let full = swatch([rgba[0], rgba[1], rgba[2]]);
    let a = rgba[3] as f32 / 255.0;
    let mix = |shift: u32| -> u32 {
        let s = ((full >> shift) & 255) as f32;
        let bg = ((PANEL_BG >> shift) & 255) as f32;
        ((bg + (s - bg) * a).round() as u32) << shift
    };
    0xFF00_0000 | mix(16) | mix(8) | mix(0)
}

/// A legend swatch of an authored theme colour, as the map shows it unlit (through the same display encode), opaque, in fluor's stored form.
fn swatch(c: [u8; 3]) -> u32 {
    let c = mahere_engine::colour::Display::default().swatch(c);
    (theme::dark(theme::fmt(((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32)) & 0x00FF_FFFF) | 0xFF00_0000
}
