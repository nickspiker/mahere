//! mahere desktop app — fluor host, CPU rendering.
//!
//! v0.2: real Washington roads. Loads highways from a Geofabrik extract
//! through the mahere-osm boundary (every point codec-quantized), rasterizes
//! them with anti-aliased distance-field strokes, class-styled, with
//! drag-pan and wheel-zoom.

use fluor::coord::Coord as Px;
use fluor::event::{CursorIcon, ElementState, Event as FEvent, MouseButton, MouseScrollDelta};
use fluor::host::app::{Context, EventResponse, FluorApp, run_app};
use fluor::paint::pack_argb;
use mahere_osm::{CLASS_COUNT, Road, RoadClass};

/// Visible RGB per class, tuned for the dark background. Trails green — this
/// is a trail app.
const CLASS_RGB: [[u8; 3]; CLASS_COUNT] = [
    [235, 145, 60],  // Motorway
    [228, 170, 62],  // Trunk
    [230, 202, 82],  // Primary
    [202, 202, 160], // Secondary
    [172, 182, 172], // Tertiary
    [122, 127, 138], // Residential
    [96, 101, 112],  // Service
    [142, 112, 82],  // Track
    [92, 200, 122],  // Path
];

/// Stroke half-width in pixels at the reference zoom (PPD_REF), per class.
const CLASS_HALF_W: [f32; CLASS_COUNT] = [1.6, 1.4, 1.2, 1.0, 0.85, 0.6, 0.45, 0.45, 0.5];

/// Minimum pixels-per-degree(lat) at which each class appears. Majors always;
/// trails only once zoomed near city scale.
const CLASS_MIN_PPD: [f64; CLASS_COUNT] =
    [0., 0., 0., 700., 700., 2500., 2500., 2500., 2500.];

/// Reference zoom for stroke widths: ~18.5 m/px.
const PPD_REF: f64 = 6000.;

const BG_RGB: u32 = 0x12141A;

/// A road prepared for drawing: points relative to a fixed local origin, in
/// degrees, plus a bbox for culling. f32 degrees lose ~0.4 m absolute, so
/// points are stored as offsets from the dataset's bbox center instead —
/// sub-centimeter at state scale.
struct PreparedRoad {
    class: RoadClass,
    pts: Vec<(f32, f32)>, // (dlat, dlon) from origin
    bbox: (f32, f32, f32, f32), // min dlat, min dlon, max dlat, max dlon
}

struct MahereApp {
    roads: Vec<PreparedRoad>,
    origin: (f64, f64), // (lat, lon) all road points are relative to
    /// View center as offsets from origin, degrees.
    center: (f64, f64),
    /// Zoom: pixels per degree of latitude.
    ppd: f64,
    canvas: Vec<u32>, // visible 0xRRGGBB working buffer
    canvas_w: usize,
    canvas_h: usize,
    dirty: bool,
    dragging: bool,
    last_cursor: (f64, f64),
}

impl MahereApp {
    fn new(roads: Vec<Road>, origin: (f64, f64), start: (f64, f64)) -> Self {
        // Sort minor-first so majors draw on top, then prepare.
        let mut roads: Vec<&Road> = roads.iter().collect();
        roads.sort_by(|a, b| b.class.cmp(&a.class));
        let prepared = roads
            .iter()
            .map(|r| {
                let pts: Vec<(f32, f32)> = r
                    .pts
                    .iter()
                    .map(|&(lat, lon)| {
                        ((lat as f64 - origin.0) as f32, (lon as f64 - origin.1) as f32)
                    })
                    .collect();
                let mut bbox = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                for &(a, o) in &pts {
                    bbox.0 = bbox.0.min(a);
                    bbox.1 = bbox.1.min(o);
                    bbox.2 = bbox.2.max(a);
                    bbox.3 = bbox.3.max(o);
                }
                PreparedRoad { class: r.class, pts, bbox }
            })
            .collect();
        MahereApp {
            roads: prepared,
            origin,
            center: (start.0 - origin.0, start.1 - origin.1),
            ppd: PPD_REF,
            canvas: Vec::new(),
            canvas_w: 0,
            canvas_h: 0,
            dirty: true,
            dragging: false,
            last_cursor: (0., 0.),
        }
    }

    /// Longitude compression at the view center.
    fn coslat(&self) -> f64 {
        (self.origin.0 + self.center.0).to_radians().cos()
    }

    fn redraw(&mut self, w: usize, h: usize) {
        self.canvas.clear();
        self.canvas.resize(w * h, BG_RGB);
        self.canvas_w = w;
        self.canvas_h = h;
        if w == 0 || h == 0 {
            return;
        }
        let ppd = self.ppd;
        let ppd_lon = ppd * self.coslat();
        let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
        // Viewport in origin-relative degrees, padded a stroke's worth.
        let vlat0 = (self.center.0 - cy / ppd) as f32;
        let vlat1 = (self.center.0 + cy / ppd) as f32;
        let vlon0 = (self.center.1 - cx / ppd_lon) as f32;
        let vlon1 = (self.center.1 + cx / ppd_lon) as f32;
        let wscale = (ppd / PPD_REF).clamp(0.35, 3.0) as f32;

        for road in &self.roads {
            let ci = road.class as usize;
            if ppd < CLASS_MIN_PPD[ci] {
                continue;
            }
            if road.bbox.2 < vlat0
                || road.bbox.0 > vlat1
                || road.bbox.3 < vlon0
                || road.bbox.1 > vlon1
            {
                continue;
            }
            let rgb = CLASS_RGB[ci];
            let half_w = (CLASS_HALF_W[ci] * wscale).max(0.45);
            let mut prev: Option<(f32, f32)> = None;
            for &(dlat, dlon) in &road.pts {
                let x = (cx + (dlon as f64 - self.center.1) * ppd_lon) as f32;
                let y = (cy - (dlat as f64 - self.center.0) * ppd) as f32;
                if let Some((px, py)) = prev {
                    draw_segment(&mut self.canvas, w, h, px, py, x, y, half_w, rgb);
                }
                prev = Some((x, y));
            }
        }
    }
}

/// Anti-aliased stroke: distance-to-segment coverage over the segment's
/// padded bbox, lerped onto the canvas in visible RGB.
#[allow(clippy::too_many_arguments)]
fn draw_segment(
    canvas: &mut [u32],
    w: usize,
    h: usize,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    half_w: f32,
    rgb: [u8; 3],
) {
    let pad = half_w + 1.0;
    let bx0 = (x0.min(x1) - pad).floor().max(0.) as usize;
    let by0 = (y0.min(y1) - pad).floor().max(0.) as usize;
    // Clamp through isize and floor at 0: a segment fully off-canvas yields a
    // negative bound, and a bare `as usize` would wrap it past the guard.
    let bx1 = ((x0.max(x1) + pad).ceil() as isize).clamp(0, w as isize) as usize;
    let by1 = ((y0.max(y1) + pad).ceil() as isize).clamp(0, h as isize) as usize;
    if bx0 >= bx1 || by0 >= by1 {
        return;
    }
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len2 = dx * dx + dy * dy;
    for py in by0..by1 {
        let fy = py as f32 + 0.5;
        let row = &mut canvas[py * w..py * w + w];
        for (px, pixel) in row[bx0..bx1].iter_mut().enumerate() {
            let fx = (bx0 + px) as f32 + 0.5;
            // Project onto the segment, clamped to its endpoints.
            let t = if len2 > 0. {
                (((fx - x0) * dx + (fy - y0) * dy) / len2).clamp(0., 1.)
            } else {
                0.
            };
            let ex = fx - (x0 + t * dx);
            let ey = fy - (y0 + t * dy);
            let d = (ex * ex + ey * ey).sqrt();
            let cov = (half_w + 0.5 - d).clamp(0., 1.);
            if cov > 0. {
                let bg = *pixel;
                let lerp = |b: u32, f: u8| -> u32 {
                    (b as f32 + (f as f32 - b as f32) * cov) as u32
                };
                *pixel = (lerp((bg >> 16) & 255, rgb[0]) << 16)
                    | (lerp((bg >> 8) & 255, rgb[1]) << 8)
                    | lerp(bg & 255, rgb[2]);
            }
        }
    }
}

impl FluorApp for MahereApp {
    type UserEvent = ();

    fn title(&self) -> &str {
        "mahere"
    }

    fn init(&mut self, _ctx: &mut Context) {}

    fn on_resize(&mut self, _width: u32, _height: u32, _ctx: &mut Context) {
        self.dirty = true;
    }

    fn on_event(&mut self, event: &FEvent, ctx: &mut Context) -> EventResponse {
        match event {
            FEvent::MouseInput { state, button: MouseButton::Left } => {
                self.dragging = *state == ElementState::Pressed;
                self.last_cursor = (ctx.cursor_x as f64, ctx.cursor_y as f64);
                EventResponse::Handled
            }
            FEvent::CursorMoved { x, y } => {
                if self.dragging {
                    let (dx, dy) = (*x as f64 - self.last_cursor.0, *y as f64 - self.last_cursor.1);
                    self.last_cursor = (*x as f64, *y as f64);
                    self.center.0 += dy / self.ppd;
                    self.center.1 -= dx / (self.ppd * self.coslat());
                    self.dirty = true;
                    ctx.window.request_redraw();
                }
                EventResponse::Handled
            }
            FEvent::MouseWheel { delta } => {
                let notches = match delta {
                    MouseScrollDelta::Lines(_, y) => *y,
                    MouseScrollDelta::Pixels(_, y) => *y / 60.,
                } as f64;
                let factor = 1.18_f64.powf(notches);
                // Zoom about the cursor: keep the geo point under it fixed.
                let (w, h) = (ctx.viewport.width_px as f64, ctx.viewport.height_px as f64);
                let (mx, my) = (ctx.cursor_x as f64 - w * 0.5, ctx.cursor_y as f64 - h * 0.5);
                let coslat = self.coslat();
                let old_ppd = self.ppd;
                self.ppd = (self.ppd * factor).clamp(40., 4_000_000.);
                let f = self.ppd / old_ppd;
                self.center.0 -= my * (1. - 1. / f) / self.ppd;
                self.center.1 += mx * (1. - 1. / f) / (self.ppd * coslat);
                self.dirty = true;
                ctx.window.request_redraw();
                EventResponse::Handled
            }
            _ => EventResponse::Pass,
        }
    }

    fn render(&mut self, target: &mut [u32], ctx: &mut Context) {
        let w = ctx.viewport.width_px as usize;
        let h = ctx.viewport.height_px as usize;
        if self.dirty || (w, h) != (self.canvas_w, self.canvas_h) {
            self.redraw(w, h);
            self.dirty = false;
        }
        let n = (w * h).min(target.len()).min(self.canvas.len());
        for (out, &rgb) in target[..n].iter_mut().zip(&self.canvas[..n]) {
            *out = pack_argb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8, 255);
        }
    }

    fn cursor_for(&self, _x: Px, _y: Px, _ctx: &Context) -> CursorIcon {
        CursorIcon::Default
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/washington-latest.osm.pbf".into());
    eprintln!("loading {path} ...");
    let t = std::time::Instant::now();
    let roads = mahere_osm::load_roads(&path).expect("failed to read extract");
    let pts: usize = roads.iter().map(|r| r.pts.len()).sum();
    eprintln!(
        "{} roads, {} points, {:.1}s",
        roads.len(),
        pts,
        t.elapsed().as_secs_f32()
    );
    // Origin: dataset bbox center, so f32 offsets stay tiny.
    let mut bbox = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for r in &roads {
        for &(lat, lon) in &r.pts {
            bbox.0 = bbox.0.min(lat as f64);
            bbox.1 = bbox.1.min(lon as f64);
            bbox.2 = bbox.2.max(lat as f64);
            bbox.3 = bbox.3.max(lon as f64);
        }
    }
    let origin = ((bbox.0 + bbox.2) * 0.5, (bbox.1 + bbox.3) * 0.5);
    let seattle = (47.6062, -122.3321);
    run_app(MahereApp::new(roads, origin, seattle)).expect("fluor event loop failed");
}
