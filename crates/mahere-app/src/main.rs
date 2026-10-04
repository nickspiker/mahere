//! mahere desktop app — fluor host, CPU rendering.
//!
//! v0.3: terrain. The DEM sampling engine (see terrain.rs) converges a
//! reservoir of exact elevation+gradient samples under the view, hillshaded
//! at splat time — move the sun with A/D (azimuth) and W/S (altitude) and
//! the cached reservoir relights without re-evaluating anything. Roads and
//! trails draw on top via the v0 stroke rasterizer.

mod terrain;

use fluor::coord::Coord as Px;
use fluor::event::{CursorIcon, ElementState, Event as FEvent, MouseButton, MouseScrollDelta};
use fluor::host::app::{Context, EventResponse, FluorApp, run_app};
use fluor::paint::pack_argb;
use mahere_osm::{CLASS_COUNT, Road, RoadClass};
use std::time::Instant;

/// Visible RGB per class, tuned for terrain underneath. Trails green.
const CLASS_RGB: [[u8; 3]; CLASS_COUNT] = [
    [245, 150, 60],  // Motorway
    [238, 175, 62],  // Trunk
    [240, 208, 84],  // Primary
    [212, 212, 168], // Secondary
    [182, 192, 182], // Tertiary
    [142, 147, 158], // Residential
    [112, 117, 128], // Service
    [152, 120, 88],  // Track
    [80, 230, 120],  // Path
];

const CLASS_HALF_W: [f32; CLASS_COUNT] = [1.6, 1.4, 1.2, 1.0, 0.85, 0.6, 0.45, 0.45, 0.55];
const CLASS_MIN_PPD: [f64; CLASS_COUNT] =
    [0., 0., 0., 700., 700., 2500., 2500., 2500., 2500.];
const PPD_REF: f64 = 6000.;

/// View state in absolute WGS84 degrees; ppd = pixels per degree latitude.
#[derive(Clone, Copy)]
pub struct Camera {
    pub lat: f64,
    pub lon: f64,
    pub ppd: f64,
}

impl Camera {
    fn coslat(&self) -> f64 {
        self.lat.to_radians().cos()
    }

    pub fn geo_to_screen(&self, lat: f64, lon: f64, w: usize, h: usize) -> (f64, f64) {
        let ppd_lon = self.ppd * self.coslat();
        (
            w as f64 * 0.5 + (lon - self.lon) * ppd_lon,
            h as f64 * 0.5 - (lat - self.lat) * self.ppd,
        )
    }

    pub fn screen_to_geo(&self, px: f64, py: f64, w: usize, h: usize) -> (f64, f64) {
        let ppd_lon = self.ppd * self.coslat();
        (
            self.lat + (h as f64 * 0.5 - py) / self.ppd,
            self.lon + (px - w as f64 * 0.5) / ppd_lon,
        )
    }
}

struct PreparedRoad {
    class: RoadClass,
    pts: Vec<(f32, f32)>,            // (dlat, dlon) from origin
    bbox: (f32, f32, f32, f32),
}

struct MahereApp {
    roads: Vec<PreparedRoad>,
    origin: (f64, f64),
    cam: Camera,
    terrain: terrain::Terrain,
    terrain_gen_seen: u64,
    sun_az: f32,
    sun_alt: f32,
    canvas: Vec<u32>,
    canvas_w: usize,
    canvas_h: usize,
    dirty: bool,
    dragging: bool,
    last_cursor: (f64, f64),
}

impl MahereApp {
    fn new(roads: Vec<Road>, origin: (f64, f64), terrain: terrain::Terrain) -> Self {
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
            cam: Camera { lat: 47.6062, lon: -122.3321, ppd: PPD_REF },
            terrain,
            terrain_gen_seen: 0,
            sun_az: 315.0,
            sun_alt: 40.0,
            canvas: Vec::new(),
            canvas_w: 0,
            canvas_h: 0,
            dirty: true,
            dragging: false,
            last_cursor: (0., 0.),
        }
    }

    fn camera_moved(&mut self, ctx: &mut Context) {
        let (w, h) = (ctx.viewport.width_px as usize, ctx.viewport.height_px as usize);
        self.terrain.note_camera(w, h, &self.cam);
        self.dirty = true;
        ctx.window.request_redraw();
    }

    fn redraw(&mut self, w: usize, h: usize) {
        self.canvas.clear();
        self.canvas.resize(w * h, 0x12141A);
        self.canvas_w = w;
        self.canvas_h = h;
        if w == 0 || h == 0 {
            return;
        }
        self.terrain
            .splat(&mut self.canvas, w, h, &self.cam, self.sun_az, self.sun_alt);

        // Roads over terrain (v0 stroke rasterizer, 1D splats come later).
        let ppd = self.cam.ppd;
        let ppd_lon = ppd * self.cam.coslat();
        let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
        let (clat, clon) = (self.cam.lat - self.origin.0, self.cam.lon - self.origin.1);
        let vlat0 = (clat - cy / ppd) as f32;
        let vlat1 = (clat + cy / ppd) as f32;
        let vlon0 = (clon - cx / ppd_lon) as f32;
        let vlon1 = (clon + cx / ppd_lon) as f32;
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
                let x = (cx + (dlon as f64 - clon) * ppd_lon) as f32;
                let y = (cy - (dlat as f64 - clat) * ppd) as f32;
                if let Some((px, py)) = prev {
                    draw_segment(&mut self.canvas, w, h, px, py, x, y, half_w, rgb);
                }
                prev = Some((x, y));
            }
        }
    }
}

/// Anti-aliased stroke: distance-to-segment coverage over the padded bbox.
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
                let lerp =
                    |b: u32, f: u8| -> u32 { (b as f32 + (f as f32 - b as f32) * cov) as u32 };
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

    fn on_resize(&mut self, width: u32, height: u32, _ctx: &mut Context) {
        self.terrain
            .note_camera(width as usize, height as usize, &self.cam);
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
                    let (dx, dy) =
                        (*x as f64 - self.last_cursor.0, *y as f64 - self.last_cursor.1);
                    self.last_cursor = (*x as f64, *y as f64);
                    self.cam.lat += dy / self.cam.ppd;
                    self.cam.lon -= dx / (self.cam.ppd * self.cam.coslat());
                    self.camera_moved(ctx);
                }
                EventResponse::Handled
            }
            FEvent::MouseWheel { delta } => {
                let notches = match delta {
                    MouseScrollDelta::Lines(_, y) => *y,
                    MouseScrollDelta::Pixels(_, y) => *y / 60.,
                } as f64;
                let factor = 1.18_f64.powf(notches);
                let (w, h) = (ctx.viewport.width_px as f64, ctx.viewport.height_px as f64);
                let (mx, my) = (ctx.cursor_x as f64 - w * 0.5, ctx.cursor_y as f64 - h * 0.5);
                let coslat = self.cam.coslat();
                let old_ppd = self.cam.ppd;
                self.cam.ppd = (self.cam.ppd * factor).clamp(40., 4_000_000.);
                let f = self.cam.ppd / old_ppd;
                self.cam.lat -= my * (1. - 1. / f) / self.cam.ppd;
                self.cam.lon += mx * (1. - 1. / f) / (self.cam.ppd * coslat);
                self.camera_moved(ctx);
                EventResponse::Handled
            }
            FEvent::KeyboardInput { event } => {
                if event.state != ElementState::Pressed {
                    return EventResponse::Pass;
                }
                // Splat-time relighting: the reservoir never re-evaluates.
                let handled = match event.text.as_deref() {
                    Some("a") => {
                        self.sun_az = (self.sun_az - 15.0).rem_euclid(360.0);
                        true
                    }
                    Some("d") => {
                        self.sun_az = (self.sun_az + 15.0).rem_euclid(360.0);
                        true
                    }
                    Some("w") => {
                        self.sun_alt = (self.sun_alt + 5.0).min(85.0);
                        true
                    }
                    Some("s") => {
                        self.sun_alt = (self.sun_alt - 5.0).max(5.0);
                        true
                    }
                    _ => false,
                };
                if handled {
                    self.dirty = true;
                    ctx.window.request_redraw();
                    EventResponse::Handled
                } else {
                    EventResponse::Pass
                }
            }
            _ => EventResponse::Pass,
        }
    }

    fn wake_at(&self) -> Option<Instant> {
        if self.terrain.converged() {
            None
        } else {
            Some(Instant::now()) // host floors this to one frame out
        }
    }

    fn tick(&mut self, ctx: &mut Context) -> bool {
        let (w, h) = (ctx.viewport.width_px as usize, ctx.viewport.height_px as usize);
        self.terrain.tick(w, h, &self.cam)
    }

    fn render(&mut self, target: &mut [u32], ctx: &mut Context) {
        let w = ctx.viewport.width_px as usize;
        let h = ctx.viewport.height_px as usize;
        let tgen = self.terrain.generation();
        if self.dirty || (w, h) != (self.canvas_w, self.canvas_h) || tgen != self.terrain_gen_seen
        {
            self.redraw(w, h);
            self.dirty = false;
            self.terrain_gen_seen = tgen;
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
    let pbf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/washington-latest.osm.pbf".into());
    eprintln!("loading {pbf} ...");
    let t = Instant::now();
    let roads = mahere_osm::load_roads(&pbf).expect("failed to read extract");
    let pts: usize = roads.iter().map(|r| r.pts.len()).sum();
    eprintln!("{} roads, {} points, {:.1}s", roads.len(), pts, t.elapsed().as_secs_f32());

    let t = Instant::now();
    let dem_paths: Vec<String> = ["n47w122", "n47w123", "n48w122", "n48w123"]
        .iter()
        .map(|t| format!("data/USGS_1_{t}.tif"))
        .collect();
    let dem = mahere_dem::DemStore::load(&dem_paths).expect("failed to load DEM tiles");
    eprintln!("{} DEM tiles, {:.1}s", dem.tile_count(), t.elapsed().as_secs_f32());

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
    let terrain = terrain::Terrain::new(dem);
    run_app(MahereApp::new(roads, origin, terrain)).expect("fluor event loop failed");
}
