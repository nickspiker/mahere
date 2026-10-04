//! mahere's host-independent map core. Everything a frontend needs to be a
//! map — camera, the terrain sampling engine, the road rasterizer, the GPS
//! pin — with no windowing or compositor coupling: frontends (desktop fluor
//! app, Android shell) feed input and blit the RGB canvas.

pub mod terrain;

use mahere_osm::{CLASS_COUNT, Road, RoadClass};

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
const CLASS_MIN_PPD: [f64; CLASS_COUNT] = [0., 0., 0., 700., 700., 2500., 2500., 2500., 2500.];
pub const PPD_REF: f64 = 6000.;
pub const BG_RGB: u32 = 0x12141A;

/// View state in absolute WGS84 degrees; ppd = pixels per degree latitude;
/// bearing = radians the view is rotated (0 = north-up; positive turns the
/// map so that bearing B of compass points screen-up).
#[derive(Clone, Copy)]
pub struct Camera {
    pub lat: f64,
    pub lon: f64,
    pub ppd: f64,
    pub bearing: f64,
}

impl Camera {
    pub fn coslat(&self) -> f64 {
        self.lat.to_radians().cos()
    }

    /// Screen basis in local east/north: up = (sin B, cos B), right =
    /// (cos B, −sin B). B = 0 is north-up.
    pub fn geo_to_screen(&self, lat: f64, lon: f64, w: usize, h: usize) -> (f64, f64) {
        let e = (lon - self.lon) * self.ppd * self.coslat();
        let n = (lat - self.lat) * self.ppd;
        let (sb, cb) = self.bearing.sin_cos();
        (
            w as f64 * 0.5 + e * cb - n * sb,
            h as f64 * 0.5 - (e * sb + n * cb),
        )
    }

    pub fn screen_to_geo(&self, px: f64, py: f64, w: usize, h: usize) -> (f64, f64) {
        let sx = px - w as f64 * 0.5;
        let sy = h as f64 * 0.5 - py; // screen-up positive
        let (sb, cb) = self.bearing.sin_cos();
        let e = sx * cb + sy * sb;
        let n = -sx * sb + sy * cb;
        (
            self.lat + n / self.ppd,
            self.lon + e / (self.ppd * self.coslat()),
        )
    }
}

#[derive(Clone, Copy)]
pub struct GpsFix {
    pub lat: f64,
    pub lon: f64,
    pub accuracy_m: f32,
}

struct PreparedRoad {
    class: RoadClass,
    pts: Vec<(f32, f32)>, // (dlat, dlon) from origin
    bbox: (f32, f32, f32, f32),
}

/// The map: all state and rendering, no host. Frontends call the input
/// methods, drive `tick` until `converged`, and blit `canvas` (0xRRGGBB,
/// row-major) after `render`.
pub struct MapCore {
    roads: Vec<PreparedRoad>,
    origin: (f64, f64),
    pub cam: Camera,
    terrain: terrain::Terrain,
    terrain_gen_seen: u64,
    pub sun_az: f32,
    pub sun_alt: f32,
    pub gps: Option<GpsFix>,
    pub canvas: Vec<u32>,
    canvas_w: usize,
    canvas_h: usize,
    dirty: bool,
    home: Camera,
}

impl MapCore {
    pub fn new(
        roads: Vec<Road>,
        terrain: terrain::Terrain,
        home: Camera,
    ) -> MapCore {
        let mut bbox = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for r in &roads {
            for &(lat, lon) in &r.pts {
                bbox.0 = bbox.0.min(lat as f64);
                bbox.1 = bbox.1.min(lon as f64);
                bbox.2 = bbox.2.max(lat as f64);
                bbox.3 = bbox.3.max(lon as f64);
            }
        }
        let origin = if roads.is_empty() {
            (home.lat, home.lon)
        } else {
            ((bbox.0 + bbox.2) * 0.5, (bbox.1 + bbox.3) * 0.5)
        };
        let mut sorted: Vec<&Road> = roads.iter().collect();
        sorted.sort_by(|a, b| b.class.cmp(&a.class));
        let prepared = sorted
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
        MapCore {
            roads: prepared,
            origin,
            cam: home,
            terrain,
            terrain_gen_seen: 0,
            sun_az: 315.0,
            sun_alt: 40.0,
            gps: None,
            canvas: Vec::new(),
            canvas_w: 0,
            canvas_h: 0,
            dirty: true,
            home,
        }
    }

    // ==================== INPUT ====================

    pub fn clamp_camera(&mut self) {
        // Past the pole cos(lat) flips sign and horizontal panning inverts.
        self.cam.lat = self.cam.lat.clamp(-85.0, 85.0);
        if self.cam.lon > 180.0 {
            self.cam.lon -= 360.0;
        } else if self.cam.lon < -180.0 {
            self.cam.lon += 360.0;
        }
    }

    /// Pan by a screen-pixel delta (bearing-aware: dragging always moves
    /// the map with the finger, whatever direction north points).
    pub fn pan(&mut self, dx: f64, dy: f64, w: usize, h: usize) {
        let (sb, cb) = self.cam.bearing.sin_cos();
        let e = dx * cb - dy * sb; // screen delta in ENU (y-down input)
        let n = -dx * sb - dy * cb;
        self.cam.lat += n / self.cam.ppd;
        self.cam.lon -= e / (self.cam.ppd * self.cam.coslat());
        self.clamp_camera();
        self.camera_moved(w, h);
    }

    /// Exact geo-anchored zoom: the geography under (ax, ay) stays there.
    pub fn zoom_about(&mut self, factor: f64, ax: f64, ay: f64, w: usize, h: usize) {
        let (alat, alon) = self.cam.screen_to_geo(ax, ay, w, h);
        self.cam.ppd = (self.cam.ppd * factor).clamp(40., 4_000_000.);
        self.place_anchor(alat, alon, ax, ay, w, h);
    }

    /// Re-solve the camera so (alat, alon) sits at screen (ax, ay), under
    /// the current ppd and bearing.
    pub fn place_anchor(&mut self, alat: f64, alon: f64, ax: f64, ay: f64, w: usize, h: usize) {
        let sx = ax - w as f64 * 0.5;
        let sy = h as f64 * 0.5 - ay;
        let (sb, cb) = self.cam.bearing.sin_cos();
        let e = sx * cb + sy * sb;
        let n = -sx * sb + sy * cb;
        self.cam.lat = alat - n / self.cam.ppd;
        // coslat varies negligibly across a screen; anchor latitude is fine.
        self.cam.lon = alon - e / (self.cam.ppd * alat.to_radians().cos());
        self.clamp_camera();
        self.camera_moved(w, h);
    }

    pub fn set_bearing(&mut self, bearing: f64) {
        self.cam.bearing = bearing.rem_euclid(core::f64::consts::TAU);
    }

    /// Set zoom directly (two-finger solve), without anchoring.
    pub fn set_ppd(&mut self, ppd: f64) {
        self.cam.ppd = ppd.clamp(40., 4_000_000.);
    }

    pub fn go_home(&mut self, w: usize, h: usize) {
        self.cam = self.home;
        self.camera_moved(w, h);
    }

    pub fn adjust_sun(&mut self, daz: f32, dalt: f32) {
        self.sun_az = (self.sun_az + daz).rem_euclid(360.0);
        self.sun_alt = (self.sun_alt + dalt).clamp(5.0, 85.0);
        self.dirty = true;
    }

    pub fn set_gps(&mut self, fix: GpsFix) {
        self.gps = Some(fix);
        self.dirty = true;
    }

    /// DEM elevation at the GPS fix, if both exist.
    pub fn gps_elevation(&self) -> Option<f32> {
        let g = self.gps?;
        self.terrain.dem().elevation(g.lat, g.lon)
    }

    pub fn camera_moved(&mut self, w: usize, h: usize) {
        self.terrain.note_camera(w, h, &self.cam);
        self.dirty = true;
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    // ==================== PROGRESS ====================

    pub fn tick(&mut self, w: usize, h: usize) -> bool {
        self.terrain.tick(w, h, &self.cam)
    }

    pub fn converged(&self) -> bool {
        self.terrain.converged()
    }

    // ==================== RENDER ====================

    /// True if render() would produce new pixels.
    pub fn needs_render(&self, w: usize, h: usize) -> bool {
        self.dirty
            || (w, h) != (self.canvas_w, self.canvas_h)
            || self.terrain.generation() != self.terrain_gen_seen
    }

    /// Render the full frame into `self.canvas` (visible 0xRRGGBB).
    pub fn render(&mut self, w: usize, h: usize) {
        self.canvas.clear();
        self.canvas.resize(w * h, BG_RGB);
        self.canvas_w = w;
        self.canvas_h = h;
        self.dirty = false;
        self.terrain_gen_seen = self.terrain.generation();
        if w == 0 || h == 0 {
            return;
        }
        let (sun_az, sun_alt) = (self.sun_az, self.sun_alt);
        self.terrain.splat(&mut self.canvas, w, h, &self.cam, sun_az, sun_alt);
        self.draw_roads(w, h);
        self.draw_gps(w, h);
    }

    fn draw_roads(&mut self, w: usize, h: usize) {
        let ppd = self.cam.ppd;
        let ppd_lon = ppd * self.cam.coslat();
        let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
        let (clat, clon) = (self.cam.lat - self.origin.0, self.cam.lon - self.origin.1);
        // Conservative cull box: half-diagonal in both axes so rotation
        // never culls a visible road.
        let diag = (cx * cx + cy * cy).sqrt();
        let vlat0 = (clat - diag / ppd) as f32;
        let vlat1 = (clat + diag / ppd) as f32;
        let vlon0 = (clon - diag / ppd_lon) as f32;
        let vlon1 = (clon + diag / ppd_lon) as f32;
        let (sb, cb) = self.cam.bearing.sin_cos();
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
                let e = (dlon as f64 - clon) * ppd_lon;
                let n = (dlat as f64 - clat) * ppd;
                let x = (cx + e * cb - n * sb) as f32;
                let y = (cy - (e * sb + n * cb)) as f32;
                if let Some((px, py)) = prev {
                    draw_segment(&mut self.canvas, w, h, px, py, x, y, half_w, rgb);
                }
                prev = Some((x, y));
            }
        }
    }

    /// GPS pin: accuracy ring, crosshair dot, and the DEM elevation in big
    /// seven-segment digits (meters) — readable in the field, zero text deps.
    fn draw_gps(&mut self, w: usize, h: usize) {
        let Some(g) = self.gps else { return };
        let (x, y) = self.cam.geo_to_screen(g.lat, g.lon, w, h);
        let (x, y) = (x as f32, y as f32);
        const PIN: [u8; 3] = [64, 156, 255];
        // Accuracy ring (meters -> px via ppd: 1 deg lat = 111_320 m).
        let px_per_m = (self.cam.ppd / 111_320.0) as f32;
        let r = (g.accuracy_m * px_per_m).clamp(6.0, 4000.0);
        let steps = (r as usize * 2).clamp(24, 720);
        let mut prev = (x + r, y);
        for i in 1..=steps {
            let a = i as f32 / steps as f32 * core::f32::consts::TAU;
            let p = (x + r * a.cos(), y + r * a.sin());
            draw_segment(&mut self.canvas, w, h, prev.0, prev.1, p.0, p.1, 0.9, PIN);
            prev = p;
        }
        // Crosshair dot.
        for (dx0, dy0, dx1, dy1) in [(-9., 0., 9., 0.), (0., -9., 0., 9.)] {
            draw_segment(&mut self.canvas, w, h, x + dx0, y + dy0, x + dx1, y + dy1, 1.6, PIN);
        }
        // Elevation readout above the pin.
        if let Some(elev) = self.terrain.dem().elevation(g.lat, g.lon) {
            let text = format!("{}", elev.round() as i64);
            draw_seven_seg(&mut self.canvas, w, h, &text, x, y - r.min(60.0) - 44.0, 26.0, PIN);
        }
    }
}

/// Anti-aliased stroke: distance-to-segment coverage over the padded bbox.
#[allow(clippy::too_many_arguments)]
pub fn draw_segment(
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

/// Seven-segment digit rendering via strokes: field-readable numerals with
/// zero font dependencies. Supports 0-9 and '-'.
pub fn draw_seven_seg(
    canvas: &mut [u32],
    w: usize,
    h: usize,
    text: &str,
    x: f32,
    y: f32,
    size: f32,
    rgb: [u8; 3],
) {
    //  _      segments bit order: 0 top, 1 top-right, 2 bottom-right,
    // |_|     3 bottom, 4 bottom-left, 5 top-left, 6 middle
    // |_|
    const GLYPHS: [u8; 10] = [
        0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101, 0b1111101, 0b0000111,
        0b1111111, 0b1101111,
    ];
    let sw = size * 0.62; // digit width
    let total = text.chars().count() as f32 * (sw + size * 0.28);
    let mut cx = x - total * 0.5;
    let hw = (size * 0.09).max(1.2);
    for ch in text.chars() {
        let segs = match ch {
            '0'..='9' => GLYPHS[ch as usize - '0' as usize],
            '-' => 0b1000000,
            _ => 0,
        };
        let (x0, x1) = (cx, cx + sw);
        let (y0, ym, y1) = (y - size, y - size * 0.5, y);
        let lines = [
            (x0, y0, x1, y0), // 0 top
            (x1, y0, x1, ym), // 1 top-right
            (x1, ym, x1, y1), // 2 bottom-right
            (x0, y1, x1, y1), // 3 bottom
            (x0, ym, x0, y1), // 4 bottom-left
            (x0, y0, x0, ym), // 5 top-left
            (x0, ym, x1, ym), // 6 middle
        ];
        for (i, &(ax, ay, bx, by)) in lines.iter().enumerate() {
            if segs >> i & 1 == 1 {
                draw_segment(canvas, w, h, ax, ay, bx, by, hw, rgb);
            }
        }
        cx += sw + size * 0.28;
    }
}
