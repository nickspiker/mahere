//! mahere's host-independent map core — the #pagetable edition. A frame is
//! fetches: 32 px block grid → diamond UV → page table → plane fetch → LUT
//! composite. No vectors, no reservoir, no source data at runtime: the
//! renderer's entire diet is baked cells served by a CellStore through the
//! clipmap residency layer. Frontends feed input and blit `canvas`.

pub mod raster;
pub mod residency;

use std::sync::Arc;
use std::time::Instant;

use mahere_coord::Coord;
use mahere_tiles::TEX;
use raster::{DEM_BASE_DEPTH, FrameLuts, LINE_BASE_DEPTH, build_hypso_lut, select_depth};
use residency::{CellStore, Entry, Residency};

pub const PPD_REF: f64 = 6000.;
pub const BG_RGB: u32 = 0x12141A;

/// View state in absolute WGS84 degrees; ppd = pixels per degree latitude;
/// bearing = radians the view is rotated (0 = north-up).
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

/// The map: all state and rendering, no host. Frontends call the input
/// methods, drive `tick` until `converged`, and blit `canvas` (0xRRGGBB,
/// row-major) after `render`.
pub struct MapCore {
    pub cam: Camera,
    res: Residency,
    luts: FrameLuts,
    luts_sun: (f32, f32, f64),
    pub sun_az: f32,
    pub sun_alt: f32,
    pub gps: Option<GpsFix>,
    pub canvas: Vec<u32>,
    canvas_w: usize,
    canvas_h: usize,
    line_depth: u8,
    dem_depth: u8,
    home: Camera,
    /// Measured cost of the last `render` call in milliseconds.
    pub last_frame_ms: f32,
    pub last_straddle_blocks: usize,
    /// Something changed since the last render (camera, sun, GPS). Reported
    /// by `tick` so shells that only draw on demand get a frame.
    dirty: bool,
}

impl MapCore {
    pub fn new(store: Arc<dyn CellStore>, home: Camera) -> MapCore {
        MapCore {
            cam: home,
            res: Residency::new(store),
            luts: FrameLuts { hypso: build_hypso_lut(), sun: [0.0, 0.0, 1.0] },
            luts_sun: (f32::NAN, f32::NAN, f64::NAN),
            sun_az: 315.0,
            sun_alt: 40.0,
            gps: None,
            canvas: Vec::new(),
            canvas_w: 0,
            canvas_h: 0,
            line_depth: LINE_BASE_DEPTH,
            dem_depth: DEM_BASE_DEPTH,
            home,
            last_frame_ms: 0.0,
            last_straddle_blocks: 0,
            dirty: true,
        }
    }

    // ==================== INPUT ====================

    pub fn clamp_camera(&mut self) {
        self.cam.lat = self.cam.lat.clamp(-85.0, 85.0);
        if self.cam.lon > 180.0 {
            self.cam.lon -= 360.0;
        } else if self.cam.lon < -180.0 {
            self.cam.lon += 360.0;
        }
    }

    /// Pan by a screen-pixel delta (bearing-aware).
    pub fn pan(&mut self, dx: f64, dy: f64, w: usize, h: usize) {
        let (sb, cb) = self.cam.bearing.sin_cos();
        let de = -dx * cb + dy * sb;
        let dn = dx * sb + dy * cb;
        self.cam.lat += dn / self.cam.ppd;
        self.cam.lon += de / (self.cam.ppd * self.cam.coslat());
        self.clamp_camera();
        self.camera_moved(w, h);
    }

    /// Exact geo-anchored zoom: the geography under (ax, ay) stays there.
    pub fn zoom_about(&mut self, factor: f64, ax: f64, ay: f64, w: usize, h: usize) {
        let (alat, alon) = self.cam.screen_to_geo(ax, ay, w, h);
        self.cam.ppd = (self.cam.ppd * factor).clamp(40., 4_000_000.);
        self.place_anchor(alat, alon, ax, ay, w, h);
    }

    /// Re-solve the camera so (alat, alon) sits at screen (ax, ay).
    pub fn place_anchor(&mut self, alat: f64, alon: f64, ax: f64, ay: f64, w: usize, h: usize) {
        let sx = ax - w as f64 * 0.5;
        let sy = h as f64 * 0.5 - ay;
        let (sb, cb) = self.cam.bearing.sin_cos();
        let e = sx * cb + sy * sb;
        let n = -sx * sb + sy * cb;
        self.cam.lat = alat - n / self.cam.ppd;
        self.cam.lon = alon - e / (self.cam.ppd * alat.to_radians().cos());
        self.clamp_camera();
        self.camera_moved(w, h);
    }

    pub fn set_bearing(&mut self, bearing: f64) {
        self.cam.bearing = bearing.rem_euclid(core::f64::consts::TAU);
        self.dirty = true;
    }

    pub fn set_ppd(&mut self, ppd: f64) {
        self.cam.ppd = ppd.clamp(40., 4_000_000.);
        self.dirty = true;
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

    /// DEM elevation at the GPS fix from resident cells (deepest first).
    pub fn gps_elevation(&self) -> Option<f32> {
        let g = self.gps?;
        self.elevation_at(g.lat, g.lon)
    }

    pub fn elevation_at(&self, lat: f64, lon: f64) -> Option<f32> {
        let c = Coord::from_lat_lon(lat, lon);
        let raw = c.raw();
        let (iu, iv) = c.uv();
        for depth in (raster::MIN_DEPTH..=DEM_BASE_DEPTH).rev() {
            let prefix = raw >> (60 - 2 * depth as u32);
            if let Some(Entry::Dem(p)) = self.res.dem.map.get(&(depth, prefix)) {
                let shift = 16 + (22 - depth as u32);
                let i = raster::tri_index((iu as i64) << 16, (iv as i64) << 16, shift);
                let eq = (p.texel[i] & 0xFFFF) as u16;
                if eq != residency::ELEV_NODATA {
                    return Some(eq as f32 / 4.0 - 500.0);
                }
            }
        }
        None
    }

    pub fn camera_moved(&mut self, _w: usize, _h: usize) {
        self.dirty = true;
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    // ==================== PROGRESS ====================

    /// Integrate async-loaded cells. True if a redraw is due: cells arrived
    /// or the camera/sun/GPS changed since the last render.
    pub fn tick(&mut self, _w: usize, _h: usize) -> bool {
        self.res.drain() > 0 || self.dirty
    }

    pub fn converged(&self) -> bool {
        self.res.converged()
    }

    pub fn needs_render(&self, _w: usize, _h: usize) -> bool {
        true
    }

    // ==================== RENDER ====================

    /// Render the full frame into `self.canvas` (visible 0xRRGGBB). Cheap by
    /// construction (fetch + composite), so it runs every host frame.
    pub fn render(&mut self, w: usize, h: usize) {
        let t0 = Instant::now();
        self.dirty = false;
        self.res.frame += 1;
        self.res.drain();
        self.canvas.resize(w * h, BG_RGB);
        self.canvas_w = w;
        self.canvas_h = h;
        if w == 0 || h == 0 {
            return;
        }
        // Sun vector: screen-space azimuth converted to world by bearing.
        if self.luts_sun != (self.sun_az, self.sun_alt, self.cam.bearing) {
            let az = (self.sun_az as f64 + self.cam.bearing.to_degrees()).to_radians();
            let alt = (self.sun_alt as f64).to_radians();
            self.luts.sun = [
                (az.sin() * alt.cos()) as f32,
                (az.cos() * alt.cos()) as f32,
                alt.sin() as f32,
            ];
            self.luts_sun = (self.sun_az, self.sun_alt, self.cam.bearing);
        }
        self.line_depth = select_depth(self.cam.ppd, self.line_depth, LINE_BASE_DEPTH);
        self.dem_depth = select_depth(self.cam.ppd, self.dem_depth, DEM_BASE_DEPTH);

        let (stats, want) = raster::render_frame(
            &mut self.canvas,
            w,
            h,
            &self.cam,
            &self.res.dem,
            &self.res.line,
            &self.luts,
            self.dem_depth,
            self.line_depth,
        );
        self.last_straddle_blocks = stats.straddle_blocks;

        // Residency: request what the lattice says we need, nearest-first.
        let center = Coord::from_lat_lon(self.cam.lat, self.cam.lon);
        let (cu, cv) = center.uv();
        self.res.want(want, (cu, cv));

        self.draw_gps(w, h);
        self.last_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    /// GPS pin: accuracy ring, crosshair, seven-segment elevation readout.
    fn draw_gps(&mut self, w: usize, h: usize) {
        let Some(g) = self.gps else { return };
        let (x, y) = self.cam.geo_to_screen(g.lat, g.lon, w, h);
        let (x, y) = (x as f32, y as f32);
        const PIN: [u8; 3] = [64, 156, 255];
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
        for (dx0, dy0, dx1, dy1) in [(-9., 0., 9., 0.), (0., -9., 0., 9.)] {
            draw_segment(&mut self.canvas, w, h, x + dx0, y + dy0, x + dx1, y + dy1, 1.6, PIN);
        }
        if let Some(elev) = self.elevation_at(g.lat, g.lon) {
            let text = format!("{}", elev.round() as i64);
            draw_seven_seg(&mut self.canvas, w, h, &text, x, y - r.min(60.0) - 44.0, 26.0, PIN);
        }
    }
}

// Keep TEX re-exported for frontends/tests that size things off cells.
pub use mahere_tiles::TEX as CELL_TEX;
const _: () = assert!(TEX == 256);

/// Anti-aliased stroke (screen-space overlay layer: GPS pin etc.).
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

/// Seven-segment digits via strokes: field-readable numerals, zero font deps.
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
    const GLYPHS: [u8; 10] = [
        0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101, 0b1111101, 0b0000111,
        0b1111111, 0b1101111,
    ];
    let sw = size * 0.62;
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
            (x0, y0, x1, y0),
            (x1, y0, x1, ym),
            (x1, ym, x1, y1),
            (x0, y1, x1, y1),
            (x0, ym, x0, y1),
            (x0, y0, x0, ym),
            (x0, ym, x1, ym),
        ];
        for (i, &(ax, ay, bx, by)) in lines.iter().enumerate() {
            if segs >> i & 1 == 1 {
                draw_segment(canvas, w, h, ax, ay, bx, by, hw, rgb);
            }
        }
        cx += sw + size * 0.28;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Empty;
    impl CellStore for Empty {
        fn get(&self, _layer: residency::Layer, _key: mahere_tiles::CellKey) -> Option<Vec<u8>> {
            None
        }
    }

    /// Shells that draw on demand rely on `tick` to report camera changes
    /// (the Android two-finger path has no redraw request of its own).
    #[test]
    fn tick_reports_camera_changes() {
        let cam = Camera { lat: 46.2, lon: -121.5, ppd: 2800.0, bearing: 0.0 };
        let mut map = MapCore::new(Arc::new(Empty), cam);
        for _ in 0..500 {
            map.render(64, 64);
            if map.converged() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        map.tick(64, 64);
        map.render(64, 64);
        assert!(!map.tick(64, 64), "nothing changed, nothing to draw");
        map.set_bearing(1.0);
        assert!(map.tick(64, 64), "bearing changed: a frame is due");
        map.render(64, 64);
        assert!(!map.tick(64, 64));
    }
}
