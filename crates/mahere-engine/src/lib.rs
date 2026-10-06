//! mahere's host-independent map core — the #pagetable edition. A frame is fetches: 32 px block grid → diamond UV → page table → plane fetch → LUT composite. No vectors, no reservoir, no source data at runtime: the renderer's entire diet is baked cells served by a CellStore through the clipmap residency layer. Frontends feed input and blit `canvas`.

pub mod plan;
pub mod raster;
pub mod residency;
pub mod sh;

use std::sync::Arc;
use std::time::Instant;

use mahere_coord::Coord;
use mahere_tiles::TEX;
pub use raster::LayerMask;
use raster::{Contours, DEM_BASE_DEPTH, ElevRange, FrameLuts, VEC_BASE_DEPTH, build_hypso_lut, select_depth};
use residency::{CellStore, Residency};

pub const PPD_REF: f64 = 6000.;
pub const BG_RGB: u32 = 0x12141A;

/// View state in absolute WGS84 degrees; ppd = pixels per degree latitude; bearing = radians the view is rotated (0 = north-up).
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

    /// Screen basis in local east/north: up = (sin B, cos B), right = (cos B, −sin B). B = 0 is north-up.
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

/// The map: all state and rendering, no host. Frontends call the input methods, drive `tick` until `converged`, and blit `canvas` (0xRRGGBB, row-major) after `render`.
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
    vec_depth: u8,
    dem_depth: u8,
    home: Camera,
    /// Measured cost of the last `render` call in milliseconds.
    pub last_frame_ms: f32,
    pub last_straddle_blocks: usize,
    /// Something changed since the last render (camera, sun, GPS). Reported by `tick` so shells that only draw on demand get a frame.
    dirty: bool,
    /// Elevation range the last frame saw; the next frame's contour interval is fit to it (one loop, a frame late — close enough).
    last_range: ElevRange,
    /// The lighting environment (device frame); folded into `luts.light` with the bearing each time either changes.
    env: sh::Sh9,
    /// Contours on screen the interval is fit to.
    pub contours_on_screen: f32,
    /// The interval the last frame drew, metres.
    pub contour_interval: f32,
    /// Degrees the device is turned clockwise from north (0 when no sensor feeds it); the lighting turns against it.
    pub device_heading: f32,
    /// The device's rotation matrix from the orientation sensor, world (east, north, up) = R · device (x right, y up the screen, z out of it); identity without a sensor.
    pub device_rot: [f32; 9],
    /// Light the terrain by where the sun actually is, from the clock and the position.
    pub real_sun: bool,
    /// Turn the map with the device so screen-up is the way the phone points.
    pub follow_heading: bool,
}

impl MapCore {
    pub fn new(store: Arc<dyn CellStore>, home: Camera) -> MapCore {
        MapCore {
            cam: home,
            res: Residency::new(store),
            luts: FrameLuts {
                hypso: build_hypso_lut(),
                sun: [0.0, 0.0, 1.0],
                mask: LayerMask::default(),
                dem_depth: DEM_BASE_DEPTH,
                contours: Contours { interval: 0.0, index_every: 5, m_per_px: 1.0 },
                light: sh::Sh9::sun_and_sky(315.0, 40.0).quadratic((0.0, 1.0)),
            },
            env: sh::Sh9::sun_and_sky(315.0, 40.0),
            luts_sun: (f32::NAN, f32::NAN, f64::NAN),
            sun_az: 315.0,
            sun_alt: 40.0,
            gps: None,
            canvas: Vec::new(),
            canvas_w: 0,
            canvas_h: 0,
            vec_depth: VEC_BASE_DEPTH,
            dem_depth: DEM_BASE_DEPTH,
            home,
            last_frame_ms: 0.0,
            last_straddle_blocks: 0,
            dirty: true,
            last_range: ElevRange::EMPTY,
            contours_on_screen: 32.0,
            contour_interval: 0.0,
            device_heading: 0.0,
            device_rot: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            real_sun: false,
            follow_heading: false,
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

    /// The client's layer filter: which of dem / land / water / line draw.
    pub fn layers(&self) -> LayerMask {
        self.luts.mask
    }

    pub fn set_layers(&mut self, mask: LayerMask) {
        self.luts.mask = mask;
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
            let key = mahere_tiles::CellKey { depth, prefix: raw >> (60 - 2 * depth as u32) };
            if let Some(e) = self.res.pool.map.get(&key).filter(|e| e.has_dem()) {
                let shift = 16 + (22 - depth as u32);
                let i = raster::tri_index((iu as i64) << 16, (iv as i64) << 16, shift);
                if let Some(eq) = e.elev_q_at(i).filter(|&eq| eq != mahere_tiles::ELEV_NODATA) {
                    return Some(mahere_tiles::dequantize_elev(eq));
                }
            }
        }
        None
    }

    /// A host that draws with the GPU only: the loader stops packing the CPU raster's texels, saving a megabyte and the normal pass per cell.
    pub fn set_gpu_only(&mut self) {
        self.res.set_pack_cpu(false);
    }

    /// The device's heading, degrees clockwise from north, from an orientation sensor: the lighting environment turns against it so the sun stays where it physically is as the phone turns (Nick's "sun top-left" with the room, not the screen).
    pub fn set_device_heading(&mut self, heading_deg: f32) {
        if (self.device_heading - heading_deg).abs() > 0.5 {
            self.device_heading = heading_deg;
            self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
            if self.follow_heading {
                self.cam.bearing = (heading_deg as f64).to_radians().rem_euclid(core::f64::consts::TAU);
            }
            self.dirty = true;
        }
    }

    /// The full orientation: the sensor's rotation matrix, row-major, world = R · device. The heading falls out of it (Android's own formula), and with the real sun on the landscape is lit exactly as the phone is held.
    pub fn set_device_rotation(&mut self, r: [f32; 9]) {
        let changed = self.device_rot.iter().zip(&r).any(|(a, b)| (a - b).abs() > 0.002);
        self.device_rot = r;
        let heading = r[1].atan2(r[4]).to_degrees().rem_euclid(360.0);
        self.set_device_heading(heading);
        if changed {
            self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
            self.dirty = true;
        }
    }

    pub fn set_real_sun(&mut self, on: bool) {
        self.real_sun = on;
        self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
        self.dirty = true;
    }

    pub fn set_follow_heading(&mut self, on: bool) {
        self.follow_heading = on;
        if on {
            self.cam.bearing = (self.device_heading as f64).to_radians().rem_euclid(core::f64::consts::TAU);
        }
        self.dirty = true;
    }

    pub fn camera_moved(&mut self, _w: usize, _h: usize) {
        self.dirty = true;
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    // ==================== PROGRESS ====================

    /// Integrate async-loaded cells. True if a redraw is due: cells arrived or the camera/sun/GPS changed since the last render.
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

    /// The resident cells, for a renderer that mirrors them (the GPU).
    pub fn pool(&self) -> &residency::Pool {
        &self.res.pool
    }

    /// The frame's style and lighting parameters.
    pub fn luts(&self) -> &FrameLuts {
        &self.luts
    }

    /// Plan a frame for a GPU: the same preparation as `render`, then the lattice, references and page table instead of pixels. Residency is driven from the plan's desired set; the contour interval is fit to the lattice's elevation range.
    pub fn plan(&mut self, w: usize, h: usize) -> plan::FramePlan {
        let t0 = Instant::now();
        self.prepare(w, h);
        let p = plan::plan_frame(w, h, &self.cam, &self.res.pool, self.dem_depth, self.vec_depth);
        self.last_straddle_blocks = p.straddle_blocks;
        self.last_range = p.elev;
        let center = Coord::from_lat_lon(self.cam.lat, self.cam.lon);
        let (cu, cv) = center.uv();
        self.res.want(p.want.clone(), (cu, cv));
        self.last_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
        p
    }

    /// The screen-space marks (GPS pin, compass) on a cleared canvas, for a renderer that composites them itself. 0xRRGGBB over black; the ink's brightness is its coverage. `with_pin` false leaves the pin out for a renderer that draws it itself (the GPU, where the pin would otherwise force a full overlay upload on every pan).
    pub fn overlay(&mut self, w: usize, h: usize, with_pin: bool) -> &[u32] {
        let mut scratch = std::mem::take(&mut self.canvas);
        scratch.clear();
        scratch.resize(w * h, 0);
        self.canvas = scratch;
        if with_pin {
            self.draw_gps(w, h);
        }
        self.draw_compass(w, h);
        &self.canvas
    }

    /// The GPS pin on screen: centre and accuracy radius in pixels, for a renderer that draws it itself.
    pub fn gps_screen(&self, w: usize, h: usize) -> Option<(f32, f32, f32)> {
        let g = self.gps?;
        let (x, y) = self.cam.geo_to_screen(g.lat, g.lon, w, h);
        let px_per_m = (self.cam.ppd / 111_320.0) as f32;
        Some((x as f32, y as f32, (g.accuracy_m * px_per_m).clamp(6.0, 4000.0)))
    }

    /// Everything a frame needs before any pixel: drain arrivals, lighting for the current sun and bearing, depth selection, the contour interval fit to the previous frame's range.
    fn prepare(&mut self, w: usize, h: usize) {
        self.dirty = false;
        self.res.frame += 1;
        self.res.drain();
        self.canvas_w = w;
        self.canvas_h = h;
        // The real sun: azimuth and altitude from the clock and the position (the fix, or the view), as a screen-relative azimuth; below the horizon the light stays low and grazing rather than going out.
        if self.real_sun {
            let (lat, lon) = self.gps.map_or((self.cam.lat, self.cam.lon), |g| (g.lat, g.lon));
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
            let (az, alt) = sh::sun_position(lat, lon, now);
            self.sun_az = ((az - self.cam.bearing.to_degrees()).rem_euclid(360.0)) as f32;
            self.sun_alt = alt.clamp(-10.0, 85.0) as f32;
        }
        // Lighting lives in the device frame: the environment (sun + sky as SH) is fixed to the screen, and normals are rotated by the bearing at lookup. The world-frame sun vector stays for the water glint.
        if self.luts_sun != (self.sun_az, self.sun_alt, self.cam.bearing) {
            let az = (self.sun_az as f64 + self.cam.bearing.to_degrees()).to_radians();
            let alt = (self.sun_alt as f64).to_radians();
            self.luts.sun = [
                (az.sin() * alt.cos()) as f32,
                (az.cos() * alt.cos()) as f32,
                alt.sin() as f32,
            ];
            if self.real_sun {
                // The real sun and the real sky in the device frame: the landscape is lit exactly as the phone is held. Below the horizon the direct light is gone and only the sky remains.
                let (lat, lon) = self.gps.map_or((self.cam.lat, self.cam.lon), |g| (g.lat, g.lon));
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
                let (az, alt) = sh::sun_position(lat, lon, now);
                let (azr, altr) = (az.to_radians(), alt.to_radians());
                let enu = [(azr.sin() * altr.cos()) as f32, (azr.cos() * altr.cos()) as f32, altr.sin() as f32];
                let r = self.device_rot;
                let dev = |v: [f32; 3]| [r[0] * v[0] + r[3] * v[1] + r[6] * v[2], r[1] * v[0] + r[4] * v[1] + r[7] * v[2], r[2] * v[0] + r[5] * v[1] + r[8] * v[2]];
                let strength = (alt as f32 / 5.0).clamp(0.0, 1.0);
                self.env = sh::Sh9::environment(dev(enu), strength, dev([0.0, 0.0, 1.0]));
            } else if self.luts_sun.0 != self.sun_az || self.luts_sun.1 != self.sun_alt || self.device_heading != 0.0 {
                self.env = sh::Sh9::sun_and_sky(self.sun_az - self.device_heading, self.sun_alt);
            }
            let (sb, cb) = self.cam.bearing.sin_cos();
            self.luts.light = self.env.quadratic((sb as f32, cb as f32));
            self.luts_sun = (self.sun_az, self.sun_alt, self.cam.bearing);
        }
        self.vec_depth = select_depth(self.cam.ppd, self.vec_depth, VEC_BASE_DEPTH);
        self.dem_depth = select_depth(self.cam.ppd, self.dem_depth, DEM_BASE_DEPTH);
        self.luts.dem_depth = self.dem_depth;
        // Contours: fit N levels to the previous frame's elevation span.
        let interval = if self.last_range.is_empty() {
            0.0
        } else {
            ((self.last_range.hi - self.last_range.lo) as f32 * 0.25 / self.contours_on_screen).max(0.5)
        };
        self.contour_interval = interval;
        self.luts.contours = Contours { interval, index_every: 5, m_per_px: (111_320.0 / self.cam.ppd) as f32 };
    }

    /// Render the full frame into `self.canvas` (visible 0xRRGGBB). Cheap by construction (fetch + composite), so it runs every host frame.
    pub fn render(&mut self, w: usize, h: usize) {
        let t0 = Instant::now();
        self.prepare(w, h);
        self.canvas.resize(w * h, BG_RGB);
        if w == 0 || h == 0 {
            return;
        }
        let (stats, want) = raster::render_frame(
            &mut self.canvas,
            w,
            h,
            &self.cam,
            &self.res.pool,
            &self.luts,
            self.dem_depth,
            self.vec_depth,
        );
        self.last_straddle_blocks = stats.straddle_blocks;
        self.last_range = stats.elev;

        // Residency: request what the lattice says we need, nearest-first.
        let center = Coord::from_lat_lon(self.cam.lat, self.cam.lon);
        let (cu, cv) = center.uv();
        self.res.want(want, (cu, cv));

        self.draw_gps(w, h);
        self.draw_compass(w, h);
        self.last_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    /// North arrow, top-right, with the view's heading below it: degrees the screen's up is turned from true north, clockwise positive, -180..180. No letters.
    fn draw_compass(&mut self, w: usize, h: usize) {
        if w < 120 || h < 160 {
            return;
        }
        const INK: [u8; 3] = [236, 238, 244];
        let (cx, cy) = (w as f32 - 46.0, 78.0);
        // Screen direction of true north: up = (sin B, cos B) in east/north, so north on screen is (-sin B, -cos B) in (x right, y down).
        let (sb, cb) = self.cam.bearing.sin_cos();
        let (nx, ny) = (-(sb as f32), -(cb as f32));
        let len = 26.0;
        let (tx, ty) = (cx + nx * len, cy + ny * len);
        let (bx, by) = (cx - nx * len * 0.6, cy - ny * len * 0.6);
        draw_segment(&mut self.canvas, w, h, bx, by, tx, ty, 1.4, INK);
        // Arrow head: two barbs back from the tip.
        let (px, py) = (-ny, nx);
        for sgn in [-1.0f32, 1.0] {
            let hx = tx - nx * 9.0 + px * sgn * 6.0;
            let hy = ty - ny * 9.0 + py * sgn * 6.0;
            draw_segment(&mut self.canvas, w, h, tx, ty, hx, hy, 1.4, INK);
        }
        let mut deg = self.cam.bearing.to_degrees().rem_euclid(360.0);
        if deg > 180.0 {
            deg -= 360.0;
        }
        let text = format!("{}", deg.round() as i64);
        draw_seven_seg(&mut self.canvas, w, h, &text, cx, cy + len + 30.0, 18.0, INK);
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
    // Clamp through isize and floor at 0: a segment fully off-canvas yields a negative bound, and a bare `as usize` would wrap it past the guard.
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
        fn get(&self, _key: mahere_tiles::CellKey) -> residency::Fetch {
            residency::Fetch::Absent
        }
    }

    /// Shells that draw on demand rely on `tick` to report camera changes (the Android two-finger path has no redraw request of its own).
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
