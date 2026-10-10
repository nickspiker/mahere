//! mahere's host-independent map core — the #pagetable edition. A frame is fetches: 32 px block grid → diamond UV → page table → plane fetch → LUT composite. No vectors, no reservoir, no source data at runtime: the renderer's entire diet is baked cells served by a CellStore through the clipmap residency layer. Frontends feed input and blit `canvas`.

pub mod colour;
pub mod plan;
pub mod probe;
pub mod raster;
pub mod residency;
pub mod sh;
pub mod theme;

use std::sync::Arc;
use std::time::Instant;

use mahere_coord::Coord;
use mahere_tiles::TEX;
pub use raster::LayerMask;
use raster::{Contours, DEM_BASE_DEPTH, ElevRange, FrameLuts, VEC_BASE_DEPTH, build_hypso_lut, select_depth};
use residency::{CellStore, Residency};

pub const PPD_REF: f64 = 6000.;
pub const BG_RGB: u32 = 0x12141A;

/// View state: the globe seen from far off, the camera pointed at (lat, lon), which sits at the screen's centre; ppd = pixels per degree of arc there, so the whole globe is a disk of radius ppd·180/π pixels; bearing = radians the view is rotated (0 = north-up at the centre). The projection is orthographic, so the near ground is the plain map it always was and zooming out flattens the globe into a circle (Nick 2026-10-09).
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

    /// The globe's radius on the screen, in pixels: pixels per radian of arc.
    pub fn radius(&self) -> f64 {
        self.ppd * 180.0 / core::f64::consts::PI
    }

    /// Screen basis in local east/north at the centre: up = (sin B, cos B), right = (cos B, −sin B). B = 0 is north-up. A point on the far side of the globe is pushed well outside the disk, so a mark there draws nowhere.
    pub fn geo_to_screen(&self, lat: f64, lon: f64, w: usize, h: usize) -> (f64, f64) {
        let (sl0, cl0) = self.lat.to_radians().sin_cos();
        let (sl, cl) = lat.to_radians().sin_cos();
        let (sd, cd) = (lon - self.lon).to_radians().sin_cos();
        let r = self.radius();
        let mut e = cl * sd * r;
        let mut n = (cl0 * sl - sl0 * cl * cd) * r;
        let cosc = sl0 * sl + cl0 * cl * cd;
        if cosc < 0.0 {
            let rho = (e * e + n * n).sqrt();
            if rho < 1e-6 {
                // The antipode itself has no direction: straight up and away.
                e = 0.0;
                n = r * 4.0;
            } else {
                let k = r * 4.0 / rho;
                e *= k;
                n *= k;
            }
        }
        let (sb, cb) = self.bearing.sin_cos();
        (
            w as f64 * 0.5 + e * cb - n * sb,
            h as f64 * 0.5 - (e * sb + n * cb),
        )
    }

    /// The ground under a screen point. Past the limb the point is taken at the limb, so a corner off the globe still names ground (the renderers paint the background there).
    pub fn screen_to_geo(&self, px: f64, py: f64, w: usize, h: usize) -> (f64, f64) {
        let sx = px - w as f64 * 0.5;
        let sy = h as f64 * 0.5 - py; // screen-up positive
        let (sb, cb) = self.bearing.sin_cos();
        let mut e = sx * cb + sy * sb;
        let mut n = -sx * sb + sy * cb;
        let r = self.radius();
        let mut rho = (e * e + n * n).sqrt();
        if rho < 1e-9 {
            return (self.lat, self.lon);
        }
        let limb = r * (1.0 - 1e-9);
        if rho > limb {
            let k = limb / rho;
            e *= k;
            n *= k;
            rho = limb;
        }
        let c = (rho / r).asin();
        let (sc, cc) = c.sin_cos();
        let (sl0, cl0) = self.lat.to_radians().sin_cos();
        let lat = (cc * sl0 + n * sc * cl0 / rho).clamp(-1.0, 1.0).asin();
        let lon = self.lon + (e * sc).atan2(rho * cc * cl0 - n * sc * sl0).to_degrees();
        (lat.to_degrees(), (lon + 180.0).rem_euclid(360.0) - 180.0)
    }

    /// The view's basis in world coordinates (the unit sphere, x toward 0°N 0°E, z toward the north pole): screen right, screen up, and the axis toward the viewer, which is the camera point itself.
    pub fn basis(&self) -> [[f64; 3]; 3] {
        let (sl, cl) = self.lat.to_radians().sin_cos();
        let (so, co) = self.lon.to_radians().sin_cos();
        let east = [-so, co, 0.0];
        let north = [-sl * co, -sl * so, cl];
        let centre = [cl * co, cl * so, sl];
        let (sb, cb) = self.bearing.sin_cos();
        let right = [east[0] * cb - north[0] * sb, east[1] * cb - north[1] * sb, east[2] * cb - north[2] * sb];
        let up = [east[0] * sb + north[0] * cb, east[1] * sb + north[1] * cb, east[2] * sb + north[2] * cb];
        [right, up, centre]
    }

    /// The camera whose view has this basis: the camera point from the axis toward the viewer, the bearing from where screen right lies between east and north there.
    pub fn from_basis(b: [[f64; 3]; 3], ppd: f64) -> Camera {
        let c = b[2];
        let lat = c[2].clamp(-1.0, 1.0).asin();
        let lon = c[1].atan2(c[0]);
        let (sl, cl) = lat.sin_cos();
        let (so, co) = lon.sin_cos();
        let east = [-so, co, 0.0];
        let north = [-sl * co, -sl * so, cl];
        let r = b[0];
        let bearing = (-dot(r, north)).atan2(dot(r, east)).rem_euclid(core::f64::consts::TAU);
        Camera { lat: lat.to_degrees(), lon: lon.to_degrees(), ppd, bearing }
    }

    /// The unit vector of a place on the globe.
    pub fn unit(lat: f64, lon: f64) -> [f64; 3] {
        let (sl, cl) = lat.to_radians().sin_cos();
        let (so, co) = lon.to_radians().sin_cos();
        [cl * co, cl * so, sl]
    }

    /// The view-space direction of a screen point on the globe, the limb for a point past it.
    pub fn view_dir(&self, px: f64, py: f64, w: usize, h: usize) -> [f64; 3] {
        let r = self.radius();
        let mut x = (px - w as f64 * 0.5) / r;
        let mut y = (h as f64 * 0.5 - py) / r;
        let rho2 = x * x + y * y;
        if rho2 >= 1.0 {
            let k = (1.0 - 1e-9) / rho2.sqrt();
            x *= k;
            y *= k;
        }
        [x, y, (1.0 - x * x - y * y).max(0.0).sqrt()]
    }

    /// Whether a screen point lies on the globe.
    pub fn on_globe(&self, px: f64, py: f64, w: usize, h: usize) -> bool {
        let sx = px - w as f64 * 0.5;
        let sy = py - h as f64 * 0.5;
        let r = self.radius();
        sx * sx + sy * sy <= r * r
    }

    /// Whether any of the screen lies off the globe.
    pub fn limb_visible(&self, w: usize, h: usize) -> bool {
        let r = self.radius();
        (w * w + h * h) as f64 * 0.25 > r * r
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// `v` turned about the unit axis `k` by `angle` (Rodrigues).
fn rotate(v: [f64; 3], k: [f64; 3], angle: f64) -> [f64; 3] {
    let (s, c) = angle.sin_cos();
    let kv = cross(k, v);
    let kd = dot(k, v) * (1.0 - c);
    [v[0] * c + kv[0] * s + k[0] * kd, v[1] * c + kv[1] * s + k[1] * kd, v[2] * c + kv[2] * s + k[2] * kd]
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
    /// The front camera is the light: its latest frame, projected, replaces the sun and sky while it is on.
    pub real_light: bool,
    /// The camera's light, in the device frame at the latest rotation, held to the sphere's own level.
    probe: Option<sh::Sh9>,
    /// The world-fixed sphere the camera paints, and the lens its frames come through.
    sphere: Option<Box<probe::Sphere>>,
    lens: Option<probe::Lens>,
    /// The real sun's direction in the device frame, as last lit.
    sun_device: [f32; 3],
    /// Turn the map with the device so screen-up is the way the phone points.
    pub follow_heading: bool,
    /// Magnetic declination at the position, degrees, east positive: true heading = the sensor's magnetic heading + this.
    pub declination_deg: f32,
    /// Whether an orientation sensor has ever reported.
    pub have_rotation: bool,
    /// The layers as chosen (the drawn mask is their `effective()`).
    layer_mask: LayerMask,
    /// Where the pin was last drawn, so a fix that leaves it still or off screen costs no frame.
    last_pin: Option<(f32, f32)>,
    /// The last plan and what it was for.
    plan_cache: Option<(PlanKey, Arc<plan::FramePlan>)>,
    /// The current measurement, if a point was tapped.
    measure: Option<Measure>,
    /// The last plan came from the cache: the view and the cells were as before.
    plan_cached: bool,
    /// The themes available (the built-ins until a host loads the vault's), the one in use, and a counter a GPU host watches to re-upload the tables.
    pub themes: Vec<theme::Theme>,
    pub theme: usize,
    style_version: u64,
    /// Bumped whenever the lighting is rebuilt, so a host can tell a light-only change from a cell arriving.
    light_version: u64,
    /// Measure from the fix rather than the screen centre, until the camera moves.
    pub lock_to_fix: bool,
}

type PlanKey = (u64, u64, u64, u64, usize, usize, u64, u8, u8);

impl MapCore {
    pub fn new(store: Arc<dyn CellStore>, home: Camera) -> MapCore {
        MapCore {
            cam: home,
            res: Residency::new(store),
            luts: FrameLuts {
                hypso: build_hypso_lut(),
                style: raster::Style::default(),
                sun: [0.0, 0.0, 1.0],
                mask: LayerMask::default(),
                dem_depth: DEM_BASE_DEPTH,
                contours: Contours { interval: 0.0, index_every: 5, m_per_px: 1.0 },
                light: sh::Sh9::sun_and_sky(315.0, 40.0).quadratic((0.0, 1.0)),
                display: colour::Display::default(),
                compressed: true,
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
            real_light: false,
            probe: None,
            sphere: None,
            lens: None,
            sun_device: [0.0, 0.0, 1.0],
            follow_heading: false,
            declination_deg: 0.0,
            have_rotation: false,
            layer_mask: LayerMask::default(),
            last_pin: None,
            plan_cache: None,
            measure: None,
            plan_cached: false,
            themes: theme::builtin(),
            theme: 0,
            style_version: 0,
            light_version: 0,
            lock_to_fix: false,
        }
    }

    // ==================== INPUT ====================

    pub fn clamp_camera(&mut self) {
        self.cam.lat = self.cam.lat.clamp(-89.9, 89.9);
        if self.cam.lon > 180.0 {
            self.cam.lon -= 360.0;
        } else if self.cam.lon < -180.0 {
            self.cam.lon += 360.0;
        }
    }

    /// Pan by a screen-pixel delta: the ground at the centre goes that far across the screen, the globe turning under the camera.
    pub fn pan(&mut self, dx: f64, dy: f64, w: usize, h: usize) {
        let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
        let (lat, lon) = self.cam.screen_to_geo(cx, cy, w, h);
        self.place_anchor(lat, lon, cx + dx, cy + dy, w, h);
    }

    /// The least pixels per degree a screen may show: the whole globe as a disk filling 94% of the shorter side.
    pub fn ppd_floor(w: usize, h: usize) -> f64 {
        (0.47 * w.min(h) as f64 * core::f64::consts::PI / 180.0).max(1.0)
    }

    /// Exact geo-anchored zoom: the geography under (ax, ay) stays there.

    pub fn zoom_about(&mut self, factor: f64, ax: f64, ay: f64, w: usize, h: usize) {
        let (alat, alon) = self.cam.screen_to_geo(ax, ay, w, h);
        self.cam.ppd = (self.cam.ppd * factor).clamp(Self::ppd_floor(w, h), 4_000_000.);
        self.place_anchor(alat, alon, ax, ay, w, h);
    }

    /// Re-solve the camera so (alat, alon) sits at screen (ax, ay): the globe turns, in view space, by the one rotation that carries the anchor from where it appears to where it should, about the axis between the two. Exact at every zoom, the limb included, and parallel transport along the way, so a pan near or across a pole does not spin the map (Nick 2026-10-09).
    pub fn place_anchor(&mut self, alat: f64, alon: f64, ax: f64, ay: f64, w: usize, h: usize) {
        let b = self.cam.basis();
        let p = Camera::unit(alat, alon);
        let from = [dot(p, b[0]), dot(p, b[1]), dot(p, b[2])];
        let to = self.cam.view_dir(ax, ay, w, h);
        let mut axis = cross(from, to);
        let n = dot(axis, axis).sqrt();
        let angle = dot(from, to).clamp(-1.0, 1.0).acos();
        if n < 1e-12 {
            if angle < 1e-9 {
                self.camera_moved(w, h);
                return;
            }
            // Antipodal: half a turn about any axis across the line, screen up.
            axis = [0.0, 1.0, 0.0];
        } else {
            axis = [axis[0] / n, axis[1] / n, axis[2] / n];
        }
        // The rotation acts in view space; the basis vectors (world) move by its rows: new row_i = Σ_j D_ij row_j, where D's columns are the turned axes.
        let e = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let cols = [rotate(e[0], axis, angle), rotate(e[1], axis, angle), rotate(e[2], axis, angle)];
        let mut nb = [[0.0; 3]; 3];
        for i in 0..3 {
            for k in 0..3 {
                nb[i][k] = cols[0][i] * b[0][k] + cols[1][i] * b[1][k] + cols[2][i] * b[2][k];
            }
        }
        self.cam = Camera::from_basis(nb, self.cam.ppd);
        self.clamp_camera();
        self.camera_moved(w, h);
    }

    pub fn set_bearing(&mut self, bearing: f64) {
        self.cam.bearing = bearing.rem_euclid(core::f64::consts::TAU);
        self.unlock_measure();
        self.dirty = true;
    }

    pub fn set_ppd(&mut self, ppd: f64) {
        self.cam.ppd = ppd.clamp(6., 4_000_000.);
        self.unlock_measure();
        self.dirty = true;
    }

    /// Look at a place: the camera point goes there, north up, at the zoom given or the one it has.
    pub fn go_to(&mut self, lat: f64, lon: f64, ppd: Option<f64>, w: usize, h: usize) {
        self.cam.lat = lat;
        self.cam.lon = lon;
        self.cam.bearing = 0.0;
        if let Some(p) = ppd {
            self.cam.ppd = p.clamp(Self::ppd_floor(w, h), 4_000_000.);
        }
        self.clamp_camera();
        self.camera_moved(w, h);
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

    /// A fix redraws only when the pin is on screen and has moved by a pixel (Nick 2026-10-06): a phone sitting still gets no frame per second from its GPS.
    pub fn set_gps(&mut self, fix: GpsFix) {
        self.gps = Some(fix);
        let (w, h) = (self.canvas_w, self.canvas_h);
        if w == 0 || h == 0 {
            self.dirty = true;
            return;
        }
        let (x, y) = self.cam.geo_to_screen(fix.lat, fix.lon, w, h);
        let (x, y) = (x as f32, y as f32);
        let px_per_m = (self.cam.ppd / 111_320.0) as f32;
        let r = (fix.accuracy_m * px_per_m).clamp(6.0, 4000.0);
        let on_screen = x + r >= 0.0 && y + r >= 0.0 && x - r <= w as f32 && y - r <= h as f32;
        let moved = self.last_pin.is_none_or(|(lx, ly)| (lx - x).abs() >= 1.0 || (ly - y).abs() >= 1.0);
        if on_screen && moved {
            self.last_pin = Some((x, y));
            self.dirty = true;
        }
    }

    /// The client's layer filter: which of dem / land / water / line draw.
    pub fn layers(&self) -> LayerMask {
        self.layer_mask
    }

    /// The client's choices as made; what draws is `effective()` of them.
    pub fn set_layers(&mut self, mask: LayerMask) {
        self.layer_mask = mask;
        self.luts.mask = mask.effective();
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
                self.cam.bearing = (self.true_heading() as f64).to_radians().rem_euclid(core::f64::consts::TAU);
            }
            self.dirty = true;
        }
    }

    /// The full orientation: the sensor's rotation matrix, row-major, world = R · device. The heading falls out of it (Android's own formula), and with the real sun on the landscape is lit exactly as the phone is held.
    pub fn set_device_rotation(&mut self, r: [f32; 9]) {
        let changed = self.device_rot.iter().zip(&r).any(|(a, b)| (a - b).abs() > 0.002);
        self.device_rot = r;
        self.have_rotation = true;
        // Nothing on screen depends on the orientation unless a mode uses it: no frame for a phone merely being held.
        if !self.real_sun && !self.follow_heading && !self.real_light {
            return;
        }
        let heading = r[1].atan2(r[4]).to_degrees().rem_euclid(360.0);
        self.set_device_heading(heading);
        if changed {
            // The painted sphere is world-fixed: the phone turning under it is a new light.
            if self.real_light {
                self.relight_probe();
            }
            self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
            self.dirty = true;
        }
    }

    /// The device's heading from true north, degrees clockwise.
    pub fn true_heading(&self) -> f32 {
        (self.device_heading + self.declination_deg).rem_euclid(360.0)
    }

    pub fn set_declination(&mut self, deg: f32) {
        if (self.declination_deg - deg).abs() > 0.05 {
            self.declination_deg = deg;
            self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
            if self.follow_heading {
                self.cam.bearing = (self.true_heading() as f64).to_radians();
            }
            self.dirty = true;
        }
    }

    pub fn set_real_sun(&mut self, on: bool) {
        self.real_sun = on;
        self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
        self.dirty = true;
    }

    pub fn set_real_light(&mut self, on: bool) {
        self.real_light = on;
        self.probe = None;
        self.sphere = None;
        self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
        self.dirty = true;
    }

    /// A new frame from the front camera, upright, in absolute units: painted onto the sphere where the phone points, and the light taken from the sphere.
    pub fn paint_probe(&mut self, w: usize, h: usize, tan_w: f32, tan_h: f32, rgb: &[[u32; 3]]) {
        if !self.real_light || rgb.len() != w * h {
            return;
        }
        if self.lens.as_ref().is_none_or(|l| !l.fits(w, h, tan_w, tan_h)) {
            self.lens = Some(probe::Lens::new(w, h, tan_w, tan_h));
        }
        let sphere = self.sphere.get_or_insert_with(|| Box::new(probe::Sphere::new()));
        sphere.paint(rgb, self.lens.as_ref().unwrap(), &self.device_rot);
        self.relight_probe();
    }

    /// The sphere's light at the current rotation.
    fn relight_probe(&mut self) {
        let Some(sphere) = &mut self.sphere else { return };
        self.probe = sphere.light(&self.device_rot);
        self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
        self.dirty = true;
    }

    /// The light as a ball the size of the gear, `size × size` display RGBA, oriented to the screen (x right, y up, z out): with real light, the painted sphere itself, every visible point the radiance in that direction; with the real sun, a white ball lit by it with the sun as a point where it stands. None when neither mode is on.
    pub fn orb(&self, size: usize) -> Option<Vec<[u8; 4]>> {
        if !(self.real_light || self.real_sun) || size == 0 {
            return None;
        }
        let t = &self.themes[self.theme.min(self.themes.len() - 1)];
        let sun = colour::lin(t.sun.map(|v| (v.sqrt() * 256.0).min(255.0) as u8));
        let mut out = vec![[0u8; 4]; size * size];
        let r = size as f32 * 0.5;
        for py in 0..size {
            for px in 0..size {
                let (x, y) = ((px as f32 + 0.5 - r) / r, (r - py as f32 - 0.5) / r);
                let rr = x * x + y * y;
                if rr > 1.0 {
                    continue;
                }
                let d = [x, y, (1.0 - rr).sqrt()];
                let lit: Option<[f32; 3]> = if self.real_light {
                    self.sphere.as_ref().map(|s| {
                        let rot = &self.device_rot;
                        s.radiance([rot[0] * d[0] + rot[1] * d[1] + rot[2] * d[2], rot[3] * d[0] + rot[4] * d[1] + rot[5] * d[2], rot[6] * d[0] + rot[7] * d[1] + rot[8] * d[2]])
                    })
                } else {
                    let s = self.sun_device;
                    let cos = d[0] * s[0] + d[1] * s[1] + d[2] * s[2];
                    // The disc of the sun: a few degrees across, white.
                    if cos > 0.996 { Some([4.0; 3]) } else { Some([sun[0] * cos.max(0.0) + 0.04, sun[1] * cos.max(0.0) + 0.04, sun[2] * cos.max(0.0) + 0.05]) }
                };
                if let Some(l) = lit {
                    let c = self.luts.display.encode(l, self.luts.compressed);
                    out[py * size + px] = [c[0], c[1], c[2], 255];
                }
            }
        }
        Some(out)
    }

    /// For the log: the sphere's level, and the light's luminance on the world's up and on the screen's normal after it, with the brightest triangle's luminance on the map's scale.
    pub fn probe_report(&self) -> Option<(f32, f32, f32, f32)> {
        let s = self.sphere.as_ref()?;
        let p = self.probe.as_ref()?;
        let lum = |e: [f32; 3]| 0.3 * e[0] + 0.6 * e[1] + 0.1 * e[2];
        let r = &self.device_rot;
        // World up in the device frame: rotᵀ · (0, 0, 1).
        let up_dev = [r[6], r[7], r[8]];
        let brightest = s.tris.iter().map(|t| lum([t[0] as f32, t[1] as f32, t[2] as f32])).fold(0.0, f32::max) * std::f32::consts::PI / s.level.max(1e-6);
        Some((s.level, lum(p.irradiance(up_dev)), lum(p.irradiance([0.0, 0.0, 1.0])), brightest))
    }

    /// How much of the sphere the camera has painted, 0..1: the triangles no longer black.
    pub fn probe_coverage(&self) -> f32 {
        self.sphere.as_ref().map_or(0.0, |s| s.tris.iter().filter(|&&v| v != [0; 3]).count() as f32 / probe::TRIS as f32)
    }

    pub fn set_follow_heading(&mut self, on: bool) {
        self.follow_heading = on;
        if on {
            self.cam.bearing = (self.true_heading() as f64).to_radians().rem_euclid(core::f64::consts::TAU);
        }
        self.dirty = true;
    }

    pub fn camera_moved(&mut self, _w: usize, _h: usize) {
        self.unlock_measure();
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

    pub fn pending_cells(&self) -> usize {
        self.res.pending_count()
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
    pub fn plan(&mut self, w: usize, h: usize) -> Arc<plan::FramePlan> {
        let t0 = Instant::now();
        self.prepare(w, h);
        // The same view over the same cells is the same plan: a frame the sensor or the GPS made dirty costs nothing here.
        let key = (self.cam.lat.to_bits(), self.cam.lon.to_bits(), self.cam.ppd.to_bits(), self.cam.bearing.to_bits(), w, h, self.res.pool_version, self.dem_depth, self.vec_depth);
        self.plan_cached = false;
        if let Some((k, p)) = &self.plan_cache {
            if *k == key {
                self.plan_cached = true;
                let p = p.clone();
                // Residency still hears the want: it re-sends an unchanged missing list once a second.
                let center = Coord::from_lat_lon(self.cam.lat, self.cam.lon);
                let (cu, cv) = center.uv();
                self.res.want(p.want.clone(), (cu, cv));
                self.last_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
                return p;
            }
        }
        let p = Arc::new(plan::plan_frame(w, h, &self.cam, &self.res.pool, self.dem_depth, self.vec_depth));
        self.last_straddle_blocks = p.straddle_blocks;
        self.last_range = p.elev;
        let center = Coord::from_lat_lon(self.cam.lat, self.cam.lon);
        let (cu, cv) = center.uv();
        self.res.want(p.want.clone(), (cu, cv));
        self.plan_cache = Some((key, p.clone()));
        self.last_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
        p
    }

    /// Switch themes: the tables, the lighting colours and the theme's layer defaults.
    /// The themes a host loaded (display-converted); the current one is re-applied by name, or the first if it is gone.
    pub fn set_themes(&mut self, themes: Vec<theme::Theme>) {
        if themes.is_empty() {
            return;
        }
        let name = self.themes.get(self.theme).map(|t| t.name.clone());
        self.themes = themes;
        let i = name.and_then(|n| self.themes.iter().position(|t| t.name == n)).unwrap_or(0);
        self.set_theme(i);
    }

    pub fn set_theme(&mut self, i: usize) {
        self.theme = i.min(self.themes.len() - 1);
        let t = self.themes[self.theme].clone();
        self.luts.hypso = t.hypso_lut();
        self.luts.style = t.style();
        self.set_layers(t.layers);
        self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
        self.style_version += 1;
        self.dirty = true;
    }

    /// Highlights compressed (exposure 2/3 into Opsin's rail) or linear (the stored range straight, 2.3 stops darker).
    pub fn compressed(&self) -> bool {
        self.luts.compressed
    }

    pub fn set_compressed(&mut self, on: bool) {
        self.luts.compressed = on;
        self.style_version += 1;
        self.dirty = true;
    }

    /// Change one field of the current theme, live: the tables and the light follow at once. Nothing is written; the host saves when the editor closes.
    pub fn edit_theme(&mut self, f: theme::Field, v: [f32; 4]) {
        let i = self.theme.min(self.themes.len() - 1);
        self.themes[i].set(f, v);
        self.themes[i].edited = true;
        let t = self.themes[i].clone();
        self.luts.hypso = t.hypso_lut();
        self.luts.style = t.style();
        self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
        self.style_version += 1;
        self.dirty = true;
    }

    /// The current theme back to what shipped, when it is a built-in: true if it changed.
    pub fn reset_theme(&mut self) -> bool {
        let i = self.theme.min(self.themes.len() - 1);
        let Some(shipped) = self.themes[i].shipped() else { return false };
        self.themes[i] = shipped;
        self.set_theme(i);
        true
    }

    /// Remove the current theme when it is the user's own; the first theme takes its place. Returns the removed theme's name.
    pub fn delete_theme(&mut self) -> Option<String> {
        let i = self.theme.min(self.themes.len() - 1);
        if self.themes[i].is_builtin() || self.themes.len() < 2 {
            return None;
        }
        let gone = self.themes.remove(i);
        self.set_theme(0);
        Some(gone.name)
    }

    /// A copy of the current theme under a new name (its name with the first free number), a user theme, selected.
    pub fn duplicate_theme(&mut self) -> usize {
        let i = self.theme.min(self.themes.len() - 1);
        let mut t = self.themes[i].clone();
        let base = t.name.trim_end_matches(|c: char| c.is_ascii_digit() || c == ' ').to_string();
        let mut n = 2;
        while self.themes.iter().any(|x| x.name == format!("{base} {n}")) {
            n += 1;
        }
        t.name = format!("{base} {n}");
        t.revision = 0;
        t.edited = true;
        self.themes.push(t);
        let j = self.themes.len() - 1;
        self.set_theme(j);
        j
    }

    pub fn current_theme(&self) -> &theme::Theme {
        &self.themes[self.theme.min(self.themes.len() - 1)]
    }

    pub fn style_version(&self) -> u64 {
        self.style_version
    }

    pub fn light_version(&self) -> u64 {
        self.light_version
    }

    pub fn plan_cached(&self) -> bool {
        self.plan_cached
    }

    pub fn pool_version(&self) -> u64 {
        self.res.pool_version
    }

    /// The resident cells, for a renderer that mirrors them and may release what it has copied.
    pub fn pool_mut(&mut self) -> &mut residency::Pool {
        &mut self.res.pool
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
        // The real sun: azimuth and altitude from the clock and the position (the fix, or the view), as a screen-relative azimuth. Nothing is clamped: below the horizon it is below the horizon.
        if self.real_sun {
            let (lat, lon) = self.gps.map_or((self.cam.lat, self.cam.lon), |g| (g.lat, g.lon));
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
            let (az, alt) = sh::sun_position(lat, lon, now);
            // The clock moves the sun a few thousandths of a degree a second: follow it in steps too small to see, so a still screen is not relit every frame.
            let az = ((az - self.cam.bearing.to_degrees()).rem_euclid(360.0)) as f32;
            if (az - self.sun_az).abs() > 0.05 || (alt as f32 - self.sun_alt).abs() > 0.05 {
                self.sun_az = az;
                self.sun_alt = alt as f32;
            }
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
            if let (true, Some(env)) = (self.real_light, self.probe.as_ref()) {
                // The camera's environment is already in the device frame; the bearing conjugates it onto world normals as for the sun and sky.
                let (sb, cb) = self.cam.bearing.sin_cos();
                self.luts.light = env.quadratic((sb as f32, cb as f32));
                self.luts_sun = (self.sun_az, self.sun_alt, self.cam.bearing);
                self.light_version += 1;
            } else if self.real_sun {
                // The real sun alone, in the device frame: the landscape is lit exactly as the phone is held, by max(0, n·sun) and nothing else. No sky, no fading: at night the sun is under the landscape and it renders black; turn the phone over and the sun lights it from below (Nick 2026-10-06).
                let (lat, lon) = self.gps.map_or((self.cam.lat, self.cam.lon), |g| (g.lat, g.lon));
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
                let (az, alt) = sh::sun_position(lat, lon, now);
                // The frames, in order. The almanac gives a true azimuth; the sensor's world frame has magnetic north on its Y axis, so the sun is first expressed against magnetic north (azimuth less the declination, east positive). Rᵀ then takes it into the device frame: x right, y up the screen, z out of it — the lighting frame, since the Activity is locked to portrait. Last, when the map's up is not where the phone points, the device-frame sun is turned by (bearing − true heading): a device turned clockwise sees a fixed world vector turn counterclockwise, so this is the lighting the phone would show turned to match the map, tilt kept. With follow heading on the two agree and nothing turns; without a sensor (R identity, heading = declination) it collapses to the map-locked rotation by the bearing.
                let frames = Frames { declination_deg: self.declination_deg, map_locked: !self.have_rotation, rot: self.device_rot, bearing_deg: self.cam.bearing.to_degrees() as f32, heading_mag_deg: self.device_heading };
                let t = &self.themes[self.theme.min(self.themes.len() - 1)];
                let (sb, cb) = self.cam.bearing.sin_cos();
                self.sun_device = frames.to_screen(az as f32, alt as f32);
                self.luts.light = sh::Quad::directional(self.sun_device, t.sun, (sb as f32, cb as f32));
                self.luts_sun = (self.sun_az, self.sun_alt, self.cam.bearing);
                self.light_version += 1;
            } else {
                if self.luts_sun.0 != self.sun_az || self.luts_sun.1 != self.sun_alt || self.device_heading != 0.0 || self.luts_sun.2.is_nan() {
                    let t = &self.themes[self.theme.min(self.themes.len() - 1)];
                    self.env = sh::Sh9::sun_and_sky_coloured(self.sun_az - self.device_heading, self.sun_alt, t.sun, t.sky);
                }
                let (sb, cb) = self.cam.bearing.sin_cos();
                self.luts.light = self.env.quadratic((sb as f32, cb as f32));
                self.luts_sun = (self.sun_az, self.sun_alt, self.cam.bearing);
                self.light_version += 1;
            }
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
        self.draw_measure(w, h);
        self.last_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    /// The measurement for a CPU frontend: the line from origin to target and a crosshair on the target.
    fn draw_measure(&mut self, w: usize, h: usize) {
        let Some(m) = self.measure_view(w, h, 2) else { return };
        const INK: [u8; 3] = [255, 196, 64];
        let (ox, oy) = m.origin_px;
        let (tx, ty) = m.target_px;
        draw_segment(&mut self.canvas, w, h, ox, oy, tx, ty, 1.3, INK);
        for (dx0, dy0, dx1, dy1) in [(-12., 0., 12., 0.), (0., -12., 0., 12.)] {
            draw_segment(&mut self.canvas, w, h, tx + dx0, ty + dy0, tx + dx1, ty + dy1, 1.6, INK);
        }
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

/// The chain from the sky to the screen, so it can be tested on paper cases: true azimuth → magnetic frame → device frame (the sensor's matrix transposed) → turned by (map bearing − true heading) so the lighting is what the phone would show turned to match the map.
#[derive(Clone, Copy)]
pub struct Frames {
    pub declination_deg: f32,
    /// True when there is no orientation sensor: the device-frame sun is turned by (bearing − heading) so the lighting is the map's, as if the phone were turned to match it. With a sensor the terrain is a relief model held in the hand and the real sun lights it however the phone is turned (Nick 2026-10-06).
    pub map_locked: bool,
    /// Row-major, world (east, magnetic north, up) = rot · device (x right, y up the screen, z out).
    pub rot: [f32; 9],
    pub bearing_deg: f32,
    pub heading_mag_deg: f32,
}

impl Frames {
    fn device(&self, v: [f32; 3]) -> [f32; 3] {
        let r = self.rot;
        let d = [r[0] * v[0] + r[3] * v[1] + r[6] * v[2], r[1] * v[0] + r[4] * v[1] + r[7] * v[2], r[2] * v[0] + r[5] * v[1] + r[8] * v[2]];
        if !self.map_locked {
            return d;
        }
        let (sd, cd) = (self.bearing_deg - (self.heading_mag_deg + self.declination_deg)).to_radians().sin_cos();
        [d[0] * cd - d[1] * sd, d[0] * sd + d[1] * cd, d[2]]
    }

    /// A direction given as true azimuth (clockwise from true north) and altitude, in the screen frame.
    pub fn to_screen(&self, az_true_deg: f32, alt_deg: f32) -> [f32; 3] {
        let (az, alt) = ((az_true_deg - self.declination_deg).to_radians(), alt_deg.to_radians());
        self.device([az.sin() * alt.cos(), az.cos() * alt.cos(), alt.sin()])
    }

    /// Earth's vertical in the screen frame.
    pub fn up(&self) -> [f32; 3] {
        self.device([0.0, 0.0, 1.0])
    }
}

#[cfg(test)]
mod frame_tests {
    use super::Frames;

    const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    /// Phone flat, top pointing east: device x is south, device y is east.
    const POINTING_EAST: [f32; 9] = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn a_south_sun_lights_from_the_bottom_of_a_north_up_map() {
        // No sensor: the sun's screen position is the map's.
        let f = Frames { declination_deg: 0.0, map_locked: true, rot: IDENTITY, bearing_deg: 0.0, heading_mag_deg: 0.0 };
        assert!(close(f.to_screen(180.0, 0.0), [0.0, -1.0, 0.0]));
        // The phone points east with the map still north-up: the same answer, the lighting is the map's.
        let f = Frames { declination_deg: 0.0, map_locked: true, rot: POINTING_EAST, bearing_deg: 0.0, heading_mag_deg: 90.0 };
        assert!(close(f.to_screen(180.0, 0.0), [0.0, -1.0, 0.0]));
        // Follow heading: the map turns with the phone and the south sun is to the right, as it physically is.
        let f = Frames { declination_deg: 0.0, map_locked: true, rot: POINTING_EAST, bearing_deg: 90.0, heading_mag_deg: 90.0 };
        assert!(close(f.to_screen(180.0, 0.0), [1.0, 0.0, 0.0]));
        // Declination 15° east with no sensor collapses to the map-locked answer exactly.
        let f = Frames { declination_deg: 15.0, map_locked: true, rot: IDENTITY, bearing_deg: 0.0, heading_mag_deg: 0.0 };
        assert!(close(f.to_screen(180.0, 0.0), [0.0, -1.0, 0.0]));
        assert!(close(f.up(), [0.0, 0.0, 1.0]));
        // A map turned 90° clockwise (screen-up east) shows the south sun on the right.
        let f = Frames { declination_deg: 0.0, map_locked: true, rot: IDENTITY, bearing_deg: 90.0, heading_mag_deg: 0.0 };
        assert!(close(f.to_screen(180.0, 0.0), [1.0, 0.0, 0.0]));
        // With a sensor the terrain is held in the hand: the phone pointing east sees the south sun on its right whatever the map's bearing.
        let f = Frames { declination_deg: 0.0, map_locked: false, rot: POINTING_EAST, bearing_deg: 0.0, heading_mag_deg: 90.0 };
        assert!(close(f.to_screen(180.0, 0.0), [1.0, 0.0, 0.0]));
    }

    #[test]
    fn a_phone_tilted_away_from_the_sun_sees_it_below_the_screen() {
        // Phone on its back with its top edge lifted 60°: the top of the screen points north and up, device y = (0, cos60, sin60) in world, and the screen faces south and up, device z = (0, −sin60, cos60).
        let (s, c) = 60f32.to_radians().sin_cos();
        let rot = [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c];
        let f = Frames { declination_deg: 0.0, map_locked: true, rot, bearing_deg: 0.0, heading_mag_deg: 0.0 };
        // A north sun 30° up is behind the screen: negative z. A south one is in front.
        let north = f.to_screen(0.0, 30.0);
        assert!(north[2] < 0.0, "{north:?}");
        let south = f.to_screen(180.0, 30.0);
        assert!(south[2] > 0.0, "{south:?}");
        // Earth's up leans toward the top of the screen and still comes out of it.
        let up = f.up();
        assert!(up[1] > 0.8 && up[2] > 0.4, "{up:?}");
    }
}

// ==================== MEASUREMENT ====================

/// A tapped point and where it is measured from: the fix while locked to it, the screen centre otherwise. A pan, zoom or rotate unlocks; a tap starts a new one.
#[derive(Clone, Copy, Debug)]
pub struct Measure {
    pub target: (f64, f64),
    pub from_fix: bool,
}

/// A measurement as seen this frame: screen points, distance and bearing from origin to target, elevation at both ends, and the profile along the way.
#[derive(Clone, Debug)]
pub struct MeasureView {
    pub origin_px: (f32, f32),
    pub target_px: (f32, f32),
    pub distance_m: f64,
    pub bearing_deg: f64,
    pub elev_origin: Option<f32>,
    pub elev_target: Option<f32>,
    /// Elevation every step from origin to target, NaN where no terrain is resident.
    pub profile: Vec<f32>,
}

impl MapCore {
    /// A tap on the map at screen (x, y): a new measurement to that point.
    pub fn tap(&mut self, x: f64, y: f64, w: usize, h: usize) {
        let target = self.cam.screen_to_geo(x, y, w, h);
        self.measure = Some(Measure { target, from_fix: self.lock_to_fix && self.gps.is_some() });
        self.dirty = true;
    }

    pub fn clear_measure(&mut self) {
        self.measure = None;
        self.dirty = true;
    }

    pub fn set_lock_to_fix(&mut self, on: bool) {
        self.lock_to_fix = on;
        if let Some(m) = &mut self.measure {
            m.from_fix = on && self.gps.is_some();
        }
        self.dirty = true;
    }

    /// The camera moved: a measurement from the fix now measures from the screen centre.
    fn unlock_measure(&mut self) {
        if let Some(m) = &mut self.measure {
            m.from_fix = false;
        }
    }

    /// The origin of the current measurement, geographic.
    fn measure_origin(&self, w: usize, h: usize) -> Option<(f64, f64)> {
        let m = self.measure?;
        if m.from_fix {
            if let Some(g) = self.gps {
                return Some((g.lat, g.lon));
            }
        }
        Some(self.cam.screen_to_geo(w as f64 * 0.5, h as f64 * 0.5, w, h))
    }

    /// The measurement for a `w × h` screen with a profile of `samples` points, if one is set.
    pub fn measure_view(&self, w: usize, h: usize, samples: usize) -> Option<MeasureView> {
        let m = self.measure?;
        let (olat, olon) = self.measure_origin(w, h)?;
        let (tlat, tlon) = m.target;
        let (ox, oy) = self.cam.geo_to_screen(olat, olon, w, h);
        let (tx, ty) = self.cam.geo_to_screen(tlat, tlon, w, h);
        // Flat-earth over the span a screen can show: metres east and north.
        let m_lat = 111_320.0;
        let m_lon = 111_320.0 * ((olat + tlat) * 0.5).to_radians().cos();
        let (de, dn) = ((tlon - olon) * m_lon, (tlat - olat) * m_lat);
        let distance_m = (de * de + dn * dn).sqrt();
        let bearing_deg = de.atan2(dn).to_degrees().rem_euclid(360.0);
        let n = samples.max(2);
        let profile = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                self.elevation_at(olat + (tlat - olat) * t, olon + (tlon - olon) * t).unwrap_or(f32::NAN)
            })
            .collect();
        Some(MeasureView {
            origin_px: (ox as f32, oy as f32),
            target_px: (tx as f32, ty as f32),
            distance_m,
            bearing_deg,
            elev_origin: self.elevation_at(olat, olon),
            elev_target: self.elevation_at(tlat, tlon),
            profile,
        })
    }

    pub fn has_measure(&self) -> bool {
        self.measure.is_some()
    }
}

#[cfg(test)]
mod globe_tests {
    use super::Camera;

    /// The projection inverts itself across the visible hemisphere, and the far side lands off the disk.
    #[test]
    fn the_globe_projection_round_trips() {
        let cam = Camera { lat: 47.0, lon: -121.0, ppd: 20.0, bearing: 0.7 };
        let (w, h) = (1200, 800);
        for &(lat, lon) in &[(47.0, -121.0), (21.3, -157.8), (64.0, -150.0), (10.0, -80.0), (47.1, -120.9), (-20.0, -170.0)] {
            let (px, py) = cam.geo_to_screen(lat, lon, w, h);
            assert!(cam.on_globe(px, py, w, h), "{lat},{lon} on the near side");
            let (la, lo) = cam.screen_to_geo(px, py, w, h);
            assert!((la - lat).abs() < 1e-6 && (lo - lon).abs() < 1e-6, "{lat},{lon} came back as {la},{lo}");
        }
        let (px, py) = cam.geo_to_screen(-47.0, 59.0, w, h);
        assert!(!cam.on_globe(px, py, w, h), "the antipode is off the disk");
        assert!((cam.radius() - 20.0 * 180.0 / core::f64::consts::PI).abs() < 1e-9);
    }

    /// A pan is a rigid slide of the map on the screen, even near a pole: the ground above the drag target ends the same distance above the centre, the bearing having followed the great circle.
    #[test]
    fn a_pan_near_the_pole_does_not_spin() {
        use super::MapCore;
        let (w, h) = (1000, 800);
        struct Nothing;
        impl crate::residency::CellStore for Nothing {
            fn get(&self, _key: mahere_tiles::CellKey) -> crate::residency::Fetch {
                crate::residency::Fetch::Absent
            }
        }
        let mut map = MapCore::new(std::sync::Arc::new(Nothing), Camera { lat: 84.0, lon: 30.0, ppd: 20.0, bearing: 0.4 });
        for _ in 0..6 {
            let (dx, dy) = (120.0, -260.0);
            let target = (w as f64 * 0.5 - dx, h as f64 * 0.5 - dy);
            let above = map.cam.screen_to_geo(target.0, target.1 - 100.0, w, h);
            map.pan(dx, dy, w, h);
            let (px, py) = map.cam.geo_to_screen(above.0, above.1, w, h);
            // A sphere's pan is not quite a rigid slide: a screen-vertical line through the target is a great circle only through the centre, so a point 100 px up lands within a couple of pixels, never spun away.
            assert!((px - w as f64 * 0.5).abs() < 3.0 && (py - (h as f64 * 0.5 - 100.0)).abs() < 3.0, "the ground above the target drifted to {px},{py}");
        }
    }

    /// A screen corner off the globe still names ground, at the limb.
    #[test]
    fn a_corner_off_the_globe_names_the_limb() {
        let cam = Camera { lat: 0.0, lon: 0.0, ppd: 6.0, bearing: 0.0 };
        let (lat, lon) = cam.screen_to_geo(0.0, 0.0, 1024, 768);
        assert!(lat.is_finite() && lon.is_finite());
        let (px, py) = cam.geo_to_screen(lat, lon, 1024, 768);
        let (dx, dy) = (px - 512.0, py - 384.0);
        assert!(((dx * dx + dy * dy).sqrt() - cam.radius()).abs() < 1e-3, "at the limb");
    }
}
