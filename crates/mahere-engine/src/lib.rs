//! mahere's host-independent map core — the #pagetable edition. A frame is fetches: 32 px block grid → diamond UV → page table → plane fetch → LUT composite. No vectors, no reservoir, no source data at runtime: the renderer's entire diet is baked cells served by a CellStore through the clipmap residency layer. Frontends feed input and blit `canvas`.

pub mod areas;
pub mod colour;
pub mod ink;
pub mod labels;
pub mod plan;
pub mod probe;
pub mod raster;
pub mod residency;
pub mod route;
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

/// View state: the globe seen from far off, orthographically. The view is a rotation of the ball, kept as its basis in world coordinates (the unit sphere, x toward 0°N 0°E, z toward the north pole): screen right, screen up, and the axis toward the viewer, which is the camera point at the screen's centre. Latitude, longitude and bearing are read off it, never stored, so no pole is special: an unmarked ball (Nick 2026-10-10). ppd = pixels per degree of arc at the centre, so the whole globe is a disk of radius ppd·180/π pixels.
#[derive(Clone, Copy)]
pub struct Camera {
    basis: [[f64; 3]; 3],
    pub ppd: f64,
}

impl Camera {
    /// The view over (lat, lon) with screen up turned `bearing` radians clockwise from north.
    pub fn new(lat: f64, lon: f64, ppd: f64, bearing: f64) -> Camera {
        let (sl, cl) = lat.to_radians().sin_cos();
        let (so, co) = lon.to_radians().sin_cos();
        let east = [-so, co, 0.0];
        let north = [-sl * co, -sl * so, cl];
        let centre = [cl * co, cl * so, sl];
        let (sb, cb) = bearing.sin_cos();
        let right = [east[0] * cb - north[0] * sb, east[1] * cb - north[1] * sb, east[2] * cb - north[2] * sb];
        let up = [east[0] * sb + north[0] * cb, east[1] * sb + north[1] * cb, east[2] * sb + north[2] * cb];
        Camera { basis: [right, up, centre], ppd }
    }

    /// The camera whose view has this basis, made orthonormal again (the axis toward the viewer kept, right squared against it, up their cross).
    pub fn from_basis(b: [[f64; 3]; 3], ppd: f64) -> Camera {
        let c = normalize(b[2]);
        let d = dot(b[0], c);
        let r = normalize([b[0][0] - d * c[0], b[0][1] - d * c[1], b[0][2] - d * c[2]]);
        let u = cross(c, r);
        Camera { basis: [r, u, c], ppd }
    }

    pub fn basis(&self) -> [[f64; 3]; 3] {
        self.basis
    }

    /// Latitude of the camera point, degrees.
    pub fn lat(&self) -> f64 {
        self.basis[2][2].clamp(-1.0, 1.0).asin().to_degrees()
    }

    /// Longitude of the camera point, degrees in (−180, 180].
    pub fn lon(&self) -> f64 {
        self.basis[2][1].atan2(self.basis[2][0]).to_degrees()
    }

    /// Screen up's turn clockwise from north at the camera point, radians in [0, 2π). At a pole north is whichever meridian the longitude names, so the pair stays consistent.
    pub fn bearing(&self) -> f64 {
        let (sl, cl) = self.lat().to_radians().sin_cos();
        let (so, co) = self.lon().to_radians().sin_cos();
        let east = [-so, co, 0.0];
        let north = [-sl * co, -sl * so, cl];
        let r = self.basis[0];
        (-dot(r, north)).atan2(dot(r, east)).rem_euclid(core::f64::consts::TAU)
    }

    pub fn coslat(&self) -> f64 {
        let z = self.basis[2][2];
        (1.0 - z * z).max(0.0).sqrt()
    }

    /// Turn the view about its own axis by `delta` radians, screen up going clockwise: pure rotation of the ball, no angle read back, so it is as good at a pole as anywhere (Nick 2026-10-10: turning at a pole spun the globe, since the bearing read there is noise).
    pub fn rotate_view(&mut self, delta: f64) {
        let (s, c) = delta.sin_cos();
        let [r, u, centre] = self.basis;
        let right = [r[0] * c - u[0] * s, r[1] * c - u[1] * s, r[2] * c - u[2] * s];
        let up = [u[0] * c + r[0] * s, u[1] * c + r[1] * s, u[2] * c + r[2] * s];
        self.basis = [right, up, centre];
    }

    /// Turn the view about its own axis so screen up is this bearing (an absolute target, which at a pole means little; gestures turn by [`Camera::rotate_view`]).
    pub fn set_bearing(&mut self, bearing: f64) {
        let delta = bearing - self.bearing();
        self.rotate_view(delta);
    }

    /// Look at a place, screen up at this bearing.
    pub fn look_at(&mut self, lat: f64, lon: f64, bearing: f64) {
        *self = Camera::new(lat, lon, self.ppd, bearing);
    }

    /// The globe's radius on the screen, in pixels: pixels per radian of arc.
    pub fn radius(&self) -> f64 {
        self.ppd * 180.0 / core::f64::consts::PI
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

    /// Where a place lands on the screen. A place on the far side of the globe is pushed well outside the disk, so a mark there draws nowhere.
    pub fn geo_to_screen(&self, lat: f64, lon: f64, w: usize, h: usize) -> (f64, f64) {
        let p = Camera::unit(lat, lon);
        let r = self.radius();
        let mut x = dot(p, self.basis[0]) * r;
        let mut y = dot(p, self.basis[1]) * r;
        if dot(p, self.basis[2]) < 0.0 {
            let rho = (x * x + y * y).sqrt();
            if rho < 1e-6 {
                // The antipode itself has no direction: straight up and away.
                x = 0.0;
                y = r * 4.0;
            } else {
                let k = r * 4.0 / rho;
                x *= k;
                y *= k;
            }
        }
        (w as f64 * 0.5 + x, h as f64 * 0.5 - y)
    }

    /// The ground under a screen point. Past the limb the point is taken at the limb, so a corner off the globe still names ground (the renderers paint the background there).
    pub fn screen_to_geo(&self, px: f64, py: f64, w: usize, h: usize) -> (f64, f64) {
        let v = self.view_dir(px, py, w, h);
        let b = self.basis;
        let p = [
            v[0] * b[0][0] + v[1] * b[1][0] + v[2] * b[2][0],
            v[0] * b[0][1] + v[1] * b[1][1] + v[2] * b[2][1],
            v[0] * b[0][2] + v[1] * b[1][2] + v[2] * b[2][2],
        ];
        (p[2].clamp(-1.0, 1.0).asin().to_degrees(), p[1].atan2(p[0]).to_degrees())
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

/// The sun's world direction from an azimuth (degrees clockwise from north) and altitude at a place: along that place's east, north and up.
fn sun_world_at(lat: f64, lon: f64, az_deg: f32, alt_deg: f32) -> [f64; 3] {
    let (sl, cl) = lat.to_radians().sin_cos();
    let (so, co) = lon.to_radians().sin_cos();
    let east = [-so, co, 0.0];
    let north = [-sl * co, -sl * so, cl];
    let up = [cl * co, cl * so, sl];
    let (sa, ca) = (az_deg as f64).to_radians().sin_cos();
    let (sh, ch) = (alt_deg as f64).to_radians().sin_cos();
    let (e, n, u) = (sa * ch, ca * ch, sh);
    [e * east[0] + n * north[0] + u * up[0], e * east[1] + n * north[1] + u * up[1], e * east[2] + n * north[2] + u * up[2]]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = dot(v, v).sqrt().max(1e-300);
    [v[0] / n, v[1] / n, v[2] / n]
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

/// A stroke in progress: its screen points, the user layer as it was before it (for a cancel), the depth it draws at, and for a selection whether it adds or takes.
struct Stroke {
    /// Screen points with the brush width at each, from the pen's pressure.
    pts: Vec<(f64, f64, f64)>,
    depth: u8,
    add: bool,
    /// The stroke's job, fixed at its start: the S Pen inks with no tool picked.
    tool: ink::Tool,
}

#[derive(Clone, Copy)]
pub struct GpsFix {
    pub lat: f64,
    pub lon: f64,
    pub accuracy_m: f32,
    /// Height above the WGS84 ellipsoid, as the receiver gives it, and its vertical accuracy; None where it gave none.
    pub height_m: Option<f32>,
    pub v_accuracy_m: Option<f32>,
}

/// The map: all state and rendering, no host. Frontends call the input methods, drive `tick` until `converged`, and blit `canvas` (0xRRGGBB, row-major) after `render`.
pub struct MapCore {
    pub cam: Camera,
    res: Residency,
    luts: FrameLuts,
    luts_sun: (f32, f32, f64),
    /// The map's own sun, fixed to the screen: azimuth clockwise from screen up, altitude. The real sun never touches these (it once did, and the session kept the almanac's last place as the map's own light; Nick 2026-10-10: "it's def lit from the left").
    pub sun_az: f32,
    pub sun_alt: f32,
    /// The real sun's last azimuth and altitude from the almanac, true north and the horizon.
    real_az: f32,
    real_alt: f32,
    /// The sun as a direction in the world (the unit sphere), fixed while the ball turns under the camera; set from the azimuth and altitude at the camera point when they change, or by the almanac. The frame takes it into view space with the ball's basis, so no bearing is read, and the poles are nothing special (Nick 2026-10-10: the lighting jumped there).
    sun_world: [f64; 3],
    pub gps: Option<GpsFix>,
    pub canvas: Vec<u32>,
    canvas_w: usize,
    canvas_h: usize,
    vec_depth: u8,
    dem_depth: u8,
    home: Camera,
    /// Measured cost of the last `render` call in milliseconds.
    pub last_frame_ms: f32,
    /// Pixels the screen's top-right marks (the compass, the heading and the zoom) move down, below a host's own controls there (the desktop's window buttons, Nick 2026-10-17); 0 on the phone.
    pub marks_top: f32,
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
    /// A tap measures (the ruler: a line from the origin, its distance and profile); off, a tap only tells what is there (Nick 2026-10-14: a plain tap shows info, not the distance line).
    pub ruler: bool,
    /// The user's own layer: ink and selection (see `ink`), what a touch does, the pen and its width in pixels.
    pub user: ink::UserLayer,
    pub tool: ink::Tool,
    pub pen: u8,
    /// The pen's colour, VSF RGB at gamma 2, and its opacity (Nick 2026-10-10: a mixer, not a selector).
    pub pen_colour: [u8; 4],
    pub pen_width: f32,
    stroke: Option<Stroke>,
    /// The selection being drawn, as a screen mask (w × h bytes, 255 inside the closed curve), and a version for a renderer to notice a change; empty when none.
    pub preview: Vec<u8>,
    pub preview_version: u64,
    user_store: Option<Arc<dyn ink::UserStore>>,
    /// What the last tap found at its point, for the measurement strip.
    tap_info: Vec<String>,
    /// Cells' names as labels last read them; None for a resident cell with none.
    names_cache: rustc_hash::FxHashMap<mahere_tiles::CellKey, Option<std::sync::Arc<mahere_tiles::NameCell>>>,
    /// The geoid of the depth-2 cell last looked in, centimetres per texel (one cell covers a few hundred kilometres).
    geoid_cell: Option<(mahere_tiles::CellKey, Arc<Vec<i16>>)>,
    /// Cells replaced by a newer copy since a label worker last asked, for it to forget their names.
    names_reloaded: Vec<mahere_tiles::CellKey>,
    /// Navigation (Nick 2026-10-13): how a tap routes there, if it does; the route found and its version; the search running.
    pub travel: Option<route::Travel>,
    route: Option<Arc<route::Route>>,
    pub route_version: u64,
    route_rx: Option<std::sync::mpsc::Receiver<Result<route::Route, String>>>,
}

type PlanKey = (u64, u64, u64, u64, usize, usize, u64, u8, u8);

impl MapCore {
    pub fn new(store: Arc<dyn CellStore>, home: Camera) -> MapCore {
        MapCore::layered(store, None, home)
    }

    /// The map over published cells: the base from `store`, imagery and infrared from their own stores, fetched while a view shows them.
    pub fn layered(store: Arc<dyn CellStore>, layers: Option<residency::LayerStores>, home: Camera) -> MapCore {
        MapCore {
            cam: home,
            res: Residency::layered(store, layers),
            luts: FrameLuts {
                hypso: build_hypso_lut(),
                style: raster::Style::default(),
                sun: [0.0, 0.0, 1.0],
                mask: LayerMask::default(),
                dem_depth: DEM_BASE_DEPTH,
                contours: Contours { interval: 0.0, index_every: 5, m_per_px: 1.0, lo_m: 0.0, hi_m: 4000.0 },
                light: sh::Sh9::sun_and_sky(315.0, 40.0).quadratic((0.0, 1.0)),
                display: colour::Display::default(),
                compressed: true,
                pole: [0.0, 1.0, 0.0],
                disk: [0.0, 0.0, 1.0e9],
            },
            env: sh::Sh9::sun_and_sky(315.0, 40.0),
            luts_sun: (f32::NAN, f32::NAN, f64::NAN),
            sun_az: 315.0,
            sun_alt: 40.0,
            real_az: f32::NAN,
            real_alt: f32::NAN,
            sun_world: sun_world_at(home.lat(), home.lon(), 315.0, 40.0),
            gps: None,
            canvas: Vec::new(),
            canvas_w: 0,
            canvas_h: 0,
            vec_depth: VEC_BASE_DEPTH,
            dem_depth: DEM_BASE_DEPTH,
            home,
            last_frame_ms: 0.0,
            marks_top: 0.0,
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
            ruler: false,
            user: ink::UserLayer::default(),
            tool: ink::Tool::None,
            pen: 0,
            pen_colour: [230, 40, 40, 255],
            pen_width: 6.0,
            stroke: None,
            preview: Vec::new(),
            preview_version: 0,
            user_store: None,
            tap_info: Vec::new(),
            names_cache: Default::default(),
            names_reloaded: Vec::new(),
            geoid_cell: None,
            travel: None,
            route: None,
            route_version: 0,
            route_rx: None,
        }
    }

    // ==================== INPUT ====================

    /// The ball stays a ball: the basis made orthonormal again after a turn.
    pub fn clamp_camera(&mut self) {
        self.cam = Camera::from_basis(self.cam.basis(), self.cam.ppd);
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

    /// Turn the view by `delta` radians clockwise.
    pub fn rotate_view(&mut self, delta: f64) {
        self.cam.rotate_view(delta);
        self.unlock_measure();
        self.dirty = true;
    }

    pub fn set_bearing(&mut self, bearing: f64) {
        self.cam.set_bearing(bearing.rem_euclid(core::f64::consts::TAU));
        self.unlock_measure();
        self.dirty = true;
    }

    pub fn set_ppd(&mut self, ppd: f64) {
        self.cam.ppd = ppd.clamp(6., 4_000_000.);
        self.unlock_measure();
        self.dirty = true;
    }

    /// Look at a place: the camera point goes there, north up, at the zoom given or the one it has.
    /// The map centred on the GPS fix, the zoom kept (Nick 2026-10-14: a go-to-my-location button); false without a fix.
    pub fn locate(&mut self, w: usize, h: usize) -> bool {
        let Some(g) = self.gps else { return false };
        self.go_to(g.lat, g.lon, None, w, h);
        true
    }

    pub fn go_to(&mut self, lat: f64, lon: f64, ppd: Option<f64>, w: usize, h: usize) {
        self.cam.look_at(lat, lon, 0.0);
        if let Some(p) = ppd {
            self.cam.ppd = p.clamp(Self::ppd_floor(w, h), 4_000_000.);
        }
        self.clamp_camera();
        self.camera_moved(w, h);
    }

    // ==================== INK AND SELECTION ====================

    /// The depth a mark is drawn at: one above what the view wants, so a texel is one to two pixels wide (the view's own depth puts texels under a pixel, for the supersampled render, and a mark there would be a sprinkle).
    fn draw_depth(&self) -> u8 {
        select_depth(self.cam.ppd, self.vec_depth, VEC_BASE_DEPTH).saturating_sub(1)
    }

    pub fn set_tool(&mut self, tool: ink::Tool) {
        if self.tool != tool {
            self.cancel_stroke();
            self.tool = tool;
            self.dirty = true;
        }
    }

    pub fn set_pen(&mut self, pen: u8) {
        self.pen = pen % ink::PENS.len() as u8;
        let c = ink::PENS[self.pen as usize];
        self.pen_colour = [c[0], c[1], c[2], 255];
    }

    pub fn set_pen_colour(&mut self, rgba: [u8; 4]) {
        self.pen_colour = rgba;
    }

    /// Where the user's cells are kept; loading what is there.
    pub fn set_user_store(&mut self, store: Arc<dyn ink::UserStore>) {
        for (k, bytes) in store.load_all() {
            self.user.load(k, &bytes);
        }
        self.user.loaded();
        self.user_store = Some(store);
        self.dirty = true;
    }

    /// The cells a stroke changed, encoded and written on a thread of their own: the encode is the slow part, and the touch must not wait for it.
    fn save_user(&mut self) {
        let dirty = self.user.take_dirty_cells();
        if dirty.is_empty() {
            return;
        }
        if let Some(store) = &self.user_store {
            let store = store.clone();
            std::thread::spawn(move || {
                for (k, cell) in dirty {
                    match cell {
                        Some(c) => store.put(k, &ink::encode_user_cell(&c)),
                        None => store.delete(k),
                    }
                }
            });
        }
    }

    /// The depth the map draws its lines at.
    pub fn vec_depth(&self) -> u8 {
        self.vec_depth
    }

    pub fn stroke_active(&self) -> bool {
        self.stroke.is_some()
    }

    /// The brush's width at a pressure: the pen's width at a middle pressure (a finger's), half of it at a touch, over one and a half times it pressed hard (Nick 2026-10-10: a legit brush).
    fn brush_width(&self, pressure: f64) -> f64 {
        self.pen_width as f64 * (0.4 + 1.2 * pressure.clamp(0.0, 1.0))
    }

    /// A touch down with a tool, at a pressure 0..1 (a finger passes 0.5): the pen stamps its first dot; the selection remembers where it started (inside the selection takes, outside adds).
    pub fn begin_stroke(&mut self, x: f64, y: f64, pressure: f64, w: usize, h: usize) {
        self.begin_stroke_as(self.tool, x, y, pressure, w, h);
    }

    /// A touch down doing a given job, whatever tool is picked: a stylus inks even with none (Nick 2026-10-13: the pen always writes on the map).
    pub fn begin_stroke_as(&mut self, tool: ink::Tool, x: f64, y: f64, pressure: f64, w: usize, h: usize) {
        if tool == ink::Tool::None {
            return;
        }
        let depth = self.draw_depth();
        let add = !self.user.selected_at(&self.cam, w, h, depth, x, y);
        let width = self.brush_width(pressure);
        self.user.begin_undo();
        self.stroke = Some(Stroke { pts: vec![(x, y, width)], depth, add, tool });
        if tool == ink::Tool::Pen {
            let c = self.pen_colour;
            self.user.stroke(&self.cam, w, h, depth, &[(x, y, width)], [c[0], c[1], c[2]], c[3] as f32 / 255.0);
        }
        self.dirty = true;
    }

    /// The touch moved: the pen stamps the segment, its width following the pressure; the selection grows its curve and the preview closes it.
    pub fn extend_stroke(&mut self, x: f64, y: f64, pressure: f64, w: usize, h: usize) {
        let Some(st) = &mut self.stroke else { return };
        let last = *st.pts.last().unwrap();
        if (x - last.0).abs() < 0.5 && (y - last.1).abs() < 0.5 {
            return;
        }
        let width = self.pen_width as f64 * (0.4 + 1.2 * pressure.clamp(0.0, 1.0));
        st.pts.push((x, y, width));
        let depth = st.depth;
        match st.tool {
            ink::Tool::Pen => {
                let seg = [last, (x, y, width)];
                let c = self.pen_colour;
                self.user.stroke(&self.cam, w, h, depth, &seg, [c[0], c[1], c[2]], c[3] as f32 / 255.0);
            }
            ink::Tool::Select => self.rebuild_preview(w, h),
            ink::Tool::None => {}
        }
        self.dirty = true;
    }

    /// The touch lifted: the pen's mark stands; the selection's curve is closed and applied. Either is saved.
    pub fn end_stroke(&mut self, w: usize, h: usize) {
        let Some(st) = self.stroke.take() else { return };
        if st.tool == ink::Tool::Select && st.pts.len() >= 3 {
            let poly: Vec<(f64, f64)> = st.pts.iter().map(|p| (p.0, p.1)).collect();
            self.user.select(&self.cam, w, h, st.depth, &poly, st.add);
        }
        self.user.commit_undo();
        self.preview.clear();
        self.preview_version += 1;
        self.save_user();
        self.dirty = true;
    }

    /// A second finger, or a tool change, mid-stroke: everything the stroke did is undone.
    pub fn cancel_stroke(&mut self) {
        if self.stroke.take().is_some() {
            self.user.rollback_undo();
            self.preview.clear();
            self.preview_version += 1;
            self.dirty = true;
        }
    }

    /// The selection's curve so far, closed back to its start and filled, as the screen mask.
    fn rebuild_preview(&mut self, w: usize, h: usize) {
        let Some(st) = &self.stroke else { return };
        let poly_v: Vec<(f64, f64)> = st.pts.iter().map(|p| (p.0, p.1)).collect();
        let poly = &poly_v;
        self.preview.clear();
        self.preview.resize(w * h, 0);
        if poly.len() < 3 {
            self.preview_version += 1;
            return;
        }
        let (mut y0, mut y1) = (f64::MAX, f64::MIN);
        for p in poly {
            y0 = y0.min(p.1);
            y1 = y1.max(p.1);
        }
        let y0 = y0.floor().max(0.0) as usize;
        let y1 = (y1.ceil().max(0.0) as usize).min(h.saturating_sub(1));
        let mut xs: Vec<f64> = Vec::new();
        for py in y0..=y1 {
            let yc = py as f64 + 0.5;
            xs.clear();
            for i in 0..poly.len() {
                let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
                if (a.1 <= yc) != (b.1 <= yc) {
                    xs.push(a.0 + (yc - a.1) / (b.1 - a.1) * (b.0 - a.0));
                }
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for pair in xs.chunks(2) {
                if pair.len() < 2 {
                    break;
                }
                let x0 = pair[0].round().max(0.0) as usize;
                let x1 = (pair[1].round().max(0.0) as usize).min(w);
                for px in x0..x1 {
                    self.preview[py * w + px] = 255;
                }
            }
        }
        self.preview_version += 1;
    }

    pub fn clear_ink(&mut self) {
        self.cancel_stroke();
        self.user.clear_ink();
        self.save_user();
        self.dirty = true;
    }

    pub fn clear_selection(&mut self) {
        self.cancel_stroke();
        self.user.clear_selection();
        self.save_user();
        self.dirty = true;
    }

    /// An area's shape shown as the selection, replacing what was selected.
    pub fn show_selection(&mut self, planes: Vec<(mahere_tiles::CellKey, mahere_tiles::CovCell)>) {
        self.cancel_stroke();
        self.user.load_selection(planes);
        self.save_user();
        self.dirty = true;
    }

    /// The depth a selection would be drawn at now: what a new area asks for by default, clamped to what areas may ask.
    pub fn area_depth(&self) -> u8 {
        self.draw_depth().clamp(areas::AREA_MIN_DEPTH, areas::AREA_MAX_DEPTH)
    }

    pub fn go_home(&mut self, w: usize, h: usize) {
        self.cam = self.home;
        self.camera_moved(w, h);
    }

    pub fn adjust_sun(&mut self, daz: f32, dalt: f32) {
        self.sun_az = (self.sun_az + daz).rem_euclid(360.0);
        self.sun_alt = (self.sun_alt + dalt).clamp(5.0, 85.0);
        self.sun_world = sun_world_at(self.cam.lat(), self.cam.lon(), self.sun_az, self.sun_alt);
        self.dirty = true;
    }

    /// The sun in view space: screen right, up, toward the viewer.
    fn sun_view(&self) -> [f32; 3] {
        let b = self.cam.basis();
        let s = self.sun_world;
        [dot(b[0], s) as f32, dot(b[1], s) as f32, dot(b[2], s) as f32]
    }

    /// A fix redraws only when the pin is on screen and has moved by a pixel (Nick 2026-10-06): a phone sitting still gets no frame per second from its GPS.
    /// The geoid's height above the WGS84 ellipsoid at a place, metres (EGM2008, baked into the planet's depth-2 cells): from the device alone, never the network.
    pub fn geoid_at(&mut self, lat: f64, lon: f64) -> Option<f32> {
        const DEPTH: u8 = 2;
        let c = Coord::from_lat_lon(lat, lon);
        let key = mahere_tiles::CellKey { depth: DEPTH, prefix: c.raw() >> (60 - 2 * DEPTH as u32) };
        if self.geoid_cell.as_ref().is_none_or(|(k, _)| *k != key) {
            let plane = match self.res.store().get_local(key) {
                residency::Fetch::Bytes(b) => mahere_tiles::decode_geoid_only(&b)?,
                _ => return None,
            };
            self.geoid_cell = Some((key, Arc::new(plane)));
        }
        let (iu, iv) = c.uv();
        let i = raster::tri_index((iu as i64) << 16, (iv as i64) << 16, 16 + (22 - DEPTH as u32));
        self.geoid_cell.as_ref().and_then(|(_, p)| p.get(i)).map(|&cm| cm as f32 / 100.0)
    }

    /// The fix's height above sea level, the receiver's height less the geoid, with its vertical accuracy (Nick 2026-10-13: the GPS readout above sea level).
    pub fn gps_sea_level(&mut self) -> Option<(f32, Option<f32>)> {
        let g = self.gps?;
        let h = g.height_m?;
        Some((h - self.geoid_at(g.lat, g.lon)?, g.v_accuracy_m))
    }

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
        // The image layers come with each cell only while a view shows them: true colour, or the infrared alone.
        let mut want = match (mask.imagery, mask.infrared) {
            (true, false) => residency::WANT_IMG,
            (true, true) => residency::WANT_NIR,
            _ => 0,
        };
        // The terrain whenever anything drawn from it shows: the shading, the tint, contours, slope bands, raw elevation, and the sea under imagery.
        if mask.dem || mask.hypso || mask.contours || mask.slope || mask.raw || mask.imagery {
            want |= residency::WANT_DEM;
        }
        self.res.set_imagery(want);
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
                self.cam.set_bearing((self.true_heading() as f64).to_radians().rem_euclid(core::f64::consts::TAU));
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
                self.cam.set_bearing((self.true_heading() as f64).to_radians());
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
            self.cam.set_bearing((self.true_heading() as f64).to_radians().rem_euclid(core::f64::consts::TAU));
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
        let routed = self.poll_route();
        self.res.drain() > 0 || self.dirty || routed
    }

    // ==================== NAVIGATION ====================

    /// How a tap routes there: by car, bike or foot, or not at all (which clears the route).
    pub fn set_travel(&mut self, travel: Option<route::Travel>) {
        if travel != self.travel {
            self.travel = travel;
            if travel.is_none() {
                self.clear_route();
            }
            self.dirty = true;
        }
    }

    fn clear_route(&mut self) {
        if self.route.take().is_some() {
            self.route_version += 1;
        }
        self.route_rx = None;
    }

    /// The route on the map, as latitude and longitude.
    pub fn route_points(&self) -> Option<&[(f64, f64)]> {
        self.route.as_ref().map(|r| r.pts.as_slice())
    }

    pub fn route(&self) -> Option<&route::Route> {
        self.route.as_deref()
    }

    /// Search for a route from where the ruler starts (the screen centre, or the fix with "Measure from me" on: Nick 2026-10-14, the route started at the fix and found nothing from outside the 1 m areas) to a point, on a thread of its own; the result comes in through [`MapCore::tick`].
    fn start_route(&mut self, to: (f64, f64), travel: route::Travel) {
        let centre = (self.cam.lat(), self.cam.lon());
        let from = match self.gps {
            Some(g) if self.lock_to_fix => (g.lat, g.lon),
            _ => centre,
        };
        let store = self.res.store();
        // The climb comes from the terrain, its own layer when cells come in layers.
        let dem = self.res.layer_stores().map_or(store.clone(), |l| l.dem);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(route::find(store, dem, from, to, travel));
        });
        self.route_rx = Some(rx);
        self.tap_info.insert(0, format!("Finding a route by {}\u{2026}", travel.word().to_lowercase()));
    }

    /// A finished search: the route kept and told at the top of the strip (its length, time and climb, then the turns), or why there is none.
    fn poll_route(&mut self) -> bool {
        let Some(rx) = &self.route_rx else { return false };
        let Ok(found) = rx.try_recv() else { return false };
        self.route_rx = None;
        self.tap_info.retain(|l| !l.starts_with("Finding a route"));
        match found {
            Ok(r) => {
                let mut lines = Vec::new();
                let mins = (r.seconds / 60.0).round() as i64;
                let time = if mins >= 60 { format!("{} h {:02} min", mins / 60, mins % 60) } else { format!("{mins} min") };
                let climb = if r.climb_m >= 1.0 { format!(" \u{b7} {} m up", r.climb_m.round() as i64) } else { String::new() };
                lines.push(format!("{} \u{b7} {:.1} km \u{b7} {time}{climb}", r.travel.word(), r.metres / 1000.0));
                for s in r.steps.iter().skip(1).take(4) {
                    let name = if s.name.is_empty() { "the way" } else { &s.name };
                    lines.push(format!("{:.1} km  {} {name}", s.at_m / 1000.0, s.turn));
                }
                for (k, l) in lines.into_iter().enumerate() {
                    self.tap_info.insert(k, l);
                }
                self.route = Some(Arc::new(r));
            }
            Err(e) => {
                self.route = None;
                self.tap_info.insert(0, format!("No route: {e}"));
            }
        }
        self.route_version += 1;
        self.dirty = true;
        true
    }

    pub fn converged(&self) -> bool {
        self.res.converged()
    }

    pub fn pending_cells(&self) -> usize {
        self.res.pending_count()
    }

    /// Cells in view still without their base: the lines are all drawn when it is none.
    pub fn bases_left(&self) -> usize {
        self.res.bases_left()
    }

    /// The elevations a GPU's map pass drew (quantised, lowest and highest), read back after the frame: the next frame's contour interval and raw elevation range come from them, and a change redraws. None when it drew no terrain, which keeps the last. Until the first arrives a frame draws unscaled.
    pub fn set_drawn_range(&mut self, range: Option<(u16, u16)>) {
        if let Some((lo, hi)) = range {
            if (lo, hi) != (self.last_range.lo, self.last_range.hi) {
                self.last_range = ElevRange { lo, hi };
                self.dirty = true;
            }
        }
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
        // The same view over the same cells, the user's included, is the same plan: a frame the sensor or the GPS made dirty costs nothing here.
        let key = (self.cam.lat().to_bits(), self.cam.lon().to_bits(), self.cam.ppd.to_bits(), self.cam.bearing().to_bits(), w, h, self.res.pool_version + (self.user.version << 40), self.dem_depth, self.vec_depth);
        self.plan_cached = false;
        if let Some((k, p)) = &self.plan_cache {
            if *k == key {
                self.plan_cached = true;
                let p = p.clone();
                // Residency still hears the want: it re-sends an unchanged missing list once a second.
                let center = Coord::from_lat_lon(self.cam.lat(), self.cam.lon());
                let (cu, cv) = center.uv();
                self.res.want(p.want.clone(), (cu, cv));
                self.last_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
                return p;
            }
        }
        let p = Arc::new(plan::plan_frame(w, h, &self.cam, &self.res.pool, self.dem_depth, self.vec_depth, &self.user));
        self.last_straddle_blocks = p.straddle_blocks;
        let center = Coord::from_lat_lon(self.cam.lat(), self.cam.lon());
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

    /// The real light as it stands, into the current theme as its own light: absolute, in the screen's frame as the phone is held now. False without a real light to take.
    pub fn capture_light(&mut self) -> bool {
        let (Some(p), Some(s)) = (&self.probe, &self.sphere) else { return false };
        let absolute = p.scaled(s.level);
        let i = self.theme.min(self.themes.len() - 1);
        self.themes[i].probe = Some(absolute);
        self.themes[i].edited = true;
        self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
        self.dirty = true;
        true
    }

    /// The current theme back to its sun and sky.
    pub fn clear_light(&mut self) {
        let i = self.theme.min(self.themes.len() - 1);
        self.themes[i].probe = None;
        self.themes[i].edited = true;
        self.luts_sun = (f32::NAN, f32::NAN, f64::NAN);
        self.dirty = true;
    }

    /// How much of the light sphere the camera has painted, 0..1, for a capture to show what it would take.
    pub fn light_seen(&self) -> Option<f32> {
        self.sphere.as_ref().map(|s| s.tris.iter().filter(|t| **t != [0; 3]).count() as f32 / s.tris.len() as f32)
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
        // A cell replaced by a newer copy: its names are read again, here and by a label worker.
        for k in std::mem::take(&mut self.res.reloaded) {
            self.names_cache.remove(&k);
            self.names_reloaded.push(k);
        }
        self.canvas_w = w;
        self.canvas_h = h;
        // The real sun: azimuth and altitude from the clock and the position (the fix, or the view), as a screen-relative azimuth. Nothing is clamped: below the horizon it is below the horizon.
        if self.real_sun {
            let (lat, lon) = self.gps.map_or((self.cam.lat(), self.cam.lon()), |g| (g.lat, g.lon));
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
            let (az, alt) = sh::sun_position(lat, lon, now);
            // The clock moves the sun a few thousandths of a degree a second: follow it in steps too small to see, so a still screen is not relit every frame.
            let az = (az.rem_euclid(360.0)) as f32;
            if !((az - self.real_az).abs() <= 0.05 && (alt as f32 - self.real_alt).abs() <= 0.05) {
                self.real_az = az;
                self.real_alt = alt as f32;
                self.sun_world = sun_world_at(lat, lon, self.real_az, self.real_alt);
            }
        }
        // Lighting lives in the view frame, where the normals now are (each turned by its own place on the ball). The map's own sun is fixed to the screen: its azimuth is clockwise from screen up and nothing about the view enters, so the lit side stays where the phone has it as the globe turns and a pole is nothing special (Nick 2026-10-10: phone-relative, not planet-relative). The real sun is a world direction, brought into view space by the basis. The key is the sun's view vector, to the thousandth, so a still screen is not relit every frame.
        let sv = if self.real_sun {
            self.sun_view()
        } else {
            let az = (self.sun_az as f64).to_radians();
            let alt = (self.sun_alt as f64).to_radians();
            [(az.sin() * alt.cos()) as f32, (az.cos() * alt.cos()) as f32, alt.sin() as f32]
        };
        let key = (if self.real_sun { self.real_az } else { self.sun_az }, if self.real_sun { self.real_alt } else { self.sun_alt }, ((sv[0] * 1000.0).round() as f64) * 1.0e6 + ((sv[1] * 1000.0).round() as f64) * 1.0e3 + (sv[2] * 1000.0).round() as f64);
        if self.luts_sun != key {
            self.luts.sun = sv;
            if let (true, Some(env)) = (self.real_light, self.probe.as_ref()) {
                // The camera's environment is already in the device frame, which is the view frame.
                self.luts.light = env.quadratic((0.0, 1.0));
                self.luts_sun = key;
                self.light_version += 1;
            } else if self.real_sun {
                // The real sun alone, in the device frame: the landscape is lit exactly as the phone is held, by max(0, n·sun) and nothing else. No sky, no fading: at night the sun is under the landscape and it renders black; turn the phone over and the sun lights it from below (Nick 2026-10-06).
                let (lat, lon) = self.gps.map_or((self.cam.lat(), self.cam.lon()), |g| (g.lat, g.lon));
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
                let (az, alt) = sh::sun_position(lat, lon, now);
                // The frames, in order. The almanac gives a true azimuth; the sensor's world frame has magnetic north on its Y axis, so the sun is first expressed against magnetic north (azimuth less the declination, east positive). Rᵀ then takes it into the device frame: x right, y up the screen, z out of it — the lighting frame, since the Activity is locked to portrait. Last, when the map's up is not where the phone points, the device-frame sun is turned by (bearing − true heading): a device turned clockwise sees a fixed world vector turn counterclockwise, so this is the lighting the phone would show turned to match the map, tilt kept. With follow heading on the two agree and nothing turns; without a sensor (R identity, heading = declination) it collapses to the map-locked rotation by the bearing.
                let frames = Frames { declination_deg: self.declination_deg, map_locked: !self.have_rotation, rot: self.device_rot, bearing_deg: self.cam.bearing().to_degrees() as f32, heading_mag_deg: self.device_heading };
                let t = &self.themes[self.theme.min(self.themes.len() - 1)];
                let (sb, cb) = self.cam.bearing().sin_cos();
                self.sun_device = frames.to_screen(az as f32, alt as f32);
                self.luts.sun = self.sun_device;
                let _ = (sb, cb);
                self.luts.light = sh::Quad::directional(self.sun_device, t.sun, (0.0, 1.0));
                self.luts_sun = key;
                self.light_version += 1;
            } else if let Some(p) = self.themes[self.theme.min(self.themes.len() - 1)].probe {
                // The theme's own light, turned with the map's sun (its azimuth from where the probe was made, top left) and held to its level, so flat ground lights to one whatever it was captured at (Nick 2026-10-10: stored absolute, normalised when used).
                let turned = p.rotated_z(-(self.sun_az - 315.0).to_radians());
                self.env = turned.scaled(1.0 / turned.level());
                self.luts.sun = self.env.dominant_dir();
                self.luts.light = self.env.quadratic((0.0, 1.0));
                self.luts_sun = key;
                self.light_version += 1;
            } else {
                // The sun and sky around the sun's view direction: its azimuth and altitude as seen on the screen.
                let t = &self.themes[self.theme.min(self.themes.len() - 1)];
                let az_v = sv[0].atan2(sv[1]).to_degrees();
                let alt_v = sv[2].clamp(-1.0, 1.0).asin().to_degrees();
                self.env = sh::Sh9::sun_and_sky_coloured(az_v, alt_v, t.sun, t.sky);
                self.luts.light = self.env.quadratic((0.0, 1.0));
                self.luts_sun = key;
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
        let (lo_m, hi_m) = if self.last_range.is_empty() {
            (0.0, 4000.0)
        } else {
            (self.last_range.lo as f32 * mahere_tiles::ELEV_STEP - mahere_tiles::ELEV_OFFSET, self.last_range.hi as f32 * mahere_tiles::ELEV_STEP - mahere_tiles::ELEV_OFFSET)
        };
        self.luts.contours = Contours { interval, index_every: 5, m_per_px: (111_320.0 / self.cam.ppd) as f32, lo_m, hi_m };
    }

    /// Render the full frame into `self.canvas` (visible 0xRRGGBB). Cheap by construction (fetch + composite), so it runs every host frame.
    pub fn render(&mut self, w: usize, h: usize) {
        let t0 = Instant::now();
        self.prepare(w, h);
        self.canvas.resize(w * h, BG_RGB);
        if w == 0 || h == 0 {
            return;
        }
        {
            // Every pixel's own frame: the pole in view coordinates and the disk, for the ball's lighting.
            let b = self.cam.basis();
            self.luts.pole = [b[0][2] as f32, b[1][2] as f32, b[2][2] as f32];
            self.luts.disk = [w as f32 * 0.5, h as f32 * 0.5, self.cam.radius() as f32];
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
            &self.user,
        );
        self.last_straddle_blocks = stats.straddle_blocks;
        self.last_range = stats.elev;
        // The selection being drawn, as a wash over the frame.
        if self.preview.len() == w * h {
            let tint = colour::Display::default().swatch([
                (ink::SEL_TINT[0].sqrt() * 255.0) as u8,
                (ink::SEL_TINT[1].sqrt() * 255.0) as u8,
                (ink::SEL_TINT[2].sqrt() * 255.0) as u8,
            ]);
            // The wash inside, and the edge (two pixels either side of the curve) stronger, as the selection itself is drawn.
            let pv = &self.preview;
            for (i, px) in self.canvas.iter_mut().enumerate() {
                let m = pv[i];
                let (x, y) = (i % w, i / w);
                let edge = (x > 0 && pv[i - 1] != m) || (x + 1 < w && pv[i + 1] != m) || (y > 0 && pv[i - w] != m) || (y + 1 < h && pv[i + w] != m);
                if m != 0 || edge {
                    let (r, g, b) = ((*px >> 16) & 255, (*px >> 8) & 255, *px & 255);
                    let a = ((if edge { ink::SEL_EDGE_ALPHA } else { ink::SEL_ALPHA }) * 255.0) as u32;
                    let mix = |c: u32, t: u32| (c * (255 - a) + t * a) / 255;
                    *px = (mix(r, tint[0] as u32) << 16) | (mix(g, tint[1] as u32) << 8) | mix(b, tint[2] as u32);
                }
            }
        }

        // Residency: request what the lattice says we need, nearest-first.
        let center = Coord::from_lat_lon(self.cam.lat(), self.cam.lon());
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

    /// Whether the view is zoomed past the finest data under the crosshair: the cell there at the view's depth is known to be absent (a 404 from the bucket, or under a cell that was). A cell still loading is not past anything.
    pub fn past_data(&self) -> bool {
        if self.zoom_depth() > self.vec_depth {
            return true;
        }
        let key = mahere_tiles::CellKey::containing(Coord::from_lat_lon(self.cam.lat(), self.cam.lon()), self.vec_depth);
        self.res.pool.map.get(&key).is_some_and(|e| e.present == 0 && e.is_absent())
    }

    /// The depth the zoom asks for: the view's depth, counting on past the finest any cell is baked to (so z keeps climbing as the map magnifies past its detail).
    pub fn zoom_depth(&self) -> u8 {
        let ideal = raster::select_depth(self.cam.ppd, 0, 30);
        if self.vec_depth == raster::VEC_BASE_DEPTH { ideal.max(self.vec_depth) } else { self.vec_depth }
    }

    /// North arrow, top-right, with the view's heading below it: degrees the screen's up is turned from true north, clockwise positive, -180..180. No letters.
    fn draw_compass(&mut self, w: usize, h: usize) {
        if w < 120 || (h as f32) < 160.0 + self.marks_top {
            return;
        }
        const INK: [u8; 3] = [236, 238, 244];
        let (cx, cy) = (w as f32 - 46.0, 78.0 + self.marks_top);
        // Screen direction of true north: up = (sin B, cos B) in east/north, so north on screen is (-sin B, -cos B) in (x right, y down).
        let (sb, cb) = self.cam.bearing().sin_cos();
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
        let mut deg = self.cam.bearing().to_degrees().rem_euclid(360.0);
        if deg > 180.0 {
            deg -= 360.0;
        }
        let text = format!("{}", deg.round() as i64);
        draw_seven_seg(&mut self.canvas, w, h, &text, cx, cy + len + 30.0, 18.0, INK);
        // The zoom: the depth of the cells the view is drawn from, under the heading (Nick 2026-10-10: to see the level and when it changes).
        // Amber past the data (Nick 2026-10-16): the cell under the crosshair at this depth is known not to exist, on the device or in the bucket; white while it is there or still on its way.
        const AMBER: [u8; 3] = [255, 168, 0];
        let zoom_ink = if self.past_data() { AMBER } else { INK };
        let zoom = format!("z{}", self.zoom_depth());
        draw_seven_seg(&mut self.canvas, w, h, &zoom, cx, cy + len + 58.0, 18.0, zoom_ink);
        // With Detail on, its legend under the zoom: a bar in each depth's colour, the depth beside it, finest at the top.
        if self.luts.mask.debug {
            let n = raster::DETAIL_COLOURS.len();
            for (k, col) in raster::DETAIL_COLOURS.iter().enumerate().rev() {
                let y = cy + len + 84.0 + (n - 1 - k) as f32 * 22.0;
                draw_segment(&mut self.canvas, w, h, cx - 22.0, y, cx - 8.0, y, 5.0, *col);
                let d = raster::DETAIL_FIRST as usize + k;
                draw_seven_seg(&mut self.canvas, w, h, &format!("{d}"), cx + 14.0, y + 7.0, 14.0, INK);
            }
        }
    }

    /// The depth of the cells the view is drawn from: what the corner shows as z.
    pub fn view_depth(&self) -> u8 {
        self.vec_depth
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
            // A z: the top and bottom bars, the diagonal drawn below.
            'z' => 0b0001001,
            _ => 0,
        };
        let (x0, x1) = (cx, cx + sw);
        let (y0, ym, y1) = (y - size, y - size * 0.5, y);
        if ch == 'z' {
            draw_segment(canvas, w, h, x1, y0, x0, y1, hw, rgb);
        }
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

    /// The zoom turns amber past the data: where no cell exists under the crosshair once the view has looked, and past the finest depth any cell is baked to, where it keeps counting.
    #[test]
    fn the_zoom_says_when_it_is_past_the_data() {
        let mut map = MapCore::new(Arc::new(Empty), Camera::new(47.6, -122.3, 20_000.0, 0.0));
        let (w, h) = (400, 300);
        for _ in 0..500 {
            map.render(w, h);
            if map.converged() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
            map.tick(w, h);
        }
        assert!(map.past_data(), "nothing anywhere: past the data at z{}", map.zoom_depth());
        let mut deep = MapCore::new(Arc::new(Empty), Camera::new(47.6, -122.3, 5_000_000.0, 0.0));
        deep.render(w, h);
        assert!(deep.zoom_depth() > raster::VEC_BASE_DEPTH, "the zoom counts on past the finest depth: z{}", deep.zoom_depth());
        assert!(deep.past_data());
    }

    /// Shells that draw on demand rely on `tick` to report camera changes (the Android two-finger path has no redraw request of its own).
    #[test]
    fn tick_reports_camera_changes() {
        let cam = Camera::new(46.2, -121.5, 2800.0, 0.0);
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
    /// What is at the target, a line each: the road, trail or boundary under the finger, motor vehicle use, who manages the land, the ground, and the elevation with its slope and the way it faces.
    pub info: Vec<String>,
    /// A ruler's measurement (line, distance, profile) or only a tap's card.
    pub ruler: bool,
    /// The target, latitude and longitude.
    pub target: (f64, f64),
}

impl MapCore {
    /// A tap on the map at screen (x, y): a new measurement to that point.
    pub fn tap(&mut self, x: f64, y: f64, w: usize, h: usize) {
        let target = self.cam.screen_to_geo(x, y, w, h);
        self.measure = Some(Measure { target, from_fix: self.lock_to_fix && self.gps.is_some() });
        self.tap_info = self.point_info(x, y, w, h);
        // Navigating: a tap is where to go.
        if let Some(t) = self.travel {
            self.start_route(target, t);
        }
        self.dirty = true;
    }

    /// A tap that landed on a label: the label's feature first in what the tap says, then the rest as a tap says it.
    pub fn tap_label(&mut self, x: f64, y: f64, w: usize, h: usize, label: Option<&mahere_tiles::Named>) {
        self.tap(x, y, w, h);
        if let Some(n) = label {
            let first = format!("{} \u{b7} {}", n.label(), named_kind(n));
            self.tap_info.retain(|l| !l.starts_with(&format!("{} \u{b7}", n.label())) && *l != n.label());
            self.tap_info.insert(0, first);
        }
    }

    pub fn clear_measure(&mut self) {
        self.measure = None;
        self.tap_info.clear();
        self.clear_route();
        self.dirty = true;
    }

    /// What is at a point on the screen (Nick 2026-10-11: the road, the national park, the elevation): from the finest resident cell there that carries vector planes, the lines searched within a finger's width, the areas read at the point itself.
    pub fn point_info(&self, x: f64, y: f64, w: usize, h: usize) -> Vec<String> {
        use mahere_tiles::CellKey;
        use mahere_tiles::landuse::{mvum, own};
        let mut cells: rustc_hash::FxHashMap<CellKey, Option<mahere_tiles::CellPlanes>> = Default::default();
        let mut planes_of = |key: CellKey| -> Option<mahere_tiles::CellPlanes> { cells.entry(key).or_insert_with(|| self.res.read_planes(key)).clone() };
        let at = |px: f64, py: f64| -> Option<(u64, i64, i64)> {
            if !self.cam.on_globe(px, py, w, h) {
                return None;
            }
            let (lat, lon) = self.cam.screen_to_geo(px, py, w, h);
            let c = Coord::from_lat_lon(lat, lon);
            let (iu, iv) = c.uv();
            Some((c.raw(), (iu as i64) << 16, (iv as i64) << 16))
        };
        let Some((raw, uq, vq)) = at(x, y) else { return Vec::new() };
        // The finest resident cell under the point that has vectors: the depth the rest of the reading uses.
        let vec_bits = residency::PRESENT_LINE | residency::PRESENT_LAND | residency::PRESENT_WATER | residency::PRESENT_OWN | residency::PRESENT_MVUM;
        let mut depth = None;
        for d in (raster::MIN_DEPTH..=self.vec_depth.max(14)).rev() {
            let key = CellKey { depth: d, prefix: raw >> (60 - 2 * d as u32) };
            if self.res.pool.map.get(&key).is_some_and(|e| e.present & vec_bits != 0) {
                depth = Some(d);
                break;
            }
        }
        let mut out = Vec::new();
        if let Some(d) = depth {
            let texel = |uq: i64, vq: i64| raster::tri_index(uq, vq, 16 + 22 - d as u32);
            let key_of = |raw: u64| CellKey { depth: d, prefix: raw >> (60 - 2 * d as u32) };
            // Lines are thin: the nearest within about 14 px, in rings a pixel apart out from the point.
            let mut line: Option<(u8, u8, Option<mahere_tiles::Named>)> = None;
            let mut route: Option<u8> = None;
            let mut near: Vec<CellKey> = Vec::new();
            'rings: for r in 0..=14 {
                let rr = r as f64;
                let n = if r == 0 { 1 } else { (std::f64::consts::TAU * rr).ceil() as usize };
                for k in 0..n {
                    let a = k as f64 / n as f64 * std::f64::consts::TAU;
                    let Some((sraw, su, sv)) = at(x + rr * a.cos(), y + rr * a.sin()) else { continue };
                    let Some(p) = planes_of(key_of(sraw)) else { continue };
                    let i = texel(su, sv);
                    let named = p.names.as_ref().and_then(|n| n.line_at(i));
                    if !near.contains(&key_of(sraw)) {
                        near.push(key_of(sraw));
                    }
                    if line.is_none() {
                        if let Some(l) = &p.line {
                            if l.cov[i] >= 96 && l.class[i] != 0 {
                                line = Some((l.class[i], l.uses_at(i), named.filter(|e| !e.is_point()).cloned()));
                            }
                        }
                    }
                    if route.is_none() {
                        if let Some(m) = &p.mvum {
                            if m.cov[i] >= 96 && m.class[i] != 0 {
                                route = Some(m.class[i]);
                            }
                        }
                    }
                    if line.is_some() && route.is_some() {
                        break 'rings;
                    }
                }
            }
            // A named point is one texel, which can be under a pixel: searched in every cell within reach of the finger, and its reach grows with its rank (Nick 2026-10-13), from 14 px for a spring or a trailhead to about 40 px for a big summit or a city; the nearest by its own reach wins.
            for (dx, dy) in [(-40.0, -40.0), (0.0, -40.0), (40.0, -40.0), (-40.0, 0.0), (40.0, 0.0), (-40.0, 40.0), (0.0, 40.0), (40.0, 40.0)] {
                if let Some((sraw, _, _)) = at(x + dx, y + dy) {
                    if !near.contains(&key_of(sraw)) {
                        near.push(key_of(sraw));
                    }
                }
            }
            let reach = |rank: u32| 14.0 + 26.0 * ((rank as f64 + 1.0).log10() / 5.5).clamp(0.0, 1.0);
            let mut point: Option<(f64, mahere_tiles::Named)> = None;
            for k in near {
                let Some(p) = planes_of(k) else { continue };
                let Some(n) = &p.names else { continue };
                if !n.entries.iter().any(|e| e.is_point()) {
                    continue;
                }
                for (i, &e) in n.line.iter().enumerate() {
                    if e == 0 || !n.entries[e as usize - 1].is_point() {
                        continue;
                    }
                    let (plat, plon) = k.texel_lat_lon(i);
                    let (px, py) = self.cam.geo_to_screen(plat, plon, w, h);
                    let entry = &n.entries[e as usize - 1];
                    let near_by = (px - x).hypot(py - y) / reach(entry.rank);
                    if near_by <= 1.0 && point.as_ref().is_none_or(|(d, _)| near_by < *d) {
                        point = Some((near_by, entry.clone()));
                    }
                }
            }
            let point = point.map(|(_, e)| e);
            if let Some(e) = &point {
                let what = mahere_osm_point_name(e.kind);
                out.push(format!("{} \u{b7} {what}", e.label()));
            }
            if let Some((class, uses, named)) = line {
                let name = theme::class_word(class);
                let name = match &named {
                    Some(e) => format!("{} \u{b7} {name}", e.label()),
                    None => name.to_string(),
                };
                use mahere_tiles::use_bits::*;
                let mut how: Vec<&str> = Vec::new();
                for (bit, word) in [(FOOT, "foot"), (BIKE, "bikes"), (HORSE, "horses"), (SKI, "ski"), (MOTOR, "motor vehicles"), (UNPAVED, "unpaved"), (RESTRICTED, "restricted")] {
                    if uses & bit != 0 {
                        how.push(word);
                    }
                }
                out.push(if how.is_empty() { name } else { format!("{name}: {}", how.join(", ")) });
            }
            if let Some(c) = route {
                out.push(format!("Motor vehicle use: {}", mvum::NAMES.get(c as usize).copied().unwrap_or("")));
            }
            if let Some(p) = planes_of(key_of(raw)) {
                let i = texel(uq, vq);
                if let Some(o) = &p.own {
                    let c = o.class[i];
                    if o.cov[i] >= 128 && c != 0 {
                        let mgr = if c >= own::WILD { c - own::WILD } else { c };
                        let name = own::NAMES.get(mgr as usize).copied().unwrap_or("Public land");
                        out.push(if c >= own::WILD { format!("{name} \u{b7} Wilderness") } else { name.to_string() });
                    } else {
                        out.push("Private or unlisted land".to_string());
                    }
                }
                // The water or land cover here, by its name where it has one.
                let area = p.names.as_ref().and_then(|n| n.area_at(i)).map(|e| e.label());
                let wet = p.water.as_ref().is_some_and(|w| w.cov[i] >= 128);
                let cover = if wet {
                    Some("Water")
                } else {
                    p.land.as_ref().filter(|l| l.cov[i] >= 128 && l.class[i] != 0).map(|l| theme::LAND_NAMES.get(l.class[i] as usize).copied().unwrap_or(""))
                };
                match (area, cover) {
                    (Some(a), Some(c)) if wet => out.push(if a.to_lowercase().contains("lake") || a.to_lowercase().contains("river") { a } else { format!("{a} \u{b7} {c}") }),
                    (Some(a), Some(c)) => out.push(format!("{a} \u{b7} {c}")),
                    (Some(a), None) => out.push(a),
                    (None, Some(c)) => out.push(c.to_string()),
                    (None, None) => {}
                }
            }
        }
        // The ground: the elevation, and from the elevations a few metres either side the slope and the way it faces.
        let (lat, lon) = self.cam.screen_to_geo(x, y, w, h);
        if let Some(e) = self.elevation_at(lat, lon) {
            let step = (mahere_tiles::texel_m(self.dem_depth) as f64 * 1.5).max(2.0);
            let (dlat, dlon) = (step / 111_320.0, step / (111_320.0 * lat.to_radians().cos().max(1e-6)));
            let face = match (self.elevation_at(lat, lon + dlon), self.elevation_at(lat, lon - dlon), self.elevation_at(lat + dlat, lon), self.elevation_at(lat - dlat, lon)) {
                (Some(ee), Some(ew), Some(en), Some(es)) => {
                    let (gx, gy) = ((ee - ew) as f64 / (2.0 * step), (en - es) as f64 / (2.0 * step));
                    let slope = gx.hypot(gy).atan().to_degrees();
                    if slope < 2.0 {
                        "  \u{b7}  flat".to_string()
                    } else {
                        // Downhill is where it faces.
                        let az = (-gx).atan2(-gy).to_degrees().rem_euclid(360.0);
                        let dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
                        format!("  \u{b7}  {}\u{b0} slope facing {}", slope.round() as i64, dirs[((az + 22.5) / 45.0) as usize % 8])
                    }
                }
                _ => String::new(),
            };
            out.push(format!("{} m{face}", e.round() as i64));
        }
        out
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
        let (tx, ty) = self.cam.geo_to_screen(tlat, tlon, w, h);
        // Without the ruler the line folds onto the target: only its mark shows.
        let (ox, oy) = if self.ruler { self.cam.geo_to_screen(olat, olon, w, h) } else { (tx, ty) };
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
            info: self.tap_info.clone(),
            ruler: self.ruler,
            target: m.target,
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
        let cam = Camera::new(47.0, -121.0, 20.0, 0.7);
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
        let mut map = MapCore::new(std::sync::Arc::new(Nothing), Camera::new(84.0, 30.0, 20.0, 0.4));
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

    /// A pen stroke lands on the frame in its colour, a cancelled one leaves nothing, and a selection washes what it encloses.
    #[test]
    fn ink_draws_and_a_cancel_undoes() {
        use super::{MapCore, ink};
        struct Nothing;
        impl crate::residency::CellStore for Nothing {
            fn get(&self, _key: mahere_tiles::CellKey) -> crate::residency::Fetch {
                crate::residency::Fetch::Absent
            }
        }
        let (w, h) = (256, 192);
        let mut map = MapCore::new(std::sync::Arc::new(Nothing), Camera::new(46.2, -122.19, 24000.0, 0.0));
        map.render(w, h);
        let (bare_on, bare_off) = (map.canvas[100 * w + 120], map.canvas[40 * w + 120]);
        map.set_tool(ink::Tool::Pen);
        map.set_pen(0);
        map.begin_stroke(60.0, 100.0, 0.5, w, h);
        map.extend_stroke(180.0, 100.0, 0.5, w, h);
        map.end_stroke(w, h);
        map.render(w, h);
        let on = map.canvas[100 * w + 120];
        assert_ne!(on, bare_on, "the stroke shows");
        assert!((on >> 16) & 255 > (on & 255), "red ink is red: {on:#08x}");
        map.begin_stroke(60.0, 40.0, 0.5, w, h);
        map.extend_stroke(180.0, 40.0, 0.5, w, h);
        map.cancel_stroke();
        map.render(w, h);
        let (lat, lon) = map.cam.screen_to_geo(120.5, 40.5, w, h);
        let raw = mahere_coord::Coord::from_lat_lon(lat, lon).raw();
        assert!(map.user.ink_at(raw, 12).is_none(), "a cancelled stroke leaves no ink");
        let p = map.canvas[40 * w + 120];
        assert!((p >> 16) & 255 < 150, "nor any red on the frame: {p:#08x} (bare {bare_off:#08x})");
        map.set_tool(ink::Tool::Select);
        map.begin_stroke(40.0, 120.0, 0.5, w, h);
        map.extend_stroke(200.0, 120.0, 0.5, w, h);
        map.extend_stroke(200.0, 180.0, 0.5, w, h);
        map.extend_stroke(40.0, 180.0, 0.5, w, h);
        assert_eq!(map.preview.len(), w * h, "the preview mask is live");
        assert_eq!(map.preview[150 * w + 120], 255, "inside the curve");
        map.end_stroke(w, h);
        assert!(map.preview.is_empty());
        assert!(map.user.selected_at(&map.cam, w, h, 12, 120.0, 150.0), "selected");
    }

    /// Turning at a pole turns about the view axis and nothing else: the camera point stays put and the ball stays a ball.
    #[test]
    fn turning_at_the_pole_only_turns() {
        let mut cam = Camera::new(89.9999, 12.0, 300.0, 0.0);
        let before = cam.basis()[2];
        for _ in 0..50 {
            cam.rotate_view(0.137);
        }
        let b = cam.basis();
        for k in 0..3 {
            assert!((b[2][k] - before[k]).abs() < 1e-12, "the camera point moved");
        }
        for i in 0..3 {
            for j in 0..3 {
                let d = b[i][0] * b[j][0] + b[i][1] * b[j][1] + b[i][2] * b[j][2];
                assert!((d - if i == j { 1.0 } else { 0.0 }).abs() < 1e-9, "not orthonormal");
            }
        }
    }

    /// Over the pole and back: the ball has no seam there. A pan straight up from 89°N crosses the pole onto the far meridian, the ground under the drag keeps its place, and the angles read off the basis rebuild the same basis.
    #[test]
    fn the_pole_is_nothing_special() {
        use super::MapCore;
        struct Nothing;
        impl crate::residency::CellStore for Nothing {
            fn get(&self, _key: mahere_tiles::CellKey) -> crate::residency::Fetch {
                crate::residency::Fetch::Absent
            }
        }
        let (w, h) = (1000, 800);
        let mut map = MapCore::new(std::sync::Arc::new(Nothing), Camera::new(89.0, 30.0, 20.0, 0.0));
        for _ in 0..4 {
            let target = (500.0, 100.0);
            let ground = map.cam.screen_to_geo(target.0, target.1, w, h);
            map.pan(0.0, 300.0, w, h);
            let (px, py) = map.cam.geo_to_screen(ground.0, ground.1, w, h);
            assert!((px - 500.0).abs() < 1e-6 && (py - 400.0).abs() < 1e-6, "the ground under the drag landed at {px},{py}");
            let again = Camera::new(map.cam.lat(), map.cam.lon(), map.cam.ppd, map.cam.bearing());
            for i in 0..3 {
                for k in 0..3 {
                    assert!((again.basis()[i][k] - map.cam.basis()[i][k]).abs() < 1e-9, "the angles do not rebuild the basis");
                }
            }
        }
        assert!(map.cam.lat() < 89.0, "the camera crossed the pole: lat {}", map.cam.lat());
    }

    /// A screen corner off the globe still names ground, at the limb.
    #[test]
    fn a_corner_off_the_globe_names_the_limb() {
        let cam = Camera::new(0.0, 0.0, 6.0, 0.0);
        let (lat, lon) = cam.screen_to_geo(0.0, 0.0, 1024, 768);
        assert!(lat.is_finite() && lon.is_finite());
        let (px, py) = cam.geo_to_screen(lat, lon, 1024, 768);
        let (dx, dy) = (px - 512.0, py - 384.0);
        assert!(((dx * dx + dy * dy).sqrt() - cam.radius()).abs() < 1e-3, "at the limb");
    }
}

/// What a named point is, by its kind in the name table.
fn mahere_osm_point_name(kind: u8) -> &'static str {
    mahere_tiles::names::POINT_NAMES.get(kind.saturating_sub(mahere_tiles::names::POINT) as usize).copied().unwrap_or("Place")
}

/// What a named thing is, in words: a point's kind, a line's class, an area's cover.
pub fn named_kind(n: &mahere_tiles::Named) -> &'static str {
    if n.is_point() {
        mahere_osm_point_name(n.kind)
    } else if n.is_area() {
        theme::LAND_NAMES.get((n.kind - mahere_tiles::names::AREA + 1) as usize).copied().filter(|s| !s.is_empty()).unwrap_or("Area")
    } else {
        theme::class_word(n.kind)
    }
}
