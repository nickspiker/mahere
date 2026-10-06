//! mahere on Android: fluor's AndroidShell hosting the shared MapCore, plus
//! the raw two-finger gesture solve and GPS.
//!
//! Gesture philosophy (Nick's spec): no gesture recognizers, no slop radius,
//! no focal-point abstraction. Two fingers are two point correspondences;
//! with rotation locked the solve is scale = finger-distance ratio and the
//! geographic midpoint pinned under the screen midpoint — applied per raw
//! MotionEvent, so the map locks to the fingers from the first event.
//! Rotation joins when the camera grows a bearing.
//!
//! Single-finger events forward into fluor's AndroidShell (which synthesizes
//! MouseInput/CursorMoved); the shell's synthetic MouseWheel (meant for
//! scrolly apps) is swallowed — zoom belongs to the two-finger solve.

use fluor::coord::Coord as Px;
use fluor::event::{CursorIcon, ElementState, Event as FEvent, MouseButton};
use fluor::host::EventResponse;
use fluor::host::android::shell::AndroidShell;
use fluor::host::app::{Context, FluorApp};
use fluor::paint::pack_argb;
use jni::JNIEnv;
use jni::objects::{JClass, JObject, JString};
use jni::sys::{jboolean, jdouble, jfloat, jint, jlong};
use mahere_engine::residency::{CellStore, DirStore, HttpStore, TieredStore, DEFAULT_CELLS_URL};
use mahere_engine::{Camera, GpsFix, MapCore};
use ndk::native_window::NativeWindow;
use std::time::Instant;

struct TwoFinger {
    /// Geography captured under each finger at gesture start.
    geo_a: (f64, f64),
    geo_b: (f64, f64),
    d0: f64,
    ppd0: f64,
    /// Screen angle of the finger vector at capture, and bearing then.
    alpha0: f64,
    bearing0: f64,
}

pub struct AndroidApp {
    map: MapCore,
    w: usize,
    h: usize,
    dragging: bool,
    last_cursor: (f64, f64),
    two: Option<TwoFinger>,
    /// Re-anchor the next CursorMoved instead of panning (finger handoff
    /// after a pinch would otherwise jump by the stale delta).
    suppress_move: bool,
    centered_once: bool,
    store: Option<std::sync::Arc<mahere_store::FlatStorage>>,
    recorder: Option<mahere_store::TrackRecorder>,
    fixes_since_save: u32,
}

impl AndroidApp {
    fn two_begin(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        let geo_a = self.map.cam.screen_to_geo(x0, y0, self.w, self.h);
        let geo_b = self.map.cam.screen_to_geo(x1, y1, self.w, self.h);
        let d0 = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt().max(1.0);
        self.two = Some(TwoFinger {
            geo_a,
            geo_b,
            d0,
            ppd0: self.map.cam.ppd,
            alpha0: (y1 - y0).atan2(x1 - x0),
            bearing0: self.map.cam.bearing,
        });
        self.dragging = false;
    }

    /// Full 4-DOF similarity solve: two finger correspondences exactly
    /// determine pan+rotate+zoom. Scale from the distance ratio, bearing
    /// from the finger-vector angle (screen angle of a fixed geo segment is
    /// -(B + its ENU angle), so B = B0 + (alpha0 - alpha)), then the
    /// geographic midpoint pinned under the screen midpoint.
    fn two_update(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        let Some(t) = &self.two else { return };
        let d = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt().max(1.0);
        let alpha = (y1 - y0).atan2(x1 - x0);
        self.map.set_ppd(t.ppd0 * d / t.d0);
        self.map.set_bearing(t.bearing0 + (t.alpha0 - alpha));
        let mid_lat = (t.geo_a.0 + t.geo_b.0) * 0.5;
        let mid_lon = (t.geo_a.1 + t.geo_b.1) * 0.5;
        let (mx, my) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        self.map.place_anchor(mid_lat, mid_lon, mx, my, self.w, self.h);
    }

    fn two_end(&mut self) {
        if self.two.take().is_some() {
            // Keep panning with whichever finger remains, re-anchored.
            self.dragging = true;
            self.suppress_move = true;
        }
    }

    fn on_gps(&mut self, fix: GpsFix) {
        if !self.centered_once {
            self.centered_once = true;
            self.map.cam.lat = fix.lat;
            self.map.cam.lon = fix.lon;
            self.map.camera_moved(self.w, self.h);
        }
        self.map.set_gps(fix);
        // Track recording: every fix into the vault (chunk-flushed), the
        // session (camera + sun) refreshed every tenth fix.
        if let Some(rec) = &mut self.recorder {
            let elev = self.map.gps_elevation().unwrap_or(f32::NAN) as f64;
            rec.on_fix(fix.lat, fix.lon, elev);
        }
        self.fixes_since_save += 1;
        if self.fixes_since_save >= 10 {
            self.fixes_since_save = 0;
            self.save_session();
        }
    }

    fn save_session(&self) {
        if let Some(store) = &self.store {
            let c = &self.map.cam;
            let _ = mahere_store::save_session(
                store,
                &mahere_store::Session {
                    lat: c.lat,
                    lon: c.lon,
                    ppd: c.ppd,
                    bearing: c.bearing,
                    sun_az: self.map.sun_az as f64,
                    sun_alt: self.map.sun_alt as f64,
                },
            );
        }
    }
}

impl FluorApp for AndroidApp {
    type UserEvent = ();

    fn title(&self) -> &str {
        "mahere"
    }

    fn init(&mut self, _ctx: &mut Context) {}

    fn on_resize(&mut self, width: u32, height: u32, _ctx: &mut Context) {
        self.w = width as usize;
        self.h = height as usize;
        self.map.camera_moved(self.w, self.h);
    }

    fn on_event(&mut self, event: &FEvent, ctx: &mut Context) -> EventResponse {
        match event {
            FEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left } => {
                if self.two.is_none() {
                    self.dragging = true;
                    self.last_cursor = (ctx.cursor_x as f64, ctx.cursor_y as f64);
                }
                EventResponse::Handled
            }
            FEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left } => {
                self.dragging = false;
                EventResponse::Handled
            }
            FEvent::CursorMoved { .. } => {
                let (x, y) = (ctx.cursor_x as f64, ctx.cursor_y as f64);
                if self.suppress_move {
                    self.suppress_move = false;
                } else if self.dragging && self.two.is_none() {
                    let (dx, dy) = (x - self.last_cursor.0, y - self.last_cursor.1);
                    self.map.pan(dx, dy, self.w, self.h);
                    ctx.window.request_redraw();
                }
                self.last_cursor = (x, y);
                EventResponse::Handled
            }
            // The shell synthesizes MouseWheel from touch-drags for scrolly
            // apps; swallowed — zoom is the two-finger solve's job.
            FEvent::MouseWheel { .. } => EventResponse::Handled,
            _ => EventResponse::Pass,
        }
    }

    fn wake_at(&self) -> Option<Instant> {
        if self.map.converged() { None } else { Some(Instant::now()) }
    }

    fn tick(&mut self, _ctx: &mut Context) -> bool {
        self.map.tick(self.w, self.h)
    }

    fn render(&mut self, target: &mut [u32], ctx: &mut Context) {
        let w = ctx.viewport.width_px as usize;
        let h = ctx.viewport.height_px as usize;
        if self.map.needs_render(w, h) {
            self.map.render(w, h);
        }
        // ANativeWindow RGBA_8888 lands R in the low byte: swap R/B vs the
        // desktop path or Puget Sound renders brown and the pin orange.
        let n = (w * h).min(target.len()).min(self.map.canvas.len());
        for (out, &rgb) in target[..n].iter_mut().zip(&self.map.canvas[..n]) {
            *out = pack_argb(rgb as u8, (rgb >> 8) as u8, (rgb >> 16) as u8, 255);
        }
    }

    fn cursor_for(&self, _x: Px, _y: Px, _ctx: &Context) -> CursorIcon {
        CursorIcon::Default
    }
}

type Shell = AndroidShell<AndroidApp>;

fn shell<'a>(ptr: jlong) -> &'a mut Shell {
    unsafe { &mut *(ptr as *mut Shell) }
}

/// Android MotionEvent action codes (masked).
const ACTION_MOVE: jint = 2;
const ACTION_POINTER_UP: jint = 6;

#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeInit(
    mut env: JNIEnv,
    _class: JClass,
    width: jint,
    height: jint,
    data_dir: JString,
) -> jlong {
    let dir: String = match env.get_string(&data_dir) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    // The vault: session, tracks, and the on-device cell cache (kete).
    let store = mahere_store::open(Some(&dir)).ok();
    // Cells stream from the bucket through the vault; `cells-local` (pushed
    // by hand) overrides for offline development.
    let local = std::path::PathBuf::from(format!("{dir}/cells-local"));
    let cell_store: std::sync::Arc<dyn CellStore> = if local.is_dir() {
        std::sync::Arc::new(DirStore(local))
    } else {
        let remote = std::sync::Arc::new(HttpStore::new(DEFAULT_CELLS_URL));
        match &store {
            Some(v) => std::sync::Arc::new(TieredStore::new(
                std::sync::Arc::new(mahere_store::VaultCells(v.clone())),
                remote,
            )),
            None => remote,
        }
    };
    let session = store.as_ref().and_then(|s| mahere_store::load_session(s));
    let cam = match session {
        Some(s) => Camera { lat: s.lat, lon: s.lon, ppd: s.ppd, bearing: s.bearing },
        // Mt St Helens until the first GPS fix recenters us.
        None => Camera { lat: 46.2000, lon: -122.1900, ppd: 12_000.0, bearing: 0.0 },
    };
    let mut map = MapCore::new(cell_store, cam);
    if let Some(s) = session {
        map.sun_az = s.sun_az as f32;
        map.sun_alt = s.sun_alt as f32;
    }
    let recorder = store.clone().map(mahere_store::TrackRecorder::new);
    let app = AndroidApp {
        map,
        w: width as usize,
        h: height as usize,
        dragging: false,
        last_cursor: (0., 0.),
        two: None,
        suppress_move: false,
        // A restored session IS the view; don't let the first fix yank it.
        centered_once: session.is_some(),
        store,
        recorder,
        fixes_since_save: 0,
    };
    Box::into_raw(Box::new(AndroidShell::new(app, width as u32, height as u32))) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeResize(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    width: jint,
    height: jint,
) {
    if ptr != 0 {
        let s = shell(ptr);
        s.resize(width as u32, height as u32);
        // A restored surface arrives with undefined buffers; same-size
        // resizes must still repaint everything.
        s.app().map.mark_dirty();
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeDraw(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    surface: JObject,
) -> jboolean {
    if ptr == 0 {
        return 0;
    }
    let window = match unsafe {
        NativeWindow::from_surface(env.get_native_interface() as *mut _, surface.as_raw() as *mut _)
    } {
        Some(w) => w,
        None => return 0,
    };
    shell(ptr).draw(&window) as jboolean
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeOnTouch(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    action: jint,
    count: jint,
    x0: jfloat,
    y0: jfloat,
    x1: jfloat,
    y1: jfloat,
) -> jint {
    if ptr == 0 {
        return 0;
    }
    let s = shell(ptr);
    if count >= 2 {
        if action == ACTION_POINTER_UP {
            s.app().two_end();
        } else if s.app().two.is_none() {
            s.app().two_begin(x0 as f64, y0 as f64, x1 as f64, y1 as f64);
        } else if action == ACTION_MOVE {
            s.app().two_update(x0 as f64, y0 as f64, x1 as f64, y1 as f64);
        }
        return 0;
    }
    if s.app().two.is_some() {
        // Last finger of a pinch lifted or moved; close the gesture first.
        s.app().two_end();
    }
    s.on_touch(action, x0, y0)
}

/// Pause = the durability moment: session saved, track chunk flushed.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeOnPause(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) {
    if ptr != 0 {
        let app = shell(ptr).app();
        app.save_session();
        if let Some(rec) = &mut app.recorder {
            rec.flush();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeOnLocation(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    lat: jdouble,
    lon: jdouble,
    accuracy: jfloat,
) {
    if ptr != 0 {
        shell(ptr).app().on_gps(GpsFix { lat, lon, accuracy_m: accuracy });
    }
}
