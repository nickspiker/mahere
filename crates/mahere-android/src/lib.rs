//! mahere on Android: fluor's AndroidShell hosting the shared MapCore, plus the raw two-finger gesture solve and GPS.
//!
//! Gesture philosophy (Nick's spec): no gesture recognizers, no slop radius, no focal-point abstraction. Two fingers are two point correspondences; with rotation locked the solve is scale = finger-distance ratio and the geographic midpoint pinned under the screen midpoint — applied per raw MotionEvent, so the map locks to the fingers from the first event.
//! Rotation joins when the camera grows a bearing.
//!
//! Single-finger events forward into fluor's AndroidShell (which synthesizes MouseInput/CursorMoved); the shell's synthetic MouseWheel (meant for scrolly apps) is swallowed — zoom belongs to the two-finger solve.

use fluor::coord::Coord as Px;
use fluor::event::{CursorIcon, ElementState, Event as FEvent, MouseButton};
use fluor::host::EventResponse;
use fluor::host::android::shell::AndroidShell;
use fluor::host::app::{Context, FluorApp};
use fluor::paint::pack_argb;
use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass, JFloatArray, JObject, JString};
use jni::sys::{jboolean, jdouble, jfloat, jint, jlong};
use mahere_engine::residency::{CellStore, DirStore, HttpStore, TieredStore, DEFAULT_CELLS_URL};
use mahere_engine::{Camera, GpsFix, MapCore};
use mahere_panel::{Controls, Panel};
use ndk::native_window::NativeWindow;
use std::time::Instant;

mod gpu_host;
use gpu_host::GpuHost;

struct TwoFinger {
    /// Geography captured under each finger at gesture start.
    geo_mid: (f64, f64),
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
    /// Pixels the finger has travelled since the press: under a dozen at release is a tap.
    travel: f64,
    /// A second finger touched during this press.
    had_two: bool,
    last_cursor: (f64, f64),
    two: Option<TwoFinger>,
    /// Re-anchor the next CursorMoved instead of panning (finger handoff after a pinch would otherwise jump by the stale delta).
    suppress_move: bool,
    centered_once: bool,
    store: Option<std::sync::Arc<mahere_store::FlatStorage>>,
    /// The dated cell cache, flushed at the pause moment.
    cells_cache: Option<std::sync::Arc<mahere_store::VaultCells>>,
    recorder: Option<std::sync::Arc<std::sync::Mutex<mahere_store::TrackRecorder>>>,
    fixes_since_save: u32,
    persist: Persist,
    /// The GPU path, created on the first draw; None after a failure means the CPU present is in use.
    gpu: Option<GpuHost>,
    gpu_failed: bool,
    /// Draws that panicked inside wgpu; the host is remade each time, until a few in a row.
    gpu_panics: u32,
    /// The latest short bracket from the front camera (its stop, the binned frame): the long frame's clipped bins take their light from it.
    probe_short: Option<(u32, mahere_engine::probe::Binned)>,
    /// The UI's scale, pinched while the panel is open; the last finger distance of such a pinch.
    ui_ru: f32,
    ui_pinch: Option<f64>,
    panel: Panel,
    /// Bytes the cell cache may hold; purged to it at the pause moment.
    cache_budget: u64,
    data_dir: Option<String>,
}

/// Vault writes, off the UI thread. The vault serialises every write through one queue, and the loader fills it with cells while the map streams: a settings save on the UI thread waited behind hundreds of them and the system killed the app as unresponsive. Touch, location and pause now hand their writes here and return at once.
struct Persist {
    tx: std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>,
    /// The latest settings not yet written; a burst of changes (a slider drag) collapses into one write.
    pending_settings: std::sync::Arc<std::sync::Mutex<Option<mahere_store::Settings>>>,
}

impl Persist {
    fn new() -> Persist {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
        std::thread::Builder::new()
            .name("mahere-persist".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            })
            .expect("persist thread");
        Persist { tx, pending_settings: Default::default() }
    }

    fn run(&self, job: impl FnOnce() + Send + 'static) {
        let _ = self.tx.send(Box::new(job));
    }

    fn settings(&self, store: std::sync::Arc<mahere_store::FlatStorage>, s: mahere_store::Settings) {
        let was_pending = self.pending_settings.lock().unwrap().replace(s).is_some();
        if !was_pending {
            let pending = self.pending_settings.clone();
            self.run(move || {
                let latest = pending.lock().unwrap().take();
                if let Some(s) = latest {
                    let _ = mahere_store::save_settings(&store, &s);
                }
            });
        }
    }
}

/// A gigabyte of cells until the slider says otherwise.
const DEFAULT_CACHE_BUDGET: u64 = 1 << 30;

/// Free bytes on the volume holding `dir`, or none if it cannot be asked.
fn free_bytes(dir: &str) -> Option<u64> {
    let c = std::ffi::CString::new(dir).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    Some(st.f_bavail as u64 * st.f_frsize as u64)
}

impl AndroidApp {
    fn controls(&self) -> Controls {
        Controls { real_sun: self.map.real_sun, real_light: self.map.real_light, follow_heading: self.map.follow_heading, lock_to_fix: self.map.lock_to_fix, cache_budget: self.cache_budget, theme: self.map.theme, compressed: self.map.compressed() }
    }

    /// A pinch with the panel open: the finger distance's ratio to the last event's scales the UI.
    fn ui_pinch(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        let d = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt().max(1.0);
        if let Some(prev) = self.ui_pinch {
            self.ui_ru = (self.ui_ru * (d / prev) as f32).clamp(mahere_panel::RU_MIN, mahere_panel::RU_MAX);
            self.panel.set_ru(self.ui_ru);
        }
        self.ui_pinch = Some(d);
    }

    /// The editor's requests: a built-in is copied before it is edited, a field change is live, and closing the editor writes the theme to the vault.
    fn apply_theme_edits(&mut self) {
        for e in self.panel.take_theme_edits() {
            let save_current = |me: &Self| {
                if let Some(store) = &me.store {
                    let (store, t) = (store.clone(), me.map.current_theme().clone());
                    me.persist.run(move || mahere_store::themes::save(&store, &t));
                }
            };
            match e {
                mahere_panel::ThemeEdit::Set(f, v) => self.map.edit_theme(f, v),
                mahere_panel::ThemeEdit::Done => save_current(self),
                mahere_panel::ThemeEdit::Duplicate => {
                    self.map.duplicate_theme();
                    save_current(self);
                    self.save_settings();
                }
                mahere_panel::ThemeEdit::Reset => {
                    if self.map.reset_theme() {
                        save_current(self);
                    }
                }
                mahere_panel::ThemeEdit::Delete => {
                    if let Some(name) = self.map.delete_theme() {
                        if let Some(store) = &self.store {
                            let store = store.clone();
                            self.persist.run(move || mahere_store::themes::delete(&store, &name));
                        }
                        self.save_settings();
                    }
                }
            }
        }
    }

    /// Apply what the panel changed and keep it.
    fn apply_controls(&mut self, ctl: Controls, mask: mahere_engine::LayerMask) {
        self.map.set_layers(mask);
        if ctl.real_sun != self.map.real_sun {
            self.map.set_real_sun(ctl.real_sun);
        }
        if ctl.real_light != self.map.real_light {
            self.map.set_real_light(ctl.real_light);
        }
        if ctl.compressed != self.map.compressed() {
            self.map.set_compressed(ctl.compressed);
        }
        if ctl.follow_heading != self.map.follow_heading {
            self.map.set_follow_heading(ctl.follow_heading);
        }
        if ctl.lock_to_fix != self.map.lock_to_fix {
            self.map.set_lock_to_fix(ctl.lock_to_fix);
        }
        self.cache_budget = ctl.cache_budget;
        if ctl.theme != self.map.theme {
            self.map.set_theme(ctl.theme);
        }
        self.save_settings();
    }

    fn save_settings(&self) {
        if let Some(store) = &self.store {
            let s = mahere_store::Settings { cache_budget: self.cache_budget, layer_bits: self.map.layers().bits() as u64, real_sun: self.map.real_sun, real_light: self.map.real_light, follow_heading: self.map.follow_heading, lock_to_fix: self.map.lock_to_fix, theme: self.map.theme as u64, compressed: self.map.compressed(), ui_ru: self.ui_ru };
            self.persist.settings(store.clone(), s);
        }
    }

    /// What the cache holds and the most the slider may allow: what is held plus the free space.
    fn cache_stats(&self) -> (u64, u64) {
        let used = self.cells_cache.as_ref().map_or(0, |c| c.cached_bytes());
        let free = self.data_dir.as_deref().and_then(free_bytes).unwrap_or(0);
        (used, used + free)
    }

    fn two_begin(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        // A second finger means a gesture, never a tap, however little the first one travelled.
        self.had_two = true;
        // The ground under the screen midpoint, which the gesture keeps there: the mean of the two fingers' latitudes and longitudes is nowhere near it at a globe zoom, where the fingers can straddle a pole or the antimeridian (Nick 2026-10-09: the map jumped a region away on the second finger).
        let geo_mid = self.map.cam.screen_to_geo((x0 + x1) * 0.5, (y0 + y1) * 0.5, self.w, self.h);
        let d0 = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt().max(1.0);
        self.two = Some(TwoFinger {
            geo_mid,
            d0,
            ppd0: self.map.cam.ppd,
            alpha0: (y1 - y0).atan2(x1 - x0),
            bearing0: self.map.cam.bearing,
        });
        self.dragging = false;
    }

    /// Full 4-DOF similarity solve: two finger correspondences determine pan+rotate+zoom. Scale from the distance ratio, bearing from the finger-vector angle (screen angle of a fixed geo segment is -(B + its ENU angle), so B = B0 + (alpha0 - alpha)), then the ground that was under the screen midpoint pinned under it.
    fn two_update(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        let Some(t) = &self.two else { return };
        let d = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt().max(1.0);
        let alpha = (y1 - y0).atan2(x1 - x0);
        self.map.set_ppd(t.ppd0 * d / t.d0);
        self.map.set_bearing(t.bearing0 + (t.alpha0 - alpha));
        let (mx, my) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        self.map.place_anchor(t.geo_mid.0, t.geo_mid.1, mx, my, self.w, self.h);
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
        // Track recording: every fix into the vault (chunk-flushed), the session (camera + sun) refreshed every tenth fix.
        if let Some(rec) = &self.recorder {
            let elev = self.map.gps_elevation().unwrap_or(f32::NAN) as f64;
            let rec = rec.clone();
            self.persist.run(move || rec.lock().unwrap().on_fix(fix.lat, fix.lon, elev));
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
            let s = mahere_store::Session { lat: c.lat, lon: c.lon, ppd: c.ppd, bearing: c.bearing, sun_az: self.map.sun_az as f64, sun_alt: self.map.sun_alt as f64 };
            let store = store.clone();
            self.persist.run(move || {
                let _ = mahere_store::save_session(&store, &s);
            });
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
                // The panel first: the gear and its rows take the tap; the map gets the rest.
                let mut mask = self.map.layers();
                let mut ctl = self.controls();
                let was_open = self.panel.is_open();
                if self.panel.tap(ctx.cursor_x as f32, ctx.cursor_y as f32, self.w, self.h, &mut mask, &mut ctl, &self.map.themes) {
                    self.apply_controls(ctl, mask);
                    self.apply_theme_edits();
                    if self.panel.take_clear_measure() {
                        self.map.clear_measure();
                    }
                    self.dragging = false;
                    return EventResponse::Handled;
                }
                // With the panel open the map takes no gesture: a tap on it closes the panel and nothing else.
                if was_open {
                    self.panel.close();
                    self.save_settings();
                    self.dragging = false;
                    return EventResponse::Handled;
                }
                if self.two.is_none() {
                    self.dragging = true;
                    self.travel = 0.0;
                    self.had_two = false;
                    self.last_cursor = (ctx.cursor_x as f64, ctx.cursor_y as f64);
                }
                EventResponse::Handled
            }
            FEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left } => {
                self.panel.release();
                // A press and release that never travelled is a tap: a new measurement to that point.
                if self.dragging && self.two.is_none() && !self.had_two && self.travel < 12.0 {
                    self.map.tap(ctx.cursor_x as f64, ctx.cursor_y as f64, self.w, self.h);
                }
                self.dragging = false;
                EventResponse::Handled
            }
            FEvent::CursorMoved { .. } => {
                let (x, y) = (ctx.cursor_x as f64, ctx.cursor_y as f64);
                let mut ctl = self.controls();
                if self.panel.drag(x as f32, y as f32, &mut ctl) {
                    let mask = self.map.layers();
                    self.apply_controls(ctl, mask);
                    self.apply_theme_edits();
                    self.last_cursor = (x, y);
                    return EventResponse::Handled;
                }
                if self.suppress_move {
                    self.suppress_move = false;
                } else if self.dragging && self.two.is_none() && !self.panel.is_open() {
                    let (dx, dy) = (x - self.last_cursor.0, y - self.last_cursor.1);
                    self.travel += dx.abs() + dy.abs();
                    if self.travel >= 12.0 {
                        self.map.pan(dx, dy, self.w, self.h);
                        ctx.window.request_redraw();
                    }
                }
                self.last_cursor = (x, y);
                EventResponse::Handled
            }
            // The shell synthesizes MouseWheel from touch-drags for scrolly apps; swallowed — zoom is the two-finger solve's job.
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
        // ANativeWindow RGBA_8888 lands R in the low byte: swap R/B vs the desktop path or Puget Sound renders brown and the pin orange.
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

/// Android has no stderr: everything the engine prints with eprintln (fetch failures, decode failures) vanished. Dup a pipe over fd 2 and relay each line to logcat under the `mahere` tag, once per process.
fn bridge_stderr_to_logcat() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        let mut fds = [0i32; 2];
        if libc::pipe(fds.as_mut_ptr()) != 0 {
            return;
        }
        libc::dup2(fds[1], 2);
        libc::close(fds[1]);
        let rd = fds[0];
        std::thread::spawn(move || {
            let tag = b"mahere\0";
            let mut buf = [0u8; 4096];
            let mut line: Vec<u8> = Vec::new();
            loop {
                let n = libc::read(rd, buf.as_mut_ptr() as *mut libc::c_void, buf.len());
                if n <= 0 {
                    break;
                }
                for &b in &buf[..n as usize] {
                    if b == b'\n' {
                        line.push(0);
                        ndk_sys::__android_log_write(ndk_sys::android_LogPriority::ANDROID_LOG_INFO.0 as i32, tag.as_ptr() as *const _, line.as_ptr() as *const _);
                        line.clear();
                    } else {
                        line.push(b);
                    }
                }
            }
        });
    });
}

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
    bridge_stderr_to_logcat();
    eprintln!("mahere native init: {dir}");
    // The vault: session, tracks, and the on-device cell cache (kete).
    let store = mahere_store::open(Some(&dir)).ok();
    // Cells stream from the bucket through the vault; `cells-local` (pushed by hand) overrides for offline development.
    let local = std::path::PathBuf::from(format!("{dir}/cells-local"));
    let mut cells_cache: Option<std::sync::Arc<mahere_store::VaultCells>> = None;
    let cell_store: std::sync::Arc<dyn CellStore> = if local.is_dir() {
        std::sync::Arc::new(DirStore(local))
    } else {
        let remote = std::sync::Arc::new(HttpStore::new(DEFAULT_CELLS_URL));
        match &store {
            Some(v) => {
                let cache = std::sync::Arc::new(mahere_store::VaultCells::new(v.clone()));
                cells_cache = Some(cache.clone());
                std::sync::Arc::new(TieredStore::new(cache, remote))
            }
            None => remote,
        }
    };
    let session = store.as_ref().and_then(|s| mahere_store::load_session(s));
    let settings = store.as_ref().and_then(|s| mahere_store::load_settings(s));
    // Mt St Helens until the first GPS fix recenters us; a saved view only if it is a view (finite, on the globe, at a zoom the map has).
    let home = Camera { lat: 46.2000, lon: -122.1900, ppd: 12_000.0, bearing: 0.0 };
    let cam = match session {
        Some(s) if s.lat.is_finite() && s.lon.is_finite() && s.ppd.is_finite() && s.bearing.is_finite() && s.lat.abs() <= 90.0 && s.lon.abs() <= 360.0 && (1.0..=4_000_000.0).contains(&s.ppd) => {
            Camera { lat: s.lat, lon: s.lon, ppd: s.ppd, bearing: s.bearing }
        }
        _ => home,
    };
    eprintln!("camera: {:.5} {:.5} ppd {:.1} bearing {:.3}", cam.lat, cam.lon, cam.ppd, cam.bearing);
    let mut map = MapCore::new(cell_store, cam);
    if let Some(s) = session {
        map.sun_az = s.sun_az as f32;
        map.sun_alt = s.sun_alt as f32;
    }
    // Themes from the vault (the built-ins written on first launch), then the saved choice, then the saved layers over the theme's defaults.
    if let Some(v) = &store {
        map.set_themes(mahere_store::themes::load_all(v));
    }
    let mut ui_ru = 1.0f32;
    if let Some(s) = settings {
        map.set_theme(s.theme as usize);
        map.set_layers(mahere_engine::LayerMask::from_bits(s.layer_bits as u32));
        map.set_real_sun(s.real_sun);
        map.set_real_light(s.real_light);
        ui_ru = s.ui_ru;
        map.set_compressed(s.compressed);
        map.set_follow_heading(s.follow_heading);
        map.set_lock_to_fix(s.lock_to_fix);
    }
    let recorder = store.clone().map(|s| std::sync::Arc::new(std::sync::Mutex::new(mahere_store::TrackRecorder::new(s))));
    let mut app = AndroidApp {
        map,
        w: width as usize,
        h: height as usize,
        dragging: false,
        travel: 0.0,
        had_two: false,
        last_cursor: (0., 0.),
        two: None,
        suppress_move: false,
        // A restored session IS the view; don't let the first fix yank it.
        centered_once: session.is_some(),
        store,
        cells_cache,
        recorder,
        fixes_since_save: 0,
        persist: Persist::new(),
        gpu: None,
        gpu_failed: false,
        gpu_panics: 0,
            probe_short: None,
            ui_ru,
            ui_pinch: None,
        panel: Panel::new(),
        cache_budget: settings.map_or(DEFAULT_CACHE_BUDGET, |s| if s.cache_budget == 0 { DEFAULT_CACHE_BUDGET } else { s.cache_budget }),
        data_dir: Some(dir.clone()),
    };
    app.panel.set_ru(ui_ru);
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
        // A restored surface arrives with undefined buffers; same-size resizes must still repaint everything.
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
    // The GPU draws when it can; fluor.s CPU present is the fallback (no Vulkan, or a surface it cannot wrap).
    let app = shell(ptr).app();
    if !app.gpu_failed {
        if app.gpu.is_none() {
            match GpuHost::new(&window) {
                Some(g) => app.gpu = Some(g),
                None => {
                    app.gpu_failed = true;
                    eprintln!("gpu: unavailable, CPU present");
                }
            }
        }
        let stats = app.cache_stats();
        let ctl = app.controls();
        let AndroidApp { gpu: Some(g), map, panel, w, h, .. } = app else {
            return shell(ptr).draw(&window) as jboolean;
        };
        if g.ensure_window(&window) {
            // A fatal error inside wgpu panics; across the JNI boundary that is an abort. Caught here, the GPU host is dropped and made again on the next frame (a lost device comes back that way); after a few in a row the CPU present takes over.
            let (w, h) = (*w as u32, *h as u32);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.draw(map, panel, ctl, stats, w, h))) {
                Ok(drawn) => return drawn as jboolean,
                Err(e) => {
                    let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                    eprintln!("gpu: draw panicked: {msg}");
                    app.gpu = None;
                    app.gpu_panics += 1;
                    if app.gpu_panics >= 3 {
                        app.gpu_failed = true;
                        eprintln!("gpu: giving up after {} panics, CPU present", app.gpu_panics);
                    }
                    return 0;
                }
            }
        }
    }
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
    // With the panel open, two fingers scale the UI (fluor's RU), not the map.
    if s.app().panel.is_open() {
        if count >= 2 && action != ACTION_POINTER_UP {
            s.app().ui_pinch(x0 as f64, y0 as f64, x1 as f64, y1 as f64);
            return 0;
        }
        if count >= 2 || s.app().ui_pinch.is_some() {
            s.app().ui_pinch = None;
            s.app().save_settings();
            return 0;
        }
        return s.on_touch(action, x0, y0);
    }
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

/// A place to look at, from a `geo:` intent (a shared location, or adb): the camera goes there, at the zoom given (pixels per degree) when positive.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeGoTo(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    lat: jdouble,
    lon: jdouble,
    ppd: jdouble,
) {
    if ptr != 0 && lat.is_finite() && lon.is_finite() {
        let s = shell(ptr);
        let (w, h) = (s.app().w, s.app().h);
        s.app().map.go_to(lat, lon, if ppd > 0.0 { Some(ppd) } else { None }, w, h);
    }
}

/// The device heading from the rotation-vector sensor, degrees clockwise from north: the lighting turns against it so the sun stays where it physically is.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeOnHeading(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    heading_deg: jfloat,
) {
    if ptr != 0 {
        shell(ptr).app().map.set_device_heading(heading_deg);
    }
}

/// The device's rotation matrix from the rotation-vector sensor, row-major, world = R · device: the full orientation, so the real sun lights the landscape exactly as the phone is held.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeOnOrientation(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    r0: jfloat,
    r1: jfloat,
    r2: jfloat,
    r3: jfloat,
    r4: jfloat,
    r5: jfloat,
    r6: jfloat,
    r7: jfloat,
    r8: jfloat,
) {
    if ptr != 0 {
        shell(ptr).app().map.set_device_rotation([r0, r1, r2, r3, r4, r5, r6, r7, r8]);
    }
}

/// Magnetic declination at the fix, degrees east positive, from Android's geomagnetic model: the sensor's north is magnetic, the almanac's is true.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeOnDeclination(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    declination_deg: jfloat,
) {
    if ptr != 0 {
        shell(ptr).app().map.set_declination(declination_deg);
    }
}

/// Pause = the durability moment: session saved, track chunk flushed.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeOnPause(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) {
    if ptr != 0 {
        // Every write goes to the persist thread: the UI thread returns at once, and the system gives a paused app time to finish.
        let app = shell(ptr).app();
        app.save_session();
        if let Some(rec) = &app.recorder {
            let rec = rec.clone();
            app.persist.run(move || rec.lock().unwrap().flush());
        }
        if let Some(c) = &app.cells_cache {
            let (c, budget) = (c.clone(), app.cache_budget);
            app.persist.run(move || {
                c.purge_to(budget);
                c.flush();
            });
        }
        app.save_settings();
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

/// Whether the engine wants the front camera: the Activity polls this every frame and opens or closes the camera to match.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeProbeWanted(_env: JNIEnv, _class: JClass, ptr: jlong) -> jboolean {
    (ptr != 0 && shell(ptr).app().map.real_light) as jboolean
}

/// The camera could not be had (permission refused, no raw front camera): the mode turns itself off and the setting follows.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeProbeDenied(_env: JNIEnv, _class: JClass, ptr: jlong) {
    if ptr != 0 {
        let app = shell(ptr).app();
        app.map.set_real_light(false);
        app.save_settings();
    }
}

/// A raw front-camera frame: samples in `buf` (`w × h`, `row_stride` bytes a row, packed 10-bit or 16-bit, Bayer order `cfa` as Android numbers it, the sensor's four `black` pedestals and its `white`), the sensor's `orientation`, the lens half-angle tangents across and down the sensor, Android's row-major XYZ→camera matrix, and the frame's `stop` on the exposure ladder (0 the base exposure, each step one stop shorter) with the long frame's. Binned, brought to absolute units by its stop, converted to VSF RGB, turned upright and painted onto the sphere. `stats` gets the clipped fraction and the 99.9th percentile level, which the Activity's exposure loop steers on.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeOnProbe(
    mut env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    buf: JByteBuffer,
    w: jint,
    h: jint,
    row_stride: jint,
    packed10: jboolean,
    cfa: jint,
    black: JFloatArray,
    white: jint,
    orientation: jint,
    tan_w: jfloat,
    tan_h: jfloat,
    xyz_to_cam: JFloatArray,
    stats: JFloatArray,
    stop: jint,
    long_stop: jint,
) {
    use mahere_engine::probe::{Cfa, Raw, bin, to_vsf, upright};
    if ptr == 0 {
        return;
    }
    let Some(cfa) = Cfa::from_android(cfa) else { return };
    let (Ok(addr), Ok(len)) = (env.get_direct_buffer_address(&buf), env.get_direct_buffer_capacity(&buf)) else { return };
    if addr.is_null() || (row_stride as usize) * (h as usize) > len {
        return;
    }
    let data = unsafe { std::slice::from_raw_parts(addr, len) };
    let mut pedestal = [0f32; 4];
    let _ = env.get_float_array_region(&black, 0, &mut pedestal);
    let raw = Raw { data, w: w as usize, h: h as usize, row_stride: row_stride as usize, packed10: packed10 != 0, cfa, black: pedestal.map(|b| b as u16), white: white as u16 };
    let t0 = std::time::Instant::now();
    let mut b = bin(&raw, 48);
    let bin_ms = t0.elapsed().as_secs_f32() * 1e3;
    let app = shell(ptr).app();
    let (stop, long_stop) = (stop.max(0) as u32, long_stop.max(0) as u32);
    // The short bracket: kept for the next long frame, which fills what it clipped from it.
    if stop > long_stop {
        app.probe_short = Some((stop, b));
        return;
    }
    let st = b.stats;
    let _ = env.set_float_array_region(&stats, 0, &[st.clipped, st.p999]);
    let mut filled = 0usize;
    if let Some((short_stop, short)) = &app.probe_short {
        filled = b.clipped.iter().filter(|&&c| c > 0).count();
        b.fill_clipped(short, stop, *short_stop);
    }
    b.shift(stop);
    let mut m = [0f32; 9];
    if env.get_float_array_region(&xyz_to_cam, 0, &mut m).is_ok() && m.iter().any(|&v| v != 0.0) {
        to_vsf(&mut b.rgb, &m);
    }
    let (uw, uh, rgb) = upright(b.w, b.h, &b.rgb, orientation);
    // Upright: the tangents turn with the frame.
    let (tw, th) = if matches!(orientation.rem_euclid(360), 90 | 270) { (tan_h, tan_w) } else { (tan_w, tan_h) };
    app.map.paint_probe(uw, uh, tw, th, &rgb);
    // Once a second or so: what the camera saw, as the exposure loop and the colour path are tuned.
    static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
    let mut last = LAST.lock().unwrap();
    if last.is_none_or(|t| t.elapsed().as_secs_f32() > 1.0) {
        *last = Some(std::time::Instant::now());
        let n = rgb.len().max(1) as u64;
        let sum = rgb.iter().fold([0u64; 3], |a, p| [a[0] + p[0] as u64, a[1] + p[1] as u64, a[2] + p[2] as u64]);
        let mean = [sum[0] / n, sum[1] / n, sum[2] / n];
        eprintln!("probe: clipped {:.5} p999 {:.3} filled {filled} bins from the bracket; stop {stop} (short {}), bin {bin_ms:.1} ms; mean {mean:?}; sphere {:.0}% painted; {:?} = (level, light on world up, on the screen, brightest triangle)", st.clipped, st.p999, app.probe_short.as_ref().map_or(0, |s| s.0), app.map.probe_coverage() * 100.0, app.map.probe_report());
    }
}
