//! mahere on Android: fluor's AndroidShell hosting the touch-device app (mahere-device: the map, the panel, the two-finger solve, GPS), with what is Android's here: the JNI entry points the activity calls, the Vulkan map host (gpu_host), raw MotionEvents into the two-finger solve, the location, heading and light sensors, the share sheet and the installer.

use fluor::host::android::shell::AndroidShell;
use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass, JFloatArray, JObject, JString};
use jni::sys::{jboolean, jdouble, jfloat, jint, jlong};
use mahere_engine::GpsFix;
use ndk::native_window::NativeWindow;
use std::time::Instant;

mod gpu_host;
use gpu_host::GpuHost;

/// The touch-device app (mahere-device), on the GPU through the Vulkan host here.
pub type AndroidApp = mahere_device::DeviceApp<GpuHost>;

impl mahere_device::MapGpu for GpuHost {
    fn labeller(&self) -> &mahere_panel::labels::Labeller {
        &self.labeller
    }
}

type Shell = AndroidShell<AndroidApp>;

fn shell<'a>(ptr: jlong) -> &'a mut Shell {
    unsafe { &mut *(ptr as *mut Shell) }
}

/// Android MotionEvent action codes (masked).
const ACTION_DOWN: jint = 0;
const ACTION_MOVE: jint = 2;
const ACTION_POINTER_UP: jint = 6;

/// Android has no stderr: everything the engine prints with eprintln (fetch failures, decode failures) vanished. Dup a pipe over fd 2 and relay each line to logcat under the `mahere` tag, once per process.
/// The log facade's records from the vault (kete, manifestus) onto stderr, and so into logcat.
struct VaultLog;

impl log::Log for VaultLog {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.target().starts_with("kete") || m.target().starts_with("manifestus")
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            eprintln!("{} {}", r.level(), r.args());
        }
    }
    fn flush(&self) {}
}

fn bridge_stderr_to_logcat() {
    if log::set_logger(&VaultLog).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
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
    let t0 = std::time::Instant::now();
    let store = mahere_store::open(Some(&dir)).ok();
    eprintln!("start: vault open at {} ms", t0.elapsed().as_millis());
    let app = AndroidApp::open(width as u32, height as u32, store, Some(&dir));
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
                Some(mut g) => {
                    g.labeller.set_fonts(app.fonts.clone());
                    app.gpu = Some(g);
                }
                None => {
                    app.gpu_failed = true;
                    eprintln!("gpu: unavailable, CPU present");
                }
            }
        }
        let stats = app.cache_stats();
        let ctl = app.controls();
        // A download's progress repaints the open panel a few times a second; the GPU host otherwise skips frames while the map is still.
        let finished = app.areas.poll();
        // A font pack on its way or just arrived: the panel paints again, loading it as it does.
        if app.panel.fonts_due() {
            app.panel.invalidate();
        }
        if app.panel.is_open() && (finished || (app.areas.downloading() && app.areas_shown.elapsed().as_millis() >= 250)) {
            app.panel.invalidate();
            app.areas_shown = Instant::now();
        }
        // Updates: what the threads found, onto the page; a verified build waits for the activity to install it.
        if app.updater.poll() {
            let view = app.updates_view();
            app.panel.set_updates(view);
        }
        if let Some(p) = app.updater.take_ready() {
            app.install = Some(p.to_string_lossy().into_owned());
        }
        let areas = app.areas.view(app.map.user.has_selection());
        let AndroidApp { gpu: Some(g), map, panel, w, h, .. } = app else {
            return shell(ptr).draw(&window) as jboolean;
        };
        if g.is_lost() {
            app.gpu = None;
            app.gpu_panics += 1;
            if app.gpu_panics >= 3 {
                app.gpu_failed = true;
                eprintln!("gpu: giving up after {} failures, CPU present", app.gpu_panics);
            }
            return 1;
        }
        if g.ensure_window(&window) {
            // A fatal error inside wgpu panics; across the JNI boundary that is an abort. Caught here, the GPU host is dropped and made again on the next frame (a lost device comes back that way); after a few in a row the CPU present takes over.
            let (w, h) = (*w as u32, *h as u32);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.draw(map, panel, ctl, stats, &areas, w, h))) {
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
    mut env: JNIEnv,
    activity: JObject,
    ptr: jlong,
    action: jint,
    count: jint,
    x0: jfloat,
    y0: jfloat,
    x1: jfloat,
    y1: jfloat,
    tool: jint,
    pressure: jfloat,
    buttons: jint,
) -> jint {
    if ptr == 0 {
        return 0;
    }
    let s = shell(ptr);
    // The touch's tool and pressure (Android's TOOL_TYPE_STYLUS is 2): a stylus draws at its pressure, a finger at a middle one.
    // An eraser tip (TOOL_TYPE_ERASER, 4) is a stylus too, and it, or a stylus with its side button held (BUTTON_STYLUS_PRIMARY, 0x20, the S Pen's), erases whatever the tool (Nick 2026-10-21).
    let stylus = tool == 2 || tool == 4;
    s.app().stylus = stylus;
    s.app().erasing = tool == 4 || (stylus && buttons & 0x20 != 0);
    s.app().pressure = if stylus { pressure.clamp(0.0, 1.0) } else { 0.5 };
    if stylus {
        s.app().stylus_seen = Some(Instant::now());
    }
    if action == ACTION_DOWN {
        s.app().first_down = (x0 as f64, y0 as f64);
    }
    // With the panel open, two fingers scale the interface or the map's lettering, by where the first came down; never the map.
    if s.app().panel.is_open() {
        if count >= 2 && action != ACTION_POINTER_UP {
            s.app().pinched = true;
            s.app().ui_pinch(x0 as f64, y0 as f64, x1 as f64, y1 as f64);
            return 0;
        }
        if count >= 2 || s.app().ui_pinch.is_some() {
            s.app().ui_pinch = None;
            if s.app().pinch_on_panel {
                s.app().save_settings();
            } else {
                let app = s.app();
                if let Some(store) = &app.store {
                    let _ = mahere_store::save_label_scale(store, app.panel.label_scale());
                }
            }
            return 0;
        }
        let r = s.on_touch(action, x0, y0);
        share_if_asked(&mut env, &activity, s);
        return r;
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
    let r = s.on_touch(action, x0, y0);
    share_if_asked(&mut env, &activity, s);
    r
}

/// A share the panel asked for during this touch: the activity opens the system's share sheet on the file.
fn share_if_asked(env: &mut JNIEnv, activity: &JObject, s: &mut Shell) {
    if let Some(path) = s.app().share_file.take() {
        eprintln!("share: {path}");
        if let Ok(jpath) = env.new_string(&path) {
            if let Err(e) = env.call_method(activity, "shareFile", "(Ljava/lang/String;)V", &[jni::objects::JValue::Object(&jpath)]) {
                eprintln!("share: {e}");
                let _ = env.exception_clear();
            }
        }
    }
}

/// The back button: the panel one page back, or closed; else a measurement shown is cleared. False when there was nothing to step back from, and the system's back applies.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeBack(_env: JNIEnv, _class: JClass, ptr: jlong) -> jboolean {
    if ptr == 0 {
        return 0;
    }
    let app = shell(ptr).app();
    if app.panel.back() {
        app.save_settings();
        return 1;
    }
    if app.map.has_measure() {
        app.map.clear_measure();
        return 1;
    }
    0
}

/// A shared area document, from a file or link the activity received: kept with the areas and opened.
/// Text the soft keyboard committed: into the search.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeOnText(mut env: JNIEnv, _class: JClass, ptr: jlong, text: jni::objects::JString) {
    if ptr == 0 {
        return;
    }
    let Ok(t) = env.get_string(&text) else { return };
    shell(ptr).on_text_input(t.into());
}

/// A key the soft keyboard sent (delete, enter), as Android numbers it; true when it was taken.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeOnKey(_env: JNIEnv, _class: JClass, ptr: jlong, code: jint) -> jboolean {
    if ptr == 0 {
        return 0;
    }
    shell(ptr).on_key_event(code) as jboolean
}

/// Whether the keyboard should come up (1), go (-1) or stay as it is (0): read every frame, as the search opens and closes from more than touches.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativePollKeyboard(_env: JNIEnv, _class: JClass, ptr: jlong) -> jint {
    if ptr == 0 {
        return 0;
    }
    shell(ptr).poll_keyboard()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeOpenArea(env: JNIEnv, _class: JClass, ptr: jlong, bytes: jni::objects::JByteArray) {
    if ptr == 0 {
        return;
    }
    if let Ok(b) = env.convert_byte_array(&bytes) {
        shell(ptr).app().open_area(&b);
    }
}

/// The phone says its pen is out or back in its slot (Samsung's broadcast), or a stylus is hovering (out, then).
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativePenOut(_env: JNIEnv, _class: JClass, ptr: jlong, out: jboolean) {
    if ptr != 0 {
        shell(ptr).app().pen_out = Some(out != 0);
    }
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
            let pinned = app.areas.pinned();
            app.persist.run(move || {
                c.set_pinned(pinned);
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
    height: jdouble,
    v_accuracy: jfloat,
) {
    if ptr != 0 {
        shell(ptr).app().on_gps(GpsFix { lat, lon, accuracy_m: accuracy, height_m: height.is_finite().then_some(height as f32), v_accuracy_m: v_accuracy.is_finite().then_some(v_accuracy) });
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
    env: JNIEnv,
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

#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeTakeInstall<'a>(env: JNIEnv<'a>, _class: JClass, ptr: jlong) -> jni::sys::jstring {
    if ptr == 0 {
        return std::ptr::null_mut();
    }
    match shell(ptr).app().install.take() {
        Some(path) => env.new_string(path).map_or(std::ptr::null_mut(), |s| s.into_raw()),
        None => std::ptr::null_mut(),
    }
}

/// Text a copy button asked for, once, for the activity to put on the clipboard.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeTakeCopy<'a>(env: JNIEnv<'a>, _class: JClass, ptr: jlong) -> jni::sys::jstring {
    if ptr == 0 {
        return std::ptr::null_mut();
    }
    match shell(ptr).take_clipboard() {
        Some(what) => env.new_string(what).map_or(std::ptr::null_mut(), |s| s.into_raw()),
        None => std::ptr::null_mut(),
    }
}

/// Whether the network is unmetered, from the activity's network callback: automatic updates download only then.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeSetUnmetered(_env: JNIEnv, _class: JClass, ptr: jlong, unmetered: jboolean) {
    if ptr != 0 {
        shell(ptr).app().updater.unmetered = unmetered != 0;
    }
}

/// Whether a track is being recorded: the activity keeps the track service running while it is.
#[unsafe(no_mangle)]
pub extern "system" fn Java_nz_mahere_app_MahereActivity_nativeWantsRecording(_env: JNIEnv, _class: JClass, ptr: jlong) -> jboolean {
    if ptr == 0 {
        return 0;
    }
    shell(ptr).app().recording as jboolean
}

/// The track service's fixes, one or a batch (a batch arrives a minute or more after it was measured): each into the track as the receiver measured it, at the time it was taken (NaN where it gave nothing), the newest onto the map.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_nz_mahere_app_MahereTrackService_nativeOnTrackFixes<'a>(env: JNIEnv<'a>, _class: JClass, ptr: jlong, lats: jni::objects::JDoubleArray<'a>, lons: jni::objects::JDoubleArray<'a>, accs: jni::objects::JFloatArray<'a>, times: jni::objects::JLongArray<'a>, heights: jni::objects::JDoubleArray<'a>, vaccs: jni::objects::JFloatArray<'a>, speeds: jni::objects::JFloatArray<'a>, bearings: jni::objects::JFloatArray<'a>) {
    if ptr == 0 {
        return;
    }
    let n = env.get_array_length(&lats).unwrap_or(0).max(0) as usize;
    let (mut la, mut lo, mut ht) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
    let (mut ac, mut va, mut sp, mut be) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let mut tm = vec![0i64; n];
    let ok = env.get_double_array_region(&lats, 0, &mut la).is_ok()
        && env.get_double_array_region(&lons, 0, &mut lo).is_ok()
        && env.get_double_array_region(&heights, 0, &mut ht).is_ok()
        && env.get_float_array_region(&accs, 0, &mut ac).is_ok()
        && env.get_float_array_region(&vaccs, 0, &mut va).is_ok()
        && env.get_float_array_region(&speeds, 0, &mut sp).is_ok()
        && env.get_float_array_region(&bearings, 0, &mut be).is_ok()
        && env.get_long_array_region(&times, 0, &mut tm).is_ok();
    if !ok {
        return;
    }
    let app = shell(ptr).app();
    if app.recording {
        if let Some(rec) = &app.recorder {
            let rec = rec.clone();
            let some = |v: f32| v.is_finite().then_some(v);
            let fixes: Vec<mahere_store::Fix> = (0..n)
                .map(|i| mahere_store::Fix { coord: mahere_coord::Coord::from_lat_lon(la[i], lo[i]).raw(), unix_ms: tm[i], height_m: ht[i].is_finite().then_some(ht[i] as f32), accuracy_m: ac[i], v_accuracy_m: some(va[i]), speed_ms: some(sp[i]), bearing_deg: some(be[i]) })
                .collect();
            app.persist.run(move || {
                let mut r = rec.lock().unwrap();
                for f in fixes {
                    r.on_gps(f);
                }
            });
        }
    }
    if n > 0 {
        app.on_gps(GpsFix { lat: la[n - 1], lon: lo[n - 1], accuracy_m: ac[n - 1], height_m: ht[n - 1].is_finite().then_some(ht[n - 1] as f32), v_accuracy_m: va[n - 1].is_finite().then_some(va[n - 1]) });
    }
}
