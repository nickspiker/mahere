//! The GPU host on Android: a wgpu (Vulkan) surface over the Activity's ANativeWindow, driven from `nativeDraw` in place of fluor's CPU present. fluor still owns input; the engine still owns residency and the frame plan; this is the glue that gets a planned frame onto the screen.

use mahere_engine::MapCore;
use mahere_gpu::{GpuMap, RenderMode};
use mahere_panel::{Controls, Panel, Readouts};
use ndk::native_window::NativeWindow;
use raw_window_handle::{AndroidDisplayHandle, AndroidNdkWindowHandle, RawDisplayHandle, RawWindowHandle};

pub struct GpuHost {
    instance: wgpu::Instance,
    // Declared before the window it references so it drops first.
    surface: wgpu::Surface<'static>,
    window: NativeWindow,
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    alpha: wgpu::CompositeAlphaMode,
    pub map: GpuMap,
    /// The map's lettering, laid out when the view rests.
    pub labeller: mahere_panel::labels::Labeller,
    configured: (u32, u32),
    overlay_stamp: Option<OverlayStamp>,
    overlay_at: std::time::Instant,
    last_cam: Option<(u64, u64, u64, u64)>,
    /// The theme tables uploaded so far.
    style_version: Option<u64>,
    /// The last frame drawn at 1×, and the last camera move.
    last_low: std::time::Instant,
    last_move: std::time::Instant,
    /// The last frame was drawn still, at the still factor: nothing more to draw until something changes.
    settled: bool,
    frames: u64,
    frame_ms: f32,
    /// What a frame costs on the GPU (2026-10-21: the Pixel 3a's Adreno 615 lost its device on the first still frame at 3×, 21 million pixels): each submitted frame's time to done, with its scale, as the queue reports it; the cost of a 1× frame learnt from them; and the still scale that cost allows.
    gpu_done: std::sync::Arc<std::sync::Mutex<Option<(u32, f32)>>>,
    /// The device was lost: the host is made again.
    lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    cost_1x: Option<f32>,
    /// The passes' GPU times summed over the report's frames (map, labels, present) and how many frames gave them; the CPU's labels, overlay and encode times summed likewise.
    pass_ms: [f32; 3],
    /// The profiling switches and when they were read.
    debug: Debug,
    debug_read: std::time::Instant,
    pass_n: u32,
    labels_ms: f32,
    overlay_ms: f32,
    encode_ms: f32,
    work_ms: f32,
    plan_ms: f32,
    sync_ms: f32,
    report: std::time::Instant,
}

/// A system property as text, read from adb's reach (`adb shell setprop debug.mahere.<name> <value>`), for profiling on a phone without touching it.
fn debug_prop(name: &str) -> Option<String> {
    let key = std::ffi::CString::new(format!("debug.mahere.{name}")).ok()?;
    let mut buf = [0u8; 92];
    let n = unsafe { libc::__system_property_get(key.as_ptr(), buf.as_mut_ptr() as *mut libc::c_char) };
    (n > 0).then(|| String::from_utf8_lossy(&buf[..n as usize]).into_owned())
}

/// The profiling switches, read every two seconds: `mask_off` (layer bits the shader leaves out, decimal), `spin` (1: draw every frame, the view still), `overlay_off` (1: the panel's overlay not repainted), `comp_off` (1: every pixel composited whole, the baked composites unused).
#[derive(Default, Clone, Copy)]
struct Debug {
    mask_off: u32,
    spin: bool,
    overlay_off: bool,
    comp_off: bool,
}

impl Debug {
    fn read() -> Debug {
        let num = |n: &str| debug_prop(n).and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(0);
        Debug { mask_off: num("mask_off"), spin: num("spin") == 1, overlay_off: num("overlay_off") == 1, comp_off: num("comp_off") == 1 }
    }
}

/// How long a still frame may take on the GPU, milliseconds: the scale is the largest whose measured cost fits (a still frame is drawn once, but far past this a mobile GPU's watchdog takes the device).
const STILL_BUDGET_MS: f32 = 100.0;
/// Plane uploads a frame makes at most: a burst of arriving cells spread over frames, never one long stall on the UI thread.
const UPLOADS_PER_FRAME: usize = 24;
/// Cells' composites baked a frame at most: each is a full composite of 131 072 texels, some 20 ms on a Pixel 3a.
const BAKES_PER_FRAME: usize = 3;

/// Supersampling while the camera moves and once it has stopped.
const MOVING_SCALE: u32 = 1;
const STILL_SCALE: u32 = 3;
/// The most pixels a still frame draws, its scale included: three times a 1080 by 2340 screen. A 1440 by 3040 screen at three times is 39 million, near a second a frame on the Note 10+'s Adreno 640 (Nick 2026-10-15), so it rests at two, the same pixels as the 1080 phones at three.
const STILL_PIXELS: u64 = 3 * 3 * 1080 * 2340;

type OverlayStamp = (u64, u64, u64, u64, Option<(u64, u64, u32)>, u32, u32, bool, u32, u64, u32, u64);

impl GpuHost {
    pub fn new(window: &NativeWindow) -> Option<GpuHost> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
        let surface = Self::surface_for(&instance, window)?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .map_err(|e| eprintln!("gpu: no adapter: {e:?}"))
        .ok()?;
        // Texture arrays hold one cell per layer: ask for the adapter.s real layer limit, not WebGPU.s 256.
        let mut limits = wgpu::Limits::default();
        limits.max_texture_array_layers = adapter.limits().max_texture_array_layers.min(2048);
        // The still frame is drawn at three times the screen, larger than WebGPU's 8192 on a tall screen (a 1440 by 3040 Note 10+ needs 9120): ask for the GPU's own limit (Nick 2026-10-15: the screen flicked to black whenever the map came to rest).
        limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d.max(limits.max_texture_dimension_2d);
        // The GPU's own clock around the passes where it has one: what a frame costs it, apart from the CPU's share.
        let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("mahere"), required_limits: limits, required_features, ..Default::default() }))
            .map_err(|e| eprintln!("gpu: no device: {e:?}"))
            .ok()?;
        // A validation error logs instead of killing the app: the map goes wrong on screen, the phone stays usable, logcat says why.
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| eprintln!("gpu: {e}")));
        // A lost device answers nothing again: marked, so the host is dropped and made afresh on the next frame (2026-10-21: the Pixel 3a stayed black after its loss, every texture write refused).
        let lost = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mark = lost.clone();
        device.set_device_lost_callback(move |reason, msg| {
            eprintln!("gpu: device lost ({reason:?}): {msg}");
            mark.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        let caps = surface.get_capabilities(&adapter);
        // A linear 8-bit format: the shader writes display values, not scene-linear ones.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let alpha = caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto);
        let mut map = GpuMap::new(&device, &queue, format);
        map.upload_budget = UPLOADS_PER_FRAME;
        map.bake_budget = BAKES_PER_FRAME;
        let info = adapter.get_info();
        eprintln!("gpu: {} ({:?}), surface {:?}", info.name, info.backend, format);
        Some(GpuHost {
            instance,
            surface,
            window: window.clone(),
            device,
            queue,
            format,
            alpha,
            map,
            labeller: mahere_panel::labels::Labeller::new(),
            configured: (0, 0),
            overlay_stamp: None,
            overlay_at: std::time::Instant::now(),
            last_cam: None,
            style_version: None,
            last_low: std::time::Instant::now(),
            last_move: std::time::Instant::now(),
            settled: false,
            frames: 0,
            frame_ms: 0.0,
            gpu_done: Default::default(),
            lost,
            cost_1x: None,
            pass_ms: [0.0; 3],
            debug: Debug::default(),
            debug_read: std::time::Instant::now() - std::time::Duration::from_secs(10),
            pass_n: 0,
            labels_ms: 0.0,
            overlay_ms: 0.0,
            encode_ms: 0.0,
            work_ms: 0.0,
            plan_ms: 0.0,
            sync_ms: 0.0,
            report: std::time::Instant::now(),
        })
    }

    fn surface_for(instance: &wgpu::Instance, window: &NativeWindow) -> Option<wgpu::Surface<'static>> {
        let rwh = RawWindowHandle::AndroidNdk(AndroidNdkWindowHandle::new(window.ptr().cast()));
        let rdh = RawDisplayHandle::Android(AndroidDisplayHandle::new());
        unsafe { instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle { raw_display_handle: rdh, raw_window_handle: rwh }) }
            .map_err(|e| eprintln!("gpu: surface: {e:?}"))
            .ok()
    }

    /// Whether the device was lost: the host must be made again.
    pub fn is_lost(&self) -> bool {
        self.lost.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The Activity recreated its Surface: a new wgpu surface over the new window. False if that failed.
    pub fn ensure_window(&mut self, window: &NativeWindow) -> bool {
        if self.window.ptr() == window.ptr() {
            return true;
        }
        match Self::surface_for(&self.instance, window) {
            Some(s) => {
                self.surface = s;
                self.window = window.clone();
                self.configured = (0, 0);
                true
            }
            None => false,
        }
    }

    /// One frame: tick, plan, mirror the pool, marks when they changed, draw, present. Returns whether more frames are wanted (cells still arriving).
    pub fn draw(&mut self, map: &mut MapCore, panel: &mut Panel, ctl: Controls, cache: (u64, u64), areas: &mahere_panel::AreasView, w: u32, h: u32) -> bool {
        if w == 0 || h == 0 {
            return false;
        }
        let t0 = std::time::Instant::now();
        if self.configured != (w, h) {
            self.surface.configure(
                &self.device,
                &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: self.format,
                    width: w,
                    height: h,
                    present_mode: wgpu::PresentMode::Fifo,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: self.alpha,
                    view_formats: vec![],
                },
            );
            self.configured = (w, h);
            tag_bt2020(&self.window);
        }
        // Nothing changed and nothing arrived: the surface keeps its last image, and the CPU keeps its budget. The poll still runs so finished uploads release their staging memory. Once a quarter second has passed since the last change, one more frame is drawn at the still factor and stays on screen.
        // The elevations the last map pass drew, if they have come back: the next frame's range.
        if let Some(r) = self.map.take_range() {
            map.set_drawn_range(r);
        }
        let fresh = self.configured != (w, h) || self.frames == 0;
        let panel_dirty = panel.take_dirty();
        if panel_dirty {
            self.overlay_stamp = None;
        }
        // A layout the label worker finished is a frame to draw.
        let lettered = self.labeller.poll();
        // Planes still owed to the GPU are a change too: a view at rest goes on drawing until they are all in, or the cells waiting draw without them.
        if self.debug_read.elapsed().as_secs() >= 2 {
            self.debug = Debug::read();
            self.debug_read = std::time::Instant::now();
            self.map.mask_off = self.debug.mask_off;
            self.map.composite = !self.debug.comp_off;
        }
        let changed = map.tick(w as usize, h as usize) || fresh || panel_dirty || lettered || self.map.uploads_pending() || self.map.bakes_pending() || self.debug.spin;
        let quiet = std::time::Duration::from_millis(250);
        if !changed && (self.settled || self.last_low.elapsed() < quiet) {
            self.device.poll(wgpu::PollType::Poll).ok();
            return !map.converged();
        }
        if self.frames == 0 {
            map.set_gpu_only();
        }
        let plan = map.plan(w as usize, h as usize);
        let t_plan = t0.elapsed().as_secs_f32() * 1000.0;
        self.map.sync(&self.device, &self.queue, map.pool_mut(), true);
        let t_sync = t0.elapsed().as_secs_f32() * 1000.0 - t_plan;
        self.plan_ms += t_plan;
        self.sync_ms += t_sync;
        // What kind of frame. Only the camera drops to 1×: a pan, zoom or rotate, or within a quarter second of one (Nick 2026-10-06). Everything else — cells streaming in, the light turning with the phone, a pin moving, the panel — is drawn at the still factor, so a loading or relit map stays sharp instead of flickering between the two. A quarter second after the last 1× frame, one more at 3× settles it.
        let c0 = map.cam;
        let cam_key = (c0.lat().to_bits(), c0.lon().to_bits(), c0.ppd.to_bits(), c0.bearing().to_bits());
        let moving = self.last_cam != Some(cam_key);
        self.last_cam = Some(cam_key);
        let now = std::time::Instant::now();
        if moving {
            self.last_move = now;
        }
        let in_motion = changed && (moving || self.last_move.elapsed() < quiet);
        let mode = RenderMode::Full;
        // The still frame's scale, at most what the GPU's largest texture holds for this screen: a scale it cannot hold fails to make its target, and the frame is lost.
        let fits = (self.device.limits().max_texture_dimension_2d / w.max(h).max(1)).max(1);
        let budget = ((STILL_PIXELS as f64 / (w as f64 * h as f64).max(1.0)).sqrt().floor() as u32).max(1);
        // The GPU's own measure: a frame's cost goes as its pixels, the scale squared; until one is known a still frame is drawn at 1×.
        // The map pass's own GPU time where the device stamps it (its scale the one it was drawn at, two frames back: near enough, the scale changes rarely); else the frame's time to done, which takes in the CPU between.
        if let Some(t) = self.map.take_pass_times() {
            for k in 0..3 {
                self.pass_ms[k] += t[k];
            }
            self.pass_n += 1;
            let s = self.map.scale.max(1) as f32;
            let one = t[0] / (s * s);
            self.cost_1x = Some(self.cost_1x.map_or(one, |c| c * 0.7 + one * 0.3));
            self.gpu_done.lock().unwrap().take();
        } else if let Some((s, ms)) = self.gpu_done.lock().unwrap().take() {
            let one = ms / (s * s) as f32;
            self.cost_1x = Some(self.cost_1x.map_or(one, |c| c * 0.7 + one * 0.3));
        }
        let affordable = self.cost_1x.map_or(1, |c| (1..=STILL_SCALE).rev().find(|&s| c * (s * s) as f32 <= STILL_BUDGET_MS).unwrap_or(1));
        self.map.scale = if in_motion { MOVING_SCALE } else { STILL_SCALE.min(fits).min(budget).min(affordable) };
        if in_motion {
            self.last_low = now;
        }
        self.settled = !in_motion;
        let c = map.cam;
        let mask = map.layers();
        let mask_bits = mahere_gpu::mask_bits(mask);
        // The overlay repaints when what it shows changes: the panel (open: its readouts follow the camera), the compass (bearing), the size. The pin is the shader's.
        // An unlocked measurement's profile follows the screen centre, so the camera is part of the stamp while one exists.
        let cam_part = if panel.is_open() || map.has_measure() { (c.lat().to_bits(), c.lon().to_bits(), c.ppd.to_bits()) } else { (0, 0, 0) };
        let measure_part = map.measure_view(w as usize, h as usize, 2).map(|m| (m.target_px.0.to_bits() as u64, m.target_px.1.to_bits() as u64, (m.distance_m.round() as u32)));
        if self.style_version != Some(map.style_version()) {
            self.map.update_style(&self.queue, map.luts());
            self.style_version = Some(map.style_version());
            self.overlay_stamp = None;
        }
        // The light version is in the stamp for the ball in the gear's place: it follows the sun, the camera and the phone's turning.
        let stamp: OverlayStamp = (cam_part.0, cam_part.1, cam_part.2, c.bearing().to_bits(), measure_part, w, h, panel.is_open(), mask_bits | (map.real_sun as u32) << 20 | (map.follow_heading as u32) << 21 | (map.real_light as u32) << 31 | if panel.is_open() && map.have_rotation { (map.true_heading().round() as u32) << 22 } else { 0 }, if map.real_sun || map.real_light { map.light_version() } else { 0 }, panel.ru().to_bits(), {
            // The areas as shown (a download's progress repaints the panel) and the zoom: either repaints at once.
            use std::hash::{Hash, Hasher};
            let mut hs = std::collections::hash_map::DefaultHasher::new();
            if panel.is_open() {
                areas.hash(&mut hs);
            }
            // The zoom shown under the compass, and whether it is past the data (it turns amber).
            map.view_depth().hash(&mut hs);
            map.zoom_depth().hash(&mut hs);
            map.past_data().hash(&mut hs);
            // The ring round the globe while its base set loads: each cell's arrival, and the zoom it circles.
            if let Some(done) = map.globe_loading() {
                ((done * 10.0).round() as u32, c.ppd.to_bits()).hash(&mut hs);
            }
            hs.finish()
        });
        // Labels: laid out again when the view rests; every frame each letter is a quad at its label's place on the map now.
        let t_labels = std::time::Instant::now();
        if mahere_panel::labels::Labeller::wanted(map) {
            self.labeller.ask(map, w as usize, h as usize, panel.font_px(), !in_motion);
            let a = self.labeller.atlas();
            self.map.set_atlas(&self.device, &self.queue, a.side as u32, &a.rg, a.version);
            let (glyphs, segments) = (self.labeller.glyphs(&map.cam, w as usize, h as usize), self.labeller.segments(&map.cam, w as usize, h as usize));
            self.map.set_labels(&self.device, &self.queue, &glyphs, &segments);
        } else {
            self.map.set_labels(&self.device, &self.queue, &[], &[]);
        }
        self.map.pin = map.gps_screen(w as usize, h as usize);
        self.map.measure = map.measure_view(w as usize, h as usize, 2).map(|m| (m.origin_px.0, m.origin_px.1, m.target_px.0, m.target_px.1));
        // A bearing or readout change repaints at most a few times a second (the orientation sensor would otherwise repaint the panel's text every frame); the panel opening, closing or a row flipping repaints at once.
        self.labels_ms += t_labels.elapsed().as_secs_f32() * 1000.0;
        let t_overlay = std::time::Instant::now();
        let structural = self.overlay_stamp.is_none_or(|s| (s.5, s.6, s.7, s.8, s.10, s.11) != (w, h, panel.is_open(), stamp.8, stamp.10, stamp.11));
        if self.overlay_stamp != Some(stamp) && (structural || self.overlay_at.elapsed().as_millis() >= 150) && !(self.debug.overlay_off && self.overlay_stamp.is_some()) {
            self.overlay_at = std::time::Instant::now();
            let mut heading = c.bearing().to_degrees().rem_euclid(360.0);
            if heading > 180.0 {
                heading -= 360.0;
            }
            let readouts = Readouts { lat: c.lat(), lon: c.lon(), gps_height: map.gps_sea_level(), elev: map.elevation_at(c.lat(), c.lon()), heading_deg: heading, m_per_px: 111_320.0 / c.ppd, frame_ms: map.last_frame_ms, resident: map.pool().map.len(), phone_heading: map.have_rotation.then(|| map.true_heading()), cache_used: cache.0, cache_max: cache.1, has_fix: map.gps.is_some() };
            let measure = map.measure_view(w as usize, h as usize, Panel::strip_samples(w as usize));
            let orb = |n: usize| map.orb(n);
            panel.paint(w as usize, h as usize, mask, ctl, &readouts, measure.as_ref(), &map.themes, areas, Some(&orb));
            let marks = map.overlay(w as usize, h as usize, false).to_vec();
            let rgba = panel.overlay_rgba(&marks, w as usize, h as usize);
            self.map.set_overlay_rgba(&self.device, &self.queue, w, h, &rgba);
            self.overlay_stamp = Some(stamp);
        }
        self.overlay_ms += t_overlay.elapsed().as_secs_f32() * 1000.0;
        self.work_ms += t0.elapsed().as_secs_f32() * 1000.0;
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.configured = (0, 0);
                return true;
            }
            Err(e) => {
                eprintln!("gpu: surface: {e:?}");
                return true;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        let t_encode = std::time::Instant::now();
        self.map.set_user(&self.device, &self.queue, &map.user, &map.preview, map.preview_version, w, h);
        self.map.render(&self.device, &self.queue, &mut enc, &view, &plan, map.luts(), w, h, mode);
        self.queue.submit([enc.finish()]);
        {
            // The frame's time to done on the GPU, reported when the queue is next polled.
            let (scale, at, done) = (self.map.scale, std::time::Instant::now(), self.gpu_done.clone());
            self.queue.on_submitted_work_done(move || {
                *done.lock().unwrap() = Some((scale, at.elapsed().as_secs_f32() * 1000.0));
            });
        }
        self.map.after_submit();
        frame.present();
        self.encode_ms += t_encode.elapsed().as_secs_f32() * 1000.0;
        // Reclaim what finished: staging buffers behind every upload and the arrays replaced by growth are only freed when the device is polled.
        self.device.poll(wgpu::PollType::Poll).ok();
        self.frames += 1;
        self.frame_ms += t0.elapsed().as_secs_f32() * 1000.0;
        if self.report.elapsed().as_secs() >= 10 {
            let n = self.frames.max(1) as f32;
            let pn = self.pass_n.max(1) as f32;
            eprintln!("gpu: a 1x frame costs {:.1} ms on the GPU, still frames at {}x; GPU per frame: map {:.1} ms, labels {:.1}, present {:.1} ({} timed); CPU per frame: labels {:.1} ms, overlay {:.1}, encode and present {:.1}", self.cost_1x.unwrap_or(f32::NAN), self.map.scale.max(1), self.pass_ms[0] / pn, self.pass_ms[1] / pn, self.pass_ms[2] / pn, self.pass_n, self.labels_ms / self.frames.max(1) as f32, self.overlay_ms / self.frames.max(1) as f32, self.encode_ms / self.frames.max(1) as f32);
            self.pass_ms = [0.0; 3];
            self.pass_n = 0;
            self.labels_ms = 0.0;
            self.overlay_ms = 0.0;
            self.encode_ms = 0.0;
            eprintln!("gpu: {} frames, per frame {:.2} ms plan + {:.2} ms sync + {:.2} ms overlay and encode + {:.2} ms waiting for the swapchain, {} resident, {} pending, {} uploads, layers {:?}, {} of {} blocks from composites, {} cells baked so far", self.frames, self.plan_ms / n, self.sync_ms / n, (self.work_ms - self.plan_ms - self.sync_ms) / n, (self.frame_ms - self.work_ms) / n, map.pool().map.len(), map.pending_cells(), self.map.uploads, self.map.layers(), self.map.fast_blocks.0, self.map.fast_blocks.1, self.map.bakes);
            self.frames = 0;
            self.frame_ms = 0.0;
            self.work_ms = 0.0;
            self.plan_ms = 0.0;
            self.sync_ms = 0.0;
            self.report = std::time::Instant::now();
        }
        // Lettering held back by cells still arriving needs another frame once its second is up.
        !map.converged() || self.labeller.pending(map, w as usize, h as usize) || panel.fonts_due() || self.map.uploads_pending() || self.map.bakes_pending()
    }
}

/// Tag the window's buffers BT.2020, gamma 2.2, full range, as fluor does for its CPU present: the map writes VSF RGB converted to BT.2020 primaries at gamma 2, and without the tag the compositor reads those bytes as sRGB and washes them out. Android names no gamma 2.0 transfer, so 2.2 is the nearest (the panel shows a touch darker than authored, the documented cost). The Vulkan swapchain sets its own sRGB dataspace when it is created, so this follows every configure. Resolved at run time: the symbol arrived in API 28 and the floor is 26, where the buffers simply stay sRGB.
fn tag_bt2020(window: &NativeWindow) {
    type SetDataSpace = unsafe extern "C" fn(*mut ndk_sys::ANativeWindow, i32) -> i32;
    const STANDARD_BT2020: i32 = 6 << 16;
    const TRANSFER_GAMMA2_2: i32 = 4 << 22;
    const RANGE_FULL: i32 = 1 << 27;
    unsafe {
        let lib = libc::dlopen(c"libandroid.so".as_ptr(), libc::RTLD_NOW);
        if lib.is_null() {
            return;
        }
        let f = libc::dlsym(lib, c"ANativeWindow_setBuffersDataSpace".as_ptr());
        if f.is_null() {
            eprintln!("gpu: no setBuffersDataSpace (API < 28), buffers stay sRGB");
            return;
        }
        let f: SetDataSpace = std::mem::transmute(f);
        let r = f(window.ptr().as_ptr(), STANDARD_BT2020 | TRANSFER_GAMMA2_2 | RANGE_FULL);
        eprintln!("gpu: buffers tagged BT.2020 gamma 2.2 ({r})");
    }
}
