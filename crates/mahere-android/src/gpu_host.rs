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
    configured: (u32, u32),
    overlay_stamp: Option<OverlayStamp>,
    overlay_at: std::time::Instant,
    last_cam: Option<(u64, u64, u64, u64)>,
    /// The theme tables uploaded so far.
    style_version: Option<u64>,
    /// The view and cells the G-buffer was written for.
    g_key: Option<((u64, u64, u64, u64), u64, u32, u32)>,
    last_change: std::time::Instant,
    relights: u64,
    /// The last frame was drawn still, at the still factor: nothing more to draw until something changes.
    settled: bool,
    frames: u64,
    frame_ms: f32,
    work_ms: f32,
    plan_ms: f32,
    sync_ms: f32,
    report: std::time::Instant,
}

/// Supersampling while the camera moves and once it has stopped.
const MOVING_SCALE: u32 = 1;
const STILL_SCALE: u32 = 3;

type OverlayStamp = (u64, u64, u64, u64, Option<(u64, u64, u32)>, u32, u32, bool, u32);

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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("mahere"), required_limits: limits, ..Default::default() }))
            .map_err(|e| eprintln!("gpu: no device: {e:?}"))
            .ok()?;
        // A validation error logs instead of killing the app: the map goes wrong on screen, the phone stays usable, logcat says why.
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| eprintln!("gpu: {e}")));
        let caps = surface.get_capabilities(&adapter);
        // A linear 8-bit format: the shader writes display values, not scene-linear ones.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let alpha = caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto);
        let map = GpuMap::new(&device, &queue, format);
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
            configured: (0, 0),
            overlay_stamp: None,
            overlay_at: std::time::Instant::now(),
            last_cam: None,
            style_version: None,
            g_key: None,
            last_change: std::time::Instant::now(),
            relights: 0,
            settled: false,
            frames: 0,
            frame_ms: 0.0,
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
    pub fn draw(&mut self, map: &mut MapCore, panel: &mut Panel, ctl: Controls, cache: (u64, u64), w: u32, h: u32) -> bool {
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
        }
        // Nothing changed and nothing arrived: the surface keeps its last image, and the CPU keeps its budget. The poll still runs so finished uploads release their staging memory. Once a quarter second has passed since the last change, one more frame is drawn at the still factor and stays on screen.
        let fresh = self.configured != (w, h) || self.frames == 0;
        let panel_dirty = panel.take_dirty();
        if panel_dirty {
            self.overlay_stamp = None;
        }
        let changed = map.tick(w as usize, h as usize) || fresh || panel_dirty;
        let mut settle_frame = false;
        if !changed {
            if self.settled || self.last_change.elapsed() < std::time::Duration::from_millis(250) {
                self.device.poll(wgpu::PollType::Poll).ok();
                return !map.converged();
            }
            settle_frame = true;
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
        // What kind of frame: anything moving or arriving draws the map at 1× and writes the G-buffer; a change that touched only the light relights that G-buffer at 1×; the settle frame draws the map once at the still factor.
        let c0 = map.cam;
        let cam_key = (c0.lat.to_bits(), c0.lon.to_bits(), c0.ppd.to_bits(), c0.bearing.to_bits());
        let moving = self.last_cam != Some(cam_key);
        self.last_cam = Some(cam_key);
        let g_key = (cam_key, map.pool_version(), w, h);
        let light_only = !moving && !settle_frame && map.plan_cached() && !panel_dirty && self.g_key == Some(g_key) && self.map.g_valid(w, h);
        let mode = if settle_frame {
            RenderMode::Full
        } else if light_only {
            RenderMode::Relight
        } else {
            RenderMode::FullWithG
        };
        self.map.scale = if settle_frame { STILL_SCALE } else { MOVING_SCALE };
        if mode == RenderMode::FullWithG {
            self.g_key = Some(g_key);
        }
        if !settle_frame {
            self.last_change = std::time::Instant::now();
        }
        self.settled = settle_frame;
        self.relights += (mode == RenderMode::Relight) as u64;
        let c = map.cam;
        let mask = map.layers();
        let mask_bits = mahere_gpu::mask_bits(mask);
        // The overlay repaints when what it shows changes: the panel (open: its readouts follow the camera), the compass (bearing), the size. The pin is the shader's.
        // An unlocked measurement's profile follows the screen centre, so the camera is part of the stamp while one exists.
        let cam_part = if panel.is_open() || map.has_measure() { (c.lat.to_bits(), c.lon.to_bits(), c.ppd.to_bits()) } else { (0, 0, 0) };
        let measure_part = map.measure_view(w as usize, h as usize, 2).map(|m| (m.target_px.0.to_bits() as u64, m.target_px.1.to_bits() as u64, (m.distance_m.round() as u32)));
        if self.style_version != Some(map.style_version()) {
            self.map.update_style(&self.queue, map.luts());
            self.style_version = Some(map.style_version());
            self.overlay_stamp = None;
        }
        let stamp: OverlayStamp = (cam_part.0, cam_part.1, cam_part.2, c.bearing.to_bits(), measure_part, w, h, panel.is_open(), mask_bits | (map.real_sun as u32) << 20 | (map.follow_heading as u32) << 21 | if panel.is_open() && map.have_rotation { (map.true_heading().round() as u32) << 22 } else { 0 });
        self.map.pin = map.gps_screen(w as usize, h as usize);
        self.map.measure = map.measure_view(w as usize, h as usize, 2).map(|m| (m.origin_px.0, m.origin_px.1, m.target_px.0, m.target_px.1));
        // A bearing or readout change repaints at most a few times a second (the orientation sensor would otherwise repaint the panel's text every frame); the panel opening, closing or a row flipping repaints at once.
        let structural = self.overlay_stamp.is_none_or(|s| (s.5, s.6, s.7, s.8) != (w, h, panel.is_open(), stamp.8));
        if self.overlay_stamp != Some(stamp) && (structural || self.overlay_at.elapsed().as_millis() >= 150) {
            self.overlay_at = std::time::Instant::now();
            let mut heading = c.bearing.to_degrees().rem_euclid(360.0);
            if heading > 180.0 {
                heading -= 360.0;
            }
            let readouts = Readouts { lat: c.lat, lon: c.lon, elev: map.elevation_at(c.lat, c.lon), heading_deg: heading, m_per_px: 111_320.0 / c.ppd, frame_ms: map.last_frame_ms, resident: map.pool().map.len(), phone_heading: map.have_rotation.then(|| map.true_heading()), cache_used: cache.0, cache_max: cache.1 };
            let measure = map.measure_view(w as usize, h as usize, Panel::strip_samples(w as usize));
            panel.paint(w as usize, h as usize, mask, ctl, &readouts, measure.as_ref(), &map.themes);
            let marks = map.overlay(w as usize, h as usize, false).to_vec();
            let rgba = panel.overlay_rgba(&marks, w as usize, h as usize);
            self.map.set_overlay_rgba(&self.device, &self.queue, w, h, &rgba);
            self.overlay_stamp = Some(stamp);
        }
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
        self.map.render(&self.device, &self.queue, &mut enc, &view, &plan, map.luts(), w, h, mode);
        self.queue.submit([enc.finish()]);
        frame.present();
        // Reclaim what finished: staging buffers behind every upload and the arrays replaced by growth are only freed when the device is polled.
        self.device.poll(wgpu::PollType::Poll).ok();
        self.frames += 1;
        self.frame_ms += t0.elapsed().as_secs_f32() * 1000.0;
        if self.report.elapsed().as_secs() >= 10 {
            let n = self.frames.max(1) as f32;
            eprintln!("gpu: {} frames ({} relit), per frame {:.2} ms plan + {:.2} ms sync + {:.2} ms overlay and encode + {:.2} ms waiting for the swapchain, {} resident, {} pending, {} uploads, layers {:?}", self.frames, self.relights, self.plan_ms / n, self.sync_ms / n, (self.work_ms - self.plan_ms - self.sync_ms) / n, (self.frame_ms - self.work_ms) / n, map.pool().map.len(), map.pending_cells(), self.map.uploads, self.map.layers());
            self.frames = 0;
            self.relights = 0;
            self.frame_ms = 0.0;
            self.work_ms = 0.0;
            self.plan_ms = 0.0;
            self.sync_ms = 0.0;
            self.report = std::time::Instant::now();
        }
        !map.converged()
    }
}
