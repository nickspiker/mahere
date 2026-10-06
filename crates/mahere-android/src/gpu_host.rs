//! The GPU host on Android: a wgpu (Vulkan) surface over the Activity's ANativeWindow, driven from `nativeDraw` in place of fluor's CPU present. fluor still owns input; the engine still owns residency and the frame plan; this is the glue that gets a planned frame onto the screen.

use mahere_engine::MapCore;
use mahere_gpu::GpuMap;
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
    frames: u64,
    frame_ms: f32,
    report: std::time::Instant,
}

type OverlayStamp = (u64, u64, u64, u64, Option<(u64, u64, u32)>, u32, u32);

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
            frames: 0,
            frame_ms: 0.0,
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
    pub fn draw(&mut self, map: &mut MapCore, w: u32, h: u32) -> bool {
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
        map.tick(w as usize, h as usize);
        let plan = map.plan(w as usize, h as usize);
        self.map.sync(&self.device, &self.queue, map.pool());
        let c = map.cam;
        let stamp: OverlayStamp = (c.lat.to_bits(), c.lon.to_bits(), c.ppd.to_bits(), c.bearing.to_bits(), map.gps.map(|g| (g.lat.to_bits(), g.lon.to_bits(), g.accuracy_m.to_bits())), w, h);
        if self.overlay_stamp != Some(stamp) {
            let px = map.overlay(w as usize, h as usize).to_vec();
            self.map.set_overlay(&self.device, &self.queue, w, h, &px);
            self.overlay_stamp = Some(stamp);
        }
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
        self.map.render(&self.device, &self.queue, &mut enc, &view, &plan, map.luts(), w, h);
        self.queue.submit([enc.finish()]);
        frame.present();
        self.frames += 1;
        self.frame_ms += t0.elapsed().as_secs_f32() * 1000.0;
        if self.report.elapsed().as_secs() >= 10 {
            eprintln!("gpu: {} frames, {:.2} ms CPU side per frame, {} resident, {} uploads", self.frames, self.frame_ms / self.frames.max(1) as f32, map.pool().map.len(), self.map.uploads);
            self.frames = 0;
            self.frame_ms = 0.0;
            self.report = std::time::Instant::now();
        }
        !map.converged()
    }
}
