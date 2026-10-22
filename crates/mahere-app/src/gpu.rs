//! The map on the GPU for the desktop (Nick 2026-10-17: the desktop was slow and its contours sparkled, the CPU raster stepping along texel edges where the phone's shader interpolates): the phone's compositor ([`GpuMap`]) on a headless device, its frame read back into the buffer fluor already presents, so the chrome, the panel and fluor's own present are as they were.
//! Vulkan on Linux and Windows, Metal on the Mac; with no adapter (or MAHERE_CPU=1) the app keeps the CPU raster.

use mahere_engine::MapCore;
use mahere_gpu::{GpuMap, RenderMode};
use std::time::{Duration, Instant};

/// Supersampling while the camera moves and once it has rested (the phone's rule: only the camera drops to 1×).
const MOVING_SCALE: u32 = 1;
const STILL_SCALE: u32 = 3;
/// The camera has rested this long: the next frame is the still one.
const QUIET: Duration = Duration::from_millis(250);

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    map: GpuMap,
    /// The frame's target and the buffer it is read back through, for one size.
    target: Option<(wgpu::Texture, wgpu::TextureView, wgpu::Buffer, u32, u32)>,
    style_version: Option<u64>,
    last_cam: Option<(u64, u64, u64, u64)>,
    last_move: Instant,
    /// The last frame was the still one: nothing more to draw until something changes.
    settled: bool,
    /// The frame as fluor's buffer takes it, 0xRRGGBB.
    pub frame: Vec<u32>,
}

impl Gpu {
    pub fn new() -> Option<Gpu> {
        if std::env::var("MAHERE_CPU").as_deref() == Ok("1") {
            return None;
        }
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL, ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false }))
            .map_err(|e| eprintln!("gpu: no adapter, the CPU draws the map: {e:?}"))
            .ok()?;
        // Texture arrays hold one cell per layer: the adapter's real layer limit, not WebGPU's 256; and its own largest texture, for the still frame at three times a large window.
        let mut limits = wgpu::Limits::default();
        limits.max_texture_array_layers = adapter.limits().max_texture_array_layers.min(2048);
        limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d.max(limits.max_texture_dimension_2d);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("mahere"), required_limits: limits, ..Default::default() }))
            .map_err(|e| eprintln!("gpu: no device, the CPU draws the map: {e:?}"))
            .ok()?;
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| eprintln!("gpu: {e}")));
        let info = adapter.get_info();
        eprintln!("gpu: {} ({:?})", info.name, info.backend);
        let map = GpuMap::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm);
        Some(Gpu { device, queue, map, target: None, style_version: None, last_cam: None, last_move: Instant::now(), settled: false, frame: Vec::new() })
    }

    /// The still frame is still to come, or composites to bake: the host keeps ticking until they are done.
    pub fn wants_frame(&self) -> bool {
        !self.settled || self.map.bakes_pending()
    }

    fn ensure_target(&mut self, w: u32, h: u32) {
        if self.target.as_ref().is_some_and(|t| (t.3, t.4) == (w, h)) {
            return;
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("map frame"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bpr = (w * 4).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("map readback"), size: bpr as u64 * h as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        self.target = Some((texture, view, buffer, w, h));
    }

    /// One frame of the map into [`Self::frame`]: plan, mirror the pool, labels, draw, read back.
    pub fn render(&mut self, map: &mut MapCore, labeller: &mut mahere_panel::labels::Labeller, font_px: f32, w: usize, h: usize) {
        if w == 0 || h == 0 {
            return;
        }
        let t0 = Instant::now();
        let (wu, hu) = (w as u32, h as u32);
        self.ensure_target(wu, hu);
        if let Some(r) = self.map.take_range() {
            map.set_drawn_range(r);
        }
        let plan = map.plan(w, h);
        self.map.sync(&self.device, &self.queue, map.pool_mut(), true);
        let c = map.cam;
        let cam_key = (c.lat().to_bits(), c.lon().to_bits(), c.ppd.to_bits(), c.bearing().to_bits());
        let now = Instant::now();
        if self.last_cam != Some(cam_key) {
            self.last_cam = Some(cam_key);
            self.last_move = now;
        }
        let in_motion = self.last_move.elapsed() < QUIET;
        // The still frame's scale, at most what the GPU's largest texture holds for this window.
        let fits = (self.device.limits().max_texture_dimension_2d / wu.max(hu).max(1)).max(1);
        self.map.scale = if in_motion { MOVING_SCALE } else { STILL_SCALE.min(fits) };
        self.settled = !in_motion;
        if self.style_version != Some(map.style_version()) {
            self.map.update_style(&self.queue, map.luts());
            self.style_version = Some(map.style_version());
        }
        if mahere_panel::labels::Labeller::wanted(map) {
            labeller.ask(map, w, h, font_px, !in_motion);
            let a = labeller.atlas();
            self.map.set_atlas(&self.device, &self.queue, a.side as u32, &a.rg, a.version);
            let (glyphs, segments) = (labeller.glyphs(&map.cam, w, h), labeller.segments(&map.cam, w, h));
            self.map.set_labels(&self.device, &self.queue, &glyphs, &segments);
        } else {
            self.map.set_labels(&self.device, &self.queue, &[], &[]);
        }
        // The pin is the shader's; the compass and the other marks come as the overlay; the panel is laid over the read-back frame by the host.
        self.map.pin = map.gps_screen(w, h);
        self.map.measure = map.measure_view(w, h, 2).map(|m| (m.origin_px.0, m.origin_px.1, m.target_px.0, m.target_px.1));
        let marks = map.overlay(w, h, false).to_vec();
        self.map.set_overlay(&self.device, &self.queue, wu, hu, &marks);
        self.map.set_user(&self.device, &self.queue, &map.user, &map.preview, map.preview_version, wu, hu);
        let (texture, view, buffer, _, _) = self.target.as_ref().unwrap();
        let bpr = (wu * 4).div_ceil(256) * 256;
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.map.render(&self.device, &self.queue, &mut enc, view, &plan, map.luts(), wu, hu, RenderMode::Full);
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(hu) } },
            wgpu::Extent3d { width: wu, height: hu, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        self.map.after_submit();
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        {
            let data = slice.get_mapped_range();
            self.frame.resize(w * h, 0);
            for (y, out) in self.frame.chunks_exact_mut(w).enumerate() {
                let row = &data[y * bpr as usize..y * bpr as usize + w * 4];
                for (o, p) in out.iter_mut().zip(row.chunks_exact(4)) {
                    *o = (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32;
                }
            }
        }
        buffer.unmap();
        // The status bar's frame time: the whole of it, read back included.
        map.last_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }
}
