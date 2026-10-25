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

/// Readback buffers in the ring: the frame drawn copies into a free one and is shown when its copy lands, never waited on unless every one is still in flight.
const RING: usize = 3;

/// A readback buffer's state: free, copied into and mapping, mapped and ready.
const FREE: u8 = 0;
const MAPPING: u8 = 1;
const READY: u8 = 2;

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    map: GpuMap,
    /// The frame's target (BGRA, so a row read back is fluor's pixel order once its colour is turned to darkness) and its size.
    target: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    /// The readback ring: each buffer, its state, and the frame it holds.
    ring: Vec<(wgpu::Buffer, std::sync::Arc<std::sync::atomic::AtomicU8>, u64)>,
    drawn: u64,
    shown: u64,
    style_version: Option<u64>,
    last_cam: Option<(u64, u64, u64, u64)>,
    last_move: Instant,
    /// The last frame was the still one: nothing more to draw until something changes.
    settled: bool,
    /// The overlay as last uploaded: the engine's marks and the panel, so an unchanged one is not built and uploaded again.
    last_marks: Vec<u32>,
    last_panel: Vec<u32>,
    /// The frame in fluor's own pixels (alpha and darkness, `pack_argb`), the newest whose readback has landed.
    pub frame: Vec<u32>,
    /// Profiling: milliseconds of the last frame's plan and sync, overlay, drawing and submitting, taking a landed frame in, and the GPU's own passes.
    pub parts: [f32; 5],
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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("mahere"), required_limits: limits, required_features: mahere_gpu::wanted_features(&adapter), ..Default::default() }))
            .map_err(|e| eprintln!("gpu: no device, the CPU draws the map: {e:?}"))
            .ok()?;
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| eprintln!("gpu: {e}")));
        let info = adapter.get_info();
        eprintln!("gpu: {} ({:?})", info.name, info.backend);
        let map = GpuMap::new(&device, &queue, wgpu::TextureFormat::Bgra8Unorm);
        Some(Gpu { device, queue, map, target: None, ring: Vec::new(), drawn: 0, shown: 0, style_version: None, last_cam: None, last_move: Instant::now(), settled: false, last_marks: Vec::new(), last_panel: Vec::new(), frame: Vec::new(), parts: [0.0; 5] })
    }

    /// The still frame is still to come, or composites to bake: the host keeps ticking until they are done.
    pub fn wants_frame(&self) -> bool {
        !self.settled || self.map.bakes_pending()
    }

    fn ensure_target(&mut self, w: u32, h: u32) {
        if self.target.as_ref().is_some_and(|t| (t.2, t.3) == (w, h)) {
            return;
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("map frame"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bpr = (w * 4).div_ceil(256) * 256;
        // A new size: the ring made again (a frame of the old size in flight is dropped with it).
        self.ring = (0..RING)
            .map(|_| (self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("map readback"), size: bpr as u64 * h as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false }), std::sync::Arc::new(std::sync::atomic::AtomicU8::new(FREE)), 0))
            .collect();
        self.shown = self.drawn;
        self.target = Some((texture, view, w, h));
    }

    /// The newest frame whose readback has landed into [`Self::frame`], fluor's pixels: the BGRA row read as u32 is alpha and colour, and the colour turned to darkness is `pack_argb`. Older landed ones are freed unread.
    fn take_landed(&mut self, w: usize, h: usize) {
        use std::sync::atomic::Ordering;
        let Some(i) = (0..self.ring.len()).filter(|&i| self.ring[i].1.load(Ordering::Acquire) == READY).max_by_key(|&i| self.ring[i].2) else { return };
        let id = self.ring[i].2;
        if id > self.shown {
            let bpr = ((w as u32 * 4).div_ceil(256) * 256) as usize;
            let data = self.ring[i].0.slice(..).get_mapped_range();
            self.frame.resize(w * h, 0);
            for (y, out) in self.frame.chunks_exact_mut(w).enumerate() {
                let row: &[u32] = bytemuck::cast_slice(&data[y * bpr..y * bpr + w * 4]);
                for (o, &p) in out.iter_mut().zip(row) {
                    *o = p ^ 0x00FF_FFFF;
                }
            }
            drop(data);
            self.shown = id;
        }
        for (b, state, held) in self.ring.iter() {
            if state.load(Ordering::Acquire) == READY && *held <= self.shown {
                b.unmap();
                state.store(FREE, Ordering::Release);
            }
        }
    }

    /// One frame of the map: plan, mirror the pool, labels, the overlay (the engine's marks and `panel`, the panel's painted pixels) when it changed, draw, and copy into a free readback buffer; [`Self::frame`] then holds the newest frame whose copy has landed.
    pub fn render(&mut self, map: &mut MapCore, labeller: &mut mahere_panel::labels::Labeller, font_px: f32, panel: &mahere_panel::Panel, panel_px: &[u32], w: usize, h: usize) {
        if w == 0 || h == 0 {
            return;
        }
        use std::sync::atomic::Ordering;
        let t0 = Instant::now();
        let (wu, hu) = (w as u32, h as u32);
        self.ensure_target(wu, hu);
        if let Some(r) = self.map.take_range() {
            map.set_drawn_range(r);
        }
        let plan = map.plan(w, h);
        self.map.sync(&self.device, &self.queue, map.pool_mut(), true);
        let t_sync = Instant::now();
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
        // The pin is the shader's; the compass and the other marks, and the panel, are the overlay the present pass lays over the map, built and uploaded only when either changed.
        self.map.pin = map.gps_screen(w, h);
        self.map.measure = map.measure_view(w, h, 2).map(|m| (m.origin_px.0, m.origin_px.1, m.target_px.0, m.target_px.1));
        let t_ov = Instant::now();
        let marks = map.overlay(w, h, false);
        // Only the rows that changed, built and uploaded (the panel's readouts and the compass are a sliver of the screen).
        let n = w * h;
        let whole = self.last_marks.len() != n || self.last_panel.len() != n || marks.len() != n || panel_px.len() != n;
        let (mut y0, mut y1) = (h, 0);
        if whole {
            (y0, y1) = (0, h);
            self.last_marks = marks.to_vec();
            self.last_panel = if panel_px.len() == n { panel_px.to_vec() } else { vec![0; n] };
        } else {
            for y in 0..h {
                let r = y * w..(y + 1) * w;
                if marks[r.clone()] != self.last_marks[r.clone()] || panel_px[r.clone()] != self.last_panel[r.clone()] {
                    self.last_marks[r.clone()].copy_from_slice(&marks[r.clone()]);
                    self.last_panel[r.clone()].copy_from_slice(&panel_px[r]);
                    y0 = y0.min(y);
                    y1 = y + 1;
                }
            }
        }
        if y0 < y1 {
            let rgba = panel.overlay_rows(&self.last_marks, w, h, y0, y1);
            self.map.set_overlay_rows(&self.device, &self.queue, wu, hu, y0 as u32, &rgba);
        }
        let ov_ms = t_ov.elapsed().as_secs_f32() * 1000.0;
        self.map.set_user(&self.device, &self.queue, &map.user, &map.preview, map.preview_version, wu, hu);
        // A free readback buffer; when every one is still in flight, the oldest is waited for.
        self.device.poll(wgpu::PollType::Poll).ok();
        self.take_landed(w, h);
        let slot = match (0..self.ring.len()).find(|&i| self.ring[i].1.load(Ordering::Acquire) == FREE) {
            Some(i) => i,
            None => {
                self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
                self.take_landed(w, h);
                (0..self.ring.len()).find(|&i| self.ring[i].1.load(Ordering::Acquire) == FREE).unwrap_or(0)
            }
        };
        let (texture, view, _, _) = self.target.as_ref().unwrap();
        let bpr = (wu * 4).div_ceil(256) * 256;
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.map.render(&self.device, &self.queue, &mut enc, view, &plan, map.luts(), wu, hu, RenderMode::Full);
        let (buffer, state, held) = &mut self.ring[slot];
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(hu) } },
            wgpu::Extent3d { width: wu, height: hu, depth_or_array_layers: 1 },
        );
        let t_sub = Instant::now();
        self.queue.submit([enc.finish()]);
        self.map.after_submit();
        self.drawn += 1;
        *held = self.drawn;
        state.store(MAPPING, Ordering::Release);
        let st = state.clone();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| st.store(if r.is_ok() { READY } else { FREE }, Ordering::Release));
        // Moving, what has landed meanwhile, without waiting: the CPU goes on while this frame draws, shown a frame later. The still frame is waited for: drawn once at rest, it is the one that must show.
        self.device.poll(if self.settled { wgpu::PollType::wait_indefinitely() } else { wgpu::PollType::Poll }).ok();
        let t_take = Instant::now();
        self.take_landed(w, h);
        let gpu_ms = self.map.take_pass_times().map_or(0.0, |p| p[0] + p[1] + p[2]);
        self.parts = [(t_sync - t0).as_secs_f32() * 1000.0, ov_ms, (t_take - t_sub).as_secs_f32() * 1000.0, t_take.elapsed().as_secs_f32() * 1000.0, gpu_ms];
        // The status bar's frame time: the CPU's share of a frame, the GPU's running beside it.
        map.last_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }
}
