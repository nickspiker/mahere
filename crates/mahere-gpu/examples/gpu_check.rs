// gpu_check [lat lon ppd [out_prefix]]: render one view both ways — the CPU raster and the GPU compositor — from the same resident cells, write both PNGs, and report how far apart they are. MAHERE_CELLS picks the cell directory (default data/cells); MAHERE_LAYERS the layer spec as in relight; MAHERE_BEARING a bearing in degrees. The receipt for the port: the two images must agree to the rounding of the lighting.
use mahere_engine::residency::DirStore;
use mahere_engine::{Camera, MapCore};
use mahere_gpu::GpuMap;
use std::sync::Arc;

fn write_png(path: &str, w: usize, h: usize, rgb: impl Fn(usize) -> [u8; 3]) {
    let f = std::fs::File::create(path).expect("png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().unwrap();
    let mut data = Vec::with_capacity(w * h * 3);
    for i in 0..w * h {
        data.extend_from_slice(&rgb(i));
    }
    wr.write_image_data(&data).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num = |i: usize, d: f64| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (lat, lon, ppd) = (num(1, 46.25), num(2, -122.14), num(3, 20000.0));
    let out = args.get(4).cloned().unwrap_or_else(|| "/tmp/claude-1000/gpu_check".into());
    let bearing = std::env::var("MAHERE_BEARING").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0).to_radians();
    let (w, h) = (1024usize, 768usize);

    let cells = std::env::var("MAHERE_CELLS").unwrap_or_else(|_| "data/cells".into());
    let store = Arc::new(DirStore(cells.into()));
    let mut map = MapCore::new(store, Camera { lat, lon, ppd, bearing });
    if let Ok(spec) = std::env::var("MAHERE_LAYERS") {
        let mut m = map.layers();
        for tok in spec.split(',') {
            match tok.trim() {
                "imagery" => m.imagery = true,
                "nocontours" => m.contours = false,
                "nohypso" => m.hypso = false,
                "nolines" => m.line = false,
                "noland" => m.land = false,
                "nowater" => m.water = false,
                "nodem" => m.dem = false,
                "slope" => m.slope = true,
                "canopy" => m.canopy = true,
                "debug" => m.debug = true,
                _ => {}
            }
        }
        map.set_layers(m);
    }
    // Settle residency: plan (which requests cells) until nothing is in flight, then twice more so the contour fit sees a full lattice.
    let t = std::time::Instant::now();
    let mut settled = 0;
    while settled < 3 {
        map.tick(w, h);
        let _ = map.plan(w, h);
        if map.converged() {
            settled += 1;
        } else {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        if t.elapsed().as_secs() > 120 {
            eprintln!("residency did not settle");
            break;
        }
    }
    eprintln!("resident: {} cells in {:.1}s", map.pool().map.len(), t.elapsed().as_secs_f32());

    // The plan first, then the CPU frame: both then fit the contour interval to the same range.
    let plan = map.plan(w, h);
    map.render(w, h);
    let cpu: Vec<u32> = map.canvas.clone();
    let cpu_ms = map.last_frame_ms;

    // The GPU frame, headless.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).expect("adapter");
    eprintln!("adapter: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);
    // Texture arrays hold one cell per layer: ask for the adapter.s real layer limit, not WebGPU.s 256.
    let mut limits = wgpu::Limits::default();
    limits.max_texture_array_layers = adapter.limits().max_texture_array_layers.min(2048);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("mahere"), required_limits: limits, ..Default::default() })).expect("device");
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut gpu = GpuMap::new(&device, &queue, format);
    // MAHERE_SS=1 samples exactly where the CPU does (top-left of each pixel, no supersampling): the exactness check. The default is the anti-aliased frame.
    if std::env::var("MAHERE_SS").as_deref() == Ok("1") {
        gpu.scale = 1;
        gpu.sample_offset = [-0.5, -0.5];
    }
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let t = std::time::Instant::now();
    gpu.sync(&device, &queue, map.pool_mut(), false);
    let sync_ms = t.elapsed().as_secs_f32() * 1000.0;
    let overlay = map.overlay(w, h, true).to_vec();
    // MAHERE_PANEL=1 lays the open control panel over the GPU frame (the CPU frame has none, so the diff below is then meaningless).
    if std::env::var("MAHERE_PANEL").as_deref() == Ok("1") {
        let mut panel = mahere_panel::Panel::new();
        panel.set_open(true);
        let c = map.cam;
        let readouts = mahere_panel::Readouts { lat: c.lat, lon: c.lon, elev: map.elevation_at(c.lat, c.lon), heading_deg: c.bearing.to_degrees(), m_per_px: 111_320.0 / c.ppd, frame_ms: cpu_ms, resident: map.pool().map.len(), phone_heading: None };
        panel.paint(w, h, map.layers(), mahere_panel::Controls::default(), &readouts);
        let rgba = panel.overlay_rgba(&overlay, w, h);
        gpu.set_overlay_rgba(&device, &queue, w as u32, h as u32, &rgba);
    } else {
        gpu.set_overlay(&device, &queue, w as u32, h as u32, &overlay);
    }
    let luts = map.luts();
    // Time the second frame: the first pays for pipeline warm-up.
    let mut frame_ms = 0.0;
    for i in 0..2 {
        let t = std::time::Instant::now();
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.render(&device, &queue, &mut enc, &view, &plan, luts, w as u32, h as u32);
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        if i == 1 {
            frame_ms = t.elapsed().as_secs_f32() * 1000.0;
        }
    }
    // Read back.
    let bpr = ((w as u32 * 4) + 255) / 256 * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: (bpr * h as u32) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h as u32) } },
        wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let data = slice.get_mapped_range();
    let mut gpu_px = vec![0u32; w * h];
    for y in 0..h {
        let row = &data[y * bpr as usize..];
        for x in 0..w {
            let p = &row[x * 4..x * 4 + 4];
            gpu_px[y * w + x] = ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32;
        }
    }
    drop(data);

    // Compare: the overlay is composited on both, so compare the whole image.
    let (mut diff_px, mut worst, mut sum) = (0usize, 0u32, 0u64);
    for i in 0..w * h {
        let (a, b) = (cpu[i], gpu_px[i]);
        let d = [((a >> 16) & 255).abs_diff((b >> 16) & 255), ((a >> 8) & 255).abs_diff((b >> 8) & 255), (a & 255).abs_diff(b & 255)];
        let m = d[0].max(d[1]).max(d[2]);
        if m > 8 {
            diff_px += 1;
        }
        worst = worst.max(m);
        sum += m as u64;
    }
    println!("resident {} cells, {} blocks ({} straddle), {} refs; uploads {}; sync {sync_ms:.1} ms; CPU frame {cpu_ms:.2} ms; GPU frame {frame_ms:.2} ms (submit to idle, {}x supersampled)", map.pool().map.len(), plan.blocks.len(), plan.straddle_blocks, plan.refs.len(), gpu.uploads, gpu.scale);
    println!("difference: {:.2}% of pixels beyond 8 levels, worst {worst}, mean {:.3}", diff_px as f64 * 100.0 / (w * h) as f64, sum as f64 / (w * h) as f64);
    write_png(&format!("{out}_cpu.png"), w, h, |i| [(cpu[i] >> 16) as u8, (cpu[i] >> 8) as u8, cpu[i] as u8]);
    write_png(&format!("{out}_gpu.png"), w, h, |i| [(gpu_px[i] >> 16) as u8, (gpu_px[i] >> 8) as u8, gpu_px[i] as u8]);
    println!("wrote {out}_cpu.png and {out}_gpu.png");
}
