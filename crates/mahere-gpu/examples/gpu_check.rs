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
    // A published directory (base/, img/, nir/) is read by layer, as the bucket serves it; a baked one as it is.
    let root = std::path::PathBuf::from(&cells);
    let mut map = if root.join("base").is_dir() {
        use mahere_engine::residency::{LayerDir, LayerStores};
        use mahere_tiles::publish::Layer;
        let layers = LayerStores { dem: Arc::new(LayerDir(root.clone(), Layer::Dem)), img: Arc::new(LayerDir(root.clone(), Layer::Img)), nir: Arc::new(LayerDir(root.clone(), Layer::Nir)) };
        MapCore::layered(Arc::new(LayerDir(root, Layer::Base)), Some(layers), Camera::new(lat, lon, ppd, bearing))
    } else {
        MapCore::new(Arc::new(DirStore(root)), Camera::new(lat, lon, ppd, bearing))
    };
    // MAHERE_THEME=n picks a built-in theme (its layers first, MAHERE_LAYERS over them).
    if let Some(t) = std::env::var("MAHERE_THEME").ok().and_then(|s| s.parse::<usize>().ok()) {
        map.set_theme(t);
    }
    if let Ok(spec) = std::env::var("MAHERE_LAYERS") {
        let mut m = map.layers();
        for tok in spec.split(',') {
            match tok.trim() {
                "imagery" => m.imagery = true,
                "raw" => m.raw = true,
                "nocontours" => m.contours = false,
                "nohypso" => m.hypso = false,
                "nolines" => m.line = false,
                "noland" => m.land = false,
                "nowater" => m.water = false,
                "nodem" => m.dem = false,
                "slope" => m.slope = true,
                "infrared" => m.infrared = true,
                "debug" => m.debug = true,
                "geoid" => m.geoid = true,
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
    // MAHERE_NO_GPU=1 stops here, after the residency and the CPU reference: for finding where memory goes without touching the GPU.
    if std::env::var("MAHERE_NO_GPU").as_deref() == Ok("1") {
        let cpu0 = std::time::Instant::now();
        map.render(w, h);
        eprintln!("cpu render {:.1} ms, plan {} refs {} blocks", cpu0.elapsed().as_secs_f32() * 1e3, map.plan(w, h).refs.len(), map.plan(w, h).blocks.len());
        let cpu = map.canvas.clone();
        write_png(&format!("{out}_cpu.png"), w, h, |i| [(cpu[i] >> 16) as u8, (cpu[i] >> 8) as u8, cpu[i] as u8]);
        println!("wrote {out}_cpu.png");
        return;
    }

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
    let panel_mode = std::env::var("MAHERE_PANEL").unwrap_or_default();
    if panel_mode == "1" || panel_mode == "closed" || panel_mode == "themes" || panel_mode == "pen" || panel_mode == "areas" || panel_mode == "area" || panel_mode == "updates" {
        let mut panel = mahere_panel::Panel::new();
        // MAHERE_TOOLBOX=1 shows the toolbox's list open.
        if std::env::var_os("MAHERE_TOOLBOX").is_some() {
            panel.show_toolbox();
        }
        panel.set_open(panel_mode != "closed");
        if panel_mode == "themes" {
            panel.show_themes();
        }
        let mut controls = mahere_panel::Controls::default();
        controls.pen_rgba = [230, 40, 40, 160];
        if panel_mode == "pen" {
            panel.show_pen(controls.pen_rgba);
        }
        // MAHERE_PANEL=areas lists two made-up areas; =area opens the second, mid-download.
        let mut areas = mahere_panel::AreasView::default();
        if panel_mode == "areas" || panel_mode == "area" {
            areas.has_selection = true;
            areas.areas = vec![
                mahere_panel::AreaView { name: "spruce otter".into(), depth: 14, cells: 1234, est_bytes: 198_000_000, fetched: Some(176_000_000), progress: None },
                mahere_panel::AreaView { name: "granite heron".into(), depth: 12, cells: 310, est_bytes: 52_000_000, fetched: None, progress: Some((120, 310, 19_000_000)) },
            ];
            panel.show_themes();
            panel.show_area(1, 12);
            if panel_mode == "areas" {
                panel.show_themes();
                panel.show_area(1, 12);
                panel.pop_page();
            }
        }
        // MAHERE_PANEL=updates opens the Updates page with a made-up state: a release to get, dev current, a download part way.
        if panel_mode == "updates" {
            panel.show_updates();
            panel.set_updates(mahere_panel::UpdatesView { this: "0.178+457980b".into(), channels: [("Get release 0.176".into(), "e677c45 \u{b7} tap to install".into(), true), ("Dev 0.178".into(), "457980b \u{b7} this build".into(), false)], status: "Downloading release 0.176+e677c45".into(), progress: Some(0.42), auto: true });
        }
        let c = map.cam;
        let readouts = mahere_panel::Readouts { lat: c.lat(), lon: c.lon(), gps_height: map.gps_sea_level(), elev: map.elevation_at(c.lat(), c.lon()), heading_deg: c.bearing().to_degrees(), m_per_px: 111_320.0 / c.ppd, frame_ms: cpu_ms, resident: map.pool().map.len(), phone_heading: None, cache_used: 0, cache_max: 1 << 30, has_fix: map.gps.is_some() };
        // MAHERE_TAP=x,y taps the map there first: the measurement strip shows what is at the point, and the lines are printed.
        if let Ok(t) = std::env::var("MAHERE_TAP") {
            let v: Vec<f64> = t.split(',').map(|x| x.parse().unwrap()).collect();
            map.tap(v[0], v[1], w, h);
            if let Some(m) = map.measure_view(w, h, 2) {
                for l in &m.info {
                    println!("tap: {l}");
                }
            }
        }
        let measure = map.measure_view(w, h, mahere_panel::Panel::strip_samples(w));
        panel.paint(w, h, map.layers(), controls, &readouts, measure.as_ref(), &map.themes, &areas, None);
        // The areas view is compared on paint, so the page is painted once more with it known.
        panel.paint(w, h, map.layers(), controls, &readouts, measure.as_ref(), &map.themes, &areas, None);
        let rgba = panel.overlay_rgba(&overlay, w, h);
        gpu.set_overlay_rgba(&device, &queue, w as u32, h as u32, &rgba);
    } else {
        gpu.set_overlay(&device, &queue, w as u32, h as u32, &overlay);
    }
    // MAHERE_INK=1: a pen stroke across the centre, then the view zoomed in and out around it; the stroke must show at every zoom on both renderers.
    if matches!(std::env::var("MAHERE_INK").as_deref(), Ok("1") | Ok("2") | Ok("3")) {
        ink_check(&device, &queue, &mut gpu, &target, &view, &mut map, w, h);
        return;
    }
    // MAHERE_LABELS=1 letters the view (MAHERE_LABEL_FONT the base size, default 15 px): the GPU draws it through its labels texture, the CPU frame has it laid over below. MAHERE_LABEL_PAN=dx,dy then pans the map before the GPU frame, which must carry the lettering with it.
    let mut labeller = mahere_panel::labels::Labeller::new();
    let mut plan = plan;
    // MAHERE_ROUTE=car|bike|foot,x,y routes from the screen centre to that pixel and prints what the strip says; with MAHERE_LABELS the line is drawn.
    if let Ok(spec) = std::env::var("MAHERE_ROUTE") {
        let v: Vec<&str> = spec.split(',').collect();
        map.set_travel(Some(match v[0] {
            "bike" => mahere_engine::route::Travel::Bike,
            "foot" => mahere_engine::route::Travel::Foot,
            _ => mahere_engine::route::Travel::Car,
        }));
        map.tap(v[1].parse().unwrap(), v[2].parse().unwrap(), w, h);
        let t = std::time::Instant::now();
        while map.route().is_none() && t.elapsed().as_secs() < 60 {
            map.tick(w, h);
            if map.measure_view(w, h, 2).is_some_and(|m| m.info.iter().any(|l| l.starts_with("No route"))) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        if let Some(m) = map.measure_view(w, h, 2) {
            for l in &m.info {
                println!("strip: {l}");
            }
        }
    }
    if std::env::var_os("MAHERE_LABELS").is_some() {
        let font = std::env::var("MAHERE_LABEL_FONT").ok().and_then(|s| s.parse().ok()).unwrap_or(15.0);
        // MAHERE_LABEL_AMOUNT=density,repeat and MAHERE_LABEL_SIZE=times set the theme's label amount and size for the check.
        let t = map.theme;
        if let Some(v) = std::env::var("MAHERE_LABEL_AMOUNT").ok().map(|s| s.split(',').map(|x| x.parse::<f32>().unwrap()).collect::<Vec<f32>>()) {
            map.themes[t].label_amount = [v[0], v[1]];
        }
        if let Some(v) = std::env::var("MAHERE_LABEL_SIZE").ok().and_then(|s| s.parse::<f32>().ok()) {
            map.themes[t].label_look[0] = v;
        }
        labeller.update(&mut map, w, h, font);
        if let Ok(p) = std::env::var("MAHERE_LABEL_PAN") {
            let v: Vec<f64> = p.split(',').map(|x| x.parse().unwrap()).collect();
            map.pan(v[0], v[1], w, h);
            plan = map.plan(w, h);
            gpu.sync(&device, &queue, map.pool_mut(), false);
        }
        // MAHERE_LABEL_TURN=degrees turns the map after the layout: the line and area names must turn with it, the points stay level.
        if let Some(deg) = std::env::var("MAHERE_LABEL_TURN").ok().and_then(|s| s.parse::<f64>().ok()) {
            map.rotate_view(deg.to_radians());
            plan = map.plan(w, h);
            gpu.sync(&device, &queue, map.pool_mut(), false);
        }
        let a = labeller.atlas();
        gpu.set_atlas(&device, &queue, a.side as u32, &a.rg, a.version);
        gpu.set_labels(&device, &queue, &labeller.glyphs(&map.cam, w, h), &labeller.segments(&map.cam, w, h));
    }
    let luts = map.luts();
    // Time the second frame: the first pays for pipeline warm-up.
    let mut frame_ms = 0.0;
    gpu.set_user(&device, &queue, &map.user, &map.preview, map.preview_version, w as u32, h as u32);
    for i in 0..2 {
        let t = std::time::Instant::now();
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.render(&device, &queue, &mut enc, &view, &plan, luts, w as u32, h as u32, mahere_gpu::RenderMode::Full);
        queue.submit([enc.finish()]);
        gpu.after_submit();
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        if i == 1 {
            frame_ms = t.elapsed().as_secs_f32() * 1000.0;
        }
    }
    // The elevations the map pass drew, read back as the phone reads them, against the CPU frame's own range.
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let q = |e: u16| e as f32 * 0.5 - 11000.0;
    match gpu.take_range() {
        Some(Some((lo, hi))) => eprintln!("gpu drew elevations {}..{} m", q(lo), q(hi)),
        Some(None) => eprintln!("gpu drew no terrain"),
        None => eprintln!("gpu range not read back"),
    }
    if std::env::var("MAHERE_RELIGHT").as_deref() == Ok("1") {
        relight_check(&device, &queue, &mut gpu, &target, &view, &plan, luts, w, h);
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
    let mut cpu = cpu;
    if std::env::var_os("MAHERE_LABELS").is_some() && std::env::var_os("MAHERE_LABEL_PAN").is_none() && std::env::var_os("MAHERE_LABEL_TURN").is_none() {
        labeller.draw_cpu(&mut cpu, &map.cam, w, h);
    }
    write_png(&format!("{out}_cpu.png"), w, h, |i| [(cpu[i] >> 16) as u8, (cpu[i] >> 8) as u8, cpu[i] as u8]);
    write_png(&format!("{out}_gpu.png"), w, h, |i| [(gpu_px[i] >> 16) as u8, (gpu_px[i] >> 8) as u8, gpu_px[i] as u8]);
    println!("wrote {out}_cpu.png and {out}_gpu.png");
}

/// MAHERE_RELIGHT=1: draw the frame with the G-buffer at 1×, relight from it, and report how far the two images are apart — the receipt that the deferred path shades the same pixels.
#[allow(dead_code)]
fn relight_check(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &mut GpuMap, target: &wgpu::Texture, view: &wgpu::TextureView, plan: &mahere_engine::plan::FramePlan, luts: &mahere_engine::raster::FrameLuts, w: usize, h: usize) {
    let read = |device: &wgpu::Device, queue: &wgpu::Queue| -> Vec<u32> {
        let bpr = ((w as u32 * 4) + 255) / 256 * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: (bpr * h as u32) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h as u32) } },
            wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let data = slice.get_mapped_range();
        let mut px = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                let p = &data[y * bpr as usize + x * 4..][..4];
                px[y * w + x] = ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32;
            }
        }
        px
    };
    let mut frame = |mode: mahere_gpu::RenderMode| {
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.render(device, queue, &mut enc, view, plan, luts, w as u32, h as u32, mode);
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
    };
    frame(mahere_gpu::RenderMode::FullWithG);
    let full = read(device, queue);
    let t = std::time::Instant::now();
    frame(mahere_gpu::RenderMode::Relight);
    let relight_ms = t.elapsed().as_secs_f32() * 1000.0;
    let relit = read(device, queue);
    let (mut worst, mut n) = (0u32, 0usize);
    for i in 0..w * h {
        let (a, b) = (full[i], relit[i]);
        let d = [((a >> 16) & 255).abs_diff((b >> 16) & 255), ((a >> 8) & 255).abs_diff((b >> 8) & 255), (a & 255).abs_diff(b & 255)];
        let m = d[0].max(d[1]).max(d[2]);
        worst = worst.max(m);
        if m > 2 {
            n += 1;
        }
    }
    println!("relight: {:.2}% of pixels beyond 2 levels of the full frame, worst {worst}; relight frame {relight_ms:.2} ms submit to idle", n as f64 * 100.0 / (w * h) as f64);
}

/// The stroke at every zoom: red pixels counted on the CPU frame and the GPU frame after each zoom step.
#[allow(clippy::too_many_arguments)]
fn ink_check(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &mut GpuMap, target: &wgpu::Texture, view: &wgpu::TextureView, map: &mut MapCore, w: usize, h: usize) {
    let read = |device: &wgpu::Device, queue: &wgpu::Queue| -> Vec<u32> {
        let bpr = ((w as u32 * 4) + 255) / 256 * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: (bpr * h as u32) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h as u32) } },
            wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let data = slice.get_mapped_range();
        let mut px = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                let p = &data[y * bpr as usize + x * 4..][..4];
                px[y * w + x] = ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32;
            }
        }
        px
    };
    let reds = |px: &[u32]| px.iter().filter(|p| (**p >> 16) & 255 > 180 && (**p >> 8) & 255 < 40 && (**p & 255) < 110).count();
    let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
    map.set_tool(mahere_engine::ink::Tool::Pen);
    map.set_pen(0);
    map.begin_stroke(cx - 100.0, cy, 0.5, w, h);
    map.extend_stroke(cx + 100.0, cy, 0.5, w, h);
    map.end_stroke(w, h);
    // How long a long scribble takes to draw and to lift: the moves and the lift, in milliseconds.
    {
        let t = std::time::Instant::now();
        map.begin_stroke(20.0, 20.0, 0.5, w, h);
        let mut n = 0;
        for i in 1..400 {
            let x = 20.0 + (i as f64) * (w as f64 - 40.0) / 400.0;
            let y = cy + ((i as f64) * 0.3).sin() * (h as f64 * 0.4);
            map.extend_stroke(x, y, 0.5, w, h);
            n += 1;
        }
        let moves_ms = t.elapsed().as_secs_f32() * 1000.0;
        let t = std::time::Instant::now();
        map.end_stroke(w, h);
        eprintln!("scribble: {n} moves in {moves_ms:.1} ms ({:.2} ms each), the lift {:.1} ms, {} user cells", moves_ms / n as f32, t.elapsed().as_secs_f32() * 1000.0, map.user.cells.len());
    }
    // MAHERE_INK=3 (with the imagery layer on): a selection drawn over imagery, as an image from both renderers: the wash inside and the edge around it must read over the imagery.
    if std::env::var("MAHERE_INK").as_deref() == Ok("3") {
        map.clear_ink();
        map.set_tool(mahere_engine::ink::Tool::Select);
        map.begin_stroke(cx - 200.0, cy - 120.0, 0.5, w, h);
        for (x, y) in [(cx + 180.0, cy - 160.0), (cx + 220.0, cy + 40.0), (cx + 60.0, cy + 170.0), (cx - 180.0, cy + 120.0), (cx - 240.0, cy - 20.0)] {
            map.extend_stroke(x, y, 0.5, w, h);
        }
        map.end_stroke(w, h);
        map.render(w, h);
        let plan = map.plan(w, h);
        gpu.set_user(device, queue, &map.user, &map.preview, map.preview_version, w as u32, h as u32);
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.render(device, queue, &mut enc, view, &plan, map.luts(), w as u32, h as u32, mahere_gpu::RenderMode::Full);
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let px = read(device, queue);
        let out = std::env::var("MAHERE_INK_OUT").unwrap_or_else(|_| "/tmp/claude-1000/selection.png".into());
        write_png(&out, w, h, |i| [(px[i] >> 16) as u8, (px[i] >> 8) as u8, px[i] as u8]);
        let cpu_out = out.replace(".png", "_cpu.png");
        write_png(&cpu_out, w, h, |i| [(map.canvas[i] >> 16) as u8, (map.canvas[i] >> 8) as u8, map.canvas[i] as u8]);
        let diff = px.iter().zip(map.canvas.iter()).filter(|(a, b)| a != b).count();
        eprintln!("wrote {out} and {cpu_out}; {diff} of {} pixels differ between the renderers", w * h);
        return;
    }
    // MAHERE_INK=2: a second, finer stroke beside the first, then the view zoomed in on the first: a magnified mark must keep clean edges where the finer cells cover it.
    if std::env::var("MAHERE_INK").as_deref() == Ok("2") {
        map.zoom_about(8.0, cx, cy, w, h);
        map.set_pen(3);
        map.begin_stroke(cx + 60.0, cy - 150.0, 0.5, w, h);
        map.extend_stroke(cx + 60.0, cy + 150.0, 0.5, w, h);
        map.end_stroke(w, h);
        map.render(w, h);
        let plan = map.plan(w, h);
        gpu.set_user(device, queue, &map.user, &map.preview, map.preview_version, w as u32, h as u32);
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.render(device, queue, &mut enc, view, &plan, map.luts(), w as u32, h as u32, mahere_gpu::RenderMode::Full);
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let px = read(device, queue);
        let out = std::env::var("MAHERE_INK_OUT").unwrap_or_else(|_| "/tmp/claude-1000/ink_zoomed.png".into());
        write_png(&out, w, h, |i| [(px[i] >> 16) as u8, (px[i] >> 8) as u8, px[i] as u8]);
        let cpu_out = out.replace(".png", "_cpu.png");
        write_png(&cpu_out, w, h, |i| [(map.canvas[i] >> 16) as u8, (map.canvas[i] >> 8) as u8, map.canvas[i] as u8]);
        eprintln!("wrote {out} and {cpu_out}");
        return;
    }
    for factor in [1.0, 2.0, 0.5, 0.5, 0.5, 8.0] {
        map.zoom_about(factor, cx, cy, w, h);
        map.render(w, h);
        let cpu = reds(&map.canvas);
        let plan = map.plan(w, h);
        gpu.set_user(device, queue, &map.user, &map.preview, map.preview_version, w as u32, h as u32);
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.render(device, queue, &mut enc, view, &plan, map.luts(), w as u32, h as u32, mahere_gpu::RenderMode::Full);
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let g = reds(&read(device, queue));
        let user_refs = plan.refs.iter().filter(|r| r.flags & mahere_engine::plan::FLAG_INK != 0).count();
        eprintln!("ppd {:.0} dem depth {}: red pixels cpu {cpu} gpu {g}; user cells {} (refs {user_refs}) at depths {:?}", map.cam.ppd, plan.dem_depth, map.user.cells.len(), map.user.cells.keys().map(|k| k.depth).collect::<std::collections::BTreeSet<_>>());
    }
}
