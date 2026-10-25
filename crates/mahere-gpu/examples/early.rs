// early [lat lon ppd [out]]: a view drawn on the GPU while its cells stream from the bucket, nothing on the device, frames saved as they arrive and once settled: what a phone shows before its tiles are in (Nick 2026-10-25: "before all the tiles are loaded it looks super scattered").
use mahere_engine::residency::{DEFAULT_CELLS_URL, HttpStore, LayerStores};
use mahere_engine::{Camera, MapCore};
use mahere_gpu::GpuMap;
use mahere_tiles::publish::Layer;
use std::sync::Arc;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let num = |i: usize, d: f64| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (lat, lon, ppd) = (num(1, 47.6), num(2, -121.4), num(3, 3000.0));
    let out = a.get(4).cloned().unwrap_or_else(|| "/tmp/claude-1000/early".into());
    let (w, h) = (900usize, 900usize);
    let http = |l: Layer| -> Arc<dyn mahere_engine::residency::CellStore> { Arc::new(HttpStore::layer(DEFAULT_CELLS_URL, l)) };
    let layers = LayerStores { dem: http(Layer::Dem), img: http(Layer::Img), nir: http(Layer::Nir) };
    let mut map = MapCore::layered(http(Layer::Base), Some(layers), Camera::new(lat, lon, ppd, 0.0));
    map.set_gpu_only();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).expect("adapter");
    let mut limits = wgpu::Limits::default();
    limits.max_texture_array_layers = adapter.limits().max_texture_array_layers.min(2048);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() })).expect("device");
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut gpu = GpuMap::new(&device, &queue, format);
    gpu.scale = 1;
    let target = device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[] });
    let view = target.create_view(&Default::default());
    let bpr = ((w as u32 * 4) + 255) / 256 * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (bpr * h as u32) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let t = std::time::Instant::now();
    let mut next_shot = 0.3f32;
    let mut shots = 0;
    loop {
        map.tick(w, h);
        if let Some(r) = gpu.take_range() {
            map.set_drawn_range(r);
        }
        let plan = map.plan(w, h);
        gpu.sync(&device, &queue, map.pool_mut(), true);
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.render(&device, &queue, &mut enc, &view, &plan, map.luts(), w as u32, h as u32, mahere_gpu::RenderMode::Full);
        let done = map.converged() && !gpu.uploads_pending() && !gpu.bakes_pending() && t.elapsed().as_secs_f32() > 1.0;
        let shoot = t.elapsed().as_secs_f32() >= next_shot || done;
        if shoot {
            enc.copy_texture_to_buffer(wgpu::TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All }, wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h as u32) } }, wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 });
        }
        queue.submit([enc.finish()]);
        gpu.after_submit();
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        if shoot {
            let slice = buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
            {
                let data = slice.get_mapped_range();
                let name = if done { format!("{out}_settled.png") } else { format!("{out}_{shots:02}.png") };
                let f = std::fs::File::create(&name).unwrap();
                let mut e = png::Encoder::new(f, w as u32, h as u32);
                e.set_color(png::ColorType::Rgba);
                let mut wr = e.write_header().unwrap();
                let rows: Vec<u8> = (0..h).flat_map(|y| data[y * bpr as usize..y * bpr as usize + w * 4].to_vec()).collect();
                wr.write_image_data(&rows).unwrap();
                eprintln!("{:.1} s: {} resident, {} pending, {:?} fast blocks -> {name}", t.elapsed().as_secs_f32(), map.pool().map.len(), map.pending_cells(), gpu.fast_blocks);
            }
            buf.unmap();
            shots += 1;
            next_shot *= 1.6;
        }
        if done || t.elapsed().as_secs() > 90 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(30));
    }
}
