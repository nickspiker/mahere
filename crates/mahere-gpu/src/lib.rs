//! The #pagetable compositor on the GPU. The engine still owns residency and the frame plan; this crate mirrors the resident cells into texture arrays (one slot per cell per plane), uploads the plan's blocks, references and page table, and draws every frame in two passes: the map at 2× into an offscreen target, then a 2×2 bin down onto the surface with the screen-space marks laid over. Any host with a wgpu device and a target view can drive it.

use bytemuck::{Pod, Zeroable};
use mahere_engine::plan::{FLAG_DEM, FLAG_IMG, FLAG_LAND, FLAG_LINE, FLAG_WATER, FramePlan, TABLE_N};
use mahere_engine::raster::{FrameLuts, LayerMask};
use mahere_engine::residency::{DEMQ_H, DEMQ_W, Pool};
use mahere_tiles::{CellKey, TEX, TRI};
use rustc_hash::FxHashMap;

/// Default supersampling factor: the map pass renders this many pixels per screen pixel on each axis, binned once.
pub const SCALE: u32 = 3;

/// The G-buffer: base + ground, normal (half-offset unorm) + water, post-light folds one and two. Four targets: five would pass the 32 bytes a sample may carry, so the colour comes from the relight pass instead.
const G_FORMATS: [wgpu::TextureFormat; 4] = [wgpu::TextureFormat::Rgba8Unorm; 4];

/// How a frame is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    /// The map pass at the current factor.
    Full,
    /// The map pass at 1×, also writing the G-buffer.
    FullWithG,
    /// Only the light changed: shade the last G-buffer, at 1×.
    Relight,
}

struct GBuffer {
    w: u32,
    h: u32,
    views: Vec<wgpu::TextureView>,
    bind: wgpu::BindGroup,
}

/// Layers a plane's texture array starts with; it doubles whenever the resident set outgrows it, up to the device's limit, so memory tracks what is on screen.
pub const INITIAL_LAYERS: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRef {
    a: [u32; 4],
    b: [u32; 4],
    c: [u32; 4],
    d: [f32; 4],
    e: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBlock {
    a: [u32; 4],
    b: [u32; 4],
    c: [f32; 4],
    d: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSlot {
    tag: u32,
    cu: u32,
    cv: u32,
    index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    size: [f32; 2],
    scale: f32,
    mask: u32,
    light: [[f32; 4]; 8],
    sun: [f32; 4],
    contour: [f32; 4],
    depths: [u32; 4],
    offset: [f32; 4],
    pin: [f32; 4],
    line_hi: [[f32; 4]; 8],
    measure: [f32; 4],
    style_water: [f32; 4],
    style_contour: [f32; 4],
    style_contour_index: [f32; 4],
    style_flat: [f32; 4],
    style_bg: [f32; 4],
    style_no_dem: [f32; 4],
    style_sea: [f32; 4],
    display: [[f32; 4]; 3],
    tone: [f32; 4],
    img_table: [[f32; 4]; 64],
    // The globe's disk in render-target pixels: centre x, y, radius squared, and 1 when the limb is on the screen.
    globe: [f32; 4],
}

/// The layer mask as the shader's bits.
pub fn mask_bits(m: LayerMask) -> u32 {
    m.bits()
}

/// One plane's texture array and its slots, keyed by cell. Grows by doubling, copying the old layers on the GPU.
struct Plane {
    label: &'static str,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    tex: wgpu::Texture,
    cap: u32,
    max: u32,
    map: FxHashMap<CellKey, u32>,
    free: Vec<u32>,
    /// Since when the array has been mostly empty, if it is.
    sparse_since: Option<std::time::Instant>,
}

impl Plane {
    fn new(device: &wgpu::Device, label: &'static str, w: u32, h: u32, format: wgpu::TextureFormat, max: u32) -> Plane {
        let cap = INITIAL_LAYERS.min(max);
        Plane { label, w, h, format, tex: plane_texture(device, label, w, h, cap, format), cap, max, map: FxHashMap::default(), free: (0..cap).rev().collect(), sparse_since: None }
    }

    /// The cell's slot, allocating one if it has none and growing the array when it is full: `Some((slot, fresh))`, or None at the device's limit. True when the array was reallocated (bind groups must be rebuilt).
    fn alloc(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: CellKey) -> (Option<(u32, bool)>, bool) {
        if let Some(&s) = self.map.get(&key) {
            return (Some((s, false)), false);
        }
        let mut grew = false;
        if self.free.is_empty() {
            if self.cap >= self.max {
                return (None, false);
            }
            let new_cap = (self.cap * 2).min(self.max);
            let new_tex = plane_texture(device, self.label, self.w, self.h, new_cap, self.format);
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("grow plane") });
            enc.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo { texture: &self.tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyTextureInfo { texture: &new_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: self.cap },
            );
            queue.submit([enc.finish()]);
            self.free.extend((self.cap..new_cap).rev());
            self.tex = new_tex;
            self.cap = new_cap;
            grew = true;
        }
        let s = self.free.pop().unwrap();
        self.map.insert(key, s);
        (Some((s, true)), grew)
    }

    fn evict_absent(&mut self, pool: &Pool) {
        let mut gone = Vec::new();
        for (k, s) in &self.map {
            if !pool.map.contains_key(k) {
                gone.push((*k, *s));
            }
        }
        for (k, s) in gone {
            self.map.remove(&k);
            self.free.push(s);
        }
    }

    /// When the slots in use have sat at a quarter of the array or less for two seconds, rebuild it smaller: the live layers are copied into the first slots on the GPU and the map rewritten. Memory follows the resident set down as well as up. True when the array changed.
    fn shrink_if_sparse(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        let used = self.map.len() as u32;
        if self.cap <= INITIAL_LAYERS || used * 4 > self.cap {
            self.sparse_since = None;
            return false;
        }
        match self.sparse_since {
            None => {
                self.sparse_since = Some(std::time::Instant::now());
                return false;
            }
            Some(t) if t.elapsed() < std::time::Duration::from_secs(2) => return false,
            _ => {}
        }
        let new_cap = (used * 2).max(INITIAL_LAYERS).next_power_of_two().min(self.cap / 2);
        let new_tex = plane_texture(device, self.label, self.w, self.h, new_cap, self.format);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shrink plane") });
        let mut new_map = FxHashMap::default();
        for (i, (k, old_slot)) in self.map.iter().enumerate() {
            enc.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo { texture: &self.tex, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: *old_slot }, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyTextureInfo { texture: &new_tex, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: i as u32 }, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: 1 },
            );
            new_map.insert(*k, i as u32);
        }
        queue.submit([enc.finish()]);
        self.tex = new_tex;
        self.cap = new_cap;
        self.map = new_map;
        self.free = (used..new_cap).rev().collect();
        self.sparse_since = None;
        true
    }

    fn view(&self) -> wgpu::TextureView {
        self.tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
    }
}

fn plane_texture(device: &wgpu::Device, label: &str, w: u32, h: u32, layers: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn write_layer(queue: &wgpu::Queue, tex: &wgpu::Texture, layer: u32, w: u32, h: u32, bytes_per_texel: u32, data: &[u8]) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: layer }, aspect: wgpu::TextureAspect::All },
        data,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * bytes_per_texel), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
}

fn storage_buffer(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
}

pub struct GpuMap {
    map_layout: wgpu::BindGroupLayout,
    present_layout: wgpu::BindGroupLayout,
    map_pipeline: wgpu::RenderPipeline,
    map_g_pipeline: wgpu::RenderPipeline,
    present_pipeline: wgpu::RenderPipeline,
    relight_pipeline: wgpu::RenderPipeline,
    relight_layout: wgpu::BindGroupLayout,
    empty_bind: wgpu::BindGroup,
    /// The G-buffer the last 1× map pass wrote, if any: what a sample was before the light, so a frame where only the light changed relights from it.
    gbuf: Option<GBuffer>,
    uniforms: wgpu::Buffer,
    blocks: wgpu::Buffer,
    blocks_cap: usize,
    refs: wgpu::Buffer,
    refs_cap: usize,
    table: wgpu::Buffer,
    lut: wgpu::Buffer,
    dem: Plane,
    line: Plane,
    lw: Plane,
    img: Plane,
    /// Each resident dem plane.s quantisation (base, step), set when it is uploaded.
    dem_scale: FxHashMap<CellKey, (f32, f32)>,
    map_bind: Option<wgpu::BindGroup>,
    /// One supersampled target per factor in use (a host draws at 2× while moving and 3× when still), each with its present bind group.
    offscreen: Vec<(u32, wgpu::Texture, wgpu::TextureView, u32, u32)>,
    present_binds: FxHashMap<u32, wgpu::BindGroup>,
    overlay: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    pub uploads: usize,
    /// Supersampling factor in use; a change takes effect at the next frame.
    pub scale: u32,
    /// Where inside a screen pixel the shader samples, in pixels; the CPU raster samples the top-left corner, which is (-0.5, -0.5) here, and the default centre is 0.
    pub sample_offset: [f32; 2],
    /// The GPS pin, drawn by the present pass: screen centre and accuracy radius in pixels.
    pub pin: Option<(f32, f32, f32)>,
    /// The measurement, drawn by the present pass: origin and target in screen pixels.
    pub measure: Option<(f32, f32, f32, f32)>,
    style_loaded: bool,
}

impl GpuMap {
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue, present_format: wgpu::TextureFormat) -> GpuMap {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mahere map"), source: wgpu::ShaderSource::Wgsl(include_str!("map.wgsl").into()) });
        let uint_array = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false };
        let storage = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None };
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..9u32)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: match i {
                    0 => wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    1..=4 => storage,
                    _ => uint_array,
                },
                count: None,
            })
            .collect();
        let map_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("map"), entries: &entries });
        let float_tex = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false };
        let present_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: float_tex, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: float_tex, count: None },
            ],
        });
        let relight_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("relight"),
            entries: &(0..4u32).map(|i| wgpu::BindGroupLayoutEntry { binding: i, visibility: wgpu::ShaderStages::FRAGMENT, ty: float_tex, count: None }).collect::<Vec<_>>(),
        });
        let pipeline = |label: &str, layouts: &[&wgpu::BindGroupLayout], vs: &str, fs: &str, formats: &[wgpu::TextureFormat]| -> wgpu::RenderPipeline {
            let targets: Vec<Option<wgpu::ColorTargetState>> = formats.iter().map(|&format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })).collect();
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: layouts, immediate_size: 0 });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(fs), compilation_options: Default::default(), targets: &targets }),
                multiview_mask: None,
                cache: None,
            })
        };
        let map_pipeline = pipeline("map", &[&map_layout], "vs_map", "fs_map", &[wgpu::TextureFormat::Rgba8Unorm]);
        let map_g_pipeline = pipeline("map with g-buffer", &[&map_layout], "vs_map", "fs_map_g", &G_FORMATS);
        let present_pipeline = pipeline("present", &[&map_layout, &present_layout], "vs_present", "fs_present", &[present_format]);
        // The relight pass reads the G-buffer at group 2 and must not bind the present group (it renders into that group's map texture), so group 1 is empty for it.
        let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("empty"), entries: &[] });
        let empty_bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("empty"), layout: &empty_layout, entries: &[] });
        let relight_pipeline = pipeline("relight", &[&map_layout, &empty_layout, &relight_layout], "vs_present", "fs_relight", &[wgpu::TextureFormat::Rgba8Unorm]);

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor { label: Some("uniforms"), size: std::mem::size_of::<Uniforms>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let blocks_cap = 4096;
        let refs_cap = 1024;
        let blocks = storage_buffer(device, "blocks", (blocks_cap * std::mem::size_of::<GpuBlock>()) as u64);
        let refs = storage_buffer(device, "refs", (refs_cap * std::mem::size_of::<GpuRef>()) as u64);
        let table = storage_buffer(device, "table", (TABLE_N * std::mem::size_of::<GpuSlot>()) as u64);
        // Style tables: 4096 hypsometric rows, then the line classes and the land classes, each 0xRRGGBB; filled by `update_style` from the theme.
        let lut = storage_buffer(device, "lut", ((4096 + 64) * 4) as u64);

        let max_layers = device.limits().max_texture_array_layers.max(1);
        let dem = Plane::new(device, "dem", DEMQ_W as u32, DEMQ_H as u32, wgpu::TextureFormat::R16Uint, max_layers);
        let line = Plane::new(device, "line", 2 * TEX as u32, TEX as u32, wgpu::TextureFormat::Rgba8Uint, max_layers);
        let lw = Plane::new(device, "land water", 2 * TEX as u32, TEX as u32, wgpu::TextureFormat::Rgba8Uint, max_layers);
        let img = Plane::new(device, "img", 2 * TEX as u32, TEX as u32, wgpu::TextureFormat::Rgba8Uint, max_layers);

        GpuMap {
            map_layout,
            present_layout,
            map_pipeline,
            map_g_pipeline,
            present_pipeline,
            relight_pipeline,
            relight_layout,
            empty_bind,
            gbuf: None,
            uniforms,
            blocks,
            blocks_cap,
            refs,
            refs_cap,
            table,
            lut,
            dem,
            line,
            lw,
            img,
            dem_scale: FxHashMap::default(),
            map_bind: None,
            offscreen: Vec::new(),
            present_binds: FxHashMap::default(),
            overlay: None,
            uploads: 0,
            scale: SCALE,
            sample_offset: [0.0, 0.0],
            pin: None,
            measure: None,
            style_loaded: false,
        }
    }

    fn make_map_bind(&mut self, device: &wgpu::Device) {
        let (dv, lv, wv, iv) = (self.dem.view(), self.line.view(), self.lw.view(), self.img.view());
        self.map_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("map"),
            layout: &self.map_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.blocks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.refs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.lut.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&dv) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&lv) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&wv) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&iv) },
            ],
        }));
    }

    /// Mirror the pool: a slot per resident plane, uploaded on first sight, freed when the cell leaves the pool.
    /// Mirror the pool: a slot per resident plane, uploaded on first sight, freed when the cell leaves the pool, the arrays shrunk when the resident set does. With `release`, a plane's CPU copy is dropped once it is on the GPU (a host that never draws on the CPU): the entry keeps its presence bits and its elevation plane for readouts.
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, pool: &mut Pool, release: bool) {
        let mut grew = false;
        for s in [&mut self.dem, &mut self.line, &mut self.lw, &mut self.img] {
            s.evict_absent(pool);
            if !grew {
                grew |= s.shrink_if_sparse(device, queue);
            }
        }
        self.dem_scale.retain(|k, _| pool.map.contains_key(k));
        let (w, h) = (2 * TEX as u32, TEX as u32);
        let mut scratch = vec![0u8; TRI * 4];
        for (key, e) in pool.map.iter_mut() {
            if let Some(d) = &e.dem_q {
                let (slot, g) = self.dem.alloc(device, queue, *key);
                grew |= g;
                if let Some((slot, true)) = slot {
                    write_layer(queue, &self.dem.tex, slot, DEMQ_W as u32, DEMQ_H as u32, 2, bytemuck::cast_slice(&d.tex));
                    self.dem_scale.insert(*key, (d.base, d.step));
                    self.uploads += 1;
                }
            }
            if let Some(l) = &e.line {
                let (slot, g) = self.line.alloc(device, queue, *key);
                grew |= g;
                if let Some((slot, true)) = slot {
                    for i in 0..TRI {
                        scratch[4 * i] = l.class[i];
                        scratch[4 * i + 1] = l.cov[i];
                        scratch[4 * i + 2] = l.mag_at(i);
                        scratch[4 * i + 3] = l.uses_at(i);
                    }
                    write_layer(queue, &self.line.tex, slot, w, h, 4, &scratch);
                    self.uploads += 1;
                    if release {
                        e.line = None;
                    }
                }
            }
            if e.land.is_some() || e.water.is_some() {
                let (slot, g) = self.lw.alloc(device, queue, *key);
                grew |= g;
                if let Some((slot, true)) = slot {
                    for i in 0..TRI {
                        let (lc, lv) = e.land.as_ref().map_or((0, 0), |l| (l.class[i], l.cov[i]));
                        scratch[4 * i] = lc;
                        scratch[4 * i + 1] = lv;
                        scratch[4 * i + 2] = e.water.as_ref().map_or(0, |wt| wt.cov[i]);
                        scratch[4 * i + 3] = 0;
                    }
                    write_layer(queue, &self.lw.tex, slot, w, h, 4, &scratch);
                    self.uploads += 1;
                    if release {
                        e.land = None;
                        e.water = None;
                    }
                }
            }
            if let Some(im) = &e.img {
                let (slot, g) = self.img.alloc(device, queue, *key);
                grew |= g;
                if let Some((slot, true)) = slot {
                    for i in 0..TRI {
                        scratch[4 * i] = im.red[i];
                        scratch[4 * i + 1] = im.green[i];
                        scratch[4 * i + 2] = im.blue[i];
                        scratch[4 * i + 3] = im.nir[i];
                    }
                    write_layer(queue, &self.img.tex, slot, w, h, 4, &scratch);
                    self.uploads += 1;
                    if release {
                        e.img = None;
                    }
                }
            }
        }
        if grew {
            self.map_bind = None;
        }
    }

    /// Layers allocated per plane: dem, line, land+water, imagery.
    pub fn layers(&self) -> [u32; 4] {
        [self.dem.cap, self.line.cap, self.lw.cap, self.img.cap]
    }

    /// The screen overlay as premultiplied RGBA bytes, `w × h`: marks, panel, whatever the host composes over the map.
    pub fn set_overlay_rgba(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, rgba: &[u8]) {
        self.ensure_overlay(device, w, h);
        let n = (w * h * 4) as usize;
        if rgba.len() >= n {
            write_layer(queue, &self.overlay.as_ref().unwrap().0, 0, w, h, 4, &rgba[..n]);
        }
    }

    /// The engine's marks alone: 0xRRGGBB over black, the ink's brightness its coverage, `w × h`.
    pub fn set_overlay(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, pixels: &[u32]) {
        let n = (w * h) as usize;
        let mut bytes = vec![0u8; n * 4];
        for (i, &p) in pixels.iter().take(n).enumerate() {
            let (r, g, b) = ((p >> 16) as u8, (p >> 8) as u8, p as u8);
            bytes[4 * i] = r;
            bytes[4 * i + 1] = g;
            bytes[4 * i + 2] = b;
            bytes[4 * i + 3] = r.max(g).max(b);
        }
        self.set_overlay_rgba(device, queue, w, h, &bytes);
    }

    fn ensure_overlay(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if self.overlay.as_ref().is_none_or(|(_, _, ow, oh)| (*ow, *oh) != (w, h)) {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("overlay"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            self.overlay = Some((t, v, w, h));
            self.present_binds.clear();
        }
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let scale = self.scale.max(1);
        if self.offscreen.iter().any(|(_, _, _, ow, oh)| (*ow, *oh) != (w, h)) {
            self.offscreen.clear();
            self.present_binds.clear();
        }
        if !self.offscreen.iter().any(|(s, ..)| *s == scale) {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("map supersampled"),
                size: wgpu::Extent3d { width: w * scale, height: h * scale, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            self.offscreen.push((scale, t, v, w, h));
        }
        if self.overlay.as_ref().is_none_or(|(_, _, ow, oh)| (*ow, *oh) != (w, h)) {
            // A blank (zero-initialised) overlay until the host paints one; set_overlay replaces it.
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("overlay"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            self.overlay = Some((t, v, w, h));
            self.present_binds.clear();
        }
        if !self.present_binds.contains_key(&scale) {
            let (_, _, mv, _, _) = self.offscreen.iter().find(|(s, ..)| *s == scale).unwrap();
            let (_, ov, _, _) = self.overlay.as_ref().unwrap();
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("present"),
                layout: &self.present_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(mv) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(ov) },
                ],
            });
            self.present_binds.insert(scale, bind);
        }
    }

    /// Draw a planned frame to `target` (`w × h`, the present format): upload the plan, the map pass at 2×, the present pass.
    /// Upload the theme's tables: the hypsometric ramp and the line and land colours. Once at start and whenever the theme changes.
    pub fn update_style(&mut self, queue: &wgpu::Queue, luts: &FrameLuts) {
        let mut lut_data = vec![0u32; 4096 + 64];
        let pack = |c: [u8; 3]| ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32;
        for (i, c) in luts.hypso.iter().enumerate() {
            lut_data[i] = pack(*c);
        }
        for (i, c) in luts.style.line.iter().enumerate() {
            lut_data[4096 + i] = pack(*c);
        }
        for (i, c) in luts.style.land.iter().enumerate() {
            lut_data[4128 + i] = pack(*c);
        }
        queue.write_buffer(&self.lut, 0, bytemuck::cast_slice(&lut_data));
        self.style_loaded = true;
    }

    /// Whether a G-buffer for a `w × h` screen exists to relight from.
    pub fn g_valid(&self, w: u32, h: u32) -> bool {
        self.gbuf.as_ref().is_some_and(|g| (g.w, g.h) == (w, h))
    }

    fn ensure_gbuffer(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if self.g_valid(w, h) {
            return;
        }
        let textures: Vec<wgpu::Texture> = G_FORMATS
            .iter()
            .map(|&format| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("g-buffer"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
            })
            .collect();
        let views: Vec<wgpu::TextureView> = textures.iter().map(|t| t.create_view(&Default::default())).collect();
        let entries: Vec<wgpu::BindGroupEntry> = views.iter().enumerate().map(|(i, v)| wgpu::BindGroupEntry { binding: i as u32, resource: wgpu::BindingResource::TextureView(v) }).collect();
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("relight"), layout: &self.relight_layout, entries: &entries });
        self.gbuf = Some(GBuffer { w, h, views, bind });
    }

    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, plan: &FramePlan, luts: &FrameLuts, w: u32, h: u32, mode: RenderMode) {
        if w == 0 || h == 0 {
            return;
        }
        if mode != RenderMode::Full {
            self.scale = 1;
        }
        if !self.style_loaded {
            self.update_style(queue, luts);
        }
        if mode == RenderMode::Relight && !self.g_valid(w, h) {
            return self.render(device, queue, encoder, target, plan, luts, w, h, RenderMode::FullWithG);
        }
        self.ensure_targets(device, w, h);
        if mode == RenderMode::FullWithG {
            self.ensure_gbuffer(device, w, h);
        }
        // References: the plan's cells with the slots they hold here; a plane without a slot (array full) is dropped from the flags so the shader falls through to a parent.
        let refs: Vec<GpuRef> = plan
            .refs
            .iter()
            .map(|r| {
                let mut flags = r.flags;
                let dem = self.dem.map.get(&r.key).copied();
                let line = self.line.map.get(&r.key).copied();
                let lw = self.lw.map.get(&r.key).copied();
                let img = self.img.map.get(&r.key).copied();
                if dem.is_none() {
                    flags &= !FLAG_DEM;
                }
                if line.is_none() {
                    flags &= !FLAG_LINE;
                }
                if lw.is_none() {
                    flags &= !(FLAG_LAND | FLAG_WATER);
                }
                if img.is_none() {
                    flags &= !FLAG_IMG;
                }
                let (base, step) = self.dem_scale.get(&r.key).copied().unwrap_or((0.0, 1.0));
                GpuRef {
                    a: [r.diamond as u32, r.key.depth as u32, r.cu, r.cv],
                    b: [dem.unwrap_or(0), line.unwrap_or(0), lw.unwrap_or(0), img.unwrap_or(0)],
                    c: [flags, 0, 0, 0],
                    d: [base, step, r.jac[0], r.jac[1]],
                    e: [r.jac[2], r.jac[3], r.jac[4], 0.0],
                }
            })
            .collect();
        let blocks: Vec<GpuBlock> = plan
            .blocks
            .iter()
            .map(|b| GpuBlock { a: [b.x, b.y, b.size, b.diamond], b: [b.u0, b.v0, 0, 0], c: [b.du_dx, b.dv_dx, b.du_dy, b.dv_dy], d: [b.twist_u, b.twist_v, 0.0, 0.0] })
            .collect();
        let table: Vec<GpuSlot> = plan.table.iter().map(|s| GpuSlot { tag: s.tag, cu: s.cu, cv: s.cv, index: s.index }).collect();
        let mut rebind = self.map_bind.is_none();
        if blocks.len() > self.blocks_cap {
            self.blocks_cap = blocks.len().next_power_of_two();
            self.blocks = storage_buffer(device, "blocks", (self.blocks_cap * std::mem::size_of::<GpuBlock>()) as u64);
            rebind = true;
        }
        if refs.len() > self.refs_cap {
            self.refs_cap = refs.len().next_power_of_two();
            self.refs = storage_buffer(device, "refs", (self.refs_cap * std::mem::size_of::<GpuRef>()) as u64);
            rebind = true;
        }
        if rebind {
            self.make_map_bind(device);
        }
        if !blocks.is_empty() {
            queue.write_buffer(&self.blocks, 0, bytemuck::cast_slice(&blocks));
        }
        if !refs.is_empty() {
            queue.write_buffer(&self.refs, 0, bytemuck::cast_slice(&refs));
        }
        queue.write_buffer(&self.table, 0, bytemuck::cast_slice(&table));
        let mut light = [[0f32; 4]; 8];
        for c in 0..3 {
            for i in 0..10 {
                let k = c * 10 + i;
                light[k / 4][k % 4] = luts.light.k[c][i];
            }
        }
        let sc = self.scale.max(1) as f32;
        let u = Uniforms {
            globe: [plan.globe[0] * sc, plan.globe[1] * sc, plan.globe[2] * plan.globe[2] * sc * sc, plan.globe[3]],
            size: [(w * self.scale.max(1)) as f32, (h * self.scale.max(1)) as f32],
            scale: self.scale.max(1) as f32,
            mask: mask_bits(luts.mask),
            light,
            sun: [luts.sun[0], luts.sun[1], luts.sun[2], 0.0],
            contour: [luts.contours.interval, luts.contours.index_every as f32, luts.contours.m_per_px, 0.0],
            depths: [plan.dem_depth as u32, plan.vec_depth as u32, plan.magnified as u32, 0],
            offset: [self.sample_offset[0], self.sample_offset[1], 0.0, 0.0],
            pin: match self.pin {
                Some((x, y, r)) => [x, y, r, 1.0],
                None => [0.0, 0.0, 0.0, 0.0],
            },
            line_hi: std::array::from_fn(|i| std::array::from_fn(|j| plan.line_mag_hi[4 * i + j].max(1) as f32)),
            measure: self.measure.map_or([0.0; 4], |(ox, oy, tx, ty)| [ox, oy, tx, ty]),
            style_water: rgb4(luts.style.water),
            style_contour: with_alpha(luts.style.contour, luts.style.contour_alpha[0]),
            style_contour_index: with_alpha(luts.style.contour_index, luts.style.contour_alpha[1]),
            style_flat: rgb4(luts.style.flat),
            style_bg: rgb4(luts.style.bg),
            style_no_dem: rgb4(luts.style.no_dem),
            style_sea: rgb4(luts.style.sea),
            display: luts.display.rows(),
            tone: [luts.compressed as u32 as f32, 0.0, 0.0, 0.0],
            img_table: {
                let t = mahere_engine::colour::img_table();
                std::array::from_fn(|i| std::array::from_fn(|j| t[4 * i + j]))
            },
        };
        let measure_on = self.measure.is_some();
        let u = Uniforms { depths: [plan.dem_depth as u32, plan.vec_depth as u32, plan.magnified as u32, measure_on as u32], ..u };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        let scale = self.scale.max(1);
        let (_, _, map_view, _, _) = self.offscreen.iter().find(|(s, ..)| *s == scale).unwrap();
        let clear = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 18.0 / 255.0, g: 20.0 / 255.0, b: 26.0 / 255.0, a: 1.0 }), store: wgpu::StoreOp::Store };
        if mode == RenderMode::FullWithG {
            let g = self.gbuf.as_ref().unwrap();
            let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = g.views.iter().map(|v| Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } })).collect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("map g-buffer"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.map_g_pipeline);
            pass.set_bind_group(0, self.map_bind.as_ref().unwrap(), &[]);
            pass.draw(0..6, 0..blocks.len() as u32);
        }
        if mode != RenderMode::Full {
            let g = self.gbuf.as_ref().unwrap();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("relight"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: map_view, depth_slice: None, resolve_target: None, ops: clear })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.relight_pipeline);
            pass.set_bind_group(0, self.map_bind.as_ref().unwrap(), &[]);
            pass.set_bind_group(1, &self.empty_bind, &[]);
            pass.set_bind_group(2, &g.bind, &[]);
            pass.draw(0..3, 0..1);
        } else {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("map"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: map_view, depth_slice: None, resolve_target: None, ops: clear })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.map_pipeline);
            pass.set_bind_group(0, self.map_bind.as_ref().unwrap(), &[]);
            pass.draw(0..6, 0..blocks.len() as u32);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.present_pipeline);
            pass.set_bind_group(0, self.map_bind.as_ref().unwrap(), &[]);
            pass.set_bind_group(1, &self.present_binds[&scale], &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// A style colour as the shader takes it: linear light.
fn rgb4(c: [u8; 3]) -> [f32; 4] {
    with_alpha(c, 0.0)
}

fn with_alpha(c: [u8; 3], a: f32) -> [f32; 4] {
    let l = mahere_engine::colour::lin(c);
    [l[0], l[1], l[2], a]
}
