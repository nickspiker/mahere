//! The #pagetable compositor on the GPU. The engine still owns residency and the frame plan; this crate mirrors the resident cells into texture arrays (one slot per cell per plane), uploads the plan's blocks, references and page table, and draws every frame in two passes: the map at 2× into an offscreen target, then a 2×2 bin down onto the surface with the screen-space marks laid over. Any host with a wgpu device and a target view can drive it.

use bytemuck::{Pod, Zeroable};
use mahere_engine::plan::{FLAG_DEM, FLAG_IMG, FLAG_INK, FLAG_LAND, FLAG_LINE, FLAG_MVUM, FLAG_OWN, FLAG_VEC, FLAG_WATER, FramePlan, NONE, PlanBlock, TABLE_N};
use std::hash::{Hash, Hasher};
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
    /// The block's candidate references: terrain, vector planes, imagery.
    e: [u32; 4],
    f: [u32; 4],
    g: [u32; 4],
    /// The block's user cells (ink, selection), finest first.
    h: [u32; 4],
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
    // The world's pole axis in view coordinates.
    pole: [f32; 4],
    // The raw elevation view's range: lowest and highest metres in view.
    raw: [f32; 4],
}

/// The layer mask as the shader's bits.
pub fn mask_bits(m: LayerMask) -> u32 {
    m.bits()
}

/// The GPU's own clock around the frame's passes (2026-10-21: a Pixel 3a ran at a frame a second, and a submit-to-done time could not tell the GPU's share from the CPU's): timestamps at the start and end of the map, labels and present passes, resolved and read back two frames behind, never waited on.
struct PassTimer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: [wgpu::Buffer; 2],
    state: [std::sync::Arc<std::sync::atomic::AtomicU8>; 2],
    copied: Option<usize>,
    /// Nanoseconds a tick.
    period: f32,
}

/// The timestamps a frame writes: map start and end, labels, present.
const PASS_STAMPS: u32 = 6;

/// The map pass's pipelines: the full composite, the same writing the G-buffer, the baked cells' fast pass, and the bake.
const KIND_FULL: u8 = 0;
const KIND_G: u8 = 1;
const KIND_FAST: u8 = 2;
const KIND_BAKE: u8 = 3;

/// The shader's layer bits the composite leaves to the frame (contours, slope bands: drawn live from the terrain) and those no composite is made under (raw elevation, the geoid, detail tinting).
const LIVE_BITS: u32 = 64 | 128;
const NO_COMPOSITE_BITS: u32 = 16 | 8192 | 32768;

/// The reference's composite is baked and current (the shader's FLAG_COMP).
const FLAG_COMP: u32 = 256;

/// Most composites held, a megabyte each: the least recently drawn goes when a new cell needs the room.
const COMP_LAYERS: u32 = 128;

fn build_map_pipeline(device: &wgpu::Device, shader: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout, layers: u32, kind: u8) -> wgpu::RenderPipeline {
    let constants = [("LAYERS", layers as f64)];
    let options = wgpu::PipelineCompilationOptions { constants: &constants, ..Default::default() };
    let (label, vs, fs, formats): (&str, &str, &str, &[wgpu::TextureFormat]) = match kind {
        KIND_G => ("map with g-buffer", "vs_map", "fs_map_g", &G_FORMATS),
        KIND_FAST => ("map from composites", "vs_map", "fs_fast", &[wgpu::TextureFormat::Rgba8Unorm]),
        KIND_BAKE => ("composite bake", "vs_present", "fs_bake", &[wgpu::TextureFormat::Rgba8Unorm; 2]),
        _ => ("map", "vs_map", "fs_map", &[wgpu::TextureFormat::Rgba8Unorm]),
    };
    let targets: Vec<Option<wgpu::ColorTargetState>> = formats.iter().map(|&format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })).collect();
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState { module: shader, entry_point: Some(vs), compilation_options: options.clone(), buffers: &[] },
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState { module: shader, entry_point: Some(fs), compilation_options: options, targets: &targets }),
        multiview_mask: None,
        cache: None,
    })
}

/// A composite as baked: what it was made from, and the frame that last drew it.
struct Baked {
    sig: u64,
    used: u64,
}

/// The dymaxion's diamonds: a layer of the geoid texture each.
const GEOID_DIAMONDS: u32 = 10;

/// One plane's texture array and its slots, keyed by cell. Grows by doubling, copying the old layers on the GPU.
struct Plane {
    label: &'static str,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    /// Usage beyond sampling and copying (a composite plane is rendered into).
    extra: wgpu::TextureUsages,
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
        Plane::with_usage(device, label, w, h, format, max, wgpu::TextureUsages::empty())
    }

    fn with_usage(device: &wgpu::Device, label: &'static str, w: u32, h: u32, format: wgpu::TextureFormat, max: u32, extra: wgpu::TextureUsages) -> Plane {
        let cap = INITIAL_LAYERS.min(max);
        Plane { label, w, h, format, extra, tex: plane_texture(device, label, w, h, cap, format, extra), cap, max, map: FxHashMap::default(), free: (0..cap).rev().collect(), sparse_since: None }
    }

    /// A view of one layer, to render into.
    fn layer_view(&self, slot: u32) -> wgpu::TextureView {
        self.tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_array_layer: slot, array_layer_count: Some(1), ..Default::default() })
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
            let new_tex = plane_texture(device, self.label, self.w, self.h, new_cap, self.format, self.extra);
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
        let new_tex = plane_texture(device, self.label, self.w, self.h, new_cap, self.format, self.extra);
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

fn plane_texture(device: &wgpu::Device, label: &str, w: u32, h: u32, layers: u32, format: wgpu::TextureFormat, extra: wgpu::TextureUsages) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC | extra,
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
    /// The map pass's pipelines, one per set of layers in use and whether it writes the G-buffer, each built the first time its layers are drawn: what the frame does not show is compiled out ([`GpuMap::map_variant`]). The module and layout they are made from.
    map_variants: FxHashMap<(u32, u8), wgpu::RenderPipeline>,
    /// Pipelines being built on a thread of their own (the composite's: until one is in, the frame draws as before), and where they arrive.
    building: rustc_hash::FxHashSet<(u32, u8)>,
    built_tx: std::sync::mpsc::Sender<((u32, u8), wgpu::RenderPipeline)>,
    built_rx: std::sync::mpsc::Receiver<((u32, u8), wgpu::RenderPipeline)>,
    /// The vector cells' composites (two planes in step) and what each was baked from; a stand-in pair of layers bound while baking, which may not read the planes it writes.
    comp0: Plane,
    comp1: Plane,
    baked: FxHashMap<CellKey, Baked>,
    /// Each resident cell's largest line magnitude per class, kept from the pool for the bake's line weights.
    line_max: FxHashMap<CellKey, [u8; 32]>,
    bake_dummy: wgpu::Texture,
    bake_uniforms: Vec<wgpu::Buffer>,
    frame_no: u64,
    /// Bumped by every theme upload, so composites made under the last theme are made again.
    style_gen: u64,
    /// Most composites baked a frame (each a full composite of a cell's 131 072 texels), and whether the frame draws from them at all.
    pub bake_budget: usize,
    pub composite: bool,
    /// Cells the last frame left to bake: the host keeps frames coming until they are done.
    bakes_left: bool,
    /// Blocks the last frame drew from composites, and all of them; cells it baked.
    pub fast_blocks: (usize, usize),
    pub baked_last: usize,
    /// Cells baked since the map was made.
    pub bakes: usize,
    shader: wgpu::ShaderModule,
    map_pl: wgpu::PipelineLayout,
    present_pipeline: wgpu::RenderPipeline,
    /// The top layer (panel, pin, measurement, selection) blended over the map and its labels.
    top_pipeline: wgpu::RenderPipeline,
    /// Labels: a quad per letter from the glyph atlas, a quad per stretch of route (Nick 2026-10-13: no rendering to a bitmap).
    glyph_pipeline: wgpu::RenderPipeline,
    segment_pipeline: wgpu::RenderPipeline,
    glyph_layout: wgpu::BindGroupLayout,
    atlas_sampler: wgpu::Sampler,
    atlas: Option<(wgpu::Texture, u64, wgpu::BindGroup)>,
    glyphs: Option<(wgpu::Buffer, u64)>,
    glyph_count: u32,
    segments: Option<(wgpu::Buffer, u64)>,
    segment_count: u32,
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
    ov: Plane,
    /// The elevations the last map pass drew (lowest, highest), written by the shader; the readback buffers and their states; which one this frame copied into.
    view_range: wgpu::Buffer,
    range_read: [wgpu::Buffer; 2],
    range_state: [std::sync::Arc<std::sync::atomic::AtomicU8>; 2],
    range_copied: Option<usize>,
    img: Plane,
    /// Each resident dem plane.s quantisation (base, step), set when it is uploaded.
    dem_scale: FxHashMap<CellKey, (f32, f32)>,
    map_bind: Option<wgpu::BindGroup>,
    /// One supersampled target per factor in use (a host draws at 1× while moving and 3× when still), each with its present bind group.
    offscreen: Vec<(u32, wgpu::Texture, wgpu::TextureView, u32, u32)>,
    present_binds: FxHashMap<u32, wgpu::BindGroup>,
    overlay: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    /// The user's cells (ink and selection) as a plane of their own, and the selection being drawn as a screen mask.
    ink: Plane,
    sel: Plane,
    user_version: u64,
    /// Each uploaded user cell's version, so a change re-uploads only the cells it touched.
    ink_uploaded: FxHashMap<CellKey, u64>,
    preview: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    preview_version: u64,
    pub uploads: usize,
    /// The most plane uploads one sync makes (unlimited by default): a host drawing on its UI thread spreads a burst of arrivals over frames, so no one frame stalls input (2026-10-21, the Pixel 3a's "not responding"); and whether planes were left for the next.
    pub upload_budget: usize,
    uploads_left: bool,
    /// The passes' GPU time, where the device can stamp it.
    timer: Option<PassTimer>,
    /// Layer bits the shader is told are off whatever the view shows: a profiling switch (a host's debug property), 0 in use.
    pub mask_off: u32,
    /// The geoid of the ten depth-0 cells, a layer per diamond (centimetres above the ellipsoid offset by 32768, 0 where none), and which diamonds are in.
    geoid_tex: wgpu::Texture,
    geoid_in: u16,
    /// When a full plane was last reported.
    full_warned: std::time::Instant,
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

/// Style entries after the 4096 hypsometric rows: line classes at 0, land at 32, ownership at 64 (the wilderness at 80), motor vehicle use at 96.
const LUT_STYLE: usize = 128;

/// A readback buffer's state: free, copied into this frame, being mapped, mapped and ready to read.
const RANGE_IDLE: u8 = 0;
const RANGE_COPIED: u8 = 1;
const RANGE_MAPPING: u8 = 2;
const RANGE_READY: u8 = 3;

impl GpuMap {
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue, present_format: wgpu::TextureFormat) -> GpuMap {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mahere map"), source: wgpu::ShaderSource::Wgsl(include_str!("map.wgsl").into()) });
        let uint_array = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false };
        let storage = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None };
        let float_array = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false };
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..16u32)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                // The elevation range the map pass writes is fragment-only: a vertex stage may not write storage.
                visibility: if i == 12 { wgpu::ShaderStages::FRAGMENT } else { wgpu::ShaderStages::VERTEX_FRAGMENT },
                ty: match i {
                    0 => wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    1..=4 => storage,
                    12 => wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    14 | 15 => float_array,
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
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: float_tex, count: None },
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
        let map_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("map"), bind_group_layouts: &[&map_layout], immediate_size: 0 });
        let present_pipeline = pipeline("present", &[&map_layout, &present_layout], "vs_present", "fs_present", &[present_format]);
        // Premultiplied over: what the labels and the top layer cover, they cover by their own opacity.
        let over = wgpu::BlendState { color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add }, alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add } };
        let blended = |label: &str, layouts: &[&wgpu::BindGroupLayout], vs: &str, fs: &str, buffers: &[wgpu::VertexBufferLayout], format: wgpu::TextureFormat| -> wgpu::RenderPipeline {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: layouts, immediate_size: 0 });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some(vs), compilation_options: Default::default(), buffers },
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(fs), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(over), write_mask: wgpu::ColorWrites::ALL })] }),
                multiview_mask: None,
                cache: None,
            })
        };
        let top_pipeline = blended("top", &[&map_layout, &present_layout], "vs_present", "fs_top", &[], present_format);
        let glyph_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glyph atlas"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("glyph atlas"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        // A letter: centre and half extents, angle, atlas rectangle, letter colour, halo colour (20 floats); a stretch of route: its ends, half width, colour (12).
        let glyph_attrs = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4];
        let glyph_pipeline = blended("glyphs", &[&map_layout, &glyph_layout], "vs_glyph", "fs_glyph", &[wgpu::VertexBufferLayout { array_stride: 80, step_mode: wgpu::VertexStepMode::Instance, attributes: &glyph_attrs }], wgpu::TextureFormat::Rgba8Unorm);
        let segment_attrs = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4];
        let segment_pipeline = blended("route", &[&map_layout], "vs_segment", "fs_segment", &[wgpu::VertexBufferLayout { array_stride: 48, step_mode: wgpu::VertexStepMode::Instance, attributes: &segment_attrs }], wgpu::TextureFormat::Rgba8Unorm);
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
        // Style tables: 4096 hypsometric rows, then the line classes, the land classes, the ownership classes with the wilderness, and the motor vehicle use classes, each 0xAARRGGBB (the ramp's alpha byte unused); filled by `update_style` from the theme.
        let lut = storage_buffer(device, "lut", ((4096 + LUT_STYLE) * 4) as u64);

        // The elevations the map pass drew, lowest and highest as quantised elevations, and two buffers to read them back through without waiting on the GPU.
        let view_range = device.create_buffer(&wgpu::BufferDescriptor { label: Some("view range"), size: 8, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let range_read: [wgpu::Buffer; 2] = std::array::from_fn(|_| device.create_buffer(&wgpu::BufferDescriptor { label: Some("view range readback"), size: 8, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
        let max_layers = device.limits().max_texture_array_layers.max(1);
        let dem = Plane::new(device, "dem", DEMQ_W as u32, DEMQ_H as u32, wgpu::TextureFormat::R16Uint, max_layers);
        let line = Plane::new(device, "line", 2 * TEX as u32, TEX as u32, wgpu::TextureFormat::Rgba8Uint, max_layers);
        let lw = Plane::new(device, "land water", 2 * TEX as u32, TEX as u32, wgpu::TextureFormat::Rgba8Uint, max_layers);
        let img = Plane::new(device, "img", 2 * TEX as u32, TEX as u32, wgpu::TextureFormat::Rgba8Uint, max_layers);
        let ink = Plane::new(device, "ink", 2 * TEX as u32, TEX as u32, wgpu::TextureFormat::Rgba8Uint, max_layers);
        let sel = Plane::new(device, "sel", 2 * TEX as u32, TEX as u32, wgpu::TextureFormat::Rgba8Uint, max_layers);
        // The overlays: ownership class and coverage, motor vehicle use class and coverage.
        let ov = Plane::new(device, "overlays", 2 * TEX as u32, TEX as u32, wgpu::TextureFormat::Rgba8Uint, max_layers);
        let comp_max = COMP_LAYERS.min(max_layers);
        let comp0 = Plane::with_usage(device, "composite base", 2 * TEX as u32, TEX as u32, wgpu::TextureFormat::Rgba8Unorm, comp_max, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let comp1 = Plane::with_usage(device, "composite fold", 2 * TEX as u32, TEX as u32, wgpu::TextureFormat::Rgba8Unorm, comp_max, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let (built_tx, built_rx) = std::sync::mpsc::channel();

        GpuMap {
            map_layout,
            present_layout,
            map_variants: FxHashMap::default(),
            building: Default::default(),
            built_tx,
            built_rx,
            comp0,
            comp1,
            baked: FxHashMap::default(),
            line_max: FxHashMap::default(),
            bake_dummy: plane_texture(device, "composite stand-in", 1, 1, 1, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::empty()),
            bake_uniforms: Vec::new(),
            frame_no: 0,
            style_gen: 0,
            bake_budget: 8,
            composite: true,
            bakes_left: false,
            fast_blocks: (0, 0),
            baked_last: 0,
            bakes: 0,
            shader,
            map_pl,
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
            ink,
            sel,
            ov,
            view_range,
            range_read,
            range_state: std::array::from_fn(|_| std::sync::Arc::new(std::sync::atomic::AtomicU8::new(RANGE_IDLE))),
            range_copied: None,
            user_version: u64::MAX,
            ink_uploaded: FxHashMap::default(),
            preview: None,
            top_pipeline,
            glyph_pipeline,
            segment_pipeline,
            glyph_layout,
            atlas_sampler,
            atlas: None,
            glyphs: None,
            glyph_count: 0,
            segments: None,
            segment_count: 0,
            preview_version: u64::MAX,
            dem_scale: FxHashMap::default(),
            map_bind: None,
            offscreen: Vec::new(),
            present_binds: FxHashMap::default(),
            overlay: None,
            uploads: 0,
            upload_budget: usize::MAX,
            uploads_left: false,
            mask_off: 0,
            timer: device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| PassTimer {
                set: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("pass times"), ty: wgpu::QueryType::Timestamp, count: PASS_STAMPS }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor { label: Some("pass times"), size: PASS_STAMPS as u64 * 8, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
                read: std::array::from_fn(|_| device.create_buffer(&wgpu::BufferDescriptor { label: Some("pass times readback"), size: PASS_STAMPS as u64 * 8, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })),
                state: std::array::from_fn(|_| std::sync::Arc::new(std::sync::atomic::AtomicU8::new(RANGE_IDLE))),
                copied: None,
                period: _queue.get_timestamp_period(),
            }),
            geoid_tex: plane_texture(device, "geoid", 2 * TEX as u32, TEX as u32, GEOID_DIAMONDS, wgpu::TextureFormat::R16Uint, wgpu::TextureUsages::empty()),
            geoid_in: 0,
            full_warned: std::time::Instant::now() - std::time::Duration::from_secs(60),
            scale: SCALE,
            sample_offset: [0.0, 0.0],
            pin: None,
            measure: None,
            style_loaded: false,
        }
    }

    fn make_map_bind(&mut self, device: &wgpu::Device) {
        self.map_bind = Some(self.bind_with(device, &self.uniforms, false));
    }

    /// The map's bind group over a uniform buffer: the frame's, reading the composites; or a bake's, the composites stood in for, since a bake writes them.
    fn bind_with(&self, device: &wgpu::Device, uniforms: &wgpu::Buffer, baking: bool) -> wgpu::BindGroup {
        let (dv, lv, wv, iv, kv, sv, ovv) = (self.dem.view(), self.line.view(), self.lw.view(), self.img.view(), self.ink.view(), self.sel.view(), self.ov.view());
        let array = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let (c0, c1) = if baking { (array(&self.bake_dummy), array(&self.bake_dummy)) } else { (self.comp0.view(), self.comp1.view()) };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(if baking { "bake" } else { "map" }),
            layout: &self.map_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.blocks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.refs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.lut.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&dv) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&lv) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&wv) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&iv) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&kv) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&sv) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&ovv) },
                wgpu::BindGroupEntry { binding: 12, resource: self.view_range.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&array(&self.geoid_tex)) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&c0) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&c1) },
            ],
        })
    }

    /// The user's layer onto the GPU: every user cell's ink and selection into the ink plane (they are few; all of them again whenever the layer changes), and the selection being drawn as a screen mask the present pass washes over the frame.
    pub fn set_user(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, user: &mahere_engine::ink::UserLayer, preview: &[u8], preview_version: u64, w: u32, h: u32) {
        if self.user_version != user.version {
            self.user_version = user.version;
            let gone: Vec<CellKey> = self.ink.map.keys().filter(|k| !user.cells.contains_key(k)).copied().collect();
            for k in gone {
                if let Some(slot) = self.ink.map.remove(&k) {
                    self.ink.free.push(slot);
                }
                if let Some(slot) = self.sel.map.remove(&k) {
                    self.sel.free.push(slot);
                }
                self.ink_uploaded.remove(&k);
            }
            let mut grew = false;
            let mut scratch = vec![0u8; TRI * 4];
            for (key, c) in user.cells.iter() {
                if self.ink_uploaded.get(key) == Some(&c.version) && self.ink.map.contains_key(key) {
                    continue;
                }
                let (slot, g) = self.ink.alloc(device, queue, *key);
                grew |= g;
                if let Some((slot, _)) = slot {
                    self.ink_uploaded.insert(*key, c.version);
                    // The ink plane as stored: premultiplied gamma-2 RGB and alpha; the selection's coverage rides in its own layer slot's plane beside it, so two planes per cell.
                    for i in 0..TRI {
                        let p = c.ink.as_ref().map_or([0; 4], |k| k.px(i));
                        scratch[4 * i] = p[0];
                        scratch[4 * i + 1] = p[1];
                        scratch[4 * i + 2] = p[2];
                        scratch[4 * i + 3] = p[3];
                    }
                    write_layer(queue, &self.ink.tex, slot, 2 * TEX as u32, TEX as u32, 4, &scratch);
                    for i in 0..TRI {
                        scratch[4 * i] = c.sel.as_ref().map_or(0, |k| k.cov[i]);
                        scratch[4 * i + 1] = 0;
                        scratch[4 * i + 2] = 0;
                        scratch[4 * i + 3] = 0;
                    }
                    let (sslot, g2) = self.sel.alloc(device, queue, *key);
                    grew |= g2;
                    if let Some((sslot, _)) = sslot {
                        write_layer(queue, &self.sel.tex, sslot, 2 * TEX as u32, TEX as u32, 4, &scratch);
                    }
                }
            }
            if grew {
                self.map_bind = None;
            }
        }
        let fresh = match &self.preview {
            Some((_, _, pw, ph)) => *pw != w || *ph != h,
            None => true,
        };
        if fresh {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("preview"),
                size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            self.preview = Some((t, v, w, h));
            self.preview_binds_dirty();
            self.preview_version = u64::MAX;
        }
        if self.preview_version != preview_version {
            self.preview_version = preview_version;
            let (t, _, pw, ph) = self.preview.as_ref().unwrap();
            let n = (*pw as usize) * (*ph as usize);
            let zeros;
            let data: &[u8] = if preview.len() == n {
                preview
            } else {
                zeros = vec![0u8; n];
                &zeros
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(*pw), rows_per_image: Some(*ph) },
                wgpu::Extent3d { width: *pw, height: *ph, depth_or_array_layers: 1 },
            );
        }
    }

    fn preview_binds_dirty(&mut self) {
        self.present_binds.clear();
    }

    /// Mirror the pool: a slot per resident plane, uploaded on first sight, freed when the cell leaves the pool.
    /// Mirror the pool: a slot per resident plane, uploaded on first sight, freed when the cell leaves the pool, the arrays shrunk when the resident set does. With `release`, a plane's CPU copy is dropped once it is on the GPU (a host that never draws on the CPU): the entry keeps its presence bits and its elevation plane for readouts.
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, pool: &mut Pool, release: bool) {
        let mut grew = false;
        for s in [&mut self.dem, &mut self.line, &mut self.lw, &mut self.img, &mut self.ov] {
            s.evict_absent(pool);
            if !grew {
                grew |= s.shrink_if_sparse(device, queue);
            }
        }
        self.dem_scale.retain(|k, _| pool.map.contains_key(k));
        self.comp0.evict_absent(pool);
        self.comp1.evict_absent(pool);
        self.baked.retain(|k, _| pool.map.contains_key(k));
        self.line_max.retain(|k, _| pool.map.contains_key(k));
        for (k, e) in pool.map.iter() {
            self.line_max.entry(*k).or_insert(e.line_mag_max);
        }
        let (w, h) = (2 * TEX as u32, TEX as u32);
        let mut scratch = vec![0u8; TRI * 4];
        // A plane full to the GPU's limit leaves a cell without a slot, and it draws as bare ground: said, not silent.
        let unplaced = |slot: &Option<(u32, bool)>, n: &mut usize| {
            if slot.is_none() {
                *n += 1;
            }
        };
        let mut no_slot = 0usize;
        let uploads_before = self.uploads;
        self.uploads_left = false;
        for (key, e) in pool.map.iter_mut() {
            // The frame's share of uploads made: the rest wait for the next sync, the cells drawing from coarser ones meanwhile.
            if self.uploads - uploads_before >= self.upload_budget {
                self.uploads_left = true;
                break;
            }
            if let Some(d) = &e.dem_q {
                let (slot, g) = self.dem.alloc(device, queue, *key);
                grew |= g;
                unplaced(&slot, &mut no_slot);
                if let Some((slot, true)) = slot {
                    write_layer(queue, &self.dem.tex, slot, DEMQ_W as u32, DEMQ_H as u32, 2, bytemuck::cast_slice(&d.tex));
                    self.dem_scale.insert(*key, (d.base, d.step));
                    self.uploads += 1;
                }
            }
            if let Some(l) = &e.line {
                let (slot, g) = self.line.alloc(device, queue, *key);
                grew |= g;
                unplaced(&slot, &mut no_slot);
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
                unplaced(&slot, &mut no_slot);
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
            if e.own.is_some() || e.mvum.is_some() {
                let (slot, g) = self.ov.alloc(device, queue, *key);
                grew |= g;
                unplaced(&slot, &mut no_slot);
                if let Some((slot, true)) = slot {
                    for i in 0..TRI {
                        let (oc, ov) = e.own.as_ref().map_or((0, 0), |o| (o.class[i], o.cov[i]));
                        let (mc, mv) = e.mvum.as_ref().map_or((0, 0), |m| (m.class[i], m.cov[i]));
                        scratch[4 * i] = oc;
                        scratch[4 * i + 1] = ov;
                        scratch[4 * i + 2] = mc;
                        scratch[4 * i + 3] = mv;
                    }
                    write_layer(queue, &self.ov.tex, slot, w, h, 4, &scratch);
                    self.uploads += 1;
                    if release {
                        e.own = None;
                        e.mvum = None;
                    }
                }
            }
            if let Some(im) = &e.img {
                let (slot, g) = self.img.alloc(device, queue, *key);
                grew |= g;
                unplaced(&slot, &mut no_slot);
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
        if no_slot > 0 && self.full_warned.elapsed() > std::time::Duration::from_secs(10) {
            self.full_warned = std::time::Instant::now();
            eprintln!("gpu: {no_slot} planes found no slot ({} cells resident, layers {:?}): they draw as bare ground", pool.map.len(), self.layers());
        }
        // The geoid, once per diamond as its depth-0 cell arrives: the cell keeps its copy, so a GPU map made afresh fills it again.
        for (key, e) in pool.map.iter() {
            let d = key.diamond() as u32;
            if key.depth != 0 || d >= GEOID_DIAMONDS || self.geoid_in & (1 << d) != 0 {
                continue;
            }
            if let Some(g) = &e.geoid {
                let q: Vec<u16> = g.iter().map(|&cm| (cm as i32 + 32768).clamp(1, 65535) as u16).collect();
                write_layer(queue, &self.geoid_tex, d, 2 * TEX as u32, TEX as u32, 2, bytemuck::cast_slice(&q));
                self.geoid_in |= 1 << d;
                eprintln!("gpu: the geoid of diamond {d} is in");
            }
        }
        // A cell whose planes went to the GPU and were released, but which holds no slot for them now (a GPU map made afresh, after a lost device or a panic): dropped from the pool, so residency loads it again from the device and it uploads whole.
        // Found 2026-10-16 on the Pixel: the ocean drew as land in cell-shaped triangles, coasts and roads gone, while cells loaded since drew right.
        use mahere_engine::residency::{PRESENT_IMG, PRESENT_LAND, PRESENT_LINE, PRESENT_MVUM, PRESENT_OWN, PRESENT_WATER};
        let stale: Vec<CellKey> = pool
            .map
            .iter()
            .filter(|(k, e)| {
                let lost = |bits: u8, held: bool, plane: &Plane| e.present & bits != 0 && !held && !plane.map.contains_key(k);
                lost(PRESENT_LINE, e.line.is_some(), &self.line)
                    || lost(PRESENT_LAND | PRESENT_WATER, e.land.is_some() || e.water.is_some(), &self.lw)
                    || lost(PRESENT_OWN | PRESENT_MVUM, e.own.is_some() || e.mvum.is_some(), &self.ov)
                    || lost(PRESENT_IMG, e.img.is_some(), &self.img)
            })
            .map(|(k, _)| *k)
            .collect();
        if !stale.is_empty() {
            eprintln!("gpu: {} cells lost their planes, loading them again", stale.len());
        }
        for k in stale {
            pool.map.remove(&k);
        }
        if grew {
            self.map_bind = None;
        }
    }

    /// The GPU's time in the map, labels and present passes of a frame drawn lately, milliseconds, once its stamps are back; None where the device cannot stamp or none are in yet. A pass a frame skipped reads 0.
    pub fn take_pass_times(&mut self) -> Option<[f32; 3]> {
        let t = self.timer.as_mut()?;
        let i = (0..2).find(|&i| t.state[i].load(std::sync::atomic::Ordering::Acquire) == RANGE_READY)?;
        let stamps: Vec<u64> = {
            let view = t.read[i].slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, u64>(&view).to_vec()
        };
        t.read[i].unmap();
        t.state[i].store(RANGE_IDLE, std::sync::atomic::Ordering::Release);
        let ms = |b: usize| if stamps[b + 1] > stamps[b] { (stamps[b + 1] - stamps[b]) as f32 * t.period / 1e6 } else { 0.0 };
        Some([ms(0), ms(2), ms(4)])
    }

    /// The map pass's pipeline for these layers (the uniform's mask bits), built now if it is the first frame to draw them: a second's compile on a slow phone, once a set.
    fn map_variant(&mut self, device: &wgpu::Device, layers: u32, kind: u8) {
        if self.map_variants.contains_key(&(layers, kind)) {
            return;
        }
        let t = std::time::Instant::now();
        let p = build_map_pipeline(device, &self.shader, &self.map_pl, layers, kind);
        eprintln!("gpu: map pipeline {kind} for layers {layers:#x} built in {} ms", t.elapsed().as_millis());
        self.map_variants.insert((layers, kind), p);
    }

    /// A pipeline the frame can do without until it is in: built on a thread of its own the first time it is asked for, None until it arrives (a slow phone compiles for seconds, and the frame must not wait).
    fn map_variant_later(&mut self, device: &wgpu::Device, layers: u32, kind: u8) -> bool {
        while let Ok((k, p)) = self.built_rx.try_recv() {
            self.building.remove(&k);
            self.map_variants.insert(k, p);
        }
        if self.map_variants.contains_key(&(layers, kind)) {
            return true;
        }
        if self.building.insert((layers, kind)) {
            let (device, shader, layout, tx) = (device.clone(), self.shader.clone(), self.map_pl.clone(), self.built_tx.clone());
            std::thread::spawn(move || {
                let t = std::time::Instant::now();
                let p = build_map_pipeline(&device, &shader, &layout, layers, kind);
                eprintln!("gpu: map pipeline {kind} for layers {layers:#x} built in {} ms, aside", t.elapsed().as_millis());
                tx.send(((layers, kind), p)).ok();
            });
        }
        false
    }

    /// Cells the last frame left to bake: the host keeps frames coming until they are done.
    pub fn bakes_pending(&self) -> bool {
        self.bakes_left
    }

    /// A composite's two slots for a cell, the least recently drawn composite given up when the planes are at their most; None when every one is in this frame's view.
    fn comp_alloc(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: CellKey) -> Option<(u32, u32)> {
        let full = |p: &Plane| !p.map.contains_key(&key) && p.free.is_empty() && p.cap >= p.max;
        if full(&self.comp0) || full(&self.comp1) {
            let victim = self.baked.iter().filter(|(_, b)| b.used < self.frame_no).min_by_key(|(_, b)| b.used).map(|(k, _)| *k)?;
            self.baked.remove(&victim);
            for p in [&mut self.comp0, &mut self.comp1] {
                if let Some(s) = p.map.remove(&victim) {
                    p.free.push(s);
                }
            }
        }
        let (a, g0) = self.comp0.alloc(device, queue, key);
        let (b, g1) = self.comp1.alloc(device, queue, key);
        if g0 || g1 {
            self.map_bind = None;
        }
        match (a, b) {
            (Some((s0, _)), Some((s1, _))) => Some((s0, s1)),
            _ => {
                for p in [&mut self.comp0, &mut self.comp1] {
                    if let Some(s) = p.map.remove(&key) {
                        p.free.push(s);
                    }
                }
                None
            }
        }
    }

    /// The line weights a cell bakes under: each class's largest magnitude over the cell and its parent three depths up (a few kilometres about it at street zoom), so a lane beside a highway falls back as it does against the view's largest, and the same everywhere in that parent, with no seam where neighbouring cells meet.
    fn bake_line_hi(&self, key: CellKey) -> [u8; 32] {
        let mut hi = self.line_max.get(&key).copied().unwrap_or([0; 32]);
        let mut k = key;
        let mut wide = None;
        for _ in 0..3 {
            if k.depth == 0 {
                break;
            }
            k = k.parent();
            if let Some(m) = self.line_max.get(&k) {
                wide = Some(*m);
            }
        }
        if let Some(m) = wide {
            for (h, v) in hi.iter_mut().zip(m) {
                *h = (*h).max(v);
            }
        }
        hi
    }

    /// Planes the last sync left for the next, its upload budget spent: the host keeps frames coming until they are in.
    pub fn uploads_pending(&self) -> bool {
        self.uploads_left
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

    /// The glyph atlas (two bytes a texel: a letter's coverage, its halo's), uploaded when its version changed.
    pub fn set_atlas(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, side: u32, rg: &[u8], version: u64) {
        if self.atlas.as_ref().is_some_and(|(t, v, _)| *v == version && t.width() == side) {
            return;
        }
        if self.atlas.as_ref().is_none_or(|(t, _, _)| t.width() != side) {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("glyph atlas"),
                size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = t.create_view(&Default::default());
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("glyph atlas"),
                layout: &self.glyph_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.atlas_sampler) }],
            });
            self.atlas = Some((t, 0, bind));
        }
        let (t, v, _) = self.atlas.as_mut().unwrap();
        if rg.len() >= (side * side * 2) as usize {
            write_layer(queue, t, 0, side, side, 2, &rg[..(side * side * 2) as usize]);
        }
        *v = version;
    }

    /// The frame's letters and route stretches, as the labeller laid them out for where the camera is now.
    pub fn set_labels(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, glyphs: &[[f32; 20]], segments: &[[f32; 12]]) {
        fn fill(device: &wgpu::Device, queue: &wgpu::Queue, slot: &mut Option<(wgpu::Buffer, u64)>, bytes: &[u8], label: &str) {
            if bytes.is_empty() {
                return;
            }
            if slot.as_ref().is_none_or(|(_, cap)| *cap < bytes.len() as u64) {
                let cap = (bytes.len() as u64).next_power_of_two().max(4096);
                *slot = Some((device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: cap, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }), cap));
            }
            queue.write_buffer(&slot.as_ref().unwrap().0, 0, bytes);
        }
        fill(device, queue, &mut self.glyphs, bytemuck::cast_slice(glyphs), "glyphs");
        fill(device, queue, &mut self.segments, bytemuck::cast_slice(segments), "route");
        self.glyph_count = glyphs.len() as u32;
        self.segment_count = segments.len() as u32;
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
        if self.preview.is_none() {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("preview"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            self.preview = Some((t, v, 1, 1));
        }
        if !self.present_binds.contains_key(&scale) {
            let (_, _, mv, _, _) = self.offscreen.iter().find(|(s, ..)| *s == scale).unwrap();
            let (_, ov, _, _) = self.overlay.as_ref().unwrap();
            let (_, pv, _, _) = self.preview.as_ref().unwrap();
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("present"),
                layout: &self.present_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(mv) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(ov) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(pv) },
                ],
            });
            self.present_binds.insert(scale, bind);
        }
    }

    /// Draw a planned frame to `target` (`w × h`, the present format): upload the plan, the map pass at 2×, the present pass.
    /// Upload the theme's tables: the hypsometric ramp and the line and land colours. Once at start and whenever the theme changes.
    pub fn update_style(&mut self, queue: &wgpu::Queue, luts: &FrameLuts) {
        let mut lut_data = vec![0u32; 4096 + LUT_STYLE];
        let pack = |c: [u8; 3]| ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32;
        // The styled tables carry their opacity in the top byte.
        let pack4 = |c: [u8; 4]| ((c[3] as u32) << 24) | ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32;
        for (i, c) in luts.hypso.iter().enumerate() {
            lut_data[i] = pack(*c);
        }
        for (i, c) in luts.style.line.iter().enumerate() {
            lut_data[4096 + i] = pack4(*c);
        }
        for (i, c) in luts.style.land.iter().enumerate() {
            lut_data[4128 + i] = pack4(*c);
        }
        for (i, c) in luts.style.own.iter().enumerate() {
            lut_data[4160 + i] = pack4(*c);
        }
        lut_data[4176] = pack4(luts.style.wild);
        for (i, c) in luts.style.mvum.iter().enumerate() {
            lut_data[4192 + i] = pack4(*c);
        }
        queue.write_buffer(&self.lut, 0, bytemuck::cast_slice(&lut_data));
        self.style_loaded = true;
        self.style_gen += 1;
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
        let mut refs: Vec<GpuRef> = plan
            .refs
            .iter()
            .map(|r| {
                let mut flags = r.flags;
                let dem = self.dem.map.get(&r.key).copied();
                let line = self.line.map.get(&r.key).copied();
                let lw = self.lw.map.get(&r.key).copied();
                let img = self.img.map.get(&r.key).copied();
                let ov = self.ov.map.get(&r.key).copied();
                if flags & FLAG_INK != 0 {
                    let ink = self.ink.map.get(&r.key).copied();
                    let sel = self.sel.map.get(&r.key).copied();
                    return GpuRef { a: [r.diamond as u32, r.key.depth as u32, r.cu, r.cv], b: [ink.unwrap_or(0), sel.unwrap_or(0), 0, 0], c: [if ink.is_some() { FLAG_INK } else { 0 }, 0, 0, 0], d: [0.0; 4], e: [0.0; 4] };
                }
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
                if ov.is_none() {
                    flags &= !(FLAG_OWN | FLAG_MVUM);
                }
                let (base, step) = self.dem_scale.get(&r.key).copied().unwrap_or((0.0, 1.0));
                GpuRef {
                    a: [r.diamond as u32, r.key.depth as u32, r.cu, r.cv],
                    b: [dem.unwrap_or(0), line.unwrap_or(0), lw.unwrap_or(0), img.unwrap_or(0)],
                    c: [flags, ov.unwrap_or(0), 0, 0],
                    d: [base, step, r.jac[0], r.jac[1]],
                    e: [r.jac[2], r.jac[3], r.jac[4], 0.0],
                }
            })
            .collect();
        // The composites: a vector cell's static layers baked once, the blocks whose cells all have theirs drawn by the small pass (Nick 2026-10-21: a Pixel 3a at a frame a second, Google's map at twenty).
        self.frame_no += 1;
        let layers = mask_bits(luts.mask) & !self.mask_off;
        let bake_mask = layers & !LIVE_BITS;
        let mut bakes: Vec<(usize, u32, u32, [u8; 32])> = Vec::new();
        let mut fast_ready = false;
        self.bakes_left = false;
        if self.composite && mode == RenderMode::Full && layers & NO_COMPOSITE_BITS == 0 {
            let bake_ready = self.map_variant_later(device, u32::MAX, KIND_BAKE);
            fast_ready = self.map_variant_later(device, layers, KIND_FAST);
            // What every composite under this theme and these layers shares.
            let mut hs = rustc_hash::FxHasher::default();
            (self.style_gen, bake_mask, luts.style.water, luts.style.flat, luts.style.bg, luts.style.no_dem, luts.style.sea).hash(&mut hs);
            let shared = hs.finish();
            let by_key: FxHashMap<CellKey, usize> = plan.refs.iter().enumerate().filter(|(_, r)| r.flags & FLAG_INK == 0).map(|(i, r)| (r.key, i)).collect();
            let mut seen = rustc_hash::FxHashSet::default();
            // The blocks' own cells first; then every vector cell the view wants at its depth (not every resident one: past the planes' room, bakes evicted the cells being drawn, a few re-baked every frame), which a pixel between a block's samples may fall in and the shader then finds by search.
            let mut wanted: Vec<usize> = plan.blocks.iter().flat_map(|b| b.cand[1]).filter(|&r| r != NONE && seen.insert(r)).map(|r| r as usize).collect();
            wanted.extend((0..plan.refs.len()).filter(|&i| plan.refs[i].wanted && plan.refs[i].key.depth == plan.vec_depth && plan.refs[i].flags & FLAG_INK == 0 && seen.insert(i as u32)));
            // Every current composite marked drawn first, so a bake making room never takes one this frame draws.
            let mut stale: Vec<(usize, u64, [u8; 32])> = Vec::new();
            for i in wanted {
                if refs[i].c[0] & FLAG_VEC == 0 {
                    continue;
                }
                let key = plan.refs[i].key;
                let hi = self.bake_line_hi(key);
                // Made from: the theme and layers, the cell's own planes, the terrain and imagery cells the bake's search can reach (this cell and its parents), the line weights.
                let mut h = rustc_hash::FxHasher::default();
                (shared, refs[i].c[0], hi).hash(&mut h);
                let mut k = key;
                loop {
                    if let Some(&j) = by_key.get(&k) {
                        (k.depth, refs[j].c[0] & (FLAG_DEM | FLAG_IMG)).hash(&mut h);
                    }
                    if k.depth == 0 {
                        break;
                    }
                    k = k.parent();
                }
                let sig = h.finish();
                let slots = self.comp0.map.get(&key).copied().zip(self.comp1.map.get(&key).copied());
                if let (Some(b), Some((s0, s1))) = (self.baked.get_mut(&key), slots) {
                    if b.sig == sig {
                        b.used = self.frame_no;
                        refs[i].c = [refs[i].c[0] | FLAG_COMP, refs[i].c[1], s0, s1];
                        continue;
                    }
                }
                stale.push((i, sig, hi));
            }
            for (i, sig, hi) in stale {
                if !bake_ready || bakes.len() >= self.bake_budget {
                    self.bakes_left = true;
                    break;
                }
                let key = plan.refs[i].key;
                // No room left (every composite held is drawn this frame): the rest draw whole, and nothing spins waiting for room.
                let Some((s0, s1)) = self.comp_alloc(device, queue, key) else { break };
                self.baked.insert(key, Baked { sig, used: self.frame_no });
                refs[i].c = [refs[i].c[0] | FLAG_COMP, refs[i].c[1], s0, s1];
                bakes.push((i, s0, s1, hi));
            }
        }
        // Blocks drawn from composites first, then the rest: two draws over one buffer.
        let fast = |b: &PlanBlock| fast_ready && b.cand[1][0] != NONE && b.cand[1].iter().all(|&r| r == NONE || refs[r as usize].c[0] & FLAG_COMP != 0);
        let gpu_block = |b: &PlanBlock| GpuBlock { a: [b.x, b.y, b.size, b.diamond], b: [b.u0, b.v0, 0, 0], c: [b.du_dx, b.dv_dx, b.du_dy, b.dv_dy], d: [b.twist_u, b.twist_v, 0.0, 0.0], e: b.cand[0], f: b.cand[1], g: b.cand[2], h: b.ink };
        let mut blocks: Vec<GpuBlock> = plan.blocks.iter().filter(|b| fast(b)).map(gpu_block).collect();
        let n_fast = blocks.len();
        blocks.extend(plan.blocks.iter().filter(|b| !fast(b)).map(gpu_block));
        self.fast_blocks = (n_fast, blocks.len());
        self.baked_last = bakes.len();
        self.bakes += bakes.len();
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
            pole: plan.pole,
            raw: [luts.contours.lo_m, luts.contours.hi_m, 0.0, 0.0],
            size: [(w * self.scale.max(1)) as f32, (h * self.scale.max(1)) as f32],
            scale: self.scale.max(1) as f32,
            mask: mask_bits(luts.mask) & !self.mask_off,
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
            style_water: rgba(luts.style.water),
            style_contour: rgba(luts.style.contour),
            style_contour_index: rgba(luts.style.contour_index),
            style_flat: rgba(luts.style.flat),
            style_bg: rgba(luts.style.bg),
            style_no_dem: rgba(luts.style.no_dem),
            style_sea: rgba(luts.style.sea),
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
        // The bakes, each a pass over its cell's texels with uniforms of its own: the cell's depth wanted throughout, its texel grid in `raw`, the live layers left out, nothing gathered.
        for (j, &(i, s0, s1, hi)) in bakes.iter().enumerate() {
            if self.bake_uniforms.len() <= j {
                self.bake_uniforms.push(device.create_buffer(&wgpu::BufferDescriptor { label: Some("bake uniforms"), size: std::mem::size_of::<Uniforms>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
            }
            let r = &plan.refs[i];
            let depth = r.key.depth as u32;
            let bu = Uniforms {
                size: [(2 * TEX) as f32, TEX as f32],
                scale: 1.0,
                mask: bake_mask,
                depths: [depth, depth, 0, 0],
                line_hi: std::array::from_fn(|a| std::array::from_fn(|b| hi[4 * a + b].max(1) as f32)),
                tone: [u.tone[0], 1.0, 0.0, 0.0],
                globe: [0.0; 4],
                raw: [r.cu as f32, r.cv as f32, r.diamond as f32, depth as f32],
                ..u
            };
            queue.write_buffer(&self.bake_uniforms[j], 0, bytemuck::bytes_of(&bu));
            let bind = self.bind_with(device, &self.bake_uniforms[j], true);
            let (v0, v1) = (self.comp0.layer_view(s0), self.comp1.layer_view(s1));
            let attach = |v| Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("bake"), color_attachments: &[attach(&v0), attach(&v1)], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
            pass.set_pipeline(&self.map_variants[&(u32::MAX, KIND_BAKE)]);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }

        // The map pass gathers the elevations it draws: reset, then copied out after the pass into a free readback buffer (none free: this frame's range is skipped, the next one's read).
        let draws_map = mode != RenderMode::Relight;
        // The map pass's pipeline for the layers this frame shows, the rest compiled out.
        if mode == RenderMode::FullWithG {
            self.map_variant(device, layers, KIND_G);
        }
        if mode == RenderMode::Full && n_fast < blocks.len() {
            self.map_variant(device, layers, KIND_FULL);
        }
        if draws_map {
            queue.write_buffer(&self.view_range, 0, bytemuck::cast_slice(&[u32::MAX, 0u32]));
        }
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
            pass.set_pipeline(&self.map_variants[&(layers, KIND_G)]);
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
                timestamp_writes: self.timer.as_ref().map(|t| wgpu::RenderPassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, self.map_bind.as_ref().unwrap(), &[]);
            if n_fast > 0 {
                pass.set_pipeline(&self.map_variants[&(layers, KIND_FAST)]);
                pass.draw(0..6, 0..n_fast as u32);
            }
            if n_fast < blocks.len() {
                pass.set_pipeline(&self.map_variants[&(layers, KIND_FULL)]);
                pass.draw(0..6, n_fast as u32..blocks.len() as u32);
            }
        }
        if draws_map {
            let idle = (0..2).find(|&i| self.range_state[i].load(std::sync::atomic::Ordering::Acquire) == RANGE_IDLE);
            if let Some(i) = idle {
                encoder.copy_buffer_to_buffer(&self.view_range, 0, &self.range_read[i], 0, 8);
                self.range_state[i].store(RANGE_COPIED, std::sync::atomic::Ordering::Release);
                self.range_copied = Some(i);
            }
        }
        // The route, then the letters, each a quad drawn into the map's own layer at its scale (Nick 2026-10-16: native while moving, the 3× layer when still), so the binning below scales them as it does the map's lines.
        let route = self.segments.as_ref().filter(|_| self.segment_count > 0);
        let letters = self.glyphs.as_ref().zip(self.atlas.as_ref()).filter(|_| self.glyph_count > 0);
        // Always, when the passes are timed: an empty pass still writes its stamps, which a resolve may not read unwritten.
        if route.is_some() || letters.is_some() || self.timer.is_some() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("labels"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: map_view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: self.timer.as_ref().map(|t| wgpu::RenderPassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(2), end_of_pass_write_index: Some(3) }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, self.map_bind.as_ref().unwrap(), &[]);
            if let Some((buf, _)) = route {
                pass.set_pipeline(&self.segment_pipeline);
                pass.set_vertex_buffer(0, buf.slice(..(self.segment_count as u64 * 48)));
                pass.draw(0..6, 0..self.segment_count);
            }
            if let Some(((buf, _), (_, _, atlas))) = letters {
                pass.set_pipeline(&self.glyph_pipeline);
                pass.set_bind_group(1, atlas, &[]);
                pass.set_vertex_buffer(0, buf.slice(..(self.glyph_count as u64 * 80)));
                pass.draw(0..6, 0..self.glyph_count);
            }
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
                timestamp_writes: self.timer.as_ref().map(|t| wgpu::RenderPassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(4), end_of_pass_write_index: Some(5) }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.present_pipeline);
            pass.set_bind_group(0, self.map_bind.as_ref().unwrap(), &[]);
            pass.set_bind_group(1, &self.present_binds[&scale], &[]);
            pass.draw(0..3, 0..1);
            // The top layer (panel, pin, measurement) at the screen's own resolution, over the binned map and its labels.
            pass.set_pipeline(&self.top_pipeline);
            pass.set_bind_group(1, &self.present_binds[&scale], &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(t) = &mut self.timer {
            let idle = (0..2).find(|&i| t.state[i].load(std::sync::atomic::Ordering::Acquire) == RANGE_IDLE);
            if let Some(i) = idle {
                encoder.resolve_query_set(&t.set, 0..PASS_STAMPS, &t.resolve, 0);
                encoder.copy_buffer_to_buffer(&t.resolve, 0, &t.read[i], 0, PASS_STAMPS as u64 * 8);
                t.state[i].store(RANGE_COPIED, std::sync::atomic::Ordering::Release);
                t.copied = Some(i);
            }
        }
    }
}

impl GpuMap {
    /// After the frame's submit: start mapping the readback buffer it copied the elevation range into. Never waits; the range arrives on a later poll.
    pub fn after_submit(&mut self) {
        if let Some(t) = &mut self.timer {
            if let Some(i) = t.copied.take() {
                t.state[i].store(RANGE_MAPPING, std::sync::atomic::Ordering::Release);
                let state = t.state[i].clone();
                t.read[i].slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    state.store(if r.is_ok() { RANGE_READY } else { RANGE_IDLE }, std::sync::atomic::Ordering::Release);
                });
            }
        }
        let Some(i) = self.range_copied.take() else { return };
        if self.range_state[i].load(std::sync::atomic::Ordering::Acquire) != RANGE_COPIED {
            return;
        }
        self.range_state[i].store(RANGE_MAPPING, std::sync::atomic::Ordering::Release);
        let state = self.range_state[i].clone();
        self.range_read[i].slice(..).map_async(wgpu::MapMode::Read, move |r| {
            state.store(if r.is_ok() { RANGE_READY } else { RANGE_IDLE }, std::sync::atomic::Ordering::Release);
        });
    }

    /// The elevations the last read-back map pass drew, as quantised elevations (lowest, highest), if one has arrived since last asked; None when it drew no terrain.
    pub fn take_range(&mut self) -> Option<Option<(u16, u16)>> {
        let mut out = None;
        for i in 0..2 {
            if self.range_state[i].load(std::sync::atomic::Ordering::Acquire) == RANGE_READY {
                let (lo, hi) = {
                    let view = self.range_read[i].slice(..).get_mapped_range();
                    let v: &[u32] = bytemuck::cast_slice(&view);
                    (v[0], v[1])
                };
                self.range_read[i].unmap();
                self.range_state[i].store(RANGE_IDLE, std::sync::atomic::Ordering::Release);
                out = Some((lo <= hi).then_some((lo.min(u16::MAX as u32) as u16, hi.min(u16::MAX as u32) as u16)));
            }
        }
        out
    }
}

/// A style colour as the shader takes it: linear light.
/// A styled colour as the shader's vec4: linear light (the shader mixes it as it is), the opacity as a fraction.
fn rgba(c: [u8; 4]) -> [f32; 4] {
    let l = mahere_engine::colour::lin([c[0], c[1], c[2]]);
    [l[0], l[1], l[2], c[3] as f32 / 255.0]
}



#[cfg(test)]
mod shader_tests {
    /// The map shader parses and validates (a WGSL mistake otherwise shows only on a phone, as a black map).
    #[test]
    fn the_map_shader_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("map.wgsl")).unwrap_or_else(|e| panic!("{}", e.emit_to_string(include_str!("map.wgsl"))));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).expect("valid");
    }
}
