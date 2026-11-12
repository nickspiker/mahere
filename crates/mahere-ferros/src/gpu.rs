//! The map on ferros's GPU (ferros GPU-KISTS.md; the owner, 2026-11-11: "Obviously get mahere on GPU proper and flawless at 120fps"): each frame the plan's lattice and the cells it draws go to the layer arena (`arena.rs`), the compose kernel (`kernel.rs`) is recorded with the frame's constants and run by the GPU service over every pixel, and fluor's ferros host shows what it wrote (`FluorApp::render_own`).
//!
//! The arena (`layers`, slot 10, CPU write-back and GPU coherent: what Mahere writes the GPU reads, and the other way, with no copy and no cache maintenance) holds the tables, the tiles, the blocks, the overlay, the frame, the kernel's words, and the cells' planes in slots of 1 MiB, each written once when its cell is first drawn and kept until a frame needs the slot for a cell it draws and this one it does not.
//! The GPU service (`gpu`, slot 9) runs a kernel and answers when it is done: one held call a frame, no polling.
//!
//! Until kists have GPU memory of their own (GPU-KISTS.md plan step 1), the arena is the one the system's GPU clients share (the wallpaper's maker uses its first 32 MiB, the attest mark the rest): a development spawn with `--dma layers` takes all 64 MiB, which neither uses once a person is attested, and nothing guards them from each other yet.

use crate::arena::{self, Layout, Slots};
use crate::kernel;
use mahere_device::{DeviceApp, NoGpu};

/// The arena in the GPU's address space (ferros payloads/gpu `ARENA_VA`), and its size: 128 MiB from a kernel that also fences the apps' surfaces in for the GPU (ferros 2026-11-11), 64 MiB before.
const ARENA_VA: u64 = 0x8000_0000;
const ARENA_GROWN: usize = 128 << 20;
const ARENA_BEFORE: usize = 64 << 20;
/// The apps' surfaces in the GPU's address space (payloads/gpu `SURFACES_VA`) and where they start in memory (ferros cachemap SURFACES): an app's surface is its physical base's offset from there.
const SURFACES_VA: u64 = 0xC000_0000;
const SURFACES_PA: u64 = 0xCA00_0000;

pub struct GpuMap {
    gpu: u64,
    arena: *mut u32,
    bytes: usize,
    /// The app's surface in the GPU's address space, when the kernel draws into it (`Out::Surface`).
    surface_va: Option<u64>,
    lay: Option<Layout>,
    slots: Slots,
    style: Option<u64>,
    overlay_key: Option<u64>,
    overlay_on: bool,
    stats: Stats,
    report_at: std::time::Instant,
}

/// Per report: frames, the GPU's own µs (as the service measures each job), the host's µs by stage, the kernel's size.
#[derive(Default)]
struct Stats {
    frames: u64,
    gpu_us: u64,
    gpu_max: u64,
    plan_us: u64,
    cells_us: u64,
    blocks_us: u64,
    overlay_us: u64,
    compile_us: u64,
    call_us: u64,
    words: usize,
    nodes: usize,
}

impl GpuMap {
    /// The GPU path when both grants are there: the service at `gpu_slot`, the arena at `arena_slot`. None (said) otherwise: the CPU raster then draws.
    pub fn open(gpu_slot: u64, arena_slot: u64, surface_slot: u64) -> Option<GpuMap> {
        if crate::ferros::sys::grant_count() <= arena_slot {
            println!("mahere: no GPU grants (slots {gpu_slot} gpu and {arena_slot} layers): the map draws on the CPU");
            return None;
        }
        let base = crate::ferros::sys::dma(arena_slot);
        if base == 0 || base == crate::ferros::sys::ERR {
            println!("mahere: no layer arena at slot {arena_slot}: the map draws on the CPU");
            return None;
        }
        // The grown arena says the kernel fences the apps' surfaces in for the GPU too: the kernel then writes the surface, no copy.
        let bytes = crate::ferros::sys::dma_size(arena_slot, ARENA_GROWN as u64).map_or(ARENA_BEFORE, |n| n as usize);
        let surface = crate::ferros::sys::dma(surface_slot);
        let surface_va = (bytes == ARENA_GROWN && surface != 0 && surface != crate::ferros::sys::ERR).then(|| SURFACES_VA + (surface - SURFACES_PA));
        println!("mahere: the map draws on the GPU (the service at slot {gpu_slot}, the layer arena at slot {arena_slot}, {} MiB), {}", bytes >> 20, if surface_va.is_some() { "straight into the surface" } else { "fluor's pixels in the arena, presented by the CPU" });
        Some(GpuMap { gpu: gpu_slot, arena: base as *mut u32, bytes, surface_va, lay: None, slots: Slots::new(0), style: None, overlay_key: None, overlay_on: false, stats: Stats::default(), report_at: std::time::Instant::now() })
    }

    fn arena(&mut self) -> &mut [u32] {
        // SAFETY: the arena is this process's mapping of a DMA region of ARENA_BYTES it was granted for its whole life; only the frame thread touches it, and the GPU only while the service's call is held.
        unsafe { core::slice::from_raw_parts_mut(self.arena, self.bytes / 4) }
    }

    /// The frame the last draw made, fluor's pixels; None when the kernel drew into the surface itself.
    pub fn frame(&self) -> Option<&[u32]> {
        let lay = self.lay.as_ref()?;
        if lay.out == kernel::Out::Surface {
            return None;
        }
        // SAFETY: as `arena`; the GPU wrote it before the service answered.
        Some(unsafe { core::slice::from_raw_parts(self.arena.add(lay.frame_at / 4), lay.w * lay.h) })
    }

    /// One frame of `app`'s map, `w` × `h`, drawn on the GPU into the arena's frame. Err: the service did not run it.
    pub fn draw(&mut self, app: &mut DeviceApp<NoGpu>, w: usize, h: usize) -> Result<(), String> {
        if self.lay.is_none_or(|l| (l.w, l.h) != (w, h)) {
            let out = if self.surface_va.is_some() { kernel::Out::Surface } else { kernel::Out::Argb8 };
            let l = Layout::new(w, h, self.bytes, out);
            println!("mahere: GPU layout for {w} x {h}: the overlay at {} MiB, the frame at {} MiB, the code at {} MiB, {} cell slots of 1 MiB from {} MiB", kernel::OVERLAY >> 20, l.frame_at >> 20, l.code_at >> 20, l.slots, l.slots_at >> 20);
            self.slots = Slots::new(l.slots);
            self.lay = Some(l);
            self.overlay_key = None;
        }
        let lay = self.lay.unwrap();
        let t0 = std::time::Instant::now();
        let plan = app.map.plan(w, h);
        let t1 = std::time::Instant::now();
        let version = app.map.pool_version();
        let arena = unsafe { core::slice::from_raw_parts_mut(self.arena, self.bytes / 4) };
        let (cand, range) = arena::cells(&mut self.slots, app.map.pool(), &plan, version, &lay, arena);
        app.map.set_drawn_range(range);
        let t2 = std::time::Instant::now();
        arena::blocks(&plan, &cand, &lay, arena)?;
        let ramp = self.style != Some(app.map.style_version());
        self.style = Some(app.map.style_version());
        arena::tables(app.map.luts(), &plan, ramp, arena);
        let t3 = std::time::Instant::now();
        self.overlay(app, w, h);
        let t4 = std::time::Instant::now();
        let c = arena::consts(app.map.luts(), &plan, &lay, self.overlay_on);
        let k = kernel::record(&c);
        if k.words.len() > arena::CODE_WORDS {
            return Err(format!("the map kernel is {} words, more than the GPU service's {}", k.words.len(), arena::CODE_WORDS));
        }
        let code = &mut self.arena()[lay.code_at / 4..lay.code_at / 4 + 2 * k.words.len()];
        for (j, w) in k.words.iter().enumerate() {
            code[2 * j] = *w as u32;
            code[2 * j + 1] = (*w >> 32) as u32;
        }
        let t5 = std::time::Instant::now();
        let us = self.run(&k, &lay, (w * h).div_ceil(16) as u32)?;
        let t6 = std::time::Instant::now();
        let s = &mut self.stats;
        let d = |a: std::time::Instant, b: std::time::Instant| (b - a).as_micros() as u64;
        s.frames += 1;
        s.gpu_us += us;
        s.gpu_max = s.gpu_max.max(us);
        s.plan_us += d(t0, t1);
        s.cells_us += d(t1, t2);
        s.blocks_us += d(t2, t3);
        s.overlay_us += d(t3, t4);
        s.compile_us += d(t4, t5);
        s.call_us += d(t5, t6);
        s.words = k.words.len();
        s.nodes = k.kept;
        if self.report_at.elapsed().as_secs() >= 10 {
            self.report_at = std::time::Instant::now();
            let s = &self.stats;
            let n = s.frames.max(1);
            println!(
                "mahere: GPU {} frames: the kernel {} us a frame on the GPU (longest {}), {} words from {} nodes; host per frame: plan {} us, cells {} us, blocks {} us, overlay {} us, record and compile {} us, the call {} us; {} cell planes written so far, {} wanted with no slot free ({} slots)",
                s.frames, s.gpu_us / n, s.gpu_max, s.words, s.nodes, s.plan_us / n, s.cells_us / n, s.blocks_us / n, s.overlay_us / n, s.compile_us / n, s.call_us / n, self.slots.written, self.slots.starved, lay.slots
            );
            self.stats = Stats::default();
        }
        Ok(())
    }

    /// The panel and the screen's marks (compass, pins, the globe's loading ring, the GPS) as the overlay, painted again when what they show changed.
    fn overlay(&mut self, app: &mut DeviceApp<NoGpu>, w: usize, h: usize) {
        use std::hash::{Hash, Hasher};
        let map = &app.map;
        let c = map.cam;
        let mut hs = std::collections::hash_map::DefaultHasher::new();
        (w, h, app.panel.is_open(), c.bearing().to_bits(), map.layers().bits(), app.panel.ru().to_bits()).hash(&mut hs);
        // Open, the panel's readouts follow the camera; closed, the pins and the GPS mark do.
        if app.panel.is_open() || map.gps.is_some() || [map.pin_a, map.pin_b, map.nav_to, map.found_at].iter().any(|p| p.is_some()) {
            (c.lat().to_bits(), c.lon().to_bits(), c.ppd.to_bits()).hash(&mut hs);
        }
        (map.view_depth(), map.zoom_depth(), map.route_version, map.globe_loading().map(|d| (d * 64.0) as u32), map.contour_major().map(f32::to_bits)).hash(&mut hs);
        let key = hs.finish();
        if self.overlay_key == Some(key) {
            return;
        }
        self.overlay_key = Some(key);
        app.paint_panel(w, h);
        // The marks on the map's own canvas, the panel's pixels over them, a band of rows at a time straight into the arena (a whole screen's copy at once is 16 MB of heap twice over: the arc's ceiling, 2026-11-11).
        let marks = app.map.overlay(w, h, true);
        let out = unsafe { core::slice::from_raw_parts_mut(self.arena.add(kernel::OVERLAY / 4), w * h) };
        let mut any = 0u32;
        for y0 in (0..h).step_by(64) {
            let y1 = (y0 + 64).min(h);
            let rgba = app.panel.overlay_rows(marks, w, h, y0, y1);
            for (o, p) in out[y0 * w..y1 * w].iter_mut().zip(rgba.chunks_exact(4)) {
                *o = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                any |= *o;
            }
        }
        self.overlay_on = any != 0;
    }

    /// The kernel run by the GPU service over `groups` workgroups of 16: its words at the arena's code offset, uniform 0 the frame and 1 the arena, then its constants. The GPU's own µs, as the service measures the job.
    fn run(&mut self, k: &valhall::ssa::Compiled, lay: &Layout, groups: u32) -> Result<u64, String> {
        let mut msg = Vec::with_capacity(18 + 8 * 16 + 16);
        msg.push(1u8);
        msg.extend_from_slice(&(0x1000u16 | if k.regs64 { 0x8000 } else { 0 }).to_le_bytes());
        msg.extend_from_slice(&(lay.code_at as u32).to_le_bytes());
        msg.extend_from_slice(&(k.words.len() as u16).to_le_bytes());
        msg.extend_from_slice(&groups.to_le_bytes());
        msg.extend_from_slice(&1u32.to_le_bytes());
        let frame = self.surface_va.unwrap_or(ARENA_VA + lay.frame_at as u64);
        let addrs = [frame, ARENA_VA];
        msg.push((addrs.len() + k.consts.len()) as u8);
        for a in addrs.iter().chain(k.consts.iter()) {
            msg.extend_from_slice(&a.to_le_bytes());
        }
        for name in [&b"mahere"[..], b"map"] {
            msg.push(name.len() as u8);
            msg.extend_from_slice(name);
        }
        let mut reply = [1u8, 0, 0, 0, 0];
        let n = crate::ferros::sys::call(self.gpu, &msg, &mut reply);
        if n == crate::ferros::sys::ERR {
            return Err(String::from("no GPU service receiving"));
        }
        if reply[0] != 0 {
            return Err(format!("the GPU service answered {} to the map kernel", reply[0]));
        }
        Ok(u32::from_le_bytes([reply[1], reply[2], reply[3], reply[4]]) as u64)
    }
}
