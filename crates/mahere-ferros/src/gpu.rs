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
    marks_key: Option<u64>,
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
    /// Overlays painted (of them the panel painted again, the marks drawn again), and their µs: the panel, the marks, the sum into the arena.
    overlays: u64,
    /// The plan's µs: preparing the view, the plan itself, the residency's want (`MapCore::plan_us`).
    plan_parts: [u64; 3],
    /// Inside the plan (`FramePlan::us`): the corners, the desired set, the blocks, the references and table, the candidates.
    plan_inner: [u64; 5],
    panels: u64,
    marks: u64,
    overlay_parts: [u64; 3],
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
        Some(GpuMap { gpu: gpu_slot, arena: base as *mut u32, bytes, surface_va, lay: None, slots: Slots::new(0), style: None, overlay_key: None, marks_key: None, overlay_on: false, stats: Stats::default(), report_at: std::time::Instant::now() })
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
            self.marks_key = None;
        }
        let lay = self.lay.unwrap();
        let t0 = std::time::Instant::now();
        let plan = app.map.plan(w, h);
        for (a, b) in self.stats.plan_parts.iter_mut().zip(app.map.plan_us) {
            *a += b;
        }
        for (a, b) in self.stats.plan_inner.iter_mut().zip(plan.us) {
            *a += b as u64;
        }
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
                "mahere: GPU {} frames: the kernel {} us a frame on the GPU (longest {}), {} words from {} nodes; host per frame: plan {} us (view {}, plan {} [corners {}, desired {}, blocks {}, refs {}, candidates {}], want {}), cells {} us, blocks {} us, overlay {} us, record and compile {} us, the call {} us; {} overlays painted ({} panels, {} marks; panel {} us, marks {} us, sum {} us each); {} cell planes written so far, {} wanted with no slot free ({} slots)",
                s.frames, s.gpu_us / n, s.gpu_max, s.words, s.nodes, s.plan_us / n, s.plan_parts[0] / n, s.plan_parts[1] / n, s.plan_inner[0] / n, s.plan_inner[1] / n, s.plan_inner[2] / n, s.plan_inner[3] / n, s.plan_inner[4] / n, s.plan_parts[2] / n, s.cells_us / n, s.blocks_us / n, s.overlay_us / n, s.compile_us / n, s.call_us / n, s.overlays, s.panels, s.marks, s.overlay_parts[0] / s.overlays.max(1), s.overlay_parts[1] / s.overlays.max(1), s.overlay_parts[2] / s.overlays.max(1), self.slots.written, self.slots.starved, lay.slots
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
        let hash = |f: &dyn Fn(&mut std::collections::hash_map::DefaultHasher)| {
            let mut hs = std::collections::hash_map::DefaultHasher::new();
            f(&mut hs);
            hs.finish()
        };
        // What the marks show (`MapCore::overlay`: the compass, the GPS, the pins, the globe's loading ring), and what the panel shows (its rows, its readouts, the zoom and the contours' spacing); each painted again only when its own changed.
        let pinned = map.gps.is_some() || [map.pin_a, map.pin_b, map.nav_to, map.found_at].iter().any(|p| p.is_some());
        let marks_key = hash(&|hs| {
            // The bearing as the compass can show it (2^-8 radians turn its arrow's tip a tenth of a pixel): the two-finger solve moves it by less every frame, and each was a repaint.
            (w, h, (c.bearing() * (1u32 << 8) as f64).round() as i64, map.globe_loading().map(|d| (d * 64.0) as u32)).hash(hs);
            if pinned {
                (c.lat().to_bits(), c.lon().to_bits(), c.ppd.to_bits()).hash(hs);
            }
        });
        let panel_key = hash(&|hs| {
            (w, h, app.panel.is_open(), map.layers().bits(), app.panel.ru().to_bits(), map.view_depth(), map.zoom_depth(), map.route_version, map.contour_major().map(f32::to_bits)).hash(hs);
            if app.panel.is_open() {
                (c.lat().to_bits(), c.lon().to_bits(), c.ppd.to_bits(), c.bearing().to_bits()).hash(hs);
            }
        });
        let (marks_new, panel_new) = (self.marks_key != Some(marks_key), self.overlay_key != Some(panel_key));
        if !marks_new && !panel_new {
            return;
        }
        self.marks_key = Some(marks_key);
        self.overlay_key = Some(panel_key);
        let t0 = std::time::Instant::now();
        if panel_new {
            app.paint_panel(w, h);
            self.stats.panels += 1;
        }
        let t1 = std::time::Instant::now();
        let marks = if marks_new {
            self.stats.marks += 1;
            app.map.overlay(w, h, true)
        } else {
            app.map.marks_canvas()
        };
        let t2 = std::time::Instant::now();
        // The panel's pixels with the marks under them (`Panel::overlay_rows`' sum, premultiplied RGBA bytes), straight into the arena, bands of rows on every core: one thread over the screen's 4 Mpx was 62 ms of every frame on the Pro (2026-11-11).
        let panel = app.panel.pixels();
        let panel = (panel.len() == w * h).then_some(panel);
        let out = unsafe { core::slice::from_raw_parts_mut(self.arena.add(kernel::OVERLAY / 4), w * h) };
        use rayon::prelude::*;
        let any = out
            .par_chunks_mut(w << 6)
            .enumerate()
            .map(|(band, rows)| {
                let at = band * (w << 6);
                let mut any = 0u32;
                for (k, o) in rows.iter_mut().enumerate() {
                    let i = at + k;
                    let (m, p) = (marks[i], panel.map_or(0, |b| b[i]));
                    if m == 0 && p >> 24 == 0 {
                        *o = 0;
                        continue;
                    }
                    let (mr, mg, mb) = ((m >> 16) & 255, (m >> 8) & 255, m & 255);
                    let ma = mr.max(mg).max(mb);
                    let (r, g, b, a) = if p >> 24 == 0 { (0, 0, 0, 0) } else { mahere_panel::Panel::premul_pub(p) };
                    let keep = 255 - a as u32;
                    let ch = |c: u8, under: u32| c as u32 + under * keep / 255;
                    *o = ch(r, mr) | ch(g, mg) << 8 | ch(b, mb) << 16 | ch(a, ma) << 24;
                    any |= *o;
                }
                any
            })
            .reduce(|| 0, |a, b| a | b);
        let t3 = std::time::Instant::now();
        let us = |a: std::time::Instant, b: std::time::Instant| (b - a).as_micros() as u64;
        self.stats.overlay_parts[0] += us(t0, t1);
        self.stats.overlay_parts[1] += us(t1, t2);
        self.stats.overlay_parts[2] += us(t2, t3);
        self.stats.overlays += 1;
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
