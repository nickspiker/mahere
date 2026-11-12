//! The map on ferros's GPU (ferros GPU-KISTS.md; the owner, 2026-11-11: "Obviously get mahere on GPU proper and flawless at 120fps"): each frame the view's plan (the faces in view and a page table of the cells standing in, mahere-engine viewplan.rs) and the cells it draws go to the layer arena (`arena.rs`), the compose kernel (`kernel.rs`) is recorded with the frame's constants and run by the GPU service over every pixel, and fluor's ferros host shows what it wrote (`FluorApp::render_own`).
//!
//! The arena (`layers`, slot 10, CPU write-back and GPU coherent: what Mahere writes the GPU reads, and the other way, with no copy and no cache maintenance) holds the tables, the faces' rows, the page table's squares, the overlay, the frame, the kernel's words, and the cells' planes in slots of 1 MiB, each written once when its cell is first drawn and kept until a frame needs the slot for a cell it draws and this one it does not.
//! The GPU service (`gpu`, slot 9) runs a kernel and answers when it is done: one held call a frame, no polling, made on a thread of its own (`GpuThread`) so the next frame is planned while the GPU draws this one.
//!
//! With a second surface (slot 11) the frames are a pipeline over a mailbox (ferros GPU-KISTS.md 1.4): frame K is planned into arena region K mod 2 while the GPU draws frame K − 1 into its buffer, and the display shows each buffer as it is finished; the show answers once the display has taken the frame, so the buffer shown before it, the one frame K + 1 draws into, is no longer read. A frame is on the glass a frame later than it was planned, and the planning, the GPU and the display's publish each get a whole frame's time.
//!
//! The arena is Mahere's own block of ferros's layer arena (ferros GPU-KISTS.md plan step 1, 2026-11-12): the kernel gives every kist the largest block free at its launch (128 MiB when nothing else holds one, 64 or 32 when the wallpaper's maker or the attest mark holds theirs), and a development spawn's `--dma layers` the same. The GPU service runs Mahere's kernel in an address space of Mahere's alone, which maps that block at `ARENA_VA` and Mahere's surfaces where they lie from `SURFACES_VA`, nothing else: an address outside them faults Mahere's job, never another app's memory.
//!
//! As a kist the GPU is handed, not asked for (the owner, 2026-11-12: "I feel like GPU should be implicit"): the kernel's three slots after its eight wants (ferros_sys::gpu `kist_slots`), 9 the service, 10 its block, 11 a second surface, the same slots a development spawn gives. On a production build they are empty slots until ferros checks GPU code, and the map draws on the CPU.

use crate::arena::{self, Layout, Slots};
use crate::kernel;
use mahere_device::{DeviceApp, NoGpu};
use mahere_engine::PixelBox;

/// Mahere's block of the arena in the GPU's address space (ferros_sys::gpu `ARENA_VA`), and the most it can be: 128 MiB, the whole arena. A kernel that cannot say a region's size (before ferros 2026-11-11) gave the whole 64 MiB arena and fenced no surface in for the GPU.
const ARENA_VA: u64 = 0x8000_0000;
const ARENA_MOST: usize = 128 << 20;
const ARENA_BEFORE: usize = 64 << 20;
/// The apps' surfaces in the GPU's address space (payloads/gpu `SURFACES_VA`) and where they start in memory (ferros cachemap SURFACES): an app's surface is its physical base's offset from there.
const SURFACES_VA: u64 = 0xC000_0000;
const SURFACES_PA: u64 = 0xCA00_0000;

pub struct GpuMap {
    gpu: GpuThread,
    arena: *mut u32,
    bytes: usize,
    /// The app's surfaces, (its `dma` base, its address on the GPU), when the kernel draws into them (`Out::Surface`): the window's own, and with a second surface grant the mailbox's other buffer.
    surfaces: Option<[(u64, u64); 2]>,
    /// Two surfaces: the frames pipelined over the mailbox.
    pipelined: bool,
    lay: Option<Layout>,
    slots: Slots,
    /// By frame region: the style its ramp was written for.
    style: [Option<u64>; 2],
    /// What the overlay shows (the panel's and the marks' keys), the box the panel's pixels cover, and whether the overlay holds anything.
    panel_key: Option<u64>,
    marks_key: Option<u64>,
    panel_box: PixelBox,
    overlay_on: bool,
    /// By frame region: the box of the overlay that changed since that region's copy was last summed (each region's copy is a resident layer in the arena, updated by this box alone).
    stale: [PixelBox; 2],
    /// The frame planned and recorded by the last draw and not yet run (pipelined).
    ready: Option<Ready>,
    /// The region (and buffer) the next frame is planned into, and the one the last finished frame is in.
    next: usize,
    shown: usize,
    /// A frame has been drawn into a buffer since the layout was made: until then a frame runs at once, so nothing unmade is shown.
    drawn: bool,
    /// By frame region: whether the frame planned there was made with the display's bake (kernel.rs `Bake`), so the window shows it baked (`baked`).
    baked: [bool; 2],
    /// The kernels compiled, by their recording's key (valhall `Recording::key`), the most recently used first: a frame of a structure met before binds its params into that one's words (`kernel`).
    kernels: Vec<(u64, valhall::ssa::Compiled)>,
    stats: Stats,
    report_at: std::time::Instant,
}

/// The compiled kernels kept: a handful of structures in a session (the layers, one face or several or the fold, the ring, the overlay), each about 4 KiB of words; a linear search over these few.
const KERNELS: usize = 1 << 3;

/// A frame planned and recorded: its region, the service's message, its kernel's size, and what it shows (`view_key`).
struct Ready {
    region: usize,
    msg: Vec<u8>,
    words: usize,
    nodes: usize,
    key: u64,
}

/// The GPU service's held call on a thread of its own: a job's message in, its answer (the GPU's µs, or why not) out.
struct GpuThread {
    jobs: std::sync::mpsc::Sender<Vec<u8>>,
    done: std::sync::mpsc::Receiver<Result<u64, String>>,
}

impl GpuThread {
    fn new(slot: u64) -> GpuThread {
        let (jobs, rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let (tx, done) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name(String::from("mahere-gpu"))
            .stack_size(1 << 16)
            .spawn(move || {
                for msg in rx {
                    let mut reply = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                    let n = crate::ferros::sys::call(slot, &msg, &mut reply);
                    let r = if n == crate::ferros::sys::ERR {
                        Err(String::from("no GPU service receiving"))
                    } else if reply[0] == 4 {
                        // ferros_sys::gpu::OUTSIDE: the kernel reached outside Mahere's own memory, and the GPU MMU stopped it in Mahere's own address space.
                        Err(format!("the map kernel reached outside Mahere's memory, at GPU address {:#X} (fault status {:#X})", u64::from_le_bytes(reply[5..13].try_into().unwrap()), u32::from_le_bytes(reply[13..17].try_into().unwrap())))
                    } else if reply[0] == 5 {
                        Err(String::from("a production build runs no unchecked GPU code"))
                    } else if reply[0] != 0 {
                        Err(format!("the GPU service answered {} to the map kernel", reply[0]))
                    } else {
                        Ok(u32::from_le_bytes([reply[1], reply[2], reply[3], reply[4]]) as u64)
                    };
                    if tx.send(r).is_err() {
                        return;
                    }
                }
            })
            .expect("the GPU thread");
        GpuThread { jobs, done }
    }

    fn submit(&self, msg: Vec<u8>) -> Result<(), String> {
        self.jobs.send(msg).map_err(|_| String::from("the GPU thread ended"))
    }

    fn wait(&self) -> Result<u64, String> {
        self.done.recv().map_err(|_| String::from("the GPU thread ended"))?
    }
}

/// Per report: frames, the GPU's own µs (as the service measures each job), the host's µs by stage, the kernel's size.
#[derive(Default)]
struct Stats {
    frames: u64,
    gpu_us: u64,
    gpu_max: u64,
    plan_us: u64,
    /// The page table (its squares, and the cells first drawn written into their slots), then the line, land and hypsometric tables.
    table_us: u64,
    luts_us: u64,
    overlay_us: u64,
    /// Recording the kernel, finding it among those compiled (and binding its params) or compiling it, writing its words; compiles and binds over the report.
    record_us: u64,
    kernel_us: u64,
    compile_us: u64,
    code_us: u64,
    compiled: u64,
    bound: u64,
    call_us: u64,
    /// Waiting for the GPU's last frame once this one was planned (pipelined; the GPU's share of a frame past the planning's).
    wait_us: u64,
    words: usize,
    nodes: usize,
    /// Overlay boxes summed into the arena, the panels and the marks painted again, and their µs over the report: the panel (painted, its box found), the marks, the sums.
    overlays: u64,
    /// The plan's µs: preparing the view, the plan itself, the residency's want (`MapCore::plan_us`).
    plan_parts: [u64; 3],
    /// Inside the plan (`ViewPlan::us`): the faces and the points, the desired set, the boxes and their stand-ins, the elevation range.
    plan_inner: [u64; 4],
    /// The last frame's faces in view, whether they were measured from the camera point, its squares and points.
    faces: usize,
    near: bool,
    squares: usize,
    samples: usize,
    panels: u64,
    marks: u64,
    overlay_parts: [u64; 3],
    /// Overlay pixels summed into the arena, all the sums together.
    summed: u64,
    /// Of the panel's µs, finding the box its pixels cover.
    panel_box_us: u64,
}

impl GpuMap {
    /// The GPU path when both grants are there: the service at `gpu_slot`, the arena at `arena_slot`. None (said) otherwise: the CPU raster then draws.
    pub fn open(gpu_slot: u64, arena_slot: u64, surface_slot: u64, second_slot: u64) -> Option<GpuMap> {
        if crate::ferros::sys::grant_count() <= arena_slot {
            println!("mahere: no GPU grants (slots {gpu_slot} gpu and {arena_slot} layers): the map draws on the CPU");
            return None;
        }
        let base = crate::ferros::sys::dma(arena_slot);
        if base == 0 || base == crate::ferros::sys::ERR {
            println!("mahere: no layer arena at slot {arena_slot}: the map draws on the CPU");
            return None;
        }
        // A kernel that says the block's size fences the apps' surfaces in for the GPU too (ferros 2026-11-11 on): the kernel then writes the surface, no copy.
        let sized = crate::ferros::sys::dma_size(arena_slot, ARENA_MOST as u64);
        let bytes = sized.map_or(ARENA_BEFORE, |n| n as usize);
        let surface = crate::ferros::sys::dma(surface_slot);
        let usable = |b: u64| b != 0 && b != crate::ferros::sys::ERR && b >= SURFACES_PA;
        let second = if crate::ferros::sys::grant_count() > second_slot { crate::ferros::sys::dma(second_slot) } else { 0 };
        let surfaces = (sized.is_some() && usable(surface)).then(|| {
            let other = if usable(second) { second } else { surface };
            [(surface, SURFACES_VA + (surface - SURFACES_PA)), (other, SURFACES_VA + (other - SURFACES_PA))]
        });
        let pipelined = surfaces.is_some_and(|s| s[0].0 != s[1].0);
        println!(
            "mahere: the map draws on the GPU (the service at slot {gpu_slot}, the layer arena at slot {arena_slot}, {} MiB; surfaces G#{surface:X} G#{second:X}), {}",
            bytes >> 20,
            if pipelined {
                "straight into two surfaces, a mailbox, the next frame planned while the GPU draws this one"
            } else if surfaces.is_some() {
                "straight into the surface"
            } else {
                "fluor's pixels in the arena, presented by the CPU"
            }
        );
        Some(GpuMap {
            gpu: GpuThread::new(gpu_slot),
            arena: base as *mut u32,
            bytes,
            surfaces,
            pipelined,
            lay: None,
            slots: Slots::new(0),
            style: [None; 2],
            panel_key: None,
            marks_key: None,
            panel_box: PixelBox::EMPTY,
            overlay_on: false,
            stale: [PixelBox::EMPTY; 2],
            ready: None,
            next: 0,
            shown: 0,
            drawn: false,
            baked: [false; 2],
            kernels: Vec::with_capacity(KERNELS),
            stats: Stats::default(),
            report_at: std::time::Instant::now(),
        })
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
        Some(unsafe { core::slice::from_raw_parts(self.arena.add((lay.regions[self.shown] + lay.frame_at) / 4), lay.w * lay.h) })
    }

    /// The buffer the last finished frame is in, when it is the mailbox's other one (fluor's `own_surface`): its `dma` base.
    pub fn own_surface(&self) -> Option<u64> {
        if !self.pipelined {
            return None;
        }
        self.surfaces.map(|s| s[self.shown].0)
    }

    /// The last finished frame was made with the display's bake (fluor's `own_baked`).
    pub fn baked(&self) -> bool {
        self.lay.is_some_and(|l| l.out == kernel::Out::Surface) && self.baked[self.shown]
    }

    /// A frame planned and not yet drawn (the last draw's, pipelined): the host should draw once more, so it reaches the glass when the map comes to rest.
    pub fn owes_frame(&self) -> bool {
        self.ready.is_some()
    }

    /// What a frame shows, as far as a frame planned again would differ: the view, the cells, the style and layers, the panel.
    fn view_key(app: &DeviceApp<NoGpu>, w: usize, h: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        let map = &app.map;
        let c = map.cam;
        let mut hs = std::collections::hash_map::DefaultHasher::new();
        (w, h, c.lat().to_bits(), c.lon().to_bits(), c.ppd.to_bits(), c.bearing().to_bits(), map.pool_version(), map.style_version(), map.layers().bits()).hash(&mut hs);
        (app.panel.is_open(), app.panel.ru().to_bits(), map.globe_loading().map(|d| d.to_bits()), map.gps.is_some()).hash(&mut hs);
        // The display's bake: a brightness change plans the frame again with the new one.
        fluor::host::ferros::bake().map(|b| b.generation).hash(&mut hs);
        hs.finish()
    }

    /// One frame of `app`'s map, `w` × `h`, drawn on the GPU into the arena's frame. Err: the service did not run it.
    /// Pipelined: the frame planned by the last draw goes to the GPU first (its buffer is free, the show of the frame before it having answered), this frame is planned into the other region meanwhile, and the GPU's frame is waited for: that one is the frame the host shows. When nothing changed since the frame waiting, it is only drawn.
    pub fn draw(&mut self, app: &mut DeviceApp<NoGpu>, w: usize, h: usize) -> Result<(), String> {
        if self.lay.is_none_or(|l| (l.w, l.h) != (w, h)) {
            let out = if self.surfaces.is_some() { kernel::Out::Surface } else { kernel::Out::Argb8 };
            let l = Layout::new(w, h, self.bytes, out).ok_or_else(|| format!("its block of the layer arena ({} MiB) leaves a {w} x {h} frame's two regions no room for four cells", self.bytes >> 20))?;
            println!("mahere: GPU layout for {w} x {h}: two frame regions of {} MiB, the overlay at {} MiB in each, the code at {} MiB, {} cell slots of 1 MiB from {} MiB", l.regions[1] >> 20, kernel::OVERLAY >> 20, l.code_at >> 20, l.slots, l.slots_at >> 20);
            self.slots = Slots::new(l.slots);
            self.lay = Some(l);
            // Both regions' overlays made whole once, from the panel and marks painted afresh.
            self.panel_key = None;
            self.marks_key = None;
            self.panel_box = PixelBox::EMPTY;
            self.stale = [PixelBox::screen(w, h); 2];
            self.style = [None; 2];
            self.ready = None;
            self.drawn = false;
        }
        let key = Self::view_key(app, w, h);
        let d = |a: std::time::Instant, b: std::time::Instant| (b - a).as_micros() as u64;
        let inflight = match self.ready.take() {
            Some(r) => {
                self.gpu.submit(r.msg.clone())?;
                if r.key == key {
                    let us = self.gpu.wait()?;
                    self.shown = r.region;
                    self.count(us, r.words, r.nodes);
                    return Ok(());
                }
                Some(r)
            }
            None => None,
        };
        let region = self.next;
        let ready = self.prepare(app, w, h, region, key)?;
        self.next ^= 1;
        let t = std::time::Instant::now();
        match inflight {
            Some(r) if self.pipelined => {
                let us = self.gpu.wait()?;
                self.shown = r.region;
                self.stats.wait_us += d(t, std::time::Instant::now());
                self.count(us, r.words, r.nodes);
                self.ready = Some(ready);
            }
            // The first frame of a motion: kept for the next draw, which the host makes at once (`owes_frame`); the glass keeps the last finished frame meanwhile.
            None if self.pipelined && self.drawn => self.ready = Some(ready),
            _ => {
                if let Some(r) = inflight {
                    self.gpu.wait()?;
                    self.shown = r.region;
                }
                let (words, nodes) = (ready.words, ready.nodes);
                self.gpu.submit(ready.msg)?;
                let us = self.gpu.wait()?;
                self.shown = region;
                self.stats.call_us += d(t, std::time::Instant::now());
                self.drawn = true;
                self.count(us, words, nodes);
            }
        }
        Ok(())
    }

    /// A frame drawn by the GPU: counted, and the report every ten seconds.
    fn count(&mut self, us: u64, words: usize, nodes: usize) {
        let s = &mut self.stats;
        s.frames += 1;
        s.gpu_us += us;
        s.gpu_max = s.gpu_max.max(us);
        s.words = words;
        s.nodes = nodes;
        if self.report_at.elapsed().as_secs() >= 10 {
            self.report_at = std::time::Instant::now();
            let s = &self.stats;
            let n = s.frames.max(1);
            let lay = self.lay.unwrap();
            println!(
                "mahere: GPU {} frames: the kernel {} us a frame on the GPU (longest {}), {} words from {} nodes, {} faces in view ({}), {} squares, {} points; host per frame: plan {} us (view {}, plan {} [faces and points {}, desired {}, boxes and stand-ins {}, range {}], want {}), page table {} us, tables {} us, overlay {} us, record {} us, the kernel {} us ({} compiled at {} us each, {} bound), its words {} us, the call {} us, the wait for the GPU's frame {} us; the overlay a frame: panel {} us, marks {} us, sum {} us; {} panels painted ({} us each, {} of it finding its box), {} marks ({} us each), {} boxes summed ({} pixels and {} us each); {} cell planes written so far, {} wanted with no slot free ({} slots)",
                s.frames, s.gpu_us / n, s.gpu_max, s.words, s.nodes, s.faces, if s.near { "near" } else { "far" }, s.squares, s.samples, s.plan_us / n, s.plan_parts[0] / n, s.plan_parts[1] / n, s.plan_inner[0] / n, s.plan_inner[1] / n, s.plan_inner[2] / n, s.plan_inner[3] / n, s.plan_parts[2] / n, s.table_us / n, s.luts_us / n, s.overlay_us / n, s.record_us / n, s.kernel_us / n, s.compiled, s.compile_us / s.compiled.max(1), s.bound, s.code_us / n, s.call_us / n, s.wait_us / n, s.overlay_parts[0] / n, s.overlay_parts[1] / n, s.overlay_parts[2] / n, s.panels, s.overlay_parts[0] / s.panels.max(1), s.panel_box_us / s.panels.max(1), s.marks, s.overlay_parts[1] / s.marks.max(1), s.overlays, s.summed / s.overlays.max(1), s.overlay_parts[2] / s.overlays.max(1), self.slots.written, self.slots.starved, lay.slots
            );
            self.stats = Stats::default();
        }
    }

    /// Frame planning into region `region`: the plan, its page table and cells, the tables, the overlay, the kernel recorded and its words in the region's code; the service's message for it.
    fn prepare(&mut self, app: &mut DeviceApp<NoGpu>, w: usize, h: usize, region: usize, key: u64) -> Result<Ready, String> {
        let lay = self.lay.unwrap();
        let t0 = std::time::Instant::now();
        let plan = app.map.plan_view(w, h);
        for (a, b) in self.stats.plan_parts.iter_mut().zip(app.map.plan_us) {
            *a += b;
        }
        for (a, b) in self.stats.plan_inner.iter_mut().zip(plan.us) {
            *a += b as u64;
        }
        let t1 = std::time::Instant::now();
        let version = app.map.pool_version();
        let arena = unsafe { core::slice::from_raw_parts_mut(self.arena, self.bytes / 4) };
        arena::view(&mut self.slots, app.map.pool(), &plan, version, &lay, region, arena)?;
        app.map.set_drawn_range(plan.range);
        let t2 = std::time::Instant::now();
        let ramp = self.style[region] != Some(app.map.style_version());
        self.style[region] = Some(app.map.style_version());
        arena::tables(app.map.luts(), &plan, ramp, &lay, region, arena);
        let t3 = std::time::Instant::now();
        self.overlay(app, w, h, region);
        let t4 = std::time::Instant::now();
        let ring = app.map.globe_ring(w, h).map(|g| kernel::Ring::from_globe(&g));
        let mut c = arena::consts(app.map.luts(), &plan, &lay, region, self.overlay_on, ring);
        // Into the surface, the display's bake last when it has one (fluor's ferros host asks it, op 65): the window is then shown baked and the display only dithers it (ferros COMPOSITOR.md "Baked").
        if lay.out == kernel::Out::Surface {
            c.bake = fluor::host::ferros::bake().map(|b| kernel::Bake { k: b.k, m: b.m, g: b.g });
        }
        self.baked[region] = c.bake.is_some();
        let rec = kernel::recording(&c);
        let t5 = std::time::Instant::now();
        self.kernel(rec);
        let t6 = std::time::Instant::now();
        let k = &self.kernels[0].1;
        if k.words.len() > arena::CODE_WORDS {
            return Err(format!("the map kernel is {} words, more than the GPU service's {}", k.words.len(), arena::CODE_WORDS));
        }
        let code_at = lay.regions[region] + lay.code_at;
        // SAFETY: as `arena`: the region's code, inside the block (`Layout::new`), read by the GPU only in the frame planned here.
        let code = unsafe { core::slice::from_raw_parts_mut(self.arena.add(code_at / 4), 2 * k.words.len()) };
        for (j, w) in k.words.iter().enumerate() {
            code[2 * j] = *w as u32;
            code[2 * j + 1] = (*w >> 32) as u32;
        }
        let frame = match self.surfaces {
            Some(s) => s[region].1,
            None => ARENA_VA + (lay.regions[region] + lay.frame_at) as u64,
        };
        let msg = Self::message(k, code_at, frame, (w * h).div_ceil(16) as u32);
        let (words, nodes) = (k.words.len(), k.kept);
        let t7 = std::time::Instant::now();
        let s = &mut self.stats;
        let d = |a: std::time::Instant, b: std::time::Instant| (b - a).as_micros() as u64;
        s.plan_us += d(t0, t1);
        s.table_us += d(t1, t2);
        s.luts_us += d(t2, t3);
        s.overlay_us += d(t3, t4);
        s.record_us += d(t4, t5);
        s.kernel_us += d(t5, t6);
        s.code_us += d(t6, t7);
        (s.faces, s.near, s.squares, s.samples) = (c.face_count, c.near, plan.roles[0].squares.len() + plan.roles[1].squares.len(), plan.samples);
        Ok(Ready { region, msg, words, nodes, key })
    }

    /// The compiled kernel for this frame's recording, at `kernels[0]`: the one of its key with this frame's params bound into it (valhall `Compiled::bind`: no compile), else compiled now and kept. The most recently used first; past `KERNELS` the least recently used goes (a frame's structure is its layers, one face or several or the fold, the ring and the overlay: a handful in a session).
    fn kernel(&mut self, rec: valhall::ssa::Recording) {
        let key = rec.key();
        match self.kernels.iter().position(|e| e.0 == key) {
            Some(j) => {
                let mut e = self.kernels.remove(j);
                e.1.bind(rec.params());
                self.kernels.insert(0, e);
                self.stats.bound += 1;
            }
            None => {
                let t = std::time::Instant::now();
                let k = rec.compile();
                self.stats.compile_us += t.elapsed().as_micros() as u64;
                self.stats.compiled += 1;
                self.kernels.truncate(KERNELS - 1);
                self.kernels.insert(0, (key, k));
            }
        }
    }

    /// The panel and the screen's marks (compass, pins, the GPS) as the overlay, a resident layer in each frame region: painted again only when what they show changed, and only the box that changed summed into the region's copy. The globe's loading ring is the kernel's own (`kernel::Ring`), so it is not here.
    fn overlay(&mut self, app: &mut DeviceApp<NoGpu>, w: usize, h: usize, region: usize) {
        use std::hash::{Hash, Hasher};
        let map = &app.map;
        let c = map.cam;
        let hash = |f: &dyn Fn(&mut std::collections::hash_map::DefaultHasher)| {
            let mut hs = std::collections::hash_map::DefaultHasher::new();
            f(&mut hs);
            hs.finish()
        };
        // What the marks show (`MapCore::overlay_marks`: the compass with its heading and zoom, the GPS, the pins), and what the panel shows (its rows, its readouts, the zoom and the contours' spacing); each painted again only when its own changed.
        let pinned = map.gps.is_some() || [map.pin_a, map.pin_b, map.nav_to, map.found_at].iter().any(|p| p.is_some());
        let marks_key = hash(&|hs| {
            // The bearing as the compass can show it (2^-8 radians turn its arrow's tip a tenth of a pixel): the two-finger solve moves it by less every frame, and each was a repaint.
            (w, h, (c.bearing() * (1u32 << 8) as f64).round() as i64, map.zoom_depth(), map.past_data(), map.layers().bits()).hash(hs);
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
        let t0 = std::time::Instant::now();
        // A change goes into both regions' stale boxes: this region's copy is brought up to date now, the other's when its next frame comes.
        let mut changed = PixelBox::EMPTY;
        if self.panel_key != Some(panel_key) {
            self.panel_key = Some(panel_key);
            app.paint_panel(w, h);
            let p = app.panel.pixels();
            let tb = std::time::Instant::now();
            let b = if p.len() == w * h { arena::panel_box(p, w) } else { PixelBox::EMPTY };
            self.stats.panel_box_us += tb.elapsed().as_micros() as u64;
            changed = changed.union(self.panel_box).union(b);
            self.panel_box = b;
            self.stats.panels += 1;
        }
        let t1 = std::time::Instant::now();
        if self.marks_key != Some(marks_key) {
            self.marks_key = Some(marks_key);
            changed = changed.union(app.map.overlay_marks(w, h, true, false).1);
            self.stats.marks += 1;
        }
        let t2 = std::time::Instant::now();
        let us = |a: std::time::Instant, b: std::time::Instant| (b - a).as_micros() as u64;
        self.stats.overlay_parts[0] += us(t0, t1);
        self.stats.overlay_parts[1] += us(t1, t2);
        for s in &mut self.stale {
            *s = s.union(changed);
        }
        self.overlay_on = !(self.panel_box.is_empty() && app.map.marks_box().is_empty());
        let b = std::mem::replace(&mut self.stale[region], PixelBox::EMPTY);
        if b.is_empty() {
            return;
        }
        // The panel's pixels with the marks under them (premultiplied RGBA bytes), straight into the arena over the stale box alone. The marks are a screen's (both keys are new together after a layout, so the marks are drawn before any sum).
        let marks = app.map.marks_canvas();
        let panel = app.panel.pixels();
        let panel = (panel.len() == w * h).then_some(panel);
        let at = self.lay.unwrap().regions[region] + kernel::OVERLAY;
        // SAFETY: as `arena`: the region's overlay, a screen's words inside the block (`Layout::new`); the GPU reads this region only in the frame planned here, run after this returns.
        let out = unsafe { core::slice::from_raw_parts_mut(self.arena.add(at / 4), w * h) };
        arena::overlay_sum(marks, panel, w, b, out);
        self.stats.overlay_parts[2] += us(t2, std::time::Instant::now());
        self.stats.overlays += 1;
        self.stats.summed += b.area() as u64;
    }

    /// The GPU service's message for the kernel over `groups` workgroups of 16: its words at the arena offset `code_at`, uniform 0 the frame and 1 the arena, then its constants.
    fn message(k: &valhall::ssa::Compiled, code_at: usize, frame: u64, groups: u32) -> Vec<u8> {
        let mut msg = Vec::with_capacity(18 + 8 * 64 + 16);
        msg.push(1u8);
        msg.extend_from_slice(&(0x1000u16 | if k.regs64 { 0x8000 } else { 0 }).to_le_bytes());
        msg.extend_from_slice(&(code_at as u32).to_le_bytes());
        msg.extend_from_slice(&(k.words.len() as u16).to_le_bytes());
        msg.extend_from_slice(&groups.to_le_bytes());
        msg.extend_from_slice(&1u32.to_le_bytes());
        let addrs = [frame, ARENA_VA];
        msg.push((addrs.len() + k.consts.len()) as u8);
        for a in addrs.iter().chain(k.consts.iter()) {
            msg.extend_from_slice(&a.to_le_bytes());
        }
        for name in [&b"mahere"[..], b"map"] {
            msg.push(name.len() as u8);
            msg.extend_from_slice(name);
        }
        msg
    }
}
