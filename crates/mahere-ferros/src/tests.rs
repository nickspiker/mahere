//! The GPU kernel against the CPU raster, on the host: the same cells, the same view, the arena built as the device builds it (`arena.rs`), the kernel's words run in valhall's emulator one pixel at a time, and every pixel compared with what the CPU's `render_frame` draws (ferros, owner 2026-11-11: "Just needs to look the same"; the emulator: "Prove on the host what you can").
//! The CPU writes the display's code (the BT.2020 matrix, Opsin's rail, √); the kernel writes VSF `g2t` codes for ferros's panel. The kernel's code is turned back into its light and through the CPU's own encode, so both are compared as the CPU's codes.

use crate::arena::{self, Layout, Slots};
use crate::kernel;
use mahere_engine::colour::Display;
use mahere_engine::ink::UserLayer;
use mahere_engine::viewplan::{ViewPlan, plan_view};
use mahere_engine::raster::{Contours, FrameLuts, LayerMask, Style, build_hypso_lut, render_frame, select_depth};
use mahere_engine::residency::{DemPacked, Entry, Pool, PRESENT_DEM, PRESENT_LAND, PRESENT_LINE, PRESENT_WATER};
use mahere_engine::Camera;
use mahere_tiles::{CellKey, ClassCell, CovCell, TRI};
use valhall::sim::{Memory, run, run_marked};

/// The kernel over `n` invocations as the device issues it: a warp (16 invocations, a workgroup) runs every instruction any of its invocations runs, a block skipped only where all sixteen skip it. The mean instructions a warp.
fn run_frame(k: &valhall::ssa::Compiled, uniforms: &[u64], n: usize, mem: &mut Memory) -> f64 {
    let mut total = 0usize;
    for w0 in (0..n.next_multiple_of(16)).step_by(16) {
        let mut ran = vec![false; k.words.len()];
        for g in w0..w0 + 16 {
            run_marked(&k.words, uniforms, [g as u32, 0], mem, &mut ran);
        }
        total += ran.iter().filter(|&&r| r).count();
    }
    total as f64 / (n.next_multiple_of(16) / 16) as f64
}

const VA: u64 = 0x8000_0000;
const BYTES: usize = 64 << 20;

/// A small hash for the cells' texels (deterministic, no generator crate).
fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^ (h >> 16)
}

/// A cell of every plane the kernel draws: terrain on a slope with a bumpy normal, line classes of every kind, land cover, water in patches.
fn cell(seed: u32) -> Entry {
    let mut dem = vec![0u64; TRI];
    let mut line = ClassCell::new_line();
    let mut land = ClassCell::new();
    let mut water = CovCell::new();
    for i in 0..TRI {
        let (tx, ty) = ((i >> 1) & 255, i >> 9);
        let h = mix(seed ^ i as u32);
        let eq = 26000 + tx as u64 * 3 + ty as u64 * 5;
        let nx = ((h & 255) as i32 - 128) * 40;
        let ny = (((h >> 8) & 255) as i32 - 128) * 40;
        let nz = ((32767f32 * 32767f32 - (nx * nx + ny * ny) as f32).sqrt()) as i32;
        dem[i] = eq | (nx as u16 as u64) << 16 | (ny as u16 as u64) << 32 | (nz as u16 as u64) << 48;
        let k = mix(h);
        if k & 7 == 0 {
            line.class[i] = (k >> 3) as u8 % 23;
            line.cov[i] = (k >> 8) as u8;
            line.mag[i] = (k >> 16) as u8;
        }
        land.class[i] = (k >> 24) as u8 % 15;
        land.cov[i] = (h >> 16) as u8;
        if (tx / 32 + ty / 32) % 3 == 0 {
            water.cov[i] = (h >> 24) as u8;
        }
    }
    Entry {
        present: PRESENT_DEM | PRESENT_LINE | PRESENT_LAND | PRESENT_WATER,
        elev_lo: 26000,
        elev_hi: 26000 + 255 * 8,
        dem: Some(DemPacked { texel: dem.into_boxed_slice() }),
        line: Some(line),
        land: Some(land),
        water: Some(water),
        layers: u8::MAX,
        exists: true,
        ..Entry::default()
    }
}

/// The frame's tables for a view: the plan's ball, the default style, a fixed sun and sky.
fn luts(plan: &ViewPlan, cam: &Camera, mask: LayerMask) -> FrameLuts {
    FrameLuts {
        pole: [plan.pole[0], plan.pole[1], plan.pole[2]],
        disk: [plan.globe[0], plan.globe[1], plan.globe[2]],
        hypso: build_hypso_lut(),
        style: Style::default(),
        sun: [0.3, 0.4, 0.866],
        mask: mask.effective(),
        dem_depth: plan.dem_depth,
        contours: Contours { interval: 2.0, index_every: 5, m_per_px: (111_320.0 / cam.ppd) as f32, lo_m: 0.0, hi_m: 4000.0 },
        light: mahere_engine::sh::Sh9::sun_and_sky(315.0, 40.0).quadratic((0.0, 1.0)),
        display: Display::default(),
        compressed: true,
    }
}

/// The frame both ways for `cam` over `pool` with `mask`; the CPU's codes and the kernel's (through the CPU's encode).
fn both(cam: &Camera, pool: &Pool, mask: LayerMask, w: usize, h: usize, out: kernel::Out) -> (Vec<u32>, Vec<u32>, usize) {
    let d = select_depth(cam.ppd, 0, 14);
    let user = UserLayer::default();
    let plan = plan_view(w, h, cam, pool, d, d);
    let luts = luts(&plan, cam, mask);
    let mut cpu = vec![0u32; w * h];
    render_frame(&mut cpu, w, h, cam, pool, &luts, d, d, &user);

    let lay = Layout::new(w, h, BYTES, out).unwrap();
    let mut words = vec![0u32; BYTES / 4];
    let mut slots = Slots::new(lay.slots);
    arena::view(&mut slots, pool, &plan, 1, &lay, 0, &mut words).unwrap();
    arena::tables(&luts, &plan, true, &lay, 0, &mut words);
    let c = arena::consts(&luts, &plan, &lay, 0, false, None);
    let k = kernel::record(&c);
    let imm = k.words.iter().filter(|w| (*w >> 48) & 0x1FF == 0x110).count();
    let loads = k.words.iter().filter(|w| (*w >> 48) & 0x1FF == 0x060).count();
    println!("kernel: {} words, {} constant moves, {} loads, regs64 {}, peak {}, {} uniform constants; kinds {:?}", k.words.len(), imm, loads, k.regs64, k.peak, k.consts.len(), &k.kinds[..k.kinds.len().min(14)]);
    // The surface (`Out::Surface`) after the arena in the emulated memory, a working value a pixel.
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.resize(BYTES + 8 * (w * h).next_multiple_of(16), 0);
    let mut mem = Memory { base: VA, bytes };
    let frame = if out == kernel::Out::Surface { VA + BYTES as u64 } else { VA + lay.frame_at as u64 };
    let mut uniforms = vec![frame, VA];
    uniforms.extend_from_slice(&k.consts);
    let issued = run_frame(&k, &uniforms, w * h, &mut mem);
    println!("issued: {issued:.1} instructions a warp of {} words", k.words.len());
    let display = Display::default();
    let gpu = (0..w * h)
        .map(|i| {
            let c = if out == kernel::Out::Surface {
                let (lo, hi) = (mem.read32(frame + 8 * i as u64), mem.read32(frame + 8 * i as u64 + 4));
                assert_eq!(hi >> 16, 0, "pixel {i} not opaque or past white's range: G#{hi:08X}{lo:08X}");
                let light = |v: u32| (v as f32 / 16384.0) * (v as f32 / 16384.0);
                display.encode([light(lo & 0xFFFF), light(lo >> 16), light(hi & 0xFFFF)], true)
            } else {
                let p = mem.read32(frame + 4 * i as u64);
                assert_eq!(p >> 24, 255, "pixel {i} not opaque: G#{p:08X}");
                let light = |sh: u32| {
                    let code = 255 - ((p >> sh) & 255);
                    let s = (code as f32 / 256.0).atanh();
                    s * s
                };
                display.encode([light(16), light(8), light(0)], true)
            };
            (c[0] as u32) << 16 | (c[1] as u32) << 8 | c[2] as u32
        })
        .collect();
    (cpu, gpu, k.words.len())
}

/// How far apart two frames are: the largest channel difference, and how many pixels differ by more than `tol` in any channel.
fn apart(a: &[u32], b: &[u32], tol: i32) -> (i32, usize) {
    let mut worst = 0;
    let mut over = 0;
    for (&p, &q) in a.iter().zip(b) {
        let d = [16, 8, 0].iter().map(|&s| (((p >> s) & 255) as i32 - ((q >> s) & 255) as i32).abs()).max().unwrap();
        worst = worst.max(d);
        if d > tol {
            over += 1;
        }
    }
    (worst, over)
}

/// Every pixel's depth-30 (u, v) from the kernel's projection (its words in the emulator) against the plan's own (f64) and the encoder's diamond: the most units apart, and the pixels whose (u, v) is further than `tol` units from the plan's.
fn located(cam: &Camera, pool: &Pool, w: usize, h: usize) -> (i64, usize, usize, usize, usize, bool) {
    let d = select_depth(cam.ppd, 0, 14);
    let plan = plan_view(w, h, cam, pool, d, d);
    let lay = Layout::new(w, h, BYTES, kernel::Out::Surface).unwrap();
    let mut words = vec![0u32; BYTES / 4];
    arena::view(&mut Slots::new(lay.slots), pool, &plan, 1, &lay, 0, &mut words).unwrap();
    let luts = luts(&plan, cam, LayerMask::default());
    let c = arena::consts(&luts, &plan, &lay, 0, false, None);
    let k = kernel::record_locate(&c);
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.resize(BYTES + 32 * (w * h).next_multiple_of(16), 0);
    let mut mem = Memory { base: VA, bytes };
    let frame = VA + BYTES as u64;
    let mut uniforms = vec![frame, VA];
    uniforms.extend_from_slice(&k.consts);
    for g in 0..(w * h).next_multiple_of(16) {
        run(&k.words, &uniforms, [g as u32, 0], &mut mem);
    }
    let (mut worst, mut far, mut wrong, mut seams) = (0i64, 0usize, 0usize, 0usize);
    // The units a pixel spans here: the kernel's (u, v) need only land within a small part of one.
    let tol = (((1u64 << 30) as f64 / (cam.radius() * 1.1071)) as i64 >> 4).max(1);
    for py in 0..h {
        for px in 0..w {
            let (sx, sy) = (px as f64, py as f64);
            if !cam.on_globe(sx, sy, w, h) {
                continue;
            }
            let p = plan.view_dir(sx, sy);
            let (mut f, mut u, mut v) = plan.locate(p);
            let i = (py * w + px) as u64;
            let at = |k: u64| mem.read32(frame + 32 * i + 4 * k);
            let (gu, gv) = (at(0), at(1));
            // The diamond the kernel took, from its face's seed; on a seam it can be the other side's (f32 against f64, within rounding of both): there the plan's (u, v) on that diamond's nearer face is the one to compare with.
            let seed = |d: u8| (d as u32).wrapping_mul(0x27D4_EB2F) ^ (plan.dem_depth as u32).wrapping_mul(0xC2B2_AE3D);
            if seed(plan.faces[f].diamond) != at(4) {
                seams += 1;
                let dot = |a: [f64; 3]| a[0] * p[0] + a[1] * p[1] + a[2] * p[2];
                f = (0..plan.faces.len()).filter(|&g| seed(plan.faces[g].diamond) == at(4)).max_by(|&a, &b| dot(plan.faces[a].centroid).total_cmp(&dot(plan.faces[b].centroid))).expect("the kernel's face is one of the plan's");
                (u, v) = plan.faces[f].uv(p);
            }
            let e = (gu as i64 - u as i64).abs().max((gv as i64 - v as i64).abs());
            worst = worst.max(e);
            far += (e > tol) as usize;
            // The terrain's square and texel for the kernel's own (u, v), as the host's page table and the CPU's texel index have them.
            let sq = plan.roles[0].square(plan.faces[f].diamond, gu, gv).map_or(0, |s| words[kernel::SQUARES / 4 + s]);
            let ts = sq >> kernel::SQUARE_SHIFT;
            let base = sq & ((1 << kernel::SQUARE_SHIFT) - 1);
            let m = (1u32 << ts) - 1;
            let ti = (((gv >> ts) & 255) << 9) | (((gu >> ts) & 255) << 1) | (((gu & m) + (gv & m)) >> ts) & 1;
            if at(2) != sq || at(3) != base + 2 * ti {
                if wrong < 4 {
                    eprintln!("({px}, {py}): the kernel's square G#{:X} texel G#{:X}, the host's G#{sq:X} G#{:X}", at(2), at(3), base + 2 * ti);
                }
                wrong += 1;
            }
        }
    }
    (worst, far, wrong, seams, c.face_count, c.near)
}

#[test]
fn the_kernel_locates_every_pixel_as_the_plan_does() {
    let (w, h) = (96usize, 80usize);
    // A town; a seam at 72° east; the north pole, five faces; a deep zoom on a corner of the icosahedron (26.57° N 0° E, six faces); the globe with its limb, from two sides.
    for (lat, lon, ppd, bearing) in [(46.2, -121.5, 6000.0, 0.7), (40.0, 72.0, 600.0, 0.2), (90.0, 0.0, 3000.0, 0.0), (26.565, 0.0, (1 << 20) as f64, 1.1), (10.0, 30.0, 0.4, 0.0), (-30.0, 150.0, 0.6, 2.0)] {
        let cam = Camera::new(lat, lon, ppd, bearing);
        // Every cell the view wants, but every third at the view's depth, which its parent stands in for.
        let d = select_depth(ppd, 0, 14);
        let mut pool = Pool::default();
        let mut want = plan_view(w, h, &cam, &pool, d, d).want;
        want.sort();
        for (n, k) in want.iter().enumerate() {
            if k.depth < d || n % 3 != 0 {
                pool.map.insert(*k, cell(n as u32));
            }
        }
        let (worst, far, wrong, seams, faces, near) = located(&cam, &pool, w, h);
        println!("view ({lat}, {lon}, ppd {ppd}): {faces} faces, near {near}: the kernel's (u, v) at most {worst} units from the plan's, {far} pixels past a sixteenth of a pixel, {seams} on a seam took the other diamond; {wrong} pixels' squares or texels not the host's");
        assert!(far * 1000 <= w * h, "view ({lat}, {lon}, {ppd}): {far} pixels located apart from the plan (at most {worst} units)");
        assert!(seams <= w, "view ({lat}, {lon}, {ppd}): {seams} pixels took another diamond than the plan's, more than a seam's line");
        assert_eq!(wrong, 0,"view ({lat}, {lon}, {ppd}): {wrong} pixels read another square or texel than the host's page table names");
    }
}

#[test]
fn the_kernel_draws_what_the_cpu_draws() {
    let (w, h) = (96usize, 80usize);
    // Zoomed to depth 11 over the Cascades, turned, so blocks interpolate with a twist and the frame spans several cells.
    let cam = Camera::new(46.2, -121.5, 6000.0, 0.7);
    let d = select_depth(cam.ppd, 0, 14);
    let mut pool = Pool::default();
    let plan = plan_view(w, h, &cam, &pool, d, d);
    let mut wanted: Vec<CellKey> = plan.want.iter().copied().filter(|k| k.depth == d).collect();
    wanted.sort();
    assert!(wanted.len() >= 2, "the view should span cells: {}", wanted.len());
    // Every wanted cell but the last, which is left to its parent (one depth up) standing in, and that parent's sibling cells are not there: the dark mesh where neither is.
    for (n, k) in wanted.iter().enumerate() {
        if n + 1 < wanted.len() {
            pool.map.insert(*k, cell(n as u32 * 7919));
        }
    }
    let last = wanted[wanted.len() - 1];
    pool.map.insert(CellKey { depth: d - 1, prefix: last.prefix >> 2 }, cell(4242));
    for (name, mask, out) in [
        ("default, into the surface", LayerMask::default(), kernel::Out::Surface),
        ("default, fluor's pixels", LayerMask::default(), kernel::Out::Argb8),
        ("no land: the hypsometric tint", LayerMask { land: false, ..LayerMask::default() }, kernel::Out::Surface),
        ("no contours, no water", LayerMask { contours: false, water: false, ..LayerMask::default() }, kernel::Out::Surface),
    ] {
        let (cpu, gpu, words) = both(&cam, &pool, mask, w, h, out);
        let (worst, over) = apart(&cpu, &gpu, 4);
        let mut distinct: Vec<u32> = cpu.clone();
        distinct.sort();
        distinct.dedup();
        let exact = cpu.iter().zip(&gpu).filter(|(a, b)| a == b).count();
        println!("{name}: {words} words; {} colours in the CPU's frame, {exact} pixels the same code; the largest difference {worst} codes, {over} of {} pixels past 4", distinct.len(), w * h);
        assert!(over * 100 <= w * h, "{name}: {over} pixels differ by more than 4 codes (largest {worst})");
    }
}

#[test]
fn the_kernel_draws_what_the_cpu_draws_where_faces_meet() {
    let (w, h) = (96usize, 80usize);
    // Across the seam at 72° east, and the north pole where five faces meet: each pixel's face read from the faces' rows. Every cell wanted, but every third at the view's depth, which its parent stands in for.
    for (lat, lon, ppd, bearing) in [(40.0, 72.0, 600.0, 0.2), (90.0, 0.0, 3000.0, 0.0)] {
        let cam = Camera::new(lat, lon, ppd, bearing);
        let d = select_depth(ppd, 0, 14);
        let mut pool = Pool::default();
        let mut want = plan_view(w, h, &cam, &pool, d, d).want;
        want.sort();
        for (n, k) in want.iter().enumerate() {
            if k.depth < d || n % 3 != 0 {
                pool.map.insert(*k, cell(n as u32 * 7919));
            }
        }
        let (cpu, gpu, words) = both(&cam, &pool, LayerMask::default(), w, h, kernel::Out::Surface);
        // The CPU's tiles whose corners lie in different diamonds: it draws those by its exact fallback at each pixel's centre (raster.rs `render_block_exact`), every other tile at the pixel's corner as the kernel does everywhere, so there the two sample half a pixel apart over these cells' noise of normals. Compared outside them; the seams themselves are checked exactly by `the_kernel_locates_every_pixel_as_the_plan_does`.
        let plan = plan_view(w, h, &cam, &pool, d, d);
        let diamond = |x: usize, y: usize| plan.faces[plan.locate(plan.view_dir(x as f64, y as f64)).0].diamond;
        let straddles = |x: usize, y: usize| {
            let (x0, y0) = (x & !31, y & !31);
            let d0 = diamond(x0, y0);
            [(x0 + 32, y0), (x0, y0 + 32), (x0 + 32, y0 + 32)].iter().any(|&(a, b)| diamond(a, b) != d0)
        };
        let keep: Vec<usize> = (0..w * h).filter(|&i| !straddles(i % w, i / w)).collect();
        let (cpu, gpu): (Vec<u32>, Vec<u32>) = keep.iter().map(|&i| (cpu[i], gpu[i])).unzip();
        let (worst, over) = apart(&cpu, &gpu, 4);
        let exact = cpu.iter().zip(&gpu).filter(|(a, b)| a == b).count();
        println!("({lat}, {lon}, ppd {ppd}): {words} words; {} pixels outside the CPU's straddling tiles, {exact} the same code; the largest difference {worst} codes, {over} past 4", keep.len());
        assert!(keep.len() >= 1 << 10, "({lat}, {lon}): too few pixels outside the straddling tiles to compare");
        assert!(over * 100 <= keep.len(), "({lat}, {lon}): {over} pixels differ by more than 4 codes (largest {worst})");
    }
}

/// The kernel's words over the arena `words`, its working values a pixel (`Out::Surface`).
fn surface(k: &valhall::ssa::Compiled, words: &[u32], w: usize, h: usize) -> Vec<[u32; 3]> {
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.resize(BYTES + 8 * (w * h).next_multiple_of(16), 0);
    let mut mem = Memory { base: VA, bytes };
    let frame = VA + BYTES as u64;
    let mut uniforms = vec![frame, VA];
    uniforms.extend_from_slice(&k.consts);
    let issued = run_frame(k, &uniforms, w * h, &mut mem);
    println!("surface: {} words, {issued:.1} issued a warp", k.words.len());
    (0..w * h)
        .map(|i| {
            let (lo, hi) = (mem.read32(frame + 8 * i as u64), mem.read32(frame + 8 * i as u64 + 4));
            [lo & 0xFFFF, lo >> 16, hi & 0xFFFF]
        })
        .collect()
}

/// The globe's loading ring drawn by the kernel from its numbers looks as the CPU's strokes of it do, summed into the overlay as every mark was: the same inside the ring, apart only along its anti-aliased edges (the kernel's edge smoothstepped and premultiplied in working values, the strokes' linear in codes), the same on the map beside it; at every sweep, short and past half a turn.
#[test]
fn the_kernel_draws_the_loading_ring_as_the_marks_did() {
    use mahere_engine::{GlobeRing, PixelBox, draw_segment};
    let (w, h) = (160usize, 120usize);
    let cam = Camera::new(10.0, 30.0, 0.5, 0.0);
    let pool = Pool::default();
    let d = select_depth(cam.ppd, 0, 14);
    let plan = plan_view(w, h, &cam, &pool, d, d);
    let luts = luts(&plan, &cam, LayerMask::default());
    let lay = Layout::new(w, h, BYTES, kernel::Out::Surface).unwrap();
    let mut words = vec![0u32; BYTES / 4];
    arena::view(&mut Slots::new(lay.slots), &pool, &plan, 1, &lay, 0, &mut words).unwrap();
    arena::tables(&luts, &plan, true, &lay, 0, &mut words);
    let (cx, cy) = Camera::centre(w, h);
    for turn in [0.01f32, 0.3, 0.5, 0.7, 1.0] {
        let g = GlobeRing { centre: [cx as f32, cy as f32], radius: cam.radius() as f32 + 0.015 * h as f32, half_w: 1.5, sweep: turn * core::f32::consts::TAU, rgb: [232, 232, 236] };
        // The CPU's strokes (`MapCore::draw_globe_ring`), summed into the overlay.
        let mut marks = vec![0u32; w * h];
        let steps = ((g.radius * g.sweep) as usize / 3).clamp(8, 1440);
        let at = |a: f32| (g.centre[0] + g.radius * a.sin(), g.centre[1] - g.radius * a.cos());
        let mut prev = at(0.0);
        let mut b = PixelBox::EMPTY;
        for i in 1..=steps {
            let p = at(i as f32 / steps as f32 * g.sweep);
            b = b.union(draw_segment(&mut marks, w, h, prev.0, prev.1, p.0, p.1, g.half_w, g.rgb));
            prev = p;
        }
        let mut cpu_words = words.clone();
        arena::overlay_sum(&marks, None, w, b, &mut cpu_words[kernel::OVERLAY / 4..kernel::OVERLAY / 4 + w * h]);
        let cpu = surface(&kernel::record(&arena::consts(&luts, &plan, &lay, 0, true, None)), &cpu_words, w, h);
        let k = kernel::record(&arena::consts(&luts, &plan, &lay, 0, false, Some(kernel::Ring::from_globe(&g))));
        let gpu = surface(&k, &words, w, h);
        let bare = surface(&kernel::record(&arena::consts(&luts, &plan, &lay, 0, false, None)), &words, w, h);
        // Working values (white 2^14): 2^6 is about a quarter of an 8-bit code near white.
        let apart = |i: usize| (0..3).map(|c| (cpu[i][c] as i32 - gpu[i][c] as i32).abs()).max().unwrap();
        let ring = (0..w * h).filter(|&i| marks[i] != 0).count();
        let inside = (0..w * h).filter(|&i| marks[i] >> 16 == 232).count();
        let inside_apart = (0..w * h).filter(|&i| marks[i] >> 16 == 232 && apart(i) > 1 << 6).count();
        let beside = (0..w * h).filter(|&i| marks[i] == 0).map(|i| (gpu[i][0] as i32 - bare[i][0] as i32).abs()).max().unwrap();
        // On the edge: the strokes overlap (3 px apart, 4 wide), each lerping its coverage over the last's, so their edge is bolder than one stroke's; the kernel's is one stroke's with its ramp smoothstepped. Each edge pixel within half of white of theirs (the most where the ring is a dot, eight strokes over two pixels), and the ring's ink in all within a quarter of theirs.
        let edge = (0..w * h).filter(|&i| marks[i] != 0).map(apart).max().unwrap();
        let ink = |f: &[[u32; 3]]| (0..w * h).map(|i| f[i][0] as i64 - bare[i][0] as i64).sum::<i64>();
        let (ink_cpu, ink_gpu) = (ink(&cpu), ink(&gpu));
        println!("sweep {turn} of a turn: {} words; {ring} pixels of ring ({inside} fully), {inside_apart} of them apart inside, the edge at most {edge} apart, beside it at most {beside}; the ring's ink {ink_gpu}, the strokes' {ink_cpu}", k.words.len());
        assert!(inside > 0 && inside_apart == 0, "sweep {turn}: {inside_apart} of {inside} pixels inside the ring apart");
        assert!(edge <= 1 << 13, "sweep {turn}: an edge pixel {edge} apart");
        assert!(beside <= 1 << 6, "sweep {turn}: a pixel beside the strokes {beside} apart");
        assert!((ink_gpu - ink_cpu).abs() * 4 <= ink_cpu, "sweep {turn}: the ring's ink {ink_gpu}, the strokes' {ink_cpu}");
    }
}

/// At the globe the faces found by fold draw the frame the search over them drew: the same face for every pixel but a few on a seam (f32 either way), so the same pixels, in fewer words.
#[test]
fn the_fold_draws_what_the_search_drew() {
    let (w, h) = (96usize, 80usize);
    for (lat, lon, ppd, bearing) in [(10.0, 30.0, 0.4, 0.0), (-30.0, 150.0, 0.6, 2.0), (60.0, -100.0, 0.5, 1.0)] {
        let cam = Camera::new(lat, lon, ppd, bearing);
        let d = select_depth(ppd, 0, 14);
        let mut pool = Pool::default();
        let mut want = plan_view(w, h, &cam, &pool, d, d).want;
        want.sort();
        for (n, k) in want.iter().enumerate() {
            if k.depth < d || n % 3 != 0 {
                pool.map.insert(*k, cell(n as u32 * 7919));
            }
        }
        let plan = plan_view(w, h, &cam, &pool, d, d);
        let luts = luts(&plan, &cam, LayerMask::default());
        let lay = Layout::new(w, h, BYTES, kernel::Out::Surface).unwrap();
        let mut words = vec![0u32; BYTES / 4];
        arena::view(&mut Slots::new(lay.slots), &pool, &plan, 1, &lay, 0, &mut words).unwrap();
        arena::tables(&luts, &plan, true, &lay, 0, &mut words);
        let c = arena::consts(&luts, &plan, &lay, 0, false, None);
        assert!(c.fold.is_some(), "({lat}, {lon}): {} faces, near {}: no fold", c.face_count, c.near);
        let folded = kernel::record(&c);
        let by_fold = surface(&folded, &words, w, h);
        // The search: its rows, a face each, where the fold's were.
        let (faces, n, _, _) = arena::faces(&plan, &lay, 0);
        for (j, f) in faces[..n].iter().enumerate() {
            words[kernel::FACES / 4 + j * kernel::FACE_WORDS..kernel::FACES / 4 + (j + 1) * kernel::FACE_WORDS].copy_from_slice(&f.row);
        }
        let searched = kernel::record(&kernel::Consts { fold: None, ..c });
        let by_search = surface(&searched, &words, w, h);
        let apart = (0..w * h).filter(|&i| by_fold[i] != by_search[i]).count();
        println!("({lat}, {lon}, ppd {ppd}): {n} faces, the fold {} words, the search {}; {apart} of {} pixels apart", folded.words.len(), searched.words.len(), w * h);
        assert!(apart <= w, "({lat}, {lon}): {apart} pixels apart, more than a seam's line");
        assert!(folded.words.len() < searched.words.len());
    }
}

/// A frame of a structure met before takes the kernel compiled for it with its own params bound in (gpu.rs's cache): the same words and uniforms as compiling it afresh, for a pan and a pinch of the same view; and a frame with another structure (the ring on) has another key.
#[test]
fn a_bound_kernel_is_the_compiled_one() {
    let (w, h) = (96usize, 80usize);
    let frame = |lat: f64, lon: f64, ppd: f64, ring: bool| {
        let cam = Camera::new(lat, lon, ppd, 0.7);
        let d = select_depth(ppd, 0, 14);
        let pool = Pool::default();
        let plan = plan_view(w, h, &cam, &pool, d, d);
        let luts = luts(&plan, &cam, LayerMask::default());
        let lay = Layout::new(w, h, BYTES, kernel::Out::Surface).unwrap();
        let ring = ring.then(|| kernel::Ring { centre: [48., 40.], radius: 30., half_w: 1.5, end: [0.5, 0.8], long: false, rgb: [232, 232, 236] });
        kernel::recording(&arena::consts(&luts, &plan, &lay, 0, false, ring))
    };
    let first = frame(46.2, -121.5, 6000.0, false);
    let key = first.key();
    let mut k = first.compile();
    for (lat, lon, ppd) in [(46.21, -121.48, 6100.0), (46.3, -121.2, 5400.0)] {
        let next = frame(lat, lon, ppd, false);
        assert_eq!(next.key(), key, "a pan to ({lat}, {lon}) changed the kernel's structure");
        let params = next.params().to_vec();
        let fresh = next.compile();
        k.bind(&params);
        assert_eq!(k.words, fresh.words, "({lat}, {lon}): the bound words");
        assert_eq!(k.consts, fresh.consts, "({lat}, {lon}): the bound uniforms");
    }
    assert_ne!(frame(46.2, -121.5, 6000.0, true).key(), key);
    println!("bound: {} words, {} params", k.words.len(), k.params.len());
}

/// What a warp issues (`run_frame`) at a town's scale with every cell there and with none (the loading mesh, as the 8a draws without the network), and at the globe with none: the measure the kernel's branches are judged by on the host, the device's time a frame following it (ferros GPU-KISTS.md, the fifth pass).
#[test]
fn issued_a_warp() {
    let (w, h) = (96usize, 80usize);
    for (name, lat, lon, ppd, full) in [("town, every cell", 46.2, -121.5, 6000.0, true), ("town, no cells", 46.2, -121.5, 6000.0, false), ("globe, no cells", 10.0, 30.0, 0.4, false)] {
        let cam = Camera::new(lat, lon, ppd, 0.7);
        let d = select_depth(ppd, 0, 14);
        let mut pool = Pool::default();
        if full {
            for (n, k) in plan_view(w, h, &cam, &pool, d, d).want.iter().enumerate() {
                pool.map.insert(*k, cell(n as u32 * 7919));
            }
        }
        let plan = plan_view(w, h, &cam, &pool, d, d);
        let luts = luts(&plan, &cam, LayerMask::default());
        let lay = Layout::new(w, h, BYTES, kernel::Out::Surface).unwrap();
        let mut words = vec![0u32; BYTES / 4];
        arena::view(&mut Slots::new(lay.slots), &pool, &plan, 1, &lay, 0, &mut words).unwrap();
        arena::tables(&luts, &plan, true, &lay, 0, &mut words);
        print!("{name}: ");
        surface(&kernel::record(&arena::consts(&luts, &plan, &lay, 0, false, None)), &words, w, h);
    }
}

/// The display's bake as the kernel's last step (ferros COMPOSITOR.md "Baked"): every pixel of a town's view, made baked, is the f64 bake of the same pixel made unbaked (l = tanh²(k·v), p = m·l, p^g × 65472, held at the field's ends), within a sixteenth of a code of the 8-bit link and a 512th of the code, husky's ODT at a white about a ninth of the E peak.
#[test]
fn the_kernel_bakes_as_the_display_does() {
    let (w, h) = (96usize, 80usize);
    let cam = Camera::new(46.2, -121.5, 6000.0, 0.7);
    let d = select_depth(6000.0, 0, 14);
    let mut pool = Pool::default();
    for (n, k) in plan_view(w, h, &cam, &pool, d, d).want.iter().enumerate() {
        pool.map.insert(*k, cell(n as u32 * 7919));
    }
    let plan = plan_view(w, h, &cam, &pool, d, d);
    let luts = luts(&plan, &cam, LayerMask::default());
    let lay = Layout::new(w, h, BYTES, kernel::Out::Surface).unwrap();
    let mut words = vec![0u32; BYTES / 4];
    arena::view(&mut Slots::new(lay.slots), &pool, &plan, 1, &lay, 0, &mut words).unwrap();
    arena::tables(&luts, &plan, true, &lay, 0, &mut words);
    let plain = arena::consts(&luts, &plan, &lay, 0, false, None);
    let bake = kernel::Bake { k: 0.113f32.sqrt().atanh(), m: [[1.359550, -0.372458, 0.012909], [-0.098444, 0.863965, -0.026116], [0.001826, -0.001101, 0.816481]], g: 1. / 2.19 };
    let working = surface(&kernel::record(&plain), &words, w, h);
    let baked = surface(&kernel::record(&kernel::Consts { bake: Some(bake), ..plain }), &words, w, h);
    let mut worst = 0f64;
    for (v, b) in working.iter().zip(&baked) {
        let l = v.map(|x| ((bake.k as f64) * (x as f64 / 16384.)).tanh().powi(2));
        for r in 0..3 {
            let p: f64 = (0..3).map(|j| bake.m[r][j] as f64 * l[j]).sum();
            let want = if p > 0. { (p.powf(bake.g as f64) * 65472.).min(65535.) } else { 0. };
            let d = (b[r] as f64 - want).abs();
            worst = worst.max(d);
            assert!(d <= 16. + want / 512., "working {v:?} channel {r}: {} against {want:.1}", b[r]);
        }
    }
    println!("baked against the f64 bake: the largest difference {worst:.1} of 65472");
}

/// The eviction race of 2026-11-12 (the 8a, early in a session: "no entry found for key" at the slot write): a pan over a town, a budget's worth of cells and none, every wanted cell arriving after each frame, and each frame written into the arena as the device writes it, the regions alternating. Every cell a plan names is still in the pool when its frame is written: the want lets go of cells out of view (the page table's margin among them) and over the budget before the stand-ins are found (`MapCore::plan_view`).
#[test]
fn a_planned_cell_is_in_the_pool_when_its_frame_is_written() {
    use mahere_engine::MapCore;
    use mahere_engine::residency::{CellStore, Fetch};
    use std::sync::Arc;
    struct Empty;
    impl CellStore for Empty {
        fn get(&self, _key: CellKey) -> Fetch {
            Fetch::Absent
        }
    }
    let (w, h) = (96usize, 80usize);
    let lay = Layout::new(w, h, BYTES, kernel::Out::Argb8).unwrap();
    let mut arena = vec![0u32; BYTES / 4];
    // Unbounded, and a budget of a dozen cells (each about 2 MiB here: the terrain packed, three vector planes).
    for budget in [usize::MAX, 24 << 20] {
        let mut map = MapCore::new(Arc::new(Empty), Camera::new(46.2, -121.5, 6000.0, 0.7));
        map.set_pool_budget(budget);
        let mut slots = Slots::new(lay.slots);
        let (mut n, mut written) = (0u32, 0u64);
        for step in 0..32 {
            map.cam = Camera::new(46.2 + step as f64 * 0.004, -121.5 + step as f64 * 0.006, 6000.0, 0.7);
            map.camera_moved(w, h);
            let plan = map.plan_view(w, h);
            for t in &plan.roles {
                for r in &t.refs {
                    assert!(map.pool().map.contains_key(&r.key), "budget {budget}, step {step}: the plan names {:?}, gone from the pool", r.key);
                }
            }
            arena::view(&mut slots, map.pool(), &plan, map.pool_version(), &lay, step & 1, &mut arena).unwrap();
            written = slots.written;
            for k in plan.want.clone() {
                if !map.pool().map.contains_key(&k) {
                    n += 1;
                    map.pool_mut().map.insert(k, cell(n));
                }
            }
        }
        assert!(written > 0, "budget {budget}: no cell was ever written");
    }
}

/// A small block of the arena (2026-11-12, the 8a: 32 MiB, the circle black; then 32 and 64 MiB, the kists started first holding the rest): at the 8a's 1080 x 2400 a frame region is 12 MiB (19 before the overlay moved to 1 MiB), so 64 MiB lays out two with 40 cell slots (26 before), 32 MiB one (two would leave 8 slots, fewer than the globe's base set takes) with 20 (13 before), and 8 MiB none (the CPU raster draws); the Pro's 1344 x 2992 in 32 MiB one region too. 128 MiB, the whole arena, two.
#[test]
fn a_small_block_lays_out_one_region_or_none() {
    let out = kernel::Out::Surface;
    let two = Layout::new(1080, 2400, 128 << 20, out).unwrap();
    assert!(two.two() && two.regions == [0, 12 << 20] && two.slots == 104, "{two:?}");
    let half = Layout::new(1080, 2400, 64 << 20, out).unwrap();
    assert!(half.two() && half.slots == 40, "{half:?}");
    let one = Layout::new(1080, 2400, 32 << 20, out).unwrap();
    assert!(!one.two() && one.regions == [0, 0] && one.slots_at == 12 << 20 && one.slots == arena::GLOBE_SLOTS, "{one:?}");
    let pro = Layout::new(1344, 2992, 32 << 20, out).unwrap();
    assert!(!pro.two() && pro.slots >= arena::MIN_SLOTS, "{pro:?}");
    assert!(Layout::new(1080, 2400, 8 << 20, out).is_none());
    // The kernel's frame in the arena too (no surface fenced in): a region of 22 MiB, so 24 MiB has no room for the cells' slots beside one.
    assert!(Layout::new(1080, 2400, 24 << 20, kernel::Out::Argb8).is_none());
    assert!(!Layout::new(1080, 2400, 48 << 20, kernel::Out::Argb8).unwrap().two());
}

/// The globe in a small block (the 8a, 2026-11-12: 64 MiB at 1080 x 2400 held 26 slots, the globe's plan naming hundreds of squares' cells, every frame the first 26 planes placed and the rest the loading mesh; here 32 MiB, 20 slots): with every cell the plan wants and the globe's depth-0 cells in the pool, the cells are held to a depth that fits, and from the second frame on (the first frame's planes are still the last's) every square holding a point of the screen names a slot.
#[test]
fn the_globe_fits_a_small_block() {
    use mahere_engine::MapCore;
    use mahere_engine::residency::{CellStore, Fetch};
    use std::sync::Arc;
    struct Empty;
    impl CellStore for Empty {
        fn get(&self, _key: CellKey) -> Fetch {
            Fetch::Absent
        }
    }
    let (w, h) = (1080usize, 2400usize);
    let bytes = 32 << 20;
    let lay = Layout::new(w, h, bytes, kernel::Out::Surface).unwrap();
    assert_eq!(lay.slots, 20);
    let mut arena = vec![0u32; bytes / 4];
    let mut map = MapCore::new(Arc::new(Empty), Camera::new(20.0, -150.0, 9.0, 0.0));
    map.camera_moved(w, h);
    let mut n = 0u32;
    for d in 0..10u8 {
        n += 1;
        map.pool_mut().map.insert(CellKey::from_grid(d, 0, 0, 0), cell(n));
    }
    let plan = map.plan_view(w, h);
    for k in plan.want.clone() {
        if !map.pool().map.contains_key(&k) {
            n += 1;
            map.pool_mut().map.insert(k, cell(n));
        }
    }
    let plan = plan_view(w, h, &map.cam, map.pool(), plan.dem_depth, plan.vec_depth);
    let planes: usize = plan.roles.iter().map(|t| t.refs.iter().filter(|r| r.on_screen).count()).sum();
    assert!(planes > lay.slots, "the view's own cells fit the slots ({planes} planes): not the case this checks");
    let mut slots = Slots::new(lay.slots);
    for frame in 0..3usize {
        arena::view(&mut slots, map.pool(), &plan, map.pool_version(), &lay, frame & 1, &mut arena).unwrap();
        if frame == 0 {
            continue;
        }
        let mut at = lay.regions[frame & 1] / 4 + kernel::SQUARES / 4;
        for t in &plan.roles {
            for (k, &sq) in t.squares.iter().enumerate() {
                if sq != mahere_engine::plan::NONE && t.refs[sq as usize].on_screen {
                    assert_ne!(arena[at + k], 0, "frame {frame}: a shown square at depth {} with no slot", t.depth);
                }
            }
            at += t.squares.len();
        }
    }
    assert!(slots.capped > 0);
}

/// The fetch storm of 2026-11-12 (the 8a: about 240 fetches failed "DNS still asking" in Mahere's first second): Mahere's twenty loaders asking for the bucket's address at once against a resolver like the ip server's (each call held until its lookup ends, a lookup a quarter second; sixteen held calls at most, the oldest answered "still asking" when a seventeenth comes). One call is held at a time, the loaders waiting on it all get the address, and none fails; a lookup that gives way ("still asking") is asked again, not failed; a lookup that ends with no address fails every loader waiting on it with one call.
#[test]
fn one_lookup_a_host_however_many_loaders_ask() {
    use crate::names::{Asked, Names};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    /// The resolver: calls held now (the most at once), calls made, and the outcome of each lookup in turn (true: found, false: "still asking", then found).
    struct Resolver {
        held: AtomicUsize,
        most: AtomicUsize,
        calls: AtomicUsize,
        gives_way: Mutex<bool>,
    }
    impl Resolver {
        fn ask(&self, found: bool) -> Asked {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let held = self.held.fetch_add(1, Ordering::SeqCst) + 1;
            self.most.fetch_max(held, Ordering::SeqCst);
            // Sixteen held at once at most, as the ip server's `RQN`: a seventeenth makes the oldest answer "still asking", which is what the loaders' storm was.
            assert!(held <= 1 << 4, "{held} RESOLVE calls held at once");
            std::thread::sleep(Duration::from_millis(1 << 8));
            self.held.fetch_sub(1, Ordering::SeqCst);
            let mut g = self.gives_way.lock().unwrap();
            if std::mem::take(&mut *g) {
                return Asked::Asking;
            }
            if found { Asked::Found(4, [93, 184, 216, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) } else { Asked::Failed(std::io::ErrorKind::NotFound, "DNS gave no address") }
        }
    }
    let loaders = mahere_engine::residency::LOADERS;
    for (gives_way, found) in [(false, true), (true, true), (false, false)] {
        let names = Names::new();
        let r = Resolver { held: AtomicUsize::new(0), most: AtomicUsize::new(0), calls: AtomicUsize::new(0), gives_way: Mutex::new(gives_way) };
        // All at once, as the loaders start on a fresh want list.
        let start = std::sync::Barrier::new(loaders);
        let got: Vec<std::io::Result<(u8, [u8; 16])>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..loaders).map(|_| s.spawn(|| {
                start.wait();
                names.address("cells.example", || r.ask(found))
            })).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (calls, most) = (r.calls.load(Ordering::SeqCst), r.most.load(Ordering::SeqCst));
        println!("{loaders} loaders, the lookup giving way first {gives_way}, found {found}: {calls} calls, {most} held at once");
        assert_eq!(most, 1, "one call held for the host at a time");
        assert_eq!(calls, 1 + gives_way as usize, "one lookup for every loader (and once more after it gave way)");
        if found {
            assert!(got.iter().all(|g| matches!(g, Ok((4, a)) if a[..4] == [93, 184, 216, 34])), "every loader has the address, none failed");
        } else {
            assert!(got.iter().all(|g| g.as_ref().is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)), "every loader waiting on a lookup with no address takes its outcome");
        }
    }
}
