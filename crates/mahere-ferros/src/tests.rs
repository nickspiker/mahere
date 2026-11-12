//! The GPU kernel against the CPU raster, on the host: the same cells, the same view, the arena built as the device builds it (`arena.rs`), the kernel's words run in valhall's emulator one pixel at a time, and every pixel compared with what the CPU's `render_frame` draws (ferros, owner 2026-11-11: "Just needs to look the same"; the emulator: "Prove on the host what you can").
//! The CPU writes the display's code (the BT.2020 matrix, Opsin's rail, √); the kernel writes VSF `g2t` codes for ferros's panel. The kernel's code is turned back into its light and through the CPU's own encode, so both are compared as the CPU's codes.

use crate::arena::{self, Layout, Slots};
use crate::kernel;
use mahere_engine::colour::Display;
use mahere_engine::ink::UserLayer;
use mahere_engine::plan::plan_frame;
use mahere_engine::raster::{Contours, FrameLuts, LayerMask, Style, build_hypso_lut, render_frame, select_depth};
use mahere_engine::residency::{DemPacked, Entry, Pool, PRESENT_DEM, PRESENT_LAND, PRESENT_LINE, PRESENT_WATER};
use mahere_engine::Camera;
use mahere_tiles::{CellKey, ClassCell, CovCell, TRI};
use valhall::sim::{Memory, run};

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

/// The frame both ways for `cam` over `pool` with `mask`; the CPU's codes and the kernel's (through the CPU's encode).
fn both(cam: &Camera, pool: &Pool, mask: LayerMask, w: usize, h: usize, out: kernel::Out) -> (Vec<u32>, Vec<u32>, usize) {
    let d = select_depth(cam.ppd, 0, 14);
    let user = UserLayer::default();
    let plan = plan_frame(w, h, cam, pool, d, d, &user);
    let luts = FrameLuts {
        pole: [plan.pole[0], plan.pole[1], plan.pole[2]],
        disk: [plan.globe[0], plan.globe[1], plan.globe[2]],
        hypso: build_hypso_lut(),
        style: Style::default(),
        sun: [0.3, 0.4, 0.866],
        mask: mask.effective(),
        dem_depth: d,
        contours: Contours { interval: 2.0, index_every: 5, m_per_px: (111_320.0 / cam.ppd) as f32, lo_m: 0.0, hi_m: 4000.0 },
        light: mahere_engine::sh::Sh9::sun_and_sky(315.0, 40.0).quadratic((0.0, 1.0)),
        display: Display::default(),
        compressed: true,
    };
    let mut cpu = vec![0u32; w * h];
    render_frame(&mut cpu, w, h, cam, pool, &luts, d, d, &user);

    let lay = Layout::new(w, h, BYTES, out).unwrap();
    let mut words = vec![0u32; BYTES / 4];
    let mut slots = Slots::new(lay.slots);
    let (cand, _) = arena::cells(&mut slots, pool, &plan, 1, &lay, &mut words);
    arena::blocks(&plan, &cand, &lay, 0, &mut words).unwrap();
    arena::tables(&luts, &plan, true, &lay, 0, &mut words);
    let c = arena::consts(&luts, &plan, &lay, 0, false);
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
    for g in 0..(w * h).next_multiple_of(16) {
        run(&k.words, &uniforms, [g as u32, 0], &mut mem);
    }
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

#[test]
fn the_kernel_draws_what_the_cpu_draws() {
    let (w, h) = (96usize, 80usize);
    // Zoomed to depth 11 over the Cascades, turned, so blocks interpolate with a twist and the frame spans several cells.
    let cam = Camera::new(46.2, -121.5, 6000.0, 0.7);
    let d = select_depth(cam.ppd, 0, 14);
    let mut pool = Pool::default();
    let plan = plan_frame(w, h, &cam, &pool, d, d, &UserLayer::default());
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
