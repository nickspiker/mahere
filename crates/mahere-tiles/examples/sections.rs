// sections <cell.vsf.zst>: where a cell's bytes go — each section encoded alone and zstd'd, and for the dem the mean coefficient magnitude per pyramid level.
use mahere_tiles::{CellPlanes, Loss, decode_cell, pyr};

fn zsize(p: &CellPlanes, loss: &Loss) -> usize {
    let plain = p.encode(loss).unwrap();
    zstd::encode_all(&plain[..], 3).unwrap().len()
}

fn main() {
    let path = std::env::args().nth(1).expect("cell path");
    let bytes = std::fs::read(&path).expect("read");
    let p = decode_cell(&bytes).expect("decode");
    println!("file {} bytes", bytes.len());
    let only = |f: &dyn Fn(&mut CellPlanes)| {
        let mut q = CellPlanes::default();
        f(&mut q);
        q
    };
    let p2 = p.clone();
    let dem = only(&|q| q.dem = p2.dem.clone());
    let vec = only(&|q| {
        q.line = p2.line.clone();
        q.land = p2.land.clone();
        q.water = p2.water.clone();
    });
    let img = only(&|q| q.img = p2.img.clone());
    for (name, part) in [("dem", &dem), ("vec", &vec), ("img", &img)] {
        println!("{name}: lossless {} bytes", zsize(part, &Loss::LOSSLESS));
    }
    for m in [0.1f32, 0.2, 0.3] {
        println!("dem at {m} m: {} bytes", zsize(&dem, &Loss { dem_m: m, img: 0 }));
    }
    for l in [4u32, 8, 16, 32] {
        println!("img at {l} levels: {} bytes", zsize(&img, &Loss { dem_m: 0.0, img: l }));
    }
    // What a per-block Rice code would cost versus the block-width packing in use.
    let rice = |c: &[i32]| -> (usize, usize) {
        let (mut rice_bits, mut block_bits) = (0usize, 0usize);
        for b in c.chunks(16) {
            let zz: Vec<u32> = b.iter().map(|&v| ((v << 1) ^ (v >> 31)) as u32).collect();
            let w = 32 - zz.iter().fold(0u32, |m, &v| m | v).leading_zeros();
            block_bits += 8 + w as usize * b.len();
            let best = (0..16u32).map(|k| 4 + zz.iter().map(|&v| (k + 1 + (v >> k)) as usize).sum::<usize>()).min().unwrap();
            rice_bits += best;
        }
        (rice_bits, block_bits)
    };
    if let Some(d) = &p.dem {
        let (lo, hi) = d.elev.iter().filter(|e| !e.is_nan()).fold((f32::MAX, f32::MIN), |(a, b), &e| (a.min(e), b.max(e)));
        let step = ((hi - lo) / 32000.0).max(0.05);
        println!("dem range {lo}..{hi}, step {step}");
        let q: Vec<i32> = d.elev.iter().map(|&e| if e.is_nan() { 0 } else { ((e - lo) / step).round() as i32 }).collect();
        for (name, steps) in [("lossless", pyr::Steps::LOSSLESS), ("leaf step 4", pyr::Steps::tapered(4))] {
            let coef = pyr::forward(&q, &steps);
            println!("{name}:");
            let mut off = 2;
            for l in 1..=pyr::LEVELS {
                let n = 2 << (2 * l);
                let c = &coef[off..off + n];
                let mean = c.iter().map(|&v| v.unsigned_abs() as f64).sum::<f64>() / n as f64;
                let zeros = c.iter().filter(|&&v| v == 0).count();
                let bits = c.iter().map(|&v| 32 - (((v << 1) ^ (v >> 31)) as u32).leading_zeros()).sum::<u32>() as f64 / n as f64;
                let (r, bw) = rice(c);
                println!("  level {l}: {n} coefs, mean |c| {mean:.2}, zeros {:.0}%, ideal bits {bits:.2}, rice {:.2}, block {:.2}", zeros as f64 * 100.0 / n as f64, r as f64 / n as f64, bw as f64 / n as f64);
                off += n;
            }
        }
    }
}
