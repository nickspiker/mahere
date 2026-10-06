// transcode <cell.vsf.zst>... [--dem-loss m] [--img-loss levels] [--out dir]: decode cells, re-encode them with the current codec at the given loss, and report sizes and the elevation error — the dial for choosing a bake's loss. With --out, the re-encoded cells are written there under their own names (a whole directory migrates with `transcode data/cells/*.zst --out data/cells-next`).
use mahere_tiles::{Loss, TRI, decode_cell};
use rayon::prelude::*;

struct Tally {
    before: usize,
    after: usize,
    n: usize,
    worst: f32,
    sum_sq: f64,
    count: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let loss = Loss {
        dem_m: value("--dem-loss").map(|v| v.parse().unwrap()).unwrap_or(0.0),
        img: value("--img-loss").map(|v| v.parse().unwrap()).unwrap_or(0),
    };
    let out = value("--out").map(std::path::PathBuf::from);
    if let Some(o) = &out {
        std::fs::create_dir_all(o).expect("out dir");
    }
    let mut files: Vec<&String> = Vec::new();
    let mut skip = false;
    for a in &args {
        if skip {
            skip = false;
        } else if a.starts_with("--") {
            skip = true;
        } else {
            files.push(a);
        }
    }
    let t = std::time::Instant::now();
    let tally = files
        .par_iter()
        .map(|f| {
            let mut t = Tally { before: 0, after: 0, n: 0, worst: 0.0, sum_sq: 0.0, count: 0 };
            let bytes = std::fs::read(f).expect("read");
            let planes = match decode_cell(&bytes) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{f}: {e}");
                    return t;
                }
            };
            let plain = planes.encode(&loss).expect("encode");
            let z = zstd::encode_all(&plain[..], 3).expect("zstd");
            let back = decode_cell(&z).expect("decode again");
            if let (Some(a), Some(b)) = (&planes.dem, &back.dem) {
                for i in 0..TRI {
                    if a.elev[i].is_nan() != b.elev[i].is_nan() {
                        panic!("{f}: nodata changed at {i}");
                    }
                    if !a.elev[i].is_nan() {
                        let d = (a.elev[i] - b.elev[i]).abs();
                        t.worst = t.worst.max(d);
                        t.sum_sq += (d as f64) * (d as f64);
                        t.count += 1;
                    }
                }
            }
            if let Some(o) = &out {
                let name = std::path::Path::new(f).file_name().unwrap();
                std::fs::write(o.join(name), &z).expect("write");
            }
            t.before = bytes.len();
            t.after = z.len();
            t.n = 1;
            t
        })
        .reduce(
            || Tally { before: 0, after: 0, n: 0, worst: 0.0, sum_sq: 0.0, count: 0 },
            |a, b| Tally { before: a.before + b.before, after: a.after + b.after, n: a.n + b.n, worst: a.worst.max(b.worst), sum_sq: a.sum_sq + b.sum_sq, count: a.count + b.count },
        );
    println!("{} cells: {} -> {} bytes ({:.2}x) in {:.1}s", tally.n, tally.before, tally.after, tally.before as f64 / tally.after.max(1) as f64, t.elapsed().as_secs_f32());
    if tally.count > 0 {
        println!("elevation error: worst {:.3} m, rms {:.3} m", tally.worst, (tally.sum_sq / tally.count as f64).sqrt());
    }
}
