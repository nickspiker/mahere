// decode_check <cell.vsf.zst>: decode on whatever CPU this runs on and report the dem section — a cross-compile target for chasing platform differences.
fn main() {
    let path = std::env::args().nth(1).expect("cell path");
    let bytes = std::fs::read(&path).expect("read");
    match mahere_tiles::decode_cell(&bytes) {
        Err(e) => println!("decode error: {e}"),
        Ok(p) => {
            match &p.dem {
                None => println!("dem: NONE"),
                Some(d) => {
                    let valid = d.elev.iter().filter(|e| !e.is_nan()).count();
                    let (lo, hi) = d.elev.iter().filter(|e| !e.is_nan()).fold((f32::MAX, f32::MIN), |(a, b), &e| (a.min(e), b.max(e)));
                    println!("dem: {valid}/{} valid, range {lo}..{hi}, elev[1000]={}", d.elev.len(), d.elev[1000]);
                    let key = mahere_tiles::CellKey { depth: 10, prefix: 0 };
                    let packed = d.pack_texels(key);
                    println!("packed[1000] = {:#018x}", packed[1000]);
                }
            }
            println!("line {} land {} water {}", p.line.is_some(), p.land.is_some(), p.water.is_some());
            // The apron against the edge it sits beside: a seam shows as a large step.
            if let Some(d) = &p.dem {
                let tex = mahere_tiles::TEX;
                let at = |tx: usize, ty: usize, half: usize| d.elev[(ty * tex + tx) * 2 + half];
                let apron = |side: usize, half: usize, i: usize| d.apron[(side * 2 + half) * tex + i];
                let mut worst = 0.0f32;
                let mut nan = 0;
                for i in 0..tex {
                    for half in 0..2 {
                        let pairs = [(apron(0, half, i), at(0, i, half)), (apron(1, half, i), at(tex - 1, i, half)), (apron(2, half, i), at(i, 0, half)), (apron(3, half, i), at(i, tex - 1, half))];
                        for (a, e) in pairs {
                            if a.is_nan() {
                                nan += 1;
                            } else if e.is_finite() {
                                worst = worst.max((a - e).abs());
                            }
                        }
                    }
                }
                println!("apron: {nan} of {} unknown, worst step to the edge texel {worst:.1} m; west apron[0..4] {:?} edge {:?}", 8 * tex, &d.apron[0..4], [at(0, 0, 0), at(0, 1, 0), at(0, 2, 0), at(0, 3, 0)]);
            }
            // The water plane's coverage spread: full, part and none, which says whether a coast is antialiased or hard.
            if let Some(w) = &p.water {
                let full = w.cov.iter().filter(|&&c| c == 255).count();
                let part = w.cov.iter().filter(|&&c| c != 0 && c != 255).count();
                println!("water texels: {full} full, {part} part, {} none", w.cov.len() - full - part);
            }
        }
    }
}
