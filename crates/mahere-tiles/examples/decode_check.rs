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
            // The water plane's coverage spread: full, part and none, which says whether a coast is antialiased or hard.
            if let Some(w) = &p.water {
                let full = w.cov.iter().filter(|&&c| c == 255).count();
                let part = w.cov.iter().filter(|&&c| c != 0 && c != 255).count();
                println!("water texels: {full} full, {part} part, {} none", w.cov.len() - full - part);
            }
        }
    }
}
