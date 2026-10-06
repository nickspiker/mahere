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
        }
    }
}
