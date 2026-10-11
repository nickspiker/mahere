//! Read a shared area document: `area <file.vsf.zst> [info|bbox|cells <depth>]`.
//! `info` (the default) prints the name, when it was made, the detail wanted, the bounds and the cell counts per depth; `bbox` prints lat0,lon0,lat1,lon1 for a baker; `cells <depth>` prints one cell name per line, the objects the bucket would hold.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(file) = args.get(1) else {
        eprintln!("usage: area <file.vsf.zst> [info|bbox|cells <depth>]");
        std::process::exit(2);
    };
    let bytes = std::fs::read(file).expect("read the area file");
    let a = mahere_tiles::area::Area::decode(&bytes).expect("not an area document");
    match args.get(2).map(String::as_str).unwrap_or("info") {
        "bbox" => {
            let (lat0, lon0, lat1, lon1) = a.bounds().expect("an empty area");
            println!("{lat0:.6},{lon0:.6},{lat1:.6},{lon1:.6}");
        }
        "cells" => {
            let depth: u8 = args.get(3).and_then(|d| d.parse().ok()).unwrap_or(a.depth);
            for k in a.cells_at(depth) {
                println!("{}", k.name());
            }
        }
        _ => {
            let (secs, _) = vsf::types::eagle_time::to_unix_ns(a.created);
            println!("name    {}", a.name);
            println!("created unix {secs}");
            println!("depth   {} ({:.2} m texels)", a.depth, mahere_tiles::texel_m(a.depth));
            println!("planes  {}", a.cells.len());
            if let Some((lat0, lon0, lat1, lon1)) = a.bounds() {
                println!("bounds  {lat0:.6},{lon0:.6},{lat1:.6},{lon1:.6}");
            }
            if let Some((lat, lon)) = a.centre() {
                println!("centre  {lat:.6},{lon:.6}");
            }
            for d in mahere_tiles::area::AREA_MIN_DEPTH..=mahere_tiles::area::AREA_MAX_DEPTH {
                let (cells, bytes) = a.estimate_to(d);
                println!("to depth {d:2}: {:6} cells at that depth, {cells:7} in all, about {:.0} MB", a.count_at(d), bytes as f64 / 1e6);
            }
        }
    }
}
