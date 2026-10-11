//! Read a shared area document: `area <file.vsf.zst> [info|bbox|cells <depth>|pieces <depth>]`.
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
        // `pieces <depth>`: the area cut into the cells of that depth it covers, each with the box of the finest cells under it, for baking a piece at a time.
        "pieces" => {
            let depth: u8 = args.get(3).and_then(|d| d.parse().ok()).unwrap_or(9);
            let mut boxes: std::collections::BTreeMap<mahere_tiles::CellKey, (f64, f64, f64, f64, usize)> = Default::default();
            for k in a.cells_at(14.max(depth)) {
                let e = boxes.entry(k.ancestor(depth)).or_insert((90.0, 180.0, -90.0, -180.0, 0));
                for (la, lo) in k.corners() {
                    e.0 = e.0.min(la);
                    e.1 = e.1.min(lo);
                    e.2 = e.2.max(la);
                    e.3 = e.3.max(lo);
                }
                e.4 += 1;
            }
            for (k, (la0, lo0, la1, lo1, n)) in boxes {
                println!("{} {la0:.6} {lo0:.6} {la1:.6} {lo1:.6} {n}", k.name());
            }
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
