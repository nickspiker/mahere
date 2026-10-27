//! areacount <area.vsf.zst> <points.txt> [published dir]: how many of the points (one "lat lon bytes" per line) fall inside the area's depth-14 cells, their bytes, and, given the published directory, what the area's cells weigh there now (base, dem, img, nir at depths 9 to 14).
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let area = mahere_tiles::area::Area::decode(&std::fs::read(&a[1]).unwrap()).unwrap();
    let mut cells: Vec<u64> = area.cells_at(14).into_iter().map(|k| k.prefix).collect();
    cells.sort_unstable();
    let (mut n, mut bytes, mut all) = (0usize, 0u64, 0usize);
    for line in std::fs::read_to_string(&a[2]).unwrap().lines() {
        let v: Vec<&str> = line.split_whitespace().collect();
        let (Ok(lat), Ok(lon)) = (v[0].parse::<f64>(), v[1].parse::<f64>()) else { continue };
        all += 1;
        let k = mahere_tiles::CellKey::containing(mahere_coord::Coord::from_lat_lon(lat, lon), 14);
        if cells.binary_search(&k.prefix).is_ok() {
            n += 1;
            bytes += v.get(2).and_then(|b| b.parse::<u64>().ok()).unwrap_or(0);
        }
    }
    println!("{n} of {all} points inside, {bytes} bytes of text");
    if let Some(dir) = a.get(3) {
        for layer in ["base", "dem", "img", "nir"] {
            let mut total = 0u64;
            for d in 9..=14u8 {
                for k in area.cells_at(d) {
                    if let Ok(m) = std::fs::metadata(std::path::Path::new(dir).join(layer).join(format!("{}.vsf", k.name()))) {
                        total += m.len();
                    }
                }
            }
            println!("{layer}: {:.1} MB", total as f64 / 1e6);
        }
    }
}
