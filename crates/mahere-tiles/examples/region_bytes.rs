//! `region_bytes <cells dir> <depth> <lat0,lon0,lat1,lon1>...`: the cells at a depth inside each box, how many exist and their bytes, for comparing regions on disk.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let depth: u8 = args[2].parse().unwrap();
    for b in &args[3..] {
        let v: Vec<f64> = b.split(',').map(|x| x.parse().unwrap()).collect();
        let keys = mahere_tiles::cells_covering(v[0], v[1], v[2], v[3], depth);
        let (mut n, mut bytes) = (0u64, 0u64);
        for k in &keys {
            if let Ok(m) = std::fs::metadata(dir.join(k.path())) {
                n += 1;
                bytes += m.len();
            }
        }
        println!("{b}: {} cells in the box, {n} on disk, {:.1} MB, {:.0} KB a cell", keys.len(), bytes as f64 / 1e6, bytes as f64 / n.max(1) as f64 / 1e3);
    }
}
