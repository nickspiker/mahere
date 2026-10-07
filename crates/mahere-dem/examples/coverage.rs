// coverage <lat0> <lon0> <lat1> <lon1> <tif>...: where the given DEM tiles have elevation over a box. Samples a 96 × 48 grid and prints it as a map (# data, . none) with the percentage covered, so a hole in a bake can be told apart from a hole in the source.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: Vec<f64> = args[..4].iter().map(|s| s.parse().unwrap()).collect();
    let (lat0, lon0, lat1, lon1) = (n[0].min(n[2]), n[1].min(n[3]), n[0].max(n[2]), n[1].max(n[3]));
    let store = mahere_dem::DemStore::load(&args[4..]).expect("load");
    let (cols, rows) = (96usize, 48usize);
    let mut hits = 0usize;
    for r in 0..rows {
        let lat = lat1 - (lat1 - lat0) * (r as f64 + 0.5) / rows as f64;
        let line: String = (0..cols)
            .map(|c| {
                let lon = lon0 + (lon1 - lon0) * (c as f64 + 0.5) / cols as f64;
                if store.elevation(lat, lon).is_some_and(|e| e.is_finite()) {
                    hits += 1;
                    '#'
                } else {
                    '.'
                }
            })
            .collect();
        println!("{line}");
    }
    println!("{:.1}% of the box has elevation", hits as f64 * 100.0 / (cols * rows) as f64);
}
