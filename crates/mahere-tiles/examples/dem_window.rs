//! `dem_window <url or file> <lat> <lon> [<lat> <lon>]...`: the elevation at each point from a window of a tiled GeoTIFF read by byte ranges (Canada's HRDEM) or from a local file, and the point in the file's grid, for checking a source before a bake takes it.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let pts: Vec<(f64, f64)> = a[2..].chunks(2).map(|p| (p[0].parse().unwrap(), p[1].parse().unwrap())).collect();
    let pad = 0.002;
    let bbox = pts.iter().fold((90.0f64, 180.0f64, -90.0f64, -180.0f64), |b, &(la, lo)| (b.0.min(la - pad), b.1.min(lo - pad), b.2.max(la + pad), b.3.max(lo + pad)));
    let t = std::time::Instant::now();
    let tile = if a[1].starts_with("http") {
        let r = mahere_tiles::remote::SharedRange::open(&a[1]).expect("open");
        let p = r.clone();
        mahere_dem::DemTile::load_window(r, &a[1], bbox, &mut |spans| p.prefetch(spans).expect("prefetch")).expect("window")
    } else {
        mahere_dem::DemTile::load_window(std::io::BufReader::new(std::fs::File::open(&a[1]).expect("file")), &a[1], bbox, &mut |_| {}).expect("window")
    };
    println!("window read in {:.1} s", t.elapsed().as_secs_f64());
    let Some(tile) = tile else {
        println!("nothing under the points");
        return;
    };
    let store = mahere_dem::DemStore::from_tiles(vec![tile]);
    for (lat, lon) in pts {
        let (x, y) = mahere_dem::canada_lambert_forward(lat, lon);
        println!("{lat:.6} {lon:.6}: {:?} m   (Canada Lambert {x:.1}, {y:.1})", store.elevation(lat, lon));
    }
}
