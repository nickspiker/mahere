// Probe: can we read a real USGS tile, and do known summits check out?
fn main() {
    let t = std::time::Instant::now();
    let store = mahere_dem::DemStore::load(&[
        "data/USGS_1_n47w122.tif".into(),
        "data/USGS_1_n47w123.tif".into(),
        "data/USGS_1_n48w122.tif".into(),
        "data/USGS_1_n48w123.tif".into(),
    ])
    .expect("load");
    println!("{} tiles in {:.1}s", store.tile_count(), t.elapsed().as_secs_f32());
    for (name, lat, lon, expect) in [
        ("Mt Rainier", 46.8523, -121.7603, 4392.0_f32),
        ("Mt Si", 47.4899, -121.7231, 1270.0),
        ("Seattle waterfront", 47.6050, -122.3400, 5.0),
        ("Puget Sound", 47.6, -122.45, 0.0),
        ("Granite Mountain", 47.4342, -121.4857, 1739.0),
    ] {
        match store.elev_and_gradient(lat, lon) {
            Some((e, (ge, gn))) => println!(
                "{name:20} {e:7.1} m (expect ~{expect:6.0})  grad E {ge:+.3} N {gn:+.3}"
            ),
            None => println!("{name:20} NO DATA"),
        }
    }
}
