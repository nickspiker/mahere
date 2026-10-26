//! search <lat> <lon> <ppd> <query> [w h]: the desktop vault's cached cells searched under that view (XDG_CONFIG_HOME / XDG_DATA_HOME pick the vault), timed.
use mahere_engine::{Camera, MapCore};
use mahere_tiles::publish::Layer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let num = |i: usize| a[i].parse::<f64>().expect("a number");
    let (lat, lon, ppd, query) = (num(1), num(2), num(3), a[4].clone());
    let (w, h) = (a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1280usize), a.get(6).and_then(|s| s.parse().ok()).unwrap_or(800usize));
    let vault = mahere_store::open(None).expect("vault");
    let (store, layers, cells) = mahere_store::cell_stores(None, Some(vault));
    let cells = cells.expect("cells");
    let keys = cells.keys(Layer::Base);
    let map = MapCore::layered(store, layers, Camera::new(lat, lon, ppd, 0.0));
    let job = map.search_job(&query, w, h, keys.clone());
    let mut cache = Default::default();
    let t = std::time::Instant::now();
    let found = mahere_engine::search::run(&job, &mut cache, &mahere_panel::labels::tier);
    eprintln!("{} cached base cells; {} found in {:.0} ms", keys.len(), found.len(), t.elapsed().as_secs_f64() * 1000.0);
    let t = std::time::Instant::now();
    let again = mahere_engine::search::run(&job, &mut cache, &mahere_panel::labels::tier);
    eprintln!("again, names cached: {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(found, again);
    for f in found.iter().take(15) {
        println!("{:<34} {:<10} kind {:3} rank {:7} fit {} size {:6.0} m  {:.5},{:.5}", f.name, f.reference, f.kind, f.rank, f.fit, f.size_m, f.lat, f.lon);
    }
}
