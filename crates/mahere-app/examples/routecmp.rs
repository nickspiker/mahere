//! routecmp: trips routed by the graph (route::find) and by stepping trixels (route::find_trixels), from the desktop vault's cells and the bucket, compared: length, time, turns, and how long each took.
use std::sync::Arc;

fn main() {
    let vault = mahere_store::open(None).expect("vault");
    let (store, layers, _) = mahere_store::cell_stores(None, Some(vault));
    let dem: Arc<dyn mahere_engine::residency::CellStore> = layers.map_or(store.clone(), |l| l.dem);
    let trips: Vec<(&str, (f64, f64), (f64, f64))> = vec![
        ("Harper to SE Cottonwood", (47.51892, -122.52005), (47.50952, -122.52059)),
        ("Harper to the Southworth ferry", (47.51892, -122.52005), (47.51212, -122.50045)),
        ("Fraser Road to Banner Forest", (47.50149, -122.51309), (47.4877, -122.5445)),
        ("Southworth to Port Orchard", (47.5120, -122.5005), (47.5404, -122.6364)),
    ];
    let cache = Arc::new(mahere_engine::route::RouteCache::default());
    for travel in [mahere_engine::route::Travel::Car, mahere_engine::route::Travel::Foot] {
        for (name, a, b) in &trips {
            let t = std::time::Instant::now();
            let new = mahere_engine::route::find_cached(store.clone(), dem.clone(), *a, *b, travel, Some(cache.clone()));
            let t_new = t.elapsed().as_secs_f64() * 1e3;
            let t = std::time::Instant::now();
            let old = mahere_engine::route::find_trixels(store.clone(), dem.clone(), *a, *b, travel);
            let t_old = t.elapsed().as_secs_f64() * 1e3;
            let say = |r: &Result<mahere_engine::route::Route, String>| match r {
                Ok(r) => format!("{:.2} km {:.1} min {} steps: {}", r.metres / 1000.0, r.seconds / 60.0, r.steps.len(), r.steps.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(" / ")),
                Err(e) => format!("none ({e})"),
            };
            println!("{} by {}:\n  graph   {:6.0} ms  {}\n  trixels {:6.0} ms  {}", name, travel.word(), t_new, say(&new), t_old, say(&old));
        }
    }
}
