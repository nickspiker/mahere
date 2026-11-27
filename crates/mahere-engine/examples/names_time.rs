//! names_time <lat> <lon> <ppd> <query> <depth>...: the base cells there fetched from the bucket at each depth, their names decoded, and a search of them for the query, timed (what a search pays for every kept cell under the screen).
use mahere_engine::residency::{CellStore, DEFAULT_CELLS_URL, Fetch, HttpStore};
use mahere_engine::{Camera, MapCore};
use mahere_tiles::CellKey;
use std::sync::Arc;

/// The cells fetched, held in memory: the search reads them as a device reads its vault.
struct Held(Vec<(CellKey, Vec<u8>)>);

impl CellStore for Held {
    fn get(&self, key: CellKey) -> Fetch {
        self.0.iter().find(|(k, _)| *k == key).map_or(Fetch::Absent, |(_, b)| Fetch::Bytes(b.clone()))
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (lat, lon, ppd): (f64, f64, f64) = (a[1].parse().expect("lat"), a[2].parse().expect("lon"), a[3].parse().expect("ppd"));
    let query = a[4].clone();
    let http = HttpStore::layer(DEFAULT_CELLS_URL, mahere_tiles::publish::Layer::Base);
    let mut held = Vec::new();
    for d in &a[5..] {
        let key = CellKey::containing(mahere_coord::Coord::from_lat_lon(lat, lon), d.parse().expect("depth"));
        let Fetch::Bytes(b) = http.get(key) else {
            println!("depth {d}: no cell");
            continue;
        };
        let t = std::time::Instant::now();
        let names = mahere_engine::labels::decode_names(&b);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("depth {d}: {} bytes, names {} in {ms:.1} ms", b.len(), names.map_or("none".to_string(), |n| format!("{} entries", n.entries.len())));
        held.push((key, b));
    }
    let keys: Vec<CellKey> = held.iter().map(|(k, _)| *k).collect();
    let map = MapCore::new(Arc::new(Held(held)), Camera::new(lat, lon, ppd, 0.0));
    for q in [query.chars().take(1).collect::<String>(), query.clone()] {
        let job = map.search_job(&q, 1344, 1496, keys.clone());
        let mut cache = Default::default();
        let t = std::time::Instant::now();
        let found = mahere_engine::search::run(&job, &mut cache, &|_| Some(0));
        println!("\"{q}\": {} found in {:.0} ms", found.len(), t.elapsed().as_secs_f64() * 1000.0);
    }
}
