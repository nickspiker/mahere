//! route_bytes <lat> <lon> <depth>...: a base cell's sections and their sizes at each depth, fetched from the bucket: what a route reads of a cell against what it fetches (2026-11-27, sizing route bundles).
use mahere_engine::residency::{CellStore, DEFAULT_CELLS_URL, Fetch, HttpStore};
use mahere_tiles::CellKey;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (lat, lon): (f64, f64) = (a[1].parse().expect("lat"), a[2].parse().expect("lon"));
    let store = HttpStore::layer(DEFAULT_CELLS_URL, mahere_tiles::publish::Layer::Base);
    for d in &a[3..] {
        let key = CellKey::containing(mahere_coord::Coord::from_lat_lon(lat, lon), d.parse().expect("depth"));
        let Fetch::Bytes(b) = store.get(key) else {
            println!("depth {d}: no cell");
            continue;
        };
        let (header, _) = vsf::VsfHeader::decode(&b).expect("a header");
        let parts: Vec<String> = header.fields.iter().map(|f| format!("{} {}", f.name, f.size_bytes)).collect();
        println!("depth {d}: {} bytes: {}", b.len(), parts.join(", "));
    }
}
