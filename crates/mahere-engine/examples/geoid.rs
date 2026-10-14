//! `geoid <cells dir> <lat,lon>...`: the geoid's height above the ellipsoid at points, from a cell directory's depth-2 cells.
use std::sync::Arc;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let store = Arc::new(mahere_engine::residency::DirStore(a[0].clone().into()));
    let mut map = mahere_engine::MapCore::new(store, mahere_engine::Camera::new(0.0, 0.0, 100.0, 0.0));
    for p in &a[1..] {
        let v: Vec<f64> = p.split(',').map(|x| x.parse().unwrap()).collect();
        println!("{p}: {:?}", map.geoid_at(v[0], v[1]));
    }
}
