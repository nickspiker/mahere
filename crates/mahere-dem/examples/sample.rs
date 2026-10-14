//! `sample <tif>... -- <lat,lon>...`: a DEM store's value at points (a check that a source reads as it should).
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let cut = a.iter().position(|x| x == "--").unwrap_or(a.len());
    let store = mahere_dem::DemStore::load(&a[..cut]).expect("load");
    for p in &a[(cut + 1).min(a.len())..] {
        let v: Vec<f64> = p.split(',').map(|x| x.parse().unwrap()).collect();
        println!("{p}: {:?}", store.elevation(v[0], v[1]));
    }
}
