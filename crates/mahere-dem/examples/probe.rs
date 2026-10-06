// probe <tif>... <lat> <lon>: which tile answers, and what it says.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (coords, tifs) = args.split_at(args.len() - 2);
    let lat: f64 = tifs[0].parse().unwrap();
    let lon: f64 = tifs[1].parse().unwrap();
    for t in coords {
        let store = mahere_dem::DemStore::load(&[t.clone()]).expect("load");
        for d in store.describe() {
            println!("  {d}");
        }
        println!("{t}: {:?}", store.elev_and_gradient(lat, lon));
    }
}
