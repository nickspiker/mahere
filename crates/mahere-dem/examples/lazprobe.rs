// lazprobe <tile.laz>... <lat> <lon>: build the 1 m intensity grid and sample a point.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (tiles, pt) = args.split_at(args.len() - 2);
    let (lat, lon): (f64, f64) = (pt[0].parse().unwrap(), pt[1].parse().unwrap());
    let t = std::time::Instant::now();
    let store = mahere_dem::IntensityStore::from_laz(tiles, 10).expect("laz");
    println!("built in {:.1}s; sample {:?}", t.elapsed().as_secs_f32(), store.sample(lat, lon));
}
