// imgprobe <naip.tif>... <lat> <lon>: load the 4-band tiles and sample a point.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (tifs, pt) = args.split_at(args.len() - 2);
    let (lat, lon): (f64, f64) = (pt[0].parse().unwrap(), pt[1].parse().unwrap());
    let t = std::time::Instant::now();
    let store = mahere_dem::ImgStore::load(tifs).expect("load");
    println!("{} tiles in {:.1}s; sample {:?}", store.tile_count(), t.elapsed().as_secs_f32(), store.sample(lat, lon));
}
