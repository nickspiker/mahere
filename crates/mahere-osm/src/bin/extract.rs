//! Build a featpack: extract + clip drawable features from a .osm.pbf.
//! Usage: extract <in.osm.pbf> <out.vsf> <lat0,lon0,lat1,lon1> [more boxes]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: extract <in.osm.pbf> <out.vsf> <lat0,lon0,lat1,lon1>...");
        std::process::exit(1);
    }
    let bboxes: Vec<(f64, f64, f64, f64)> = args[3..]
        .iter()
        .map(|b| {
            let v: Vec<f64> = b.split(',').map(|x| x.parse().unwrap()).collect();
            (v[0].min(v[2]), v[1].min(v[3]), v[0].max(v[2]), v[1].max(v[3]))
        })
        .collect();
    let t = std::time::Instant::now();
    let feats = mahere_osm::load_features(&args[1]).expect("read pbf");
    eprintln!("{} features loaded in {:.1}s", feats.len(), t.elapsed().as_secs_f32());
    mahere_osm::write_featpack(&args[2], &feats, &bboxes).expect("write featpack");
    let size = std::fs::metadata(&args[2]).unwrap().len();
    eprintln!("wrote {} ({:.1} MB)", args[2], size as f64 / 1e6);
}
