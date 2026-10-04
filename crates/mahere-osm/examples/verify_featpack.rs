fn main() {
    let path = std::env::args().nth(1).unwrap_or("data/featpack-fieldtest.vsf".into());
    match mahere_osm::read_featpack(&path) {
        Ok(f) => {
            let mut by_class = [0usize; mahere_osm::CLASS_COUNT];
            let mut pts = 0usize;
            for r in &f {
                by_class[r.class as usize] += 1;
                pts += r.pts.len();
            }
            println!("OK: {} features, {} points", f.len(), pts);
            println!("by class: {:?}", by_class);
        }
        Err(e) => println!("READ FAILED: {e}"),
    }
}
