//! geocode <published dir> "<address>" [lat lon] [--route lat lon car|bike|foot]: the places a typed address could be, from the address documents under the directory (`addr/`), best first; near `lat lon` when it names no ZIP; with --route, a route from that place to the best one, over the directory's cells.
use std::sync::Arc;

struct Dir(std::path::PathBuf);
impl mahere_engine::geocode::DocStore for Dir {
    fn doc(&self, path: &str) -> Option<Vec<u8>> {
        std::fs::read(self.0.join(path)).ok()
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let root = std::path::PathBuf::from(&a[1]);
    let near = match (a.get(3).and_then(|x| x.parse().ok()), a.get(4).and_then(|x| x.parse().ok())) {
        (Some(lat), Some(lon)) => (lat, lon),
        _ => (0.0, 0.0),
    };
    let t = std::time::Instant::now();
    let found = mahere_engine::geocode::geocode(&a[2], near, &Dir(root.clone()));
    println!("{} found in {:.0} ms", found.len(), t.elapsed().as_secs_f64() * 1e3);
    for f in &found {
        println!("  {:<40} {:.6} {:.6}  {} (score {})", f.label, f.lat, f.lon, f.detail, f.score);
    }
    if let (Some(i), Some(best)) = (a.iter().position(|x| x == "--route"), found.first()) {
        let from: (f64, f64) = (a[i + 1].parse().unwrap(), a[i + 2].parse().unwrap());
        let travel = match a.get(i + 3).map(String::as_str) {
            Some("bike") => mahere_engine::route::Travel::Bike,
            Some("foot") => mahere_engine::route::Travel::Foot,
            _ => mahere_engine::route::Travel::Car,
        };
        let base: Arc<dyn mahere_engine::residency::CellStore> = Arc::new(mahere_engine::residency::LayerDir(root.clone(), mahere_tiles::publish::Layer::Base));
        let dem: Arc<dyn mahere_engine::residency::CellStore> = Arc::new(mahere_engine::residency::LayerDir(root, mahere_tiles::publish::Layer::Dem));
        let t = std::time::Instant::now();
        match mahere_engine::route::find(base, dem, from, (best.lat, best.lon), travel) {
            Ok(r) => {
                println!("{} {:.1} km, {} h {:02} min, found in {:.1} s", r.travel.word(), r.metres / 1000.0, (r.seconds / 3600.0) as u32, ((r.seconds / 60.0) as u32) % 60, t.elapsed().as_secs_f64());
                let all = std::env::var_os("ALL_STEPS").is_some();
                let steps: Vec<&mahere_engine::route::Step> = if all { r.steps.iter().collect() } else { r.steps.iter().take(4).chain(r.steps.iter().rev().take(4).collect::<Vec<_>>().into_iter().rev()).collect() };
                for s in steps {
                    println!("  {:>7.1} km  {} {}", s.at_m / 1000.0, s.turn, if s.name.is_empty() { "the way" } else { &s.name });
                }
            }
            Err(e) => println!("no route: {e}"),
        }
    }
}
