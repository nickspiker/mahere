//! `route <cells dir> <lat lon> <lat lon> [car|bike|foot]`: the best route between two points from a cell directory, its length, time, climb and turns.
use std::sync::Arc;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n = |i: usize| a[i].parse::<f64>().expect("a number");
    let travel = match a.get(6).map(String::as_str) {
        Some("bike") => mahere_engine::route::Travel::Bike,
        Some("foot") => mahere_engine::route::Travel::Foot,
        _ => mahere_engine::route::Travel::Car,
    };
    let store = Arc::new(mahere_engine::residency::DirStore(a[1].clone().into()));
    let t = std::time::Instant::now();
    match mahere_engine::route::find(store, (n(2), n(3)), (n(4), n(5)), travel) {
        Ok(r) => {
            println!("{} {:.2} km, {:.0} min, {:.0} m climbed, found in {:.0} ms", r.travel.word(), r.metres / 1000.0, r.seconds / 60.0, r.climb_m, t.elapsed().as_secs_f64() * 1e3);
            for s in &r.steps {
                println!("  {:>6.2} km  {} {}", s.at_m / 1000.0, s.turn, if s.name.is_empty() { "(unnamed)" } else { &s.name });
            }
        }
        Err(e) => println!("no route: {e} ({:.0} ms)", t.elapsed().as_secs_f64() * 1e3),
    }
}
