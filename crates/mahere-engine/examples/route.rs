//! `route <cells dir> <lat lon> <lat lon> [car|bike|foot]`: the best route between two points from a cell directory, its length, time, climb and turns.
//! MAHERE_SLOW_NET=<ms>: the store as ferros's network has it (2026-11-27): eight fetches at once, each `ms` long, and a local read finding only what was fetched (the device's vault).
use std::sync::{Arc, Condvar, Mutex};

use mahere_engine::residency::{CellStore, Fetch};
use mahere_tiles::CellKey;

/// A store behind a slow network of eight connections, remembering what came through it.
struct Slow {
    inner: Arc<dyn CellStore>,
    ms: u64,
    busy: Mutex<usize>,
    free: Condvar,
    kept: Mutex<Vec<(CellKey, Vec<u8>)>>,
}

impl CellStore for Slow {
    fn get(&self, key: CellKey) -> Fetch {
        if let Some(b) = self.kept.lock().unwrap().iter().find(|(k, _)| *k == key).map(|(_, b)| b.clone()) {
            return Fetch::Bytes(b);
        }
        let mut busy = self.free.wait_while(self.busy.lock().unwrap(), |b| *b >= 8).unwrap();
        *busy += 1;
        drop(busy);
        std::thread::sleep(std::time::Duration::from_millis(self.ms));
        let got = self.inner.get(key);
        *self.busy.lock().unwrap() -= 1;
        self.free.notify_one();
        if let Fetch::Bytes(b) = &got {
            self.kept.lock().unwrap().push((key, b.clone()));
        }
        got
    }

    fn get_local(&self, key: CellKey) -> Fetch {
        self.kept.lock().unwrap().iter().find(|(k, _)| *k == key).map_or(Fetch::Absent, |(_, b)| Fetch::Bytes(b.clone()))
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n = |i: usize| a[i].parse::<f64>().expect("a number");
    let travel = match a.get(6).map(String::as_str) {
        Some("bike") => mahere_engine::route::Travel::Bike,
        Some("foot") => mahere_engine::route::Travel::Foot,
        _ => mahere_engine::route::Travel::Car,
    };
    // A published directory (base/, dem/ under it) or a flat one.
    let root = std::path::PathBuf::from(&a[1]);
    let (store, dem): (Arc<dyn mahere_engine::residency::CellStore>, Arc<dyn mahere_engine::residency::CellStore>) = if a[1] == "bucket" {
        // The bucket itself, as a phone with nothing on it reads it.
        use mahere_engine::residency::{HttpStore, DEFAULT_CELLS_URL};
        (Arc::new(HttpStore::layer(DEFAULT_CELLS_URL, mahere_tiles::publish::Layer::Base)), Arc::new(HttpStore::layer(DEFAULT_CELLS_URL, mahere_tiles::publish::Layer::Dem)))
    } else if root.join("base").is_dir() {
        (Arc::new(mahere_engine::residency::LayerDir(root.clone(), mahere_tiles::publish::Layer::Base)), Arc::new(mahere_engine::residency::LayerDir(root, mahere_tiles::publish::Layer::Dem)))
    } else {
        let s = Arc::new(mahere_engine::residency::DirStore(root));
        (s.clone(), s)
    };
    let slow = |s: Arc<dyn CellStore>, ms: u64| -> Arc<dyn CellStore> { Arc::new(Slow { inner: s, ms, busy: Mutex::new(0), free: Condvar::new(), kept: Mutex::new(Vec::new()) }) };
    let (store, dem) = match std::env::var("MAHERE_SLOW_NET").ok().and_then(|v| v.parse::<u64>().ok()) {
        Some(ms) => (slow(store, ms), slow(dem, ms)),
        None => (store, dem),
    };
    let t = std::time::Instant::now();
    match mahere_engine::route::find(store, dem, (n(2), n(3)), (n(4), n(5)), travel) {
        Ok(r) => {
            let longest = r.pts.windows(2).map(|p| ((p[0].0 - p[1].0).powi(2) + ((p[0].1 - p[1].1) * p[0].0.to_radians().cos()).powi(2)).sqrt() * 111_320.0).fold(0.0, f64::max);
            println!("{} {:.2} km, {:.0} min, {:.0} m climbed, found in {:.0} ms; {} points, the longest piece {:.0} m", r.travel.word(), r.metres / 1000.0, r.seconds / 60.0, r.climb_m, t.elapsed().as_secs_f64() * 1e3, r.pts.len(), longest);
            for s in &r.steps {
                println!("  {:>6.2} km  {} {}", s.at_m / 1000.0, s.turn, if s.name.is_empty() { "(unnamed)" } else { &s.name });
            }
        }
        Err(e) => println!("no route: {e} ({:.0} ms)", t.elapsed().as_secs_f64() * 1e3),
    }
}
