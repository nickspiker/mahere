//! `route <cells dir | bucket> <lat lon> <lat lon> [car|bike|foot]`: the best route between two points from a cell directory or the bucket, its length, time, climb and turns. A published directory's route bundles (`route/`) and the bucket's are read past the coarsest depth, as a device reads them; MAHERE_NO_BUNDLES=1 reads every cell from its base.
//! MAHERE_SLOW_NET=<ms>: the stores as ferros's network has them (2026-11-27): eight fetches at once over every layer, each waiting `ms` for its first byte, every body through one link of MAHERE_SLOW_MBPS (default 1) megabytes a second; a local read finds only what was fetched (the device's vault). The bytes fetched are counted, and the base cells read are set against bundling schemes.
use std::sync::{Arc, Condvar, Mutex};

use mahere_engine::residency::{CellStore, Fetch};
use mahere_tiles::CellKey;
use mahere_tiles::publish::Layer;

/// The modelled network: eight connections, one link.
struct Net {
    ms: u64,
    mbps: f64,
    busy: Mutex<usize>,
    free: Condvar,
    link: Mutex<()>,
    fetches: Mutex<(usize, usize)>,
}

/// A store behind the modelled network, remembering what came through it.
struct Slow {
    inner: Arc<dyn CellStore>,
    net: Arc<Net>,
    kept: Mutex<Vec<(CellKey, Vec<u8>)>>,
}

impl CellStore for Slow {
    fn get(&self, key: CellKey) -> Fetch {
        if let Some(b) = self.kept.lock().unwrap().iter().find(|(k, _)| *k == key).map(|(_, b)| b.clone()) {
            return Fetch::Bytes(b);
        }
        let n = &self.net;
        let mut busy = n.free.wait_while(n.busy.lock().unwrap(), |b| *b >= 8).unwrap();
        *busy += 1;
        drop(busy);
        std::thread::sleep(std::time::Duration::from_millis(n.ms));
        let got = self.inner.get(key);
        if let Fetch::Bytes(b) = &got {
            let _link = n.link.lock().unwrap();
            std::thread::sleep(std::time::Duration::from_secs_f64(b.len() as f64 / (n.mbps * 1e6)));
            let mut f = n.fetches.lock().unwrap();
            f.0 += 1;
            f.1 += b.len();
        }
        *n.busy.lock().unwrap() -= 1;
        n.free.notify_one();
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
    // A published directory (base/, dem/, route/ under it), the bucket, or a flat directory.
    let root = std::path::PathBuf::from(&a[1]);
    let layer = |l: Layer| -> Arc<dyn CellStore> {
        if a[1] == "bucket" {
            // The bucket itself, as a phone with nothing on it reads it.
            Arc::new(mahere_engine::residency::HttpStore::layer(mahere_engine::residency::DEFAULT_CELLS_URL, l))
        } else {
            Arc::new(mahere_engine::residency::LayerDir(root.clone(), l))
        }
    };
    let (store, dem, bundles): (Arc<dyn CellStore>, Arc<dyn CellStore>, Option<Arc<dyn CellStore>>) = if a[1] == "bucket" || root.join("base").is_dir() {
        (layer(Layer::Base), layer(Layer::Dem), std::env::var_os("MAHERE_NO_BUNDLES").is_none().then(|| layer(Layer::Route)))
    } else {
        let s = Arc::new(mahere_engine::residency::DirStore(root.clone()));
        (s.clone(), s, None)
    };
    let net = std::env::var("MAHERE_SLOW_NET").ok().and_then(|v| v.parse::<u64>().ok()).map(|ms| {
        let mbps = std::env::var("MAHERE_SLOW_MBPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1.);
        Arc::new(Net { ms, mbps, busy: Mutex::new(0), free: Condvar::new(), link: Mutex::new(()), fetches: Mutex::new((0, 0)) })
    });
    let slow = |s: Arc<dyn CellStore>, net: &Arc<Net>| Arc::new(Slow { inner: s, net: net.clone(), kept: Mutex::new(Vec::new()) });
    let slow_base = net.as_ref().map(|n| slow(store.clone(), n));
    let (store, dem, bundles) = match (&net, &slow_base) {
        (Some(n), Some(s)) => (s.clone() as Arc<dyn CellStore>, slow(dem, n) as Arc<dyn CellStore>, bundles.map(|b| slow(b, n) as Arc<dyn CellStore>)),
        _ => (store, dem, bundles),
    };
    let t = std::time::Instant::now();
    match mahere_engine::route::find_cached(store, bundles, dem, (n(2), n(3)), (n(4), n(5)), travel, Some(Arc::new(Default::default()))) {
        Ok(r) => {
            let longest = r.pts.windows(2).map(|p| ((p[0].0 - p[1].0).powi(2) + ((p[0].1 - p[1].1) * p[0].0.to_radians().cos()).powi(2)).sqrt() * 111_320.0).fold(0.0, f64::max);
            println!("{} {:.2} km, {:.0} min, {:.0} m climbed, found in {:.0} ms; {} points, the longest piece {:.0} m", r.travel.word(), r.metres / 1000.0, r.seconds / 60.0, r.climb_m, t.elapsed().as_secs_f64() * 1e3, r.pts.len(), longest);
            for s in &r.steps {
                println!("  {:>6.2} km  {} {}", s.at_m / 1000.0, s.turn, if s.name.is_empty() { "(unnamed)" } else { &s.name });
            }
        }
        Err(e) => println!("no route: {e} ({:.0} ms)", t.elapsed().as_secs_f64() * 1e3),
    }
    if let (Some(n), Some(s)) = (&net, &slow_base) {
        let (count, bytes) = *n.fetches.lock().unwrap();
        println!("network: {count} fetches, {:.1} MB", bytes as f64 / 1e6);
        if std::env::var_os("MAHERE_BUNDLE_MODEL").is_some() {
            bundle_model(&s.kept.lock().unwrap());
        }
    }
}

/// The cells a route fetched, as route bundles would fetch them: for each scheme (cells a bundle holds, as how many depths down from its key, by depth 8 to 14), the bundles fetched and their bytes, a bundle's taken as its depth's mean route section (and crossing at the coarsest) times its cells.
fn bundle_model(kept: &[(CellKey, Vec<u8>)]) {
    let mut by_depth: Vec<Vec<(CellKey, usize, usize)>> = vec![Vec::new(); 15];
    for (k, b) in kept {
        if !(8..=14).contains(&k.depth) {
            continue;
        }
        let Ok((h, _)) = vsf::VsfHeader::decode(b) else { continue };
        let size = |n: &str| h.fields.iter().find(|f| f.name == n).map_or(0, |f| f.size_bytes);
        by_depth[k.depth as usize].push((*k, size("route"), size("crossing")));
    }
    let whole: usize = kept.iter().map(|(_, b)| b.len()).sum();
    println!("now: {} cells fetched, {:.1} MB", kept.len(), whole as f64 / 1e6);
    for (name, ks) in [("each cell alone, route only", [0u8; 7]), ("4 a bundle", [0, 1, 1, 1, 1, 1, 1]), ("16 at 12-14, 4 at 9-11", [0, 1, 1, 1, 2, 2, 2]), ("16 a bundle from 10", [0, 1, 2, 2, 2, 2, 2]), ("64 at 13-14", [0, 1, 1, 1, 2, 3, 3])] {
        let (mut fetches, mut bytes) = (0usize, 0f64);
        let mut parts = Vec::new();
        for d in 8..=14usize {
            let cells = &by_depth[d];
            if cells.is_empty() {
                continue;
            }
            let k = ks[d - 8];
            let mut bundles: Vec<CellKey> = cells.iter().map(|c| c.0.ancestor(d as u8 - k)).collect();
            bundles.sort_by_key(|b| (b.depth, b.prefix));
            bundles.dedup();
            let mean = cells.iter().map(|c| c.1 + if d == 8 { c.2 } else { 0 }).sum::<usize>() as f64 / cells.len() as f64;
            let b = bundles.len() as f64 * mean * (1u64 << (2 * k)) as f64;
            fetches += bundles.len();
            bytes += b;
            parts.push(format!("d{d} {}", bundles.len()));
        }
        println!("{name}: {fetches} fetches, {:.1} MB ({})", bytes / 1e6, parts.join(", "));
    }
}
