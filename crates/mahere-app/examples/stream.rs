// Headless streaming check: published cells from the bucket through a fresh vault, as the phone loads them. Times a view from nothing to settled, then a pan of half a screen and a zoom in after the view has had a few seconds to settle, with the bucket requests each made.
// cargo run --release -p mahere-app --example stream -- [vault dir] [lat lon]
use mahere_engine::residency::{CellStore, Checked, DEFAULT_CELLS_URL, Fetch, HttpStore, LayerStores, RemoteStore, TieredStore};
use mahere_engine::{Camera, MapCore};
use mahere_tiles::CellKey;
use mahere_tiles::publish::Layer;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

static FETCHES: AtomicUsize = AtomicUsize::new(0);

struct Counting(HttpStore, Layer);
impl Counting {
    /// Count a request, and name one that took over a second.
    fn timed<T>(&self, key: CellKey, what: &str, f: impl FnOnce() -> T) -> T {
        FETCHES.fetch_add(1, Ordering::Relaxed);
        let t = std::time::Instant::now();
        let r = f();
        if t.elapsed().as_secs_f32() > 1.0 {
            eprintln!("slow {what} {} {}: {:.1} s", self.1.dir(), key.name(), t.elapsed().as_secs_f32());
        }
        r
    }
}
impl RemoteStore for Counting {
    fn fetch(&self, key: CellKey) -> Fetch {
        self.timed(key, "fetch", || self.0.fetch(key))
    }
    fn fetch_named(&self, key: CellKey, name: &str) -> Fetch {
        self.timed(key, "named fetch", || self.0.fetch_named(key, name))
    }
    fn check(&self, key: CellKey, tag: Option<&str>) -> Checked {
        self.timed(key, if tag.is_some() { "check" } else { "get" }, || self.0.check(key, tag))
    }
}

/// Render and tick until the view settles; the seconds it took.
fn settle(map: &mut MapCore, w: usize, h: usize) -> f32 {
    let t = std::time::Instant::now();
    for _ in 0..6000 {
        map.render(w, h);
        if map.converged() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        map.tick(w, h);
    }
    t.elapsed().as_secs_f32()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vault_dir = args.get(1).cloned().unwrap_or_else(|| "/tmp/claude-1000/mahere-stream-vault".into());
    let lat: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(47.596);
    let lon: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(-120.661);
    let vault = mahere_store::open(Some(&vault_dir)).expect("vault");
    let cache = Arc::new(mahere_store::VaultCells::new(vault.clone()));
    let tiered = |c: Arc<dyn mahere_engine::residency::CellCache>, l: Layer| -> Arc<dyn CellStore> { Arc::new(TieredStore::new(c, Arc::new(Counting(HttpStore::layer(DEFAULT_CELLS_URL, l), l)))) };
    let layers = LayerStores { dem: tiered(cache.layer(Layer::Dem), Layer::Dem), img: tiered(cache.layer(Layer::Img), Layer::Img), nir: tiered(cache.layer(Layer::Nir), Layer::Nir) };
    let mut map = MapCore::layered(tiered(cache.clone(), Layer::Base), Some(layers), Camera::new(lat, lon, 12000.0, 0.0));
    let (w, h) = (1080usize, 2340usize);
    let report = |what: &str, s: f32, before: usize| eprintln!("{what}: {s:.2} s, {} bucket requests", FETCHES.load(Ordering::Relaxed) - before);
    let n = FETCHES.load(Ordering::Relaxed);
    report("view from nothing", settle(&mut map, w, h), n);
    std::thread::sleep(std::time::Duration::from_secs(4));
    let n = FETCHES.load(Ordering::Relaxed);
    map.pan(w as f64 * 0.5, 0.0, w, h);
    report("pan half a screen", settle(&mut map, w, h), n);
    std::thread::sleep(std::time::Duration::from_secs(4));
    let n = FETCHES.load(Ordering::Relaxed);
    map.zoom_about(2.0, w as f64 * 0.5, h as f64 * 0.5, w, h);
    report("zoom in", settle(&mut map, w, h), n);
}
