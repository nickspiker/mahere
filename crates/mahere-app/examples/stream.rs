// Headless streaming check: cells come from the bucket through the vault, exactly as the phone does it. Run twice — the second run should be served from the vault (no fetches). Pass a vault dir to keep it out of the real one.
use mahere_engine::residency::{DEFAULT_CELLS_URL, Fetch, HttpStore, RemoteStore, TieredStore};
use mahere_tiles::CellKey;
use mahere_engine::{Camera, MapCore};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting(HttpStore, AtomicUsize);
impl RemoteStore for Counting {
    fn fetch(&self, key: CellKey) -> Fetch {
        self.1.fetch_add(1, Ordering::Relaxed);
        self.0.fetch(key)
    }
}

fn main() {
    let vault_dir = std::env::args().nth(1).unwrap_or_else(|| "/tmp/claude-1000/mahere-stream-vault".into());
    let vault = mahere_store::open(Some(&vault_dir)).expect("vault");
    let remote = Arc::new(Counting(HttpStore::new(DEFAULT_CELLS_URL), AtomicUsize::new(0)));
    let store = Arc::new(TieredStore::new(
        Arc::new(mahere_store::VaultCells(vault.clone())),
        remote.clone(),
    ));
    let mut map = MapCore::new(store, Camera { lat: 46.20, lon: -122.19, ppd: 12000.0, bearing: 0.0 });
    let (w, h) = (1024usize, 768usize);
    let t = std::time::Instant::now();
    for _ in 0..3000 {
        map.render(w, h);
        if map.converged() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
        map.tick(w, h);
    }
    map.render(w, h);
    let fetched = remote.1.load(Ordering::Relaxed);
    {
        let f = std::fs::File::create("/tmp/claude-1000/stream_view.png").unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
        enc.set_color(png::ColorType::Rgb);
        let mut wr = enc.write_header().unwrap();
        let mut buf = Vec::with_capacity(w * h * 3);
        for &p in &map.canvas {
            buf.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8]);
        }
        wr.write_image_data(&buf).unwrap();
    }
    let data_px = map.canvas.iter().filter(|&&p| p != 0x12141a).count();
    eprintln!(
        "converged in {:.1}s, {} remote fetches, {:.0}% of pixels have data, frame {:.2} ms",
        t.elapsed().as_secs_f32(),
        fetched,
        100.0 * data_px as f32 / (w * h) as f32,
        map.last_frame_ms
    );
}
