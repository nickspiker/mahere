// Headless check of font packs (Nick 2026-10-16): a view lettered before its packs arrive and after, both saved as PNG.
// cargo run --release -p mahere-app --example font_packs -- <out dir> [lat lon ppd]
use std::sync::Arc;

use mahere_engine::residency::{DEFAULT_CELLS_URL, HttpStore, LayerStores, TieredStore};
use mahere_engine::{Camera, MapCore};
use mahere_tiles::publish::Layer;

fn save(path: &str, px: &[u32], w: usize, h: usize) {
    let f = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    let mut wr = enc.write_header().unwrap();
    let buf: Vec<u8> = px.iter().flat_map(|&p| [(p >> 16) as u8, (p >> 8) as u8, p as u8]).collect();
    wr.write_image_data(&buf).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = a.get(1).cloned().unwrap_or_else(|| "/tmp/claude-1000".into());
    let lat: f64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(39.91);
    let lon: f64 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(116.40);
    let ppd: f64 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(2000.0);
    let vault = mahere_store::open(Some(&format!("{out}/fontcheck-vault"))).expect("vault");
    let cache = mahere_store::CellFiles::open(&std::path::Path::new(&format!("{out}/fontcheck-vault")).join("cells"));
    let tiered = |c: Arc<dyn mahere_engine::residency::CellCache>, l: Layer| -> Arc<dyn mahere_engine::residency::CellStore> { Arc::new(TieredStore::new(c, Arc::new(HttpStore::layer(DEFAULT_CELLS_URL, l)))) };
    let layers = LayerStores { dem: tiered(cache.layer(Layer::Dem), Layer::Dem), img: tiered(cache.layer(Layer::Img), Layer::Img), nir: tiered(cache.layer(Layer::Nir), Layer::Nir) };
    let mut map = MapCore::layered(tiered(cache.clone(), Layer::Base), Some(layers), Camera::new(lat, lon, ppd, 0.0));
    let (w, h) = (1200usize, 900usize);
    for _ in 0..6000 {
        map.render(w, h);
        if map.converged() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        map.tick(w, h);
    }
    let fonts = mahere_panel::fonts::FontPacks::new(mahere_store::VaultFonts::new(Some(vault.vault().clone())), mahere_update::AUTHOR_PUBKEY);
    let mut labels = mahere_panel::labels::Labeller::new();
    labels.set_fonts(fonts.clone());
    let shot = |labels: &mahere_panel::labels::Labeller, map: &mut MapCore, name: &str| {
        map.render(w, h);
        let mut px = map.canvas.clone();
        labels.draw_cpu(&mut px, &map.cam, w, h);
        save(&format!("{out}/{name}"), &px, w, h);
    };
    labels.update(&mut map, w, h, 14.0);
    shot(&labels, &mut map, "fontcheck-before.png");
    let t = std::time::Instant::now();
    while fonts.busy() && t.elapsed().as_secs() < 300 {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    eprintln!("packs arrived: {} in {:.1} s", fonts.arrived(), t.elapsed().as_secs_f32());
    labels.update(&mut map, w, h, 14.0);
    shot(&labels, &mut map, "fontcheck-after.png");
}
