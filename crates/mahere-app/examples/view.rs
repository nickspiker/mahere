// One view rendered from the published cells through a fresh vault, as a phone would, saved as PNG: for checking what a phone shows.
// cargo run --release -p mahere-app --example view -- <out.png> <lat> <lon> <ppd> <w> <h> [theme name]
// CELLS=<baked dir> renders a local bake instead (a check before upload); the labels are drawn over the map either way.
// MAHERE_ROUTE=<lat>,<lon>,<lat>,<lon>[,car|bike|foot] draws directions between two points over it (a check of the line against the trail).
use std::sync::Arc;

use mahere_engine::residency::{DEFAULT_CELLS_URL, HttpStore, LayerStores, TieredStore};
use mahere_engine::{Camera, MapCore};
use mahere_tiles::publish::Layer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let num = |i: usize| a[i].parse::<f64>().expect("a number");
    let (out, lat, lon, ppd, w, h) = (a[1].clone(), num(2), num(3), num(4), num(5) as usize, num(6) as usize);
    let dir = format!("{out}.vault");
    let cache = mahere_store::VaultCells::open(mahere_store::open(Some(&dir)).expect("vault"));
    let tiered = |c: Arc<dyn mahere_engine::residency::CellCache>, l: Layer| -> Arc<dyn mahere_engine::residency::CellStore> { Arc::new(TieredStore::new(c, Arc::new(HttpStore::layer(DEFAULT_CELLS_URL, l)))) };
    let layers = LayerStores { dem: tiered(cache.layer(Layer::Dem), Layer::Dem), img: tiered(cache.layer(Layer::Img), Layer::Img), nir: tiered(cache.layer(Layer::Nir), Layer::Nir) };
    let mut map = match std::env::var("CELLS") {
        Ok(dir) => {
            let (store, layers, _) = mahere_store::cell_stores(Some(std::path::Path::new(&dir)), None);
            MapCore::layered(store, layers, Camera::new(lat, lon, ppd, 0.0))
        }
        Err(_) => MapCore::layered(tiered(cache.clone(), Layer::Base), Some(layers), Camera::new(lat, lon, ppd, 0.0)),
    };
    map.set_themes(mahere_engine::theme::builtin());
    if let Some(name) = a.get(7) {
        let i = mahere_engine::theme::builtin().iter().position(|t| t.name == *name).expect("a built-in theme");
        map.set_theme(i);
    }
    if let Ok(spec) = std::env::var("MAHERE_ROUTE") {
        let v: Vec<&str> = spec.split(',').collect();
        let n = |i: usize| v[i].trim().parse::<f64>().expect("a number");
        let travel = match v.get(4).map(|s| s.trim()) {
            Some("car") => mahere_engine::route::Travel::Car,
            Some("bike") => mahere_engine::route::Travel::Bike,
            _ => mahere_engine::route::Travel::Foot,
        };
        map.set_route_mode(mahere_engine::RouteMode::Directions);
        map.set_travel(Some(travel));
        map.set_pin(mahere_engine::Pin::A, (n(0), n(1)));
        map.set_pin(mahere_engine::Pin::B, (n(2), n(3)));
    }
    for _ in 0..12000 {
        map.render(w, h);
        if map.converged() && !map.routing() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        map.tick(w, h);
    }
    if let Some(r) = map.route() {
        eprintln!("route {:.2} km, {} points", r.metres / 1000.0, r.pts.len());
    }
    map.render(w, h);
    let mut labels = mahere_panel::labels::Labeller::new();
    labels.update(&mut map, w, h, 15.0);
    map.render(w, h);
    labels.draw_cpu(&mut map.canvas, &map.cam, w, h);
    let f = std::fs::File::create(&out).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    let mut wr = enc.write_header().unwrap();
    let buf: Vec<u8> = map.canvas.iter().flat_map(|&p| [(p >> 16) as u8, (p >> 8) as u8, p as u8]).collect();
    wr.write_image_data(&buf).unwrap();
    eprintln!("vec depth {}", map.vec_depth());
}
