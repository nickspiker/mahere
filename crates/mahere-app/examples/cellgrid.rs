// Diagnostic: the Adams view with dymaxion cell outlines drawn over it —
// depth-10 dem cell edges in yellow, depth-8 in cyan — so the actual tile
// footprints (diamond-UV rhombi, not lat/lon rectangles) are visible.
use mahere_coord::Coord;
use mahere_engine::{Camera, MapCore, residency::DirStore};

fn save(path: &str, canvas: &[u32], w: usize, h: usize) {
    let f = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    let mut wr = enc.write_header().unwrap();
    let mut buf = Vec::with_capacity(w * h * 3);
    for &p in canvas {
        buf.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8]);
    }
    wr.write_image_data(&buf).unwrap();
}

fn main() {
    let ppd: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2800.0);
    let store = std::sync::Arc::new(DirStore("data/cells".into()));
    let cam = Camera { lat: 46.2024, lon: -121.4909, ppd, bearing: 0.0 };
    let mut map = MapCore::new(store, cam);
    let (w, h) = (1024usize, 768usize);
    for _ in 0..600 {
        map.render(w, h);
        if map.converged() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
        map.tick(w, h);
    }
    map.render(w, h);
    let mut canvas = map.canvas.clone();
    let key = |px: usize, py: usize, depth: u8| -> u64 {
        let (lat, lon) = cam.screen_to_geo(px as f64, py as f64, w, h);
        Coord::from_lat_lon(lat, lon).raw() >> (60 - 2 * depth as u32)
    };
    for py in 0..h - 1 {
        for px in 0..w - 1 {
            for (depth, color) in [(10u8, 0xFFE000u32), (8u8, 0x00E0FF)] {
                let k = key(px, py, depth);
                if k != key(px + 1, py, depth) || k != key(px, py + 1, depth) {
                    canvas[py * w + px] = color;
                }
            }
        }
    }
    save("/tmp/claude-1000/adams_cellgrid.png", &canvas, w, h);
}
