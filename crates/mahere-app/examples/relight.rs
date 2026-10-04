// Headless engine check: converge the reservoir over Mt Rainier and splat
// at two sun azimuths. Distinct relief in both = hillshade works; the pair
// differing = splat-time relighting works (zero re-evaluation between them).
use mahere_engine::{Camera, MapCore, terrain::Terrain};

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
    let dem = mahere_dem::DemStore::load(&[
        "data/USGS_1_n47w122.tif".into(),
        "data/USGS_1_n47w123.tif".into(),
    ])
    .unwrap();
    let mut map = MapCore::new(
        Vec::new(),
        Terrain::new(dem),
        Camera { lat: 46.8523, lon: -121.7603, ppd: 2800.0, bearing: 0.0 },
    );
    let (w, h) = (1024usize, 768usize);
    map.camera_moved(w, h);
    let start = std::time::Instant::now();
    let mut ticks = 0;
    while !map.converged() {
        map.tick(w, h);
        ticks += 1;
    }
    eprintln!("converged in {} ticks, {:.1}s", ticks, start.elapsed().as_secs_f32());
    map.render(w, h);
    save("/tmp/claude-1000/rainier_nw.png", &map.canvas, w, h);
    map.sun_az = 135.0;
    map.sun_alt = 25.0;
    map.mark_dirty();
    let start = std::time::Instant::now();
    map.render(w, h);
    eprintln!("relight: {:.0} ms", start.elapsed().as_secs_f32() * 1000.0);
    save("/tmp/claude-1000/rainier_se.png", &map.canvas, w, h);
    // Rotation sanity: 90 degrees CW, re-render, save.
    map.set_bearing(90f64.to_radians());
    map.camera_moved(w, h);
    while !map.converged() {
        map.tick(w, h);
    }
    map.render(w, h);
    save("/tmp/claude-1000/rainier_rot90.png", &map.canvas, w, h);
}
