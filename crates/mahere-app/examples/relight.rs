// Headless pipeline check: load cells for Mt Adams, settle residency, render
// at two sun azimuths from the SAME resident cells (zero re-bakes), plus a
// rotation render. Prints frame times — the perf receipt.
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

fn settle(map: &mut MapCore, w: usize, h: usize) {
    // render issues want-lists; tick drains. Loop until the loader is idle.
    for _ in 0..600 {
        map.render(w, h);
        if map.converged() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
        map.tick(w, h);
    }
}

fn main() {
    // relight [lat lon ppd [out.png]]: render that view (default Mt Adams).
    let args: Vec<String> = std::env::args().collect();
    let num = |i: usize, d: f64| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let cam = Camera { lat: num(1, 46.2000), lon: num(2, -122.1900), ppd: num(3, 12_000.0), bearing: 0.0 };
    let out = args.get(4).cloned().unwrap_or_else(|| "/tmp/claude-1000/adams_nw.png".into());
    let store = std::sync::Arc::new(DirStore("data/cells".into()));
    let mut map = MapCore::new(store, cam);
    let (w, h) = (1024usize, 768usize);
    if std::env::var("MAHERE_DEBUG").is_ok() {
        // Residency debug: a cold frame mid-stream, then the settled one.
        let mut m = map.layers();
        m.debug = true;
        map.set_layers(m);
        map.render(w, h);
        std::thread::sleep(std::time::Duration::from_millis(60));
        map.tick(w, h);
        map.render(w, h);
        save("/tmp/claude-1000/residency_cold.png", &map.canvas, w, h);
    }
    settle(&mut map, w, h);
    map.render(w, h);
    eprintln!("frame: {:.2} ms ({} straddle blocks)", map.last_frame_ms, map.last_straddle_blocks);
    save(&out, &map.canvas, w, h);
    if args.len() > 1 {
        return;
    }
    // find first magenta pixel and probe its chain
    map.sun_az = 135.0;
    map.sun_alt = 25.0;
    let t = std::time::Instant::now();
    map.render(w, h);
    eprintln!("relight frame: {:.2} ms", t.elapsed().as_secs_f32() * 1000.0);
    save("/tmp/claude-1000/adams_se.png", &map.canvas, w, h);
    map.set_bearing(90f64.to_radians());
    settle(&mut map, w, h);
    map.render(w, h);
    eprintln!("rotated frame: {:.2} ms", map.last_frame_ms);
    save("/tmp/claude-1000/adams_rot90.png", &map.canvas, w, h);
    // Timing sweep: 20 frames, report mean.
    let t = std::time::Instant::now();
    for _ in 0..20 {
        map.render(w, h);
    }
    eprintln!("mean over 20 frames: {:.2} ms", t.elapsed().as_secs_f32() * 50.0);
}
