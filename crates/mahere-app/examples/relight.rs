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
    let store = std::sync::Arc::new(DirStore("data/cells".into()));
    let mut map = MapCore::new(store, Camera { lat: 46.2024, lon: -121.4909, ppd: 2800.0, bearing: 0.0 });
    let (w, h) = (1024usize, 768usize);
    settle(&mut map, w, h);
    map.render(w, h);
    eprintln!("frame: {:.2} ms ({} straddle blocks)", map.last_frame_ms, map.last_straddle_blocks);
    save("/tmp/claude-1000/adams_nw.png", &map.canvas, w, h);
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
