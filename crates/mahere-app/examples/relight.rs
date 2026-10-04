// Headless engine check: converge the reservoir over Mt Rainier and splat
// at two sun azimuths. Distinct relief in both = hillshade works; the pair
// differing = splat-time relighting works (zero re-evaluation between them).
//
// Build quirk: examples can't reach a bin crate's modules, so include the
// engine source directly and provide the Camera it expects.
#[path = "../src/terrain.rs"]
mod terrain;

mod main_shim {
    #[derive(Clone, Copy)]
    pub struct Camera {
        pub lat: f64,
        pub lon: f64,
        pub ppd: f64,
    }
    impl Camera {
        fn coslat(&self) -> f64 {
            self.lat.to_radians().cos()
        }
        pub fn geo_to_screen(&self, lat: f64, lon: f64, w: usize, h: usize) -> (f64, f64) {
            let ppd_lon = self.ppd * self.coslat();
            (
                w as f64 * 0.5 + (lon - self.lon) * ppd_lon,
                h as f64 * 0.5 - (lat - self.lat) * self.ppd,
            )
        }
        pub fn screen_to_geo(&self, px: f64, py: f64, w: usize, h: usize) -> (f64, f64) {
            let ppd_lon = self.ppd * self.coslat();
            (
                self.lat + (h as f64 * 0.5 - py) / self.ppd,
                self.lon + (px - w as f64 * 0.5) / ppd_lon,
            )
        }
    }
}
use main_shim as crate_shim;
// terrain.rs does `use crate::Camera;` — satisfy it:
use crate_shim::Camera;

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
    let mut t = terrain::Terrain::new(dem);
    let cam = Camera { lat: 46.8523, lon: -121.7603, ppd: 2800.0 };
    let (w, h) = (1024usize, 768usize);
    let start = std::time::Instant::now();
    let mut ticks = 0;
    while !t.converged() {
        t.tick(w, h, &cam);
        ticks += 1;
    }
    eprintln!("converged in {} ticks, {:.1}s", ticks, start.elapsed().as_secs_f32());
    let mut canvas = vec![0x12141Au32; w * h];
    t.splat(&mut canvas, w, h, &cam, 315.0, 40.0);
    save("/tmp/claude-1000/rainier_nw.png", &canvas, w, h);
    let start = std::time::Instant::now();
    t.splat(&mut canvas, w, h, &cam, 135.0, 25.0);
    eprintln!("relight splat: {:.0} ms", start.elapsed().as_secs_f32() * 1000.0);
    save("/tmp/claude-1000/rainier_se.png", &canvas, w, h);
}
