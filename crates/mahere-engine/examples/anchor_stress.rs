// anchor_stress: random cameras, anchors and zooms; reports the worst anchor error after a zoom and after a pan.
use mahere_engine::{Camera, MapCore};
use std::sync::Arc;
struct Nothing;
impl mahere_engine::residency::CellStore for Nothing {
    fn get(&self, _key: mahere_tiles::CellKey) -> mahere_engine::residency::Fetch {
        mahere_engine::residency::Fetch::Absent
    }
}
fn main() {
    let (w, h) = (1080, 2340);
    let mut seed = 12345u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 1_000_000) as f64 / 1_000_000.0
    };
    let mut worst = (0.0f64, String::new());
    let mut bad_fine = 0;
    let mut bad_coarse = 0;
    for _ in 0..2000 {
        let cam = Camera { lat: rnd() * 170.0 - 85.0, lon: rnd() * 360.0 - 180.0, ppd: 10f64.powf(1.0 + rnd() * 5.0), bearing: rnd() * 6.28 };
        let mut map = MapCore::new(Arc::new(Nothing), cam);
        let (ax, ay) = (rnd() * w as f64, rnd() * h as f64);
        if !map.cam.on_globe(ax, ay, w, h) {
            continue;
        }
        let anchor = map.cam.screen_to_geo(ax, ay, w, h);
        let f = 0.5 + rnd() * 1.5;
        map.zoom_about(f, ax, ay, w, h);
        let (qx, qy) = map.cam.geo_to_screen(anchor.0, anchor.1, w, h);
        let err = (qx - ax).hypot(qy - ay);
        if err > 1.0 {
            if cam.ppd > 100.0 { bad_fine += 1; println!("fine fail: zoom {f:.2} at ({ax:.0},{ay:.0}) cam {:.2},{:.2} ppd {:.1} bearing {:.2} -> ({qx:.1},{qy:.1})", cam.lat, cam.lon, cam.ppd, cam.bearing); } else { bad_coarse += 1; }
        }
        if err > worst.0 {
            worst = (err, format!("zoom {f:.2} at ({ax:.0},{ay:.0}) cam {:.2},{:.2} ppd {:.1} bearing {:.2} -> ({qx:.1},{qy:.1})", cam.lat, cam.lon, cam.ppd, cam.bearing));
        }
        let (dx, dy) = (rnd() * 600.0 - 300.0, rnd() * 600.0 - 300.0);
        let before = map.cam.screen_to_geo(w as f64 * 0.5 - dx, h as f64 * 0.5 - dy, w, h);
        map.pan(dx, dy, w, h);
        let (qx, qy) = map.cam.geo_to_screen(before.0, before.1, w, h);
        let err = (qx - w as f64 * 0.5).hypot(qy - h as f64 * 0.5);
        if err > worst.0 {
            worst = (err, format!("pan ({dx:.0},{dy:.0}) cam {:.2},{:.2} ppd {:.1} bearing {:.2} -> centre error ({qx:.1},{qy:.1})", map.cam.lat, map.cam.lon, map.cam.ppd, map.cam.bearing));
        }
    }
    println!("worst error {:.3} px: {}", worst.0, worst.1);
    println!("zoom failures over 1 px: {bad_fine} at ppd > 100, {bad_coarse} at the globe zooms");
}
