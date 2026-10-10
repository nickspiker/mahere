// proj_check: round-trips screen points through the globe projection and anchors a zoom, printing where things land.
use mahere_engine::{Camera, MapCore};
use std::sync::Arc;
struct Nothing;
impl mahere_engine::residency::CellStore for Nothing {
    fn get(&self, _key: mahere_tiles::CellKey) -> mahere_engine::residency::Fetch {
        mahere_engine::residency::Fetch::Absent
    }
}
fn main() {
    let cam = Camera { lat: 47.6, lon: -120.7, ppd: 24000.0, bearing: 0.7 };
    let (w, h) = (1080, 2300);
    for &(px, py) in &[(100.0, 100.0), (900.0, 2000.0), (540.0, 1150.0), (50.0, 2250.0)] {
        let (lat, lon) = cam.screen_to_geo(px, py, w, h);
        let (qx, qy) = cam.geo_to_screen(lat, lon, w, h);
        println!("screen ({px},{py}) -> {lat:.6},{lon:.6} -> ({qx:.3},{qy:.3})");
    }
    let mut map = MapCore::new(Arc::new(Nothing), cam);
    let (ax, ay) = (200.0, 1800.0);
    let anchor = map.cam.screen_to_geo(ax, ay, w, h);
    map.zoom_about(2.0, ax, ay, w, h);
    let (qx, qy) = map.cam.geo_to_screen(anchor.0, anchor.1, w, h);
    println!("after zoom x2 about ({ax},{ay}) the anchor is at ({qx:.3},{qy:.3}); cam {:.6},{:.6} bearing {:.4}", map.cam.lat, map.cam.lon, map.cam.bearing);
    map.pan(300.0, -500.0, w, h);
    let (qx, qy) = map.cam.geo_to_screen(anchor.0, anchor.1, w, h);
    println!("after pan (300,-500) the anchor is at ({qx:.3},{qy:.3}) expected ({},{})", ax + 300.0, ay - 500.0);
    // The phone's pinch, as two_begin/two_update do it.
    let (x0, y0, x1, y1) = (300.0, 1000.0, 700.0, 1400.0);
    let geo_a = map.cam.screen_to_geo(x0, y0, w, h);
    let geo_b = map.cam.screen_to_geo(x1, y1, w, h);
    let d0 = ((x1 - x0) as f64).hypot(y1 - y0);
    let alpha0 = ((y1 - y0) as f64).atan2(x1 - x0);
    let (ppd0, bearing0) = (map.cam.ppd, map.cam.bearing);
    let (nx0, ny0, nx1, ny1) = (200.0, 900.0, 900.0, 1600.0);
    let d = ((nx1 - nx0) as f64).hypot(ny1 - ny0);
    let alpha = ((ny1 - ny0) as f64).atan2(nx1 - nx0);
    map.set_ppd(ppd0 * d / d0);
    map.set_bearing(bearing0 + (alpha0 - alpha));
    map.place_anchor((geo_a.0 + geo_b.0) * 0.5, (geo_a.1 + geo_b.1) * 0.5, (nx0 + nx1) * 0.5, (ny0 + ny1) * 0.5, w, h);
    let a = map.cam.geo_to_screen(geo_a.0, geo_a.1, w, h);
    let b = map.cam.geo_to_screen(geo_b.0, geo_b.1, w, h);
    println!("pinch: finger a at ({:.1},{:.1}) wanted ({nx0},{ny0}); b at ({:.1},{:.1}) wanted ({nx1},{ny1})", a.0, a.1, b.0, b.1);
}

#[allow(dead_code)]
fn pinch() {}
