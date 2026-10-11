//! `cell_info <cells dir> <lat> <lon> <depth>`: what the cell there holds, plane by plane.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (lat, lon, depth): (f64, f64, u8) = (a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap());
    let k = mahere_tiles::CellKey::containing(mahere_coord::Coord::from_lat_lon(lat, lon), depth);
    let path = std::path::Path::new(&a[1]).join(k.path());
    let Ok(bytes) = std::fs::read(&path) else { println!("{}: not there", k.name()); return };
    let p = mahere_tiles::decode_cell(&bytes).expect("decode");
    let tri = mahere_tiles::TRI as f64;
    let pct = |n: usize| n as f64 * 100.0 / tri;
    println!("{} ({} bytes)", k.name(), bytes.len());
    println!("  dem   {}", p.dem.as_ref().map_or("none".into(), |d| format!("{:.0}% with elevation", pct(d.elev.iter().filter(|e| !e.is_nan()).count()))));
    println!("  water {}", p.water.as_ref().map_or("none".into(), |w| format!("{:.0}% covered", pct(w.cov.iter().filter(|&&c| c > 0).count()))));
    println!("  land  {}", p.land.as_ref().map_or("none".into(), |l| format!("{:.0}% covered", pct(l.cov.iter().filter(|&&c| c > 0).count()))));
    println!("  line  {}", p.line.as_ref().map_or("none".into(), |l| format!("{:.1}% covered", pct(l.cov.iter().filter(|&&c| c > 0).count()))));
    println!("  img   {}", if p.img.is_some() { "present" } else { "none" });
    println!("  own   {}", p.own.as_ref().map_or("none".into(), |o| format!("{:.0}% covered", pct(o.cov.iter().filter(|&&c| c > 0).count()))));
    println!("  mvum  {}", p.mvum.as_ref().map_or("none".into(), |m| format!("{:.1}% covered", pct(m.cov.iter().filter(|&&c| c > 0).count()))));
}
