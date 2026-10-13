//! `cell_info <cells dir> <lat> <lon> <depth>`: what the cell there holds, plane by plane, and what is named at the point and in the cell.
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
    let Some(n) = &p.names else { return println!("  names none") };
    let named = |plane: &[u16]| pct(plane.iter().filter(|&&i| i != 0).count());
    println!("  names {} entries, {:.1}% of texels on a named line, {:.0}% in a named area ({} and {} byte index)", n.entries.len(), named(&n.line), named(&n.area), if n.line.is_empty() { 0 } else if n.entries.len() < 256 { 1 } else { 2 }, if n.area.is_empty() { 0 } else if n.entries.len() < 256 { 1 } else { 2 });
    // At the point: its texel's names.
    let (iu, iv) = mahere_coord::Coord::from_lat_lon(lat, lon).uv();
    let shift = 30 - depth as u32 - mahere_tiles::TEX_BITS as u32;
    let s = (1u64 << shift) as f64;
    let (tx, ty, half) = mahere_tiles::tri_at(iu as f64 / s, iv as f64 / s);
    let i = mahere_tiles::tri_idx(tx % mahere_tiles::TEX, ty % mahere_tiles::TEX, half);
    println!("  here  line: {}, area: {}", n.line_at(i).map_or("-".into(), |e| e.label()), n.area_at(i).map_or("-".into(), |e| e.label()));
    let mut list: Vec<&mahere_tiles::Named> = n.entries.iter().collect();
    list.sort_by_key(|e| (!e.is_point(), e.kind, std::cmp::Reverse(e.rank)));
    for e in list.iter().take(40) {
        println!("    kind {:3} rank {:>9}  {}", e.kind, e.rank, e.label());
    }
    if list.len() > 40 {
        println!("    ... {} more", list.len() - 40);
    }
}
