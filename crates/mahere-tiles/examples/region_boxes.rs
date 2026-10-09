// region_boxes [depth]: the latitude/longitude box of every region cell (depth 4 by default), widened a little past its edges, as an osmium extract config on stdout. A region that crosses the antimeridian is given two boxes, "<name>-a" and "<name>-b", to be merged after.
use mahere_coord::uv_to_lat_lon;
use mahere_tiles::CellKey;

fn main() {
    let depth: u8 = std::env::args().nth(1).map_or(4, |v| v.parse().unwrap());
    let out_dir = std::env::args().nth(2).unwrap_or_else(|| "regions".into());
    let per_side = 1u64 << depth;
    let mut extracts = Vec::new();
    for diamond in 0..10u8 {
        for cu in 0..per_side {
            for cv in 0..per_side {
                let key = CellKey::from_grid(diamond, depth, cu, cv);
                let (u0, v0, size) = key.uv_rect();
                let n = 48;
                let (mut lat_lo, mut lat_hi) = (90.0f64, -90.0f64);
                let mut lons = Vec::new();
                let mut pole = false;
                for i in 0..=n {
                    for j in 0..=n {
                        let u = u0 + size * (-0.03 + 1.06 * i as f64 / n as f64);
                        let v = v0 + size * (-0.03 + 1.06 * j as f64 / n as f64);
                        let (lat, lon) = uv_to_lat_lon(diamond, u, v);
                        lat_lo = lat_lo.min(lat);
                        lat_hi = lat_hi.max(lat);
                        if lat.abs() > 89.5 {
                            pole = true;
                        }
                        lons.push((lon + 180.0).rem_euclid(360.0) - 180.0);
                    }
                }
                let name = format!("{}-{}-{}", diamond, cu, cv);
                let lat_lo = (lat_lo - 0.05).max(-90.0);
                let lat_hi = (lat_hi + 0.05).min(90.0);
                let (lon_lo, lon_hi) = (lons.iter().cloned().fold(180.0, f64::min), lons.iter().cloned().fold(-180.0, f64::max));
                // A region at a pole or across the antimeridian spans the longitudes wrapped around: whole-width boxes, or two.
                if pole {
                    extracts.push((name, lat_lo, -180.0, lat_hi, 180.0));
                } else if lon_hi - lon_lo > 180.0 {
                    let east_lo = lons.iter().cloned().filter(|&l| l > 0.0).fold(180.0, f64::min);
                    let west_hi = lons.iter().cloned().filter(|&l| l < 0.0).fold(-180.0, f64::max);
                    extracts.push((format!("{name}-a"), lat_lo, (east_lo - 0.05).max(-180.0), lat_hi, 180.0));
                    extracts.push((format!("{name}-b"), lat_lo, -180.0, lat_hi, (west_hi + 0.05).min(180.0)));
                } else {
                    extracts.push((name, lat_lo, (lon_lo - 0.05).max(-180.0), lat_hi, (lon_hi + 0.05).min(180.0)));
                }
            }
        }
    }
    println!("{{\"directory\": \"{out_dir}\", \"extracts\": [");
    for (i, (name, la0, lo0, la1, lo1)) in extracts.iter().enumerate() {
        let comma = if i + 1 < extracts.len() { "," } else { "" };
        println!("  {{\"output\": \"{name}.osm.pbf\", \"output_format\": \"pbf,add_metadata=false\", \"bbox\": {{\"left\": {lo0:.4}, \"bottom\": {la0:.4}, \"right\": {lo1:.4}, \"top\": {la1:.4}}}}}{comma}");
    }
    println!("]}}");
    eprintln!("{} extracts", extracts.len());
}
