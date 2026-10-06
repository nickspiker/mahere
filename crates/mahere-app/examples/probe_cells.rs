// probe_cells <lat> <lon>: for every depth, what the cell on disk holds at
// that point — dem present / nodata, vector planes present, coverage.
use mahere_coord::Coord;
use mahere_engine::raster::tri_index;
use mahere_tiles::{CellKey, decode_cell};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lat: f64 = args[1].parse().unwrap();
    let lon: f64 = args[2].parse().unwrap();
    let c = Coord::from_lat_lon(lat, lon);
    let (iu, iv) = c.uv();
    for depth in 6..=14u8 {
        let key = CellKey::containing(c, depth);
        let path = std::path::Path::new("data/cells").join(key.path());
        let Ok(bytes) = std::fs::read(&path) else {
            println!("d{depth:02} {}: absent", key.name());
            continue;
        };
        let p = decode_cell(&bytes).unwrap();
        let shift = 16 + (22 - depth as u32);
        let i = tri_index((iu as i64) << 16, (iv as i64) << 16, shift);
        let dem = match &p.dem {
            None => "no dem".to_string(),
            Some(d) if d.elev[i].is_nan() => {
                let valid = d.elev.iter().filter(|e| !e.is_nan()).count();
                format!("dem NODATA here ({valid}/131072 valid)")
            }
            Some(d) => format!("elev {:.1}", d.elev[i]),
        };
        let vec = match (&p.line, &p.land, &p.water) {
            (None, None, None) => "no vec".to_string(),
            (l, a, w) => format!(
                "line {} land {} water {}",
                l.as_ref().map_or("-".into(), |x| format!("c{} v{}", x.class[i], x.cov[i])),
                a.as_ref().map_or("-".into(), |x| format!("c{} v{}", x.class[i], x.cov[i])),
                w.as_ref().map_or("-".into(), |x| format!("v{}", x.cov[i])),
            ),
        };
        println!("d{depth:02} {}: {dem}; {vec}", key.name());
    }
}
