//! The loader: bake a region's cells into a directory whose layout is the
//! future R2 bucket. Usage:
//!   mahere-load --pbf <file> --dem <tif>... --out <dir> \
//!     --bbox lat0,lon0,lat1,lon1 [--line-base 13] [--line-min 7] [--dem-depth 12]
use std::path::Path;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pbf = arg(&args, "--pbf").expect("--pbf");
    let out = arg(&args, "--out").expect("--out");
    let out = Path::new(&out);
    let bbox: Vec<f64> = arg(&args, "--bbox")
        .expect("--bbox")
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let (lat0, lon0, lat1, lon1) =
        (bbox[0].min(bbox[2]), bbox[1].min(bbox[3]), bbox[0].max(bbox[2]), bbox[1].max(bbox[3]));
    let line_base: u8 = arg(&args, "--line-base").map(|v| v.parse().unwrap()).unwrap_or(13);
    let line_min: u8 = arg(&args, "--line-min").map(|v| v.parse().unwrap()).unwrap_or(7);
    let dem_depth: u8 = arg(&args, "--dem-depth").map(|v| v.parse().unwrap()).unwrap_or(12);
    let tifs: Vec<String> = {
        let i = args.iter().position(|a| a == "--dem").expect("--dem") + 1;
        args[i..].iter().take_while(|a| !a.starts_with("--")).cloned().collect()
    };

    let t = std::time::Instant::now();
    let feats = mahere_osm::load_features(&pbf).expect("pbf");
    let feats: Vec<_> = feats
        .into_iter()
        .filter(|r| {
            r.pts.iter().any(|&(la, lo)| {
                (la as f64) >= lat0 && (la as f64) <= lat1 && (lo as f64) >= lon0 && (lo as f64) <= lon1
            })
        })
        .collect();
    eprintln!("{} features in box, {:.1}s", feats.len(), t.elapsed().as_secs_f32());

    let t = std::time::Instant::now();
    let cells = mahere_tiles::bake_lines(&feats, line_base, line_min);
    eprintln!("{} line cells (depths {line_min}..={line_base}), {:.1}s", cells.len(), t.elapsed().as_secs_f32());
    let t = std::time::Instant::now();
    for (key, cell) in &cells {
        mahere_tiles::write_line_cell(out, *key, cell).expect("write line cell");
    }
    eprintln!("line cells written, {:.1}s", t.elapsed().as_secs_f32());

    let t = std::time::Instant::now();
    let dem = mahere_dem::DemStore::load(&tifs).expect("dem");
    let keys = mahere_tiles::cells_covering(lat0, lon0, lat1, lon1, dem_depth);
    eprintln!("{} dem cells at depth {dem_depth}, source loaded {:.1}s", keys.len(), t.elapsed().as_secs_f32());
    let t = std::time::Instant::now();
    let baked = mahere_tiles::bake_dem(&dem, &keys);
    for (key, cell) in &baked {
        mahere_tiles::write_dem_cell(out, *key, cell).expect("write dem cell");
    }
    eprintln!("dem baked + written, {:.1}s", t.elapsed().as_secs_f32());

    // Size census.
    let du = std::process::Command::new("du").args(["-sh", out.to_str().unwrap()]).output();
    if let Ok(o) = du {
        eprintln!("total: {}", String::from_utf8_lossy(&o.stdout).trim());
    }
}
