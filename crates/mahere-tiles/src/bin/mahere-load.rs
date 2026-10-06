//! The loader: bake a region's cells into a directory whose layout is the bucket. Cells already in the directory are merged over, so regions can be baked one at a time (and a 1 m region over a 10 m one). Usage:
//!   mahere-load --pbf <file> --dem <tif>... --out <dir> \
//!     --bbox lat0,lon0,lat1,lon1 [--vec-base 13] [--dem-base 11] [--min 6]
use std::path::Path;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pbf = arg(&args, "--pbf").expect("--pbf");
    let out = arg(&args, "--out").expect("--out");
    let out = Path::new(&out);
    let bbox: Vec<f64> = arg(&args, "--bbox").expect("--bbox").split(',').map(|x| x.parse().unwrap()).collect();
    let (lat0, lon0, lat1, lon1) =
        (bbox[0].min(bbox[2]), bbox[1].min(bbox[3]), bbox[0].max(bbox[2]), bbox[1].max(bbox[3]));
    let vec_base: u8 = arg(&args, "--vec-base").map(|v| v.parse().unwrap()).unwrap_or(13);
    let dem_base: u8 = arg(&args, "--dem-base").map(|v| v.parse().unwrap()).unwrap_or(11);
    let min_depth: u8 = arg(&args, "--min").map(|v| v.parse().unwrap()).unwrap_or(6);
    let tifs: Vec<String> = {
        let i = args.iter().position(|a| a == "--dem").expect("--dem") + 1;
        args[i..].iter().take_while(|a| !a.starts_with("--")).cloned().collect()
    };
    let in_box = |la: f32, lo: f32| {
        (la as f64) >= lat0 && (la as f64) <= lat1 && (lo as f64) >= lon0 && (lo as f64) <= lon1
    };

    let t = std::time::Instant::now();
    let feats = mahere_osm::load_features(&pbf).expect("pbf");
    let roads: Vec<_> = feats.roads.into_iter().filter(|r| r.pts.iter().any(|&(la, lo)| in_box(la, lo))).collect();
    let areas: Vec<_> = feats
        .areas
        .into_iter()
        .filter(|a| a.rings.iter().flatten().any(|&(la, lo)| in_box(la, lo)))
        .collect();
    eprintln!("{} lines, {} areas in box, {:.1}s", roads.len(), areas.len(), t.elapsed().as_secs_f32());

    let t = std::time::Instant::now();
    let line = mahere_tiles::bake_lines(&roads, vec_base, min_depth);
    eprintln!("line layer: {} cells, {:.1}s", line.len(), t.elapsed().as_secs_f32());
    let t = std::time::Instant::now();
    let (land, water) = mahere_tiles::bake_areas(&areas, vec_base, min_depth);
    eprintln!("land {} cells, water {} cells, {:.1}s", land.len(), water.len(), t.elapsed().as_secs_f32());

    let t = std::time::Instant::now();
    let dem = mahere_dem::DemStore::load(&tifs).expect("dem");
    eprintln!("dem source loaded ({} tiles) {:.1}s", dem.tile_count(), t.elapsed().as_secs_f32());
    let t = std::time::Instant::now();
    let keys = mahere_tiles::cells_covering(lat0, lon0, lat1, lon1, dem_base);
    let mut dem_cells = mahere_tiles::bake_dem(&dem, &keys);
    eprintln!("depth {dem_base}: {} dem cells sampled, {:.1}s", dem_cells.len(), t.elapsed().as_secs_f32());
    drop(dem);
    let t = std::time::Instant::now();
    let pyramid = mahere_tiles::dem_pyramid(&dem_cells, min_depth);
    eprintln!("dem pyramid {}..{min_depth}: {} cells, {:.1}s", dem_base - 1, pyramid.len(), t.elapsed().as_secs_f32());
    dem_cells.extend(pyramid);

    let t = std::time::Instant::now();
    let cells = mahere_tiles::assemble(dem_cells, line, land, water);
    let n = cells.len();
    use rayon::prelude::*;
    cells.par_iter().for_each(|(key, cell)| {
        mahere_tiles::write_cell(out, *key, cell).expect("write cell");
    });
    eprintln!("{n} cells written (merged over existing), {:.1}s", t.elapsed().as_secs_f32());

    let du = std::process::Command::new("du").args(["-sh", out.to_str().unwrap()]).output();
    if let Ok(o) = du {
        eprintln!("total: {}", String::from_utf8_lossy(&o.stdout).trim());
    }
}
