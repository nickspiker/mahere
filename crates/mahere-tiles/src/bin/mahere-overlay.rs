//! The overlays: land ownership (PAD-US) and motor vehicle use (USFS MVUM) baked into an area's cells, merged over what is there, terrain and imagery untouched. Reads the shapefiles `scripts/landuse-prep.py` writes for the area.
//!   mahere-overlay --out <cells dir> --area <file.vsf.zst> --src <dir with own.shp, wild.shp, mvum.shp> [--under <cell name>] [--base 14] [--min 9]
//! With --under, only the cells under that cell are made, and the polygon fill scans only its texels, so a large area runs a piece at a time as the loader does.
use std::path::Path;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = arg(&args, "--out").expect("--out");
    let out = Path::new(&out);
    let area = mahere_tiles::area::Area::decode(&std::fs::read(arg(&args, "--area").expect("--area")).expect("area file")).expect("area document");
    let src = arg(&args, "--src").expect("--src");
    let under = arg(&args, "--under").map(|n| mahere_tiles::CellKey::from_name(&n).expect("--under: a cell name"));
    let base: u8 = arg(&args, "--base").map(|v| v.parse().unwrap()).unwrap_or(14);
    let min: u8 = arg(&args, "--min").map(|v| v.parse().unwrap()).unwrap_or(9);
    let t = std::time::Instant::now();
    let sets: std::collections::HashMap<u8, std::collections::HashSet<mahere_tiles::CellKey>> = (min..=base).map(|d| (d, area.cells_at(d).into_iter().filter(|k| under.is_none_or(|p| k.depth >= p.depth && k.ancestor(p.depth) == p)).collect())).collect();
    let keep = |k: &mahere_tiles::CellKey| sets.get(&k.depth).is_some_and(|s| s.contains(k));
    // The base-depth cells alone decide what the fill and the stamps make; the pyramid then builds every level above from them.
    let keep_base = |k: &mahere_tiles::CellKey| k.depth != base || keep(k);
    let own = mahere_osm::landuse::load_ownership(&format!("{src}/own.shp")).expect("own.shp");
    let wild = mahere_osm::landuse::load_wilderness(&format!("{src}/wild.shp")).expect("wild.shp");
    let mvum = mahere_osm::landuse::load_mvum(&format!("{src}/mvum.shp")).expect("mvum.shp");
    eprintln!("{} ownership areas, {} wilderness, {} routes, {:.1}s", own.len(), wild.len(), mvum.len(), t.elapsed().as_secs_f32());
    let t = std::time::Instant::now();
    let mut own_cells = mahere_tiles::bake_ownership(&own, &wild, base, min, &keep_base, under);
    own_cells.retain(|k, c| keep(k) && c.cov.iter().any(|&v| v != 0));
    let mut mvum_cells = mahere_tiles::bake_mvum(&mvum, base, min, &keep_base);
    mvum_cells.retain(|k, c| keep(k) && c.cov.iter().any(|&v| v != 0));
    eprintln!("ownership {} cells, motor vehicle use {} cells, {:.1}s", own_cells.len(), mvum_cells.len(), t.elapsed().as_secs_f32());
    let t = std::time::Instant::now();
    let mut keys: Vec<mahere_tiles::CellKey> = own_cells.keys().chain(mvum_cells.keys()).copied().collect();
    keys.sort();
    keys.dedup();
    use rayon::prelude::*;
    let loss = mahere_tiles::Loss { dem_m: 0.2, img: 16 };
    keys.par_iter().for_each(|k| {
        let cell = mahere_tiles::Cell { own: own_cells.get(k).cloned(), mvum: mvum_cells.get(k).cloned(), ..Default::default() };
        mahere_tiles::write_cell(out, *k, &cell, &loss).expect("write cell");
    });
    eprintln!("{} cells written (merged over existing), {:.1}s", keys.len(), t.elapsed().as_secs_f32());
}
