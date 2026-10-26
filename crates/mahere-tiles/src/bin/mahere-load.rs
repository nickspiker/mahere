//! The loader: bake a region's cells into a directory whose layout is the bucket. Cells already in the directory are merged over, so regions can be baked one at a time (and a 1 m region over a 10 m one). Usage:
//!   mahere-load --pbf <file> --dem <tif>... --out <dir> \
//!     --bbox lat0,lon0,lat1,lon1 [--vec-base 13] [--dem-base 11] [--min 6] [--dem-loss m] [--img-loss levels]
//!     [--area <file.vsf.zst>] [--under <cell name>]
//! --area bakes only the cells a shared area's shape covers, every layer; --under bakes only the cells under one cell (a piece of a large area, so a run holds only that piece's sources), its box the area's cells under it when --bbox is not given. Run pieces with --min at the piece's depth and they never write the same cell twice.
use std::path::Path;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pbf = arg(&args, "--pbf").unwrap_or_default();
    let out = arg(&args, "--out").expect("--out");
    let out = Path::new(&out);
    // --area <file>: a shared area document; the dem and imagery bake only the cells its shape covers, and its bounds stand in for --bbox.
    let area = arg(&args, "--area").map(|f| mahere_tiles::area::Area::decode(&std::fs::read(&f).expect("area file")).expect("area document"));
    let under: Option<mahere_tiles::CellKey> = arg(&args, "--under").map(|n| mahere_tiles::CellKey::from_name(&n).expect("--under: a cell name"));
    let bbox: Vec<f64> = match (arg(&args, "--bbox"), &area, under) {
        (Some(b), _, _) => b.split(',').map(|x| x.parse().unwrap()).collect(),
        (None, Some(a), Some(p)) => {
            // The box of the area's finest cells under the piece, their corners, padded by a few hundred metres for the sources' edges.
            let depth = 14u8.max(p.depth);
            let (mut la0, mut lo0, mut la1, mut lo1) = (90.0f64, 180.0f64, -90.0f64, -180.0f64);
            for k in a.cells_at(depth).into_iter().filter(|k| k.ancestor(p.depth) == p) {
                for (la, lo) in k.corners() {
                    la0 = la0.min(la);
                    la1 = la1.max(la);
                    lo0 = lo0.min(lo);
                    lo1 = lo1.max(lo);
                }
            }
            assert!(la0 <= la1, "the area has nothing under {}", p.name());
            vec![la0 - 0.003, lo0 - 0.004, la1 + 0.003, lo1 + 0.004]
        }
        (None, Some(a), None) => {
            let (lat0, lon0, lat1, lon1) = a.bounds().expect("an empty area");
            vec![lat0, lon0, lat1, lon1]
        }
        (None, None, _) => panic!("--bbox or --area"),
    };
    let (lat0, lon0, lat1, lon1) =
        (bbox[0].min(bbox[2]), bbox[1].min(bbox[3]), bbox[0].max(bbox[2]), bbox[1].max(bbox[3]));
    // What this run may write: the area's cells at every depth, under the piece when there is one.
    let area_sets: std::collections::HashMap<u8, std::collections::HashSet<mahere_tiles::CellKey>> = match &area {
        Some(a) => (0..=20u8).map(|d| (d, a.cells_at(d).into_iter().filter(|k| under.is_none_or(|p| k.depth < p.depth || k.ancestor(p.depth) == p)).collect())).collect(),
        None => Default::default(),
    };
    let keep = |k: &mahere_tiles::CellKey| -> bool {
        let in_piece = under.is_none_or(|p| k.depth >= p.depth && k.ancestor(p.depth) == p);
        let in_shape = area.is_none() || area_sets.get(&k.depth).is_some_and(|s| s.contains(k));
        in_piece && in_shape
    };
    let in_area = |keys: Vec<mahere_tiles::CellKey>, _depth: u8| -> Vec<mahere_tiles::CellKey> { keys.into_iter().filter(|k| keep(k)).collect() };
    let vec_base: u8 = arg(&args, "--vec-base").map(|v| v.parse().unwrap()).unwrap_or(13);
    let dem_base: u8 = arg(&args, "--dem-base").map(|v| v.parse().unwrap()).unwrap_or(11);
    let min_depth: u8 = arg(&args, "--min").map(|v| v.parse().unwrap()).unwrap_or(6);
    // Lossy dials for the pyramid codec: --dem-loss metres, --img-loss 8-bit levels, both at the finest level (coarser levels taper to lossless). Defaults 0.2 m and 16: under the lidar noise and invisible in the composite; pass 0 for lossless.
    let loss = mahere_tiles::Loss {
        dem_m: arg(&args, "--dem-loss").map(|v| v.parse().unwrap()).unwrap_or(0.2),
        img: arg(&args, "--img-loss").map(|v| v.parse().unwrap()).unwrap_or(16),
    };
    let list = |flag: &str| -> Vec<String> {
        match args.iter().position(|a| a == flag) {
            Some(i) => args[i + 1..].iter().take_while(|a| !a.starts_with("--")).cloned().collect(),
            None => Vec::new(),
        }
    };
    // --vectors-only: lines, land cover and water alone, merged over the bake (its terrain and imagery carried as they are), for re-running the vector pass without reading the rasters again.
    let vectors_only = args.iter().any(|a| a == "--vectors-only");
    // --terrain-only: the elevation alone, merged over the bake (its vectors, imagery and overlays kept), for re-running the terrain after a fix in its sampling.
    let terrain_only = args.iter().any(|a| a == "--terrain-only");
    let tifs = list("--dem");
    assert!(vectors_only || !tifs.is_empty(), "--dem <tif>...");
    // Imagery: --naip <tif>... (red, green, blue, near-infrared in one file, or true colour alone), --nir <tif>... (a separate near-infrared band, as Sentinel-2 ships it), --img-base <depth> (default the vector base, never past depth 14).
    let naip = list("--naip");
    let nir = list("--nir");
    let img_base_arg: Option<u8> = arg(&args, "--img-base").map(|v| v.parse().unwrap());
    let in_box = |la: f32, lo: f32| {
        (la as f64) >= lat0 && (la as f64) <= lat1 && (lo as f64) >= lon0 && (lo as f64) <= lon1
    };

    let t = std::time::Instant::now();
    let feats = if terrain_only { mahere_osm::Features::default() } else { mahere_osm::load_features(&pbf).expect("pbf") };
    // Lines are clipped to the box with a margin of a cell or so: a county line or a highway leaving the box would otherwise stamp lone-line cells along its whole length, eight objects for every one with terrain.
    let margin = 0.01;
    let clip = |la: f32, lo: f32| (la as f64) >= lat0 - margin && (la as f64) <= lat1 + margin && (lo as f64) >= lon0 - margin && (lo as f64) <= lon1 + margin;
    let mut roads: Vec<mahere_osm::Road> = Vec::new();
    for r in feats.roads {
        let mut run: Vec<(f32, f32)> = Vec::new();
        for (i, &p) in r.pts.iter().enumerate() {
            // Keep a point inside the margin, plus the one just outside on either end so the line reaches the edge.
            let inside = clip(p.0, p.1);
            let prev_in = i > 0 && clip(r.pts[i - 1].0, r.pts[i - 1].1);
            let next_in = i + 1 < r.pts.len() && clip(r.pts[i + 1].0, r.pts[i + 1].1);
            if inside || prev_in || next_in {
                run.push(p);
            }
            if !inside && !next_in && run.len() >= 2 {
                roads.push(mahere_osm::Road { class: r.class, pts: std::mem::take(&mut run), weight: r.weight, uses: r.uses, name: r.name, travel: r.travel });
            } else if !inside && !next_in {
                run.clear();
            }
        }
        if run.len() >= 2 {
            roads.push(mahere_osm::Road { class: r.class, pts: run, weight: r.weight, uses: r.uses, name: r.name, travel: r.travel });
        }
    }
    // A polygon belongs to the box when its own box overlaps it, not when a vertex lies inside: a piece in the middle of a lake or a forest has none of its shoreline or edge, and dropped the whole polygon (Flathead Lake, 2026-10-11).
    let _ = &in_box;
    let areas: Vec<_> = feats
        .areas
        .into_iter()
        .filter(|a| {
            let (mut la0, mut lo0, mut la1, mut lo1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            for &(la, lo) in a.rings.iter().flatten() {
                la0 = la0.min(la as f64);
                la1 = la1.max(la as f64);
                lo0 = lo0.min(lo as f64);
                lo1 = lo1.max(lo as f64);
            }
            la0 <= lat1 && la1 >= lat0 && lo0 <= lon1 && lo1 >= lon0
        })
        .collect();
    // Named points inside the box: places, summits, passes, trailheads.
    let points: Vec<mahere_osm::names::Point> = feats.points.into_iter().filter(|p| in_box(p.lat, p.lon)).collect();
    let names = feats.names;
    eprintln!("{} lines, {} areas, {} named points in box, {} names, {:.1}s", roads.len(), areas.len(), points.len(), names.len().saturating_sub(1), t.elapsed().as_secs_f32());

    let t = std::time::Instant::now();
    let (mut line, mut ids) = mahere_tiles::bake_lines_named(&roads, &points, &names, vec_base, min_depth);
    line.retain(|k, _| keep(k));
    eprintln!("line layer: {} cells, {:.1}s", line.len(), t.elapsed().as_secs_f32());
    let t = std::time::Instant::now();
    let (mut land, mut water, area_ids) = mahere_tiles::bake_areas_named(&areas, &names, vec_base, min_depth);
    land.retain(|k, _| keep(k));
    water.retain(|k, _| keep(k));
    for (k, a) in area_ids {
        ids.entry(k).or_default().area = a.area;
    }
    ids.retain(|k, _| keep(k));
    eprintln!("land {} cells, water {} cells, {} named cells, {:.1}s", land.len(), water.len(), ids.len(), t.elapsed().as_secs_f32());

    // The ways a route follows, a trixel wide at the base depth.
    let t = std::time::Instant::now();
    let routes = mahere_tiles::route::bake_routes(&roads, &names, &feats.junctions, vec_base);
    // The pyramid above: the major roads' maze at every coarser depth, for the long middle of a trip.
    let mut routes = mahere_tiles::route::pyramid_routes(routes, vec_base, min_depth);
    routes.retain(|k, _| keep(k));
    eprintln!("routes: {} cells, {} stacked, {} joins, {} nodes, {} corridors, {:.1}s", routes.len(), routes.values().map(|r| r.stack.len()).sum::<usize>(), routes.values().map(|r| r.joins.len()).sum::<usize>(), routes.values().map(|r| r.nodes.len()).sum::<usize>(), routes.values().map(|r| r.corridors.len()).sum::<usize>(), t.elapsed().as_secs_f32());
    let with_routes = |mut cells: std::collections::HashMap<mahere_tiles::CellKey, mahere_tiles::Cell>| {
        for (k, r) in routes {
            cells.entry(k).or_default().route = Some(r);
        }
        cells
    };

    if vectors_only {
        let t = std::time::Instant::now();
        let cells = with_routes(mahere_tiles::assemble_named(Vec::new(), line, land, water, Vec::new(), ids, &names));
        let n = cells.len();
        use rayon::prelude::*;
        cells.par_iter().for_each(|(key, cell)| {
            mahere_tiles::write_cell(out, *key, cell, &loss).expect("write cell");
        });
        eprintln!("{n} cells written (vectors only, merged over existing), {:.1}s", t.elapsed().as_secs_f32());
        return;
    }
    let t = std::time::Instant::now();
    let dem = mahere_dem::DemStore::load(&tifs).expect("dem");
    eprintln!("dem source loaded ({} tiles) {:.1}s", dem.tile_count(), t.elapsed().as_secs_f32());
    let t = std::time::Instant::now();
    let keys = in_area(mahere_tiles::cells_covering(lat0, lon0, lat1, lon1, dem_base), dem_base);
    let mut dem_cells = mahere_tiles::bake_dem(&dem, &keys);
    eprintln!("depth {dem_base}: {} dem cells sampled, {:.1}s", dem_cells.len(), t.elapsed().as_secs_f32());
    let t = std::time::Instant::now();
    let pyramid = mahere_tiles::dem_pyramid(&dem_cells, min_depth);
    eprintln!("dem pyramid {}..{min_depth}: {} cells, {:.1}s", dem_base - 1, pyramid.len(), t.elapsed().as_secs_f32());
    dem_cells.extend(pyramid);
    // A piece or an area has edges inside the map: there the aprons come from the source, so pieces meet without a seam in the shading.
    if area.is_some() || under.is_some() {
        use rayon::prelude::*;
        let t = std::time::Instant::now();
        dem_cells.par_iter_mut().for_each(|(k, c)| mahere_tiles::fill_apron_gaps_from_source(&dem, *k, c));
        eprintln!("aprons from the source: {:.1}s", t.elapsed().as_secs_f32());
    }
    drop(dem);

    let mut img_cells: Vec<(mahere_tiles::CellKey, mahere_tiles::ImgCell)> = Vec::new();
    if !naip.is_empty() && !terrain_only {
        let t = std::time::Instant::now();
        let naip_store = if naip.is_empty() { None } else { Some(mahere_dem::ImgStore::load(&naip).expect("naip")) };
        eprintln!("imagery sources loaded {:.1}s", t.elapsed().as_secs_f32());
        let t = std::time::Instant::now();
        let nir_store = if nir.is_empty() { None } else { Some(mahere_dem::ImgStore::load(&nir).expect("nir")) };
        let img_base = img_base_arg.unwrap_or(vec_base).min(mahere_tiles::IMG_MAX_DEPTH);
        let keys = in_area(mahere_tiles::cells_covering(lat0, lon0, lat1, lon1, img_base), img_base);
        img_cells = mahere_tiles::bake_img(naip_store.as_ref(), nir_store.as_ref(), &keys);
        eprintln!("depth {img_base}: {} img cells sampled, {:.1}s", img_cells.len(), t.elapsed().as_secs_f32());
        let pyramid = mahere_tiles::img_pyramid(&img_cells, min_depth);
        eprintln!("img pyramid: {} cells", pyramid.len());
        img_cells.extend(pyramid);
    }

    let t = std::time::Instant::now();
    let cells = with_routes(mahere_tiles::assemble_named(dem_cells, line, land, water, img_cells, ids, &names));
    let n = cells.len();
    use rayon::prelude::*;
    cells.par_iter().for_each(|(key, cell)| {
        mahere_tiles::write_cell(out, *key, cell, &loss).expect("write cell");
    });
    eprintln!("{n} cells written (merged over existing), {:.1}s", t.elapsed().as_secs_f32());

    let du = std::process::Command::new("du").args(["-sh", out.to_str().unwrap()]).output();
    if let Ok(o) = du {
        eprintln!("total: {}", String::from_utf8_lossy(&o.stdout).trim());
    }
}
