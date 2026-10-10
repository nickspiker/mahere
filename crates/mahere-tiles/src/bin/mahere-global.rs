//! The global terrain bake: the whole Earth from a set of one-degree DEM tiles (Copernicus GLO-30), one coarse region at a time, resumable. Usage:
//!   mahere-global --src <tile dir> --index <tile list> --out <cell dir> [--depth 8] [--region 4] [--dem-loss 1.0] [--bbox lat0,lon0,lat1,lon1] [--top]
//!
//! Each region is a depth-`region` cell. Its depth-`depth` cells are box-filtered from the source tiles the region touches (four samples per triangle, so a 108 m triangle reads 54 m data), their aprons sampled from the source, then its pyramid built down to the region cell itself. A region whose tiles have not all downloaded yet is skipped and picked up by the next run; a region over open sea writes only its region cell, at sea level, which every deeper view falls back to. Finished regions are listed in `<out>/.global-done`, so a run can stop and resume anywhere.
//! `--top` builds the depths above the regions from the region cells already written, once every region is done.
//!
//! Bathymetry: `--bathy <dir of GEBCO GeoTIFFs>` (see `bathymetry`).
//! Imagery: `--img <tile dir> --img-index <tile list>` bakes the imagery layer instead, into the same cells (their terrain carried over untouched), from one-degree four-band tiles (the ESA WorldCover Sentinel-2 composite's 37 m level, pulled by `cog-level`): every texel box-filtered from sixteen samples, regions over open sea skipped, finished regions in `<out>/.global-img-done`; with `--top` it builds the imagery above the regions. `--img-loss` sets the codec's loss (default 16 levels).
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use mahere_coord::uv_to_lat_lon;
use mahere_tiles::{Cell, CellKey, DemCell, Loss};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

/// A tile name's one-degree square: `Copernicus_DSM_COG_10_N47_00_W121_00_DEM` → (47, -121), and `ESA_WorldCover_10m_2021_v200_N47W121_S2RGBNIR.tif` → (47, -121).
fn square_of(name: &str) -> Option<(i32, i32)> {
    let name = name.trim_end_matches('/');
    let name = name.rsplit('/').next().unwrap_or(name);
    let parts: Vec<&str> = name.split('_').collect();
    // WorldCover packs both into one part: N47W121.
    if let Some(p) = parts.iter().find(|p| p.len() == 7 && (p.starts_with('N') || p.starts_with('S')) && (p.as_bytes()[3] == b'E' || p.as_bytes()[3] == b'W')) {
        let lat: i32 = p[1..3].parse().ok()?;
        let lon: i32 = p[4..7].parse().ok()?;
        return Some((if p.starts_with('S') { -lat } else { lat }, if p.as_bytes()[3] == b'W' { -lon } else { lon }));
    }
    let lat_s = parts.iter().find(|p| p.starts_with('N') || p.starts_with('S'))?;
    let lon_s = parts.iter().find(|p| (p.starts_with('E') || p.starts_with('W')) && p.len() == 4)?;
    let lat: i32 = lat_s[1..].parse().ok()?;
    let lon: i32 = lon_s[1..].parse().ok()?;
    Some((if lat_s.starts_with('S') { -lat } else { lat }, if lon_s.starts_with('W') { -lon } else { lon }))
}

/// The one-degree squares a region touches, from a grid of sample points reaching a little past its edges (aprons and the filter's half-texel reach).
fn squares_of_region(key: CellKey) -> HashSet<(i32, i32)> {
    let (u0, v0, size) = key.uv_rect();
    let d = key.diamond();
    let n = 96;
    let mut out = HashSet::new();
    for i in 0..=n {
        for j in 0..=n {
            let u = u0 + size * (-0.02 + 1.04 * i as f64 / n as f64);
            let v = v0 + size * (-0.02 + 1.04 * j as f64 / n as f64);
            let (lat, lon) = uv_to_lat_lon(d, u, v);
            let lon = (lon + 180.0).rem_euclid(360.0) - 180.0;
            out.insert((lat.floor() as i32, lon.floor() as i32));
        }
    }
    out
}

/// A cell at sea level everywhere, apron included.
fn sea_cell() -> DemCell {
    let mut c = mahere_tiles::DemCell::empty();
    c.elev.iter_mut().for_each(|e| *e = 0.0);
    c.apron.iter_mut().for_each(|e| *e = 0.0);
    c
}

fn write(out: &Path, key: CellKey, dem: DemCell, loss: &Loss) {
    let cell = Cell { dem: Some(dem), ..Default::default() };
    mahere_tiles::write_cell(out, key, &cell, loss).expect("write cell");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(arg(&args, "--out").expect("--out"));
    let depth: u8 = arg(&args, "--depth").map_or(8, |v| v.parse().unwrap());
    let region_depth: u8 = arg(&args, "--region").map_or(4, |v| v.parse().unwrap());
    let loss = Loss { dem_m: arg(&args, "--dem-loss").map_or(1.0, |v| v.parse().unwrap()), img: 0 };
    std::fs::create_dir_all(&out).unwrap();

    // Land cover from a class raster (ESA WorldCover's 37 m level), rivers from HydroRIVERS, and the rest of the lines from OSM region extracts, each into the same cells.
    if let Some(dir) = arg(&args, "--bathy") {
        bathymetry(&out, &dir, depth, region_depth, bbox_arg(&args), &loss);
        return;
    }
    if let Some(dir) = arg(&args, "--land") {
        if args.iter().any(|a| a == "--top") {
            top_vectors(&out, region_depth);
        } else {
            land_cover(&out, &dir, &arg(&args, "--land-index").expect("--land-index"), depth, region_depth, bbox_arg(&args));
        }
        return;
    }
    if let Some(shp) = arg(&args, "--rivers") {
        rivers(&out, &shp, depth, region_depth, bbox_arg(&args));
        return;
    }
    if let Some(dir) = arg(&args, "--osm") {
        if args.iter().any(|a| a == "--top") {
            top_vectors(&out, region_depth);
        } else {
            osm_lines(&out, &dir, depth, region_depth, bbox_arg(&args));
        }
        return;
    }
    if let Some(img_dir) = arg(&args, "--img") {
        let loss = Loss { dem_m: 0.0, img: arg(&args, "--img-loss").map_or(16, |v| v.parse().unwrap()) };
        if args.iter().any(|a| a == "--top") {
            top_img(&out, region_depth, &loss);
        } else {
            imagery(&out, &img_dir, &arg(&args, "--img-index").expect("--img-index"), depth, region_depth, &loss);
        }
        return;
    }
    if args.iter().any(|a| a == "--top") {
        top(&out, region_depth, &loss);
        return;
    }

    let src = PathBuf::from(arg(&args, "--src").expect("--src"));
    let index = arg(&args, "--index").expect("--index");
    // Every square the source publishes, and the ones downloaded so far.
    let published: HashSet<(i32, i32)> = std::fs::read_to_string(&index).unwrap().lines().filter_map(square_of).collect();
    let mut local: HashMap<(i32, i32), String> = HashMap::new();
    for e in std::fs::read_dir(&src).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if let Some(sq) = square_of(&name) {
            let tif = e.path().join(format!("{name}.tif"));
            if tif.exists() {
                local.insert(sq, tif.to_string_lossy().to_string());
            }
        }
    }
    eprintln!("source: {} squares published, {} downloaded", published.len(), local.len());

    let bbox: Option<Vec<f64>> = arg(&args, "--bbox").map(|b| b.split(',').map(|x| x.parse().unwrap()).collect());
    let done_path = out.join(".global-done");
    let done: HashSet<(u8, u64)> = std::fs::read_to_string(&done_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .collect();
    let mut done_file = std::fs::OpenOptions::new().create(true).append(true).open(&done_path).unwrap();

    let per_side = 1u64 << region_depth;
    let (mut baked, mut sea, mut waiting, mut skipped) = (0, 0, 0, 0);
    let t_all = std::time::Instant::now();
    for diamond in 0..10u8 {
        for cu in 0..per_side {
            for cv in 0..per_side {
                let region = CellKey::from_grid(diamond, region_depth, cu, cv);
                if done.contains(&(region.depth, region.prefix)) {
                    continue;
                }
                let squares = squares_of_region(region);
                if let Some(b) = &bbox {
                    let inside = squares.iter().any(|&(la, lo)| (la as f64) + 1.0 > b[0] && (la as f64) < b[2] && (lo as f64) + 1.0 > b[1] && (lo as f64) < b[3]);
                    if !inside {
                        skipped += 1;
                        continue;
                    }
                }
                let land: Vec<(i32, i32)> = squares.iter().copied().filter(|s| published.contains(s)).collect();
                if land.iter().any(|s| !local.contains_key(s)) {
                    waiting += 1;
                    continue;
                }
                let t = std::time::Instant::now();
                if land.is_empty() {
                    // Open sea: the region cell alone, which every deeper view falls back to.
                    write(&out, region, sea_cell(), &loss);
                    sea += 1;
                } else {
                    let paths: Vec<String> = land.iter().map(|s| local[s].clone()).collect();
                    let mut store = mahere_dem::DemStore::load(&paths).expect("load tiles");
                    store.set_land_squares(published.clone());
                    let shift = depth - region_depth;
                    let keys: Vec<CellKey> = (0..1u64 << shift)
                        .flat_map(|i| (0..1u64 << shift).map(move |j| (i, j)))
                        .map(|(i, j)| CellKey::from_grid(diamond, depth, (cu << shift) + i, (cv << shift) + j))
                        .collect();
                    let base = mahere_tiles::bake_dem_filtered(&store, &keys);
                    let mut pyramid = mahere_tiles::dem_pyramid(&base, region_depth);
                    // The pyramid's aprons come from the source too: its neighbours live in other regions.
                    for (k, c) in pyramid.iter_mut() {
                        mahere_tiles::fill_apron_from_source(&store, *k, c);
                    }
                    let n = base.len() + pyramid.len();
                    use rayon::prelude::*;
                    base.into_par_iter().chain(pyramid.into_par_iter()).for_each(|(k, c)| write(&out, k, c, &loss));
                    baked += 1;
                    eprintln!("region {} {}/{}/{}: {} tiles, {n} cells, {:.1}s", baked, diamond, cu, cv, paths.len(), t.elapsed().as_secs_f32());
                }
                writeln!(done_file, "{} {}", region.depth, region.prefix).unwrap();
            }
        }
    }
    eprintln!("{baked} land regions baked, {sea} sea, {waiting} waiting for tiles, {skipped} outside the box; {:.0}s", t_all.elapsed().as_secs_f32());
}

/// The depths above the regions, from the region cells on disk: each level the mean of its four children, aprons from the neighbours (the whole level is in memory by then).
fn top(out: &Path, region_depth: u8, loss: &Loss) {
    let per_side = 1u64 << region_depth;
    let mut regions: Vec<(CellKey, DemCell)> = Vec::new();
    for diamond in 0..10u8 {
        for cu in 0..per_side {
            for cv in 0..per_side {
                let k = CellKey::from_grid(diamond, region_depth, cu, cv);
                let Ok(bytes) = std::fs::read(out.join(k.path())) else { continue };
                if let Ok(p) = mahere_tiles::decode_cell(&bytes) {
                    if let Some(d) = p.dem {
                        regions.push((k, DemCell { elev: d.elev, apron: d.apron }));
                    }
                }
            }
        }
    }
    eprintln!("{} region cells", regions.len());
    let upper = mahere_tiles::fill_aprons(mahere_tiles::dem_pyramid(&regions, 0));
    eprintln!("{} cells above the regions", upper.len());
    for (k, c) in upper {
        write(out, k, c, loss);
    }
}


/// The depth of the open sea's cells: 430 m triangles from GEBCO's 460 m grid, two levels above the land, the first use of a coarser level under a finer one.
const BATHY_DEPTH: u8 = 6;
/// A land-source texel this close to zero is the sea it wrote there.
const SEA_EPS: f32 = 0.75;

/// The seabed from GEBCO's global grid. Every region gets the seabed at [`BATHY_DEPTH`], wet wherever it is below zero, and the pyramid above it to the region cell; a land region's depth-`depth` cells (the ones with land, the only ones there are) have the sea texels the land source wrote as zero replaced by the seabed and marked wet, the land untouched, and their pyramid lies over the seabed level where they reach. Every level written replaces the old one (imagery carried over raw). Resumable through `<out>/.global-bathy-done`; a `--top` run afterwards rebuilds the levels above the regions.
fn bathymetry(out: &Path, dir: &str, depth: u8, region_depth: u8, bbox: Option<Vec<f64>>, loss: &Loss) {
    use rayon::prelude::*;
    let mut paths: Vec<String> = Vec::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.ends_with(".tif") {
            continue;
        }
        if let (Some((n, s, w, ea)), Some(b)) = (gebco_bounds(&name), &bbox) {
            if !(n > b[0] && s < b[2] && w < b[3] && ea > b[1]) {
                continue;
            }
        }
        paths.push(e.path().to_string_lossy().to_string());
    }
    eprintln!("bathymetry: {} GEBCO tiles", paths.len());
    let store = mahere_dem::DemStore::load(&paths).expect("gebco tiles");
    let done_path = out.join(".global-bathy-done");
    let done = done_list(&done_path);
    let mut done_file = std::fs::OpenOptions::new().create(true).append(true).open(&done_path).unwrap();
    let (mut land, mut sea) = (0, 0);
    let t_all = std::time::Instant::now();
    for (diamond, cu, cv, region) in regions(region_depth) {
        if done.contains(&(region.depth, region.prefix)) || !region_in_box(&squares_of_region(region), &bbox) {
            continue;
        }
        let t = std::time::Instant::now();
        let keys = region_keys(diamond, depth, region_depth, cu, cv);
        let existing: Vec<(CellKey, mahere_tiles::CellPlanes)> = keys
            .par_iter()
            .filter_map(|k| std::fs::read(out.join(k.path())).ok().and_then(|b| mahere_tiles::decode_cell(&b).ok()).map(|p| (*k, p)))
            .collect();
        let with_dem: Vec<(CellKey, mahere_tiles::DemPlanes, Option<mahere_tiles::CovCell>)> =
            existing.into_iter().filter_map(|(k, p)| p.dem.map(|d| (k, d, p.water))).collect();
        // The land cells, their sea texels patched.
        let mut base8: Vec<(CellKey, DemCell)> = Vec::new();
        let mut water8: HashMap<CellKey, mahere_tiles::CovCell> = HashMap::new();
        if !with_dem.is_empty() {
            land += 1;
            let seabed: HashMap<CellKey, DemCell> = mahere_tiles::bake_dem_filtered(&store, &with_dem.iter().map(|(k, _, _)| *k).collect::<Vec<_>>()).into_iter().collect();
            let mut patched: Vec<(CellKey, Cell)> = Vec::new();
            for (k, old, old_water) in with_dem {
                let mut merged = DemCell { elev: old.elev.clone(), apron: old.apron.clone() };
                let mut w = old_water.unwrap_or_else(mahere_tiles::CovCell::new);
                if let Some(g) = seabed.get(&k) {
                    let mut patch = DemCell::empty();
                    let mut n = 0;
                    for i in 0..merged.elev.len() {
                        if old.elev[i].abs() <= SEA_EPS && g.elev[i] < 0.0 {
                            patch.elev[i] = g.elev[i];
                            merged.elev[i] = g.elev[i];
                            w.cov[i] = 255;
                            n += 1;
                        }
                    }
                    for i in 0..merged.apron.len() {
                        if old.apron[i].abs() <= SEA_EPS && g.apron[i] < 0.0 {
                            patch.apron[i] = g.apron[i];
                            merged.apron[i] = g.apron[i];
                        }
                    }
                    if n > 0 {
                        patched.push((k, Cell { dem: Some(patch), water: Some(w.clone()), ..Default::default() }));
                    }
                }
                water8.insert(k, w);
                base8.push((k, merged));
            }
            patched.into_par_iter().for_each(|(k, c)| mahere_tiles::write_cell(out, k, &c, loss).expect("write cell"));
        } else {
            sea += 1;
        }
        // The whole region's seabed at the bathymetry depth, the land pyramid laid over it where the land cells reach: a texel with land children takes their mean, every other the seabed.
        let keys6 = region_keys(diamond, BATHY_DEPTH, region_depth, cu, cv);
        let mut level6: HashMap<CellKey, DemCell> = mahere_tiles::bake_dem_filtered(&store, &keys6).into_iter().collect();
        let mut water6: HashMap<CellKey, mahere_tiles::CovCell> = level6
            .iter()
            .map(|(k, c)| {
                let mut w = mahere_tiles::CovCell::new();
                for i in 0..c.elev.len() {
                    if c.elev[i] < 0.0 {
                        w.cov[i] = 255;
                    }
                }
                (*k, w)
            })
            .collect();
        let from_land = mahere_tiles::dem_pyramid(&base8, BATHY_DEPTH);
        let wfrom_land = mahere_tiles::pyramid_cov(water8, depth, BATHY_DEPTH);
        let mut level7: Vec<(CellKey, Cell)> = Vec::new();
        for (k, c) in from_land {
            if k.depth > BATHY_DEPTH {
                level7.push((k, Cell { dem: Some(c), water: wfrom_land.get(&k).cloned(), ..Default::default() }));
                continue;
            }
            let g = level6.entry(k).or_insert_with(DemCell::empty);
            let w = water6.entry(k).or_insert_with(mahere_tiles::CovCell::new);
            let lw = wfrom_land.get(&k);
            for i in 0..c.elev.len() {
                if !c.elev[i].is_nan() {
                    g.elev[i] = c.elev[i];
                    w.cov[i] = lw.map_or(0, |l| l.cov[i]);
                }
            }
        }
        // No apron may stay unknown: a rewrite keeps the old apron where the new is unknown, and the old one is the sea level the land source wrote, a kilometres-deep step at the region's edge that drew as a line along it. Siblings fill what they can, the seabed grid the rest.
        let level7: Vec<(CellKey, Cell)> = {
            let dems: Vec<(CellKey, DemCell)> = mahere_tiles::fill_aprons(level7.iter().map(|(k, c)| (*k, c.dem.clone().unwrap())).collect());
            let mut by_key: HashMap<CellKey, Cell> = level7.into_iter().collect();
            dems.into_iter()
                .map(|(k, mut d)| {
                    apron_from_seabed(&store, k, &mut d);
                    let mut c = by_key.remove(&k).unwrap();
                    c.dem = Some(d);
                    (k, c)
                })
                .collect()
        };
        level7.into_par_iter().for_each(|(k, c)| mahere_tiles::write_cell(out, k, &c, loss).expect("write cell"));
        let mut base6: Vec<(CellKey, DemCell)> = mahere_tiles::fill_aprons(level6.into_iter().collect());
        base6.par_iter_mut().for_each(|(k, c)| apron_from_seabed(&store, *k, c));
        let mut upper = mahere_tiles::fill_aprons(mahere_tiles::dem_pyramid(&base6, region_depth));
        upper.par_iter_mut().for_each(|(k, c)| apron_from_seabed(&store, *k, c));
        let wupper = mahere_tiles::pyramid_cov(water6.clone(), BATHY_DEPTH, region_depth);
        let n = base6.len() + upper.len();
        base6.into_par_iter().chain(upper.into_par_iter()).for_each(|(k, c)| {
            let w = if k.depth == BATHY_DEPTH { water6.get(&k).cloned() } else { wupper.get(&k).cloned() };
            let cell = Cell { dem: Some(c), water: w, ..Default::default() };
            mahere_tiles::write_cell(out, k, &cell, loss).expect("write cell");
        });
        writeln!(done_file, "{} {}", region.depth, region.prefix).unwrap();
        eprintln!("bathy region {}/{}/{}: {} land cells, {n} seabed cells from depth {BATHY_DEPTH} up, {:.1}s", diamond, cu, cv, base8.len(), t.elapsed().as_secs_f32());
    }
    eprintln!("bathymetry pass: {land} land regions patched, {sea} sea regions baked; {:.0}s", t_all.elapsed().as_secs_f32());
}

/// The apron entries still unknown, sampled from the seabed grid (which carries the land too, coarsely: an edge texel's slope, nothing more).
fn apron_from_seabed(store: &mahere_dem::DemStore, key: CellKey, cell: &mut DemCell) {
    if cell.apron.iter().all(|a| !a.is_nan()) {
        return;
    }
    let mut fresh = DemCell::empty();
    mahere_tiles::fill_apron_from_source(store, key, &mut fresh);
    for (a, f) in cell.apron.iter_mut().zip(fresh.apron.iter()) {
        if a.is_nan() {
            *a = *f;
        }
    }
}

/// The bounds (north, south, west, east) in a GEBCO tile's name, `gebco_2024_n90.0_s0.0_w-180.0_e-90.0.tif`.
fn gebco_bounds(name: &str) -> Option<(f64, f64, f64, f64)> {
    let stem = name.strip_suffix(".tif")?;
    let (mut n, mut s, mut w, mut e) = (None, None, None, None);
    for part in stem.split('_') {
        let (k, v) = part.split_at(1);
        let Ok(v) = v.parse::<f64>() else { continue };
        match k {
            "n" => n = Some(v),
            "s" => s = Some(v),
            "w" => w = Some(v),
            "e" => e = Some(v),
            _ => {}
        }
    }
    Some((n?, s?, w?, e?))
}

/// The regions in diamond order, as (diamond, cu, cv, key).
fn regions(region_depth: u8) -> impl Iterator<Item = (u8, u64, u64, CellKey)> {
    let per_side = 1u64 << region_depth;
    (0..10u8).flat_map(move |d| (0..per_side).flat_map(move |cu| (0..per_side).map(move |cv| (d, cu, cv, CellKey::from_grid(d, region_depth, cu, cv)))))
}

fn done_list(path: &Path) -> HashSet<(u8, u64)> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .collect()
}

/// The imagery bake, region by region, like the terrain's.
fn imagery(out: &Path, dir: &str, index: &str, depth: u8, region_depth: u8, loss: &Loss) {
    let published: HashSet<(i32, i32)> = std::fs::read_to_string(index).unwrap().lines().filter_map(square_of).collect();
    let mut local: HashMap<(i32, i32), String> = HashMap::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.ends_with(".tif") {
            if let Some(sq) = square_of(&name) {
                local.insert(sq, e.path().to_string_lossy().to_string());
            }
        }
    }
    eprintln!("imagery: {} squares published, {} pulled", published.len(), local.len());
    let done_path = out.join(".global-img-done");
    let done = done_list(&done_path);
    let mut done_file = std::fs::OpenOptions::new().create(true).append(true).open(&done_path).unwrap();
    let (mut baked, mut sea, mut waiting) = (0, 0, 0);
    let t_all = std::time::Instant::now();
    for (diamond, cu, cv, region) in regions(region_depth) {
        if done.contains(&(region.depth, region.prefix)) {
            continue;
        }
        let land: Vec<(i32, i32)> = squares_of_region(region).into_iter().filter(|s| published.contains(s)).collect();
        if land.iter().any(|s| !local.contains_key(s)) {
            waiting += 1;
            continue;
        }
        let t = std::time::Instant::now();
        if land.is_empty() {
            sea += 1;
        } else {
            let paths: Vec<String> = land.iter().map(|s| local[s].clone()).collect();
            let store = mahere_dem::ImgStore::load(&paths).expect("load tiles");
            let shift = depth - region_depth;
            let keys: Vec<CellKey> = (0..1u64 << shift)
                .flat_map(|i| (0..1u64 << shift).map(move |j| (i, j)))
                .map(|(i, j)| CellKey::from_grid(diamond, depth, (cu << shift) + i, (cv << shift) + j))
                .collect();
            let base = mahere_tiles::bake_img_filtered(&store, &keys);
            drop(store);
            let pyramid = mahere_tiles::img_pyramid(&base, region_depth);
            let n = base.len() + pyramid.len();
            use rayon::prelude::*;
            base.into_par_iter().chain(pyramid.into_par_iter()).for_each(|(k, c)| {
                let cell = Cell { img: Some(c), ..Default::default() };
                mahere_tiles::write_cell(out, k, &cell, loss).expect("write cell");
            });
            baked += 1;
            eprintln!("img region {baked} {diamond}/{cu}/{cv}: {} tiles, {n} cells, {:.1}s", paths.len(), t.elapsed().as_secs_f32());
        }
        writeln!(done_file, "{} {}", region.depth, region.prefix).unwrap();
    }
    eprintln!("imagery pass: {baked} regions baked, {sea} sea, {waiting} waiting for tiles; {:.0}s", t_all.elapsed().as_secs_f32());
}

/// The imagery above the regions, from the region cells' imagery on disk.
fn top_img(out: &Path, region_depth: u8, loss: &Loss) {
    let mut cells: Vec<(CellKey, mahere_tiles::ImgCell)> = Vec::new();
    for (_, _, _, k) in regions(region_depth) {
        let Ok(bytes) = std::fs::read(out.join(k.path())) else { continue };
        if let Ok(p) = mahere_tiles::decode_cell(&bytes) {
            if let Some(im) = p.img {
                cells.push((k, im));
            }
        }
    }
    eprintln!("{} region cells with imagery", cells.len());
    let upper = mahere_tiles::img_pyramid(&cells, 0);
    eprintln!("{} cells above the regions", upper.len());
    for (k, c) in upper {
        let cell = Cell { img: Some(c), ..Default::default() };
        mahere_tiles::write_cell(out, k, &cell, loss).expect("write cell");
    }
}

fn bbox_arg(args: &[String]) -> Option<Vec<f64>> {
    arg(args, "--bbox").map(|b| b.split(',').map(|x| x.parse().unwrap()).collect())
}

/// Whether a region touches the box (lat0, lon0, lat1, lon1), by its one-degree squares.
fn region_in_box(squares: &HashSet<(i32, i32)>, b: &Option<Vec<f64>>) -> bool {
    match b {
        None => true,
        Some(b) => squares.iter().any(|&(la, lo)| (la as f64) + 1.0 > b[0] && (la as f64) < b[2] && (lo as f64) + 1.0 > b[1] && (lo as f64) < b[3]),
    }
}

/// The cells of a region at `depth`.
fn region_keys(diamond: u8, depth: u8, region_depth: u8, cu: u64, cv: u64) -> Vec<CellKey> {
    let shift = depth - region_depth;
    (0..1u64 << shift).flat_map(|i| (0..1u64 << shift).map(move |j| (i, j))).map(|(i, j)| CellKey::from_grid(diamond, depth, (cu << shift) + i, (cv << shift) + j)).collect()
}

/// ESA WorldCover's eleven classes onto the land table, water to the water layer.
fn worldcover_class(b: u8) -> mahere_tiles::Cover {
    use mahere_tiles::Cover;
    match b {
        10 => Cover::Land(5),
        20 => Cover::Land(4),
        30 => Cover::Land(1),
        40 => Cover::Land(2),
        50 => Cover::Land(12),
        60 => Cover::Land(8),
        70 => Cover::Land(9),
        80 => Cover::Water,
        90 | 95 => Cover::Land(6),
        100 => Cover::Land(1),
        _ => Cover::None,
    }
}

/// The land cover bake, region by region: the 3° WorldCover tiles a region touches, loaded, sampled, the land and water layers written over the cells.
fn land_cover(out: &Path, dir: &str, index: &str, depth: u8, region_depth: u8, bbox: Option<Vec<f64>>) {
    let published: HashSet<(i32, i32)> = std::fs::read_to_string(index).unwrap().lines().filter_map(square_of).collect();
    let mut local: HashMap<(i32, i32), String> = HashMap::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.ends_with(".tif") {
            if let Some(sq) = square_of(&name) {
                local.insert(sq, e.path().to_string_lossy().to_string());
            }
        }
    }
    eprintln!("land cover: {} tiles published, {} pulled", published.len(), local.len());
    let done_path = out.join(".global-land-done");
    let done = done_list(&done_path);
    let mut done_file = std::fs::OpenOptions::new().create(true).append(true).open(&done_path).unwrap();
    let (mut baked, mut sea, mut waiting) = (0, 0, 0);
    let t_all = std::time::Instant::now();
    for (diamond, cu, cv, region) in regions(region_depth) {
        if done.contains(&(region.depth, region.prefix)) {
            continue;
        }
        let squares = squares_of_region(region);
        if !region_in_box(&squares, &bbox) {
            continue;
        }
        // The 3° tile each 1° square sits in.
        let tiles: HashSet<(i32, i32)> = squares.iter().map(|&(la, lo)| (la.div_euclid(3) * 3, lo.div_euclid(3) * 3)).filter(|t| published.contains(t)).collect();
        if tiles.iter().any(|t| !local.contains_key(t)) {
            waiting += 1;
            continue;
        }
        let t = std::time::Instant::now();
        if tiles.is_empty() {
            sea += 1;
        } else {
            let paths: Vec<String> = tiles.iter().map(|s| local[s].clone()).collect();
            let store = mahere_dem::ImgStore::load(&paths).expect("load tiles");
            let keys = region_keys(diamond, depth, region_depth, cu, cv);
            let (land, water) = mahere_tiles::bake_cover_filtered(&store, &keys, &worldcover_class);
            drop(store);
            let land: HashMap<CellKey, mahere_tiles::ClassCell> = land.into_iter().collect();
            let water: HashMap<CellKey, mahere_tiles::CovCell> = water.into_iter().collect();
            let land = mahere_tiles::pyramid_class(land, depth, region_depth, mahere_tiles::ClassMerge::Dominant);
            let water = mahere_tiles::pyramid_cov(water, depth, region_depth);
            let mut cells: HashMap<CellKey, Cell> = HashMap::new();
            for (k, l) in land {
                cells.entry(k).or_default().land = Some(l);
            }
            for (k, w) in water {
                cells.entry(k).or_default().water = Some(w);
            }
            let n = cells.len();
            use rayon::prelude::*;
            cells.into_par_iter().for_each(|(k, c)| mahere_tiles::write_cell(out, k, &c, &Loss { dem_m: 0.0, img: 0 }).expect("write cell"));
            baked += 1;
            eprintln!("land region {baked} {diamond}/{cu}/{cv}: {} tiles, {n} cells, {:.1}s", paths.len(), t.elapsed().as_secs_f32());
        }
        writeln!(done_file, "{} {}", region.depth, region.prefix).unwrap();
    }
    eprintln!("land cover pass: {baked} regions baked, {sea} sea, {waiting} waiting for tiles; {:.0}s", t_all.elapsed().as_secs_f32());
}

/// Lines written over the cells, the pyramid from `depth` down to the region.
fn write_lines(out: &Path, roads: &[mahere_osm::Road], depth: u8, region_depth: u8) -> usize {
    let cells = mahere_tiles::bake_lines(roads, depth, region_depth);
    let n = cells.len();
    use rayon::prelude::*;
    cells.into_par_iter().for_each(|(k, l)| {
        let cell = Cell { line: Some(l), ..Default::default() };
        mahere_tiles::write_cell(out, k, &cell, &Loss { dem_m: 0.0, img: 0 }).expect("write cell");
    });
    n
}

/// The rivers: every HydroRIVERS reach with a discharge of a hundredth of a cubic metre a second or more, bucketed into the regions its points' one-degree squares belong to, baked per region.
fn rivers(out: &Path, shp: &str, depth: u8, region_depth: u8, bbox: Option<Vec<f64>>) {
    let t = std::time::Instant::now();
    let roads = mahere_osm::hydro::load_hydrorivers(shp, 0.01).expect("hydrorivers");
    eprintln!("{} reaches, {:.0}s", roads.len(), t.elapsed().as_secs_f32());
    let all: Vec<(u8, u64, u64, CellKey)> = regions(region_depth).collect();
    let mut square_regions: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (i, (_, _, _, region)) in all.iter().enumerate() {
        for sq in squares_of_region(*region) {
            square_regions.entry(sq).or_default().push(i);
        }
    }
    let mut per_region: Vec<Vec<usize>> = vec![Vec::new(); all.len()];
    for (ri, r) in roads.iter().enumerate() {
        let mut seen: Vec<usize> = Vec::new();
        for &(la, lo) in &r.pts {
            let lon = ((lo as f64) + 180.0).rem_euclid(360.0) - 180.0;
            if let Some(rs) = square_regions.get(&(la.floor() as i32, lon.floor() as i32)) {
                for &x in rs {
                    if !seen.contains(&x) {
                        seen.push(x);
                    }
                }
            }
        }
        for x in seen {
            per_region[x].push(ri);
        }
    }
    let done_path = out.join(".global-rivers-done");
    let done = done_list(&done_path);
    let mut done_file = std::fs::OpenOptions::new().create(true).append(true).open(&done_path).unwrap();
    let mut baked = 0;
    for (i, (diamond, cu, cv, region)) in all.iter().enumerate() {
        if done.contains(&(region.depth, region.prefix)) || !region_in_box(&squares_of_region(*region), &bbox) {
            continue;
        }
        let t = std::time::Instant::now();
        if !per_region[i].is_empty() {
            let subset: Vec<mahere_osm::Road> = per_region[i].iter().map(|&ri| roads[ri].clone()).collect();
            let n = write_lines(out, &subset, depth, region_depth);
            baked += 1;
            eprintln!("rivers region {baked} {diamond}/{cu}/{cv}: {} reaches, {n} cells, {:.1}s", subset.len(), t.elapsed().as_secs_f32());
        }
        writeln!(done_file, "{} {}", region.depth, region.prefix).unwrap();
    }
    eprintln!("rivers: {baked} regions baked");
}

/// Every line but the waterways (those are HydroRIVERS') from a region's OSM extract, `<dir>/<diamond>-<cu>-<cv>.osm.pbf`.
fn osm_lines(out: &Path, dir: &str, depth: u8, region_depth: u8, bbox: Option<Vec<f64>>) {
    let done_path = out.join(".global-osm-done");
    let done = done_list(&done_path);
    let mut done_file = std::fs::OpenOptions::new().create(true).append(true).open(&done_path).unwrap();
    let (mut baked, mut missing) = (0, 0);
    for (diamond, cu, cv, region) in regions(region_depth) {
        if done.contains(&(region.depth, region.prefix)) || !region_in_box(&squares_of_region(region), &bbox) {
            continue;
        }
        let path = Path::new(dir).join(format!("{diamond}-{cu}-{cv}.osm.pbf"));
        if !path.exists() {
            missing += 1;
            continue;
        }
        let t = std::time::Instant::now();
        let feats = mahere_osm::load_features(path.to_str().unwrap()).expect("osm extract");
        // At a hundred metres a texel the fine classes (streets, service roads, tracks, paths) are noise, and the waterways are HydroRIVERS': the roads that carry a map at this scale, rail, power and the boundaries.
        use mahere_osm::RoadClass as C;
        let roads: Vec<mahere_osm::Road> = feats.roads.into_iter().filter(|r| !matches!(r.class, C::Residential | C::Service | C::Track | C::Path | C::Waterway)).collect();
        let n = if roads.is_empty() { 0 } else { write_lines(out, &roads, depth, region_depth) };
        baked += 1;
        eprintln!("osm region {baked} {diamond}/{cu}/{cv}: {} lines, {n} cells, {:.1}s", roads.len(), t.elapsed().as_secs_f32());
        writeln!(done_file, "{} {}", region.depth, region.prefix).unwrap();
    }
    eprintln!("osm: {baked} regions baked, {missing} extracts missing");
}

/// The line, land and water layers above the regions, from the region cells on disk.
fn top_vectors(out: &Path, region_depth: u8) {
    let mut line: HashMap<CellKey, mahere_tiles::ClassCell> = HashMap::new();
    let mut land: HashMap<CellKey, mahere_tiles::ClassCell> = HashMap::new();
    let mut water: HashMap<CellKey, mahere_tiles::CovCell> = HashMap::new();
    for (_, _, _, k) in regions(region_depth) {
        let Ok(bytes) = std::fs::read(out.join(k.path())) else { continue };
        if let Ok(p) = mahere_tiles::decode_cell(&bytes) {
            if let Some(l) = p.line {
                line.insert(k, l);
            }
            if let Some(l) = p.land {
                land.insert(k, l);
            }
            if let Some(w) = p.water {
                water.insert(k, w);
            }
        }
    }
    eprintln!("{} line, {} land, {} water region cells", line.len(), land.len(), water.len());
    let line = mahere_tiles::pyramid_class(line, region_depth, 0, mahere_tiles::ClassMerge::Major);
    let land = mahere_tiles::pyramid_class(land, region_depth, 0, mahere_tiles::ClassMerge::Dominant);
    let water = mahere_tiles::pyramid_cov(water, region_depth, 0);
    let mut cells: HashMap<CellKey, Cell> = HashMap::new();
    for (k, l) in line.into_iter().filter(|(k, _)| k.depth < region_depth) {
        cells.entry(k).or_default().line = Some(l);
    }
    for (k, l) in land.into_iter().filter(|(k, _)| k.depth < region_depth) {
        cells.entry(k).or_default().land = Some(l);
    }
    for (k, w) in water.into_iter().filter(|(k, _)| k.depth < region_depth) {
        cells.entry(k).or_default().water = Some(w);
    }
    eprintln!("{} cells above the regions", cells.len());
    for (k, c) in cells {
        mahere_tiles::write_cell(out, k, &c, &Loss { dem_m: 0.0, img: 0 }).expect("write cell");
    }
}
