//! The global terrain bake: the whole Earth from a set of one-degree DEM tiles (Copernicus GLO-30), one coarse region at a time, resumable. Usage:
//!   mahere-global --src <tile dir> --index <tile list> --out <cell dir> [--depth 8] [--region 4] [--dem-loss 1.0] [--bbox lat0,lon0,lat1,lon1] [--top]
//!
//! Each region is a depth-`region` cell. Its depth-`depth` cells are box-filtered from the source tiles the region touches (four samples per triangle, so a 108 m triangle reads 54 m data), their aprons sampled from the source, then its pyramid built down to the region cell itself. A region whose tiles have not all downloaded yet is skipped and picked up by the next run; a region over open sea writes only its region cell, at sea level, which every deeper view falls back to. Finished regions are listed in `<out>/.global-done`, so a run can stop and resume anywhere.
//! `--top` builds the depths above the regions from the region cells already written, once every region is done.
//!
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
