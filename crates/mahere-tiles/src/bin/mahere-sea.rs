//! The sea in an area's cells (Nick 2026-10-16: Puget Sound drew as land in the Seattle area, whose OSM extract marks the sea only by its coastline): OSM's water polygons rasterised into each cell's water plane and merged over what is there (lakes and rivers kept, the higher coverage wins), terrain and imagery untouched.
//!   mahere-sea --out <cells dir> --area <file.vsf.zst> --shp <water_polygons.shp> [--base 14] [--min 9]
//! The shapefile is read once for the area's box; each depth-9 piece is then filled from the polygons clipped to its own box, so a run holds one piece's cells at a time.
use std::collections::HashSet;
use std::path::Path;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = arg(&args, "--out").expect("--out");
    let out = Path::new(&out);
    let area = mahere_tiles::area::Area::decode(&std::fs::read(arg(&args, "--area").expect("--area")).expect("area file")).expect("area document");
    let shp = arg(&args, "--shp").expect("--shp");
    let base: u8 = arg(&args, "--base").map(|v| v.parse().unwrap()).unwrap_or(14);
    let min: u8 = arg(&args, "--min").map(|v| v.parse().unwrap()).unwrap_or(9);
    let t = std::time::Instant::now();
    let bbox = area.bounds().expect("an area with a shape");
    let sea = mahere_osm::landuse::load_sea(&shp, bbox).expect("water polygons");
    eprintln!("{} sea polygons touch the area, {:.1}s", sea.len(), t.elapsed().as_secs_f32());
    let sets: Vec<HashSet<mahere_tiles::CellKey>> = (0..=base).map(|d| if d >= min { area.cells_at(d).into_iter().collect() } else { HashSet::new() }).collect();
    let (mut edited, mut pieces) = (0usize, 0usize);
    for piece in area.cells_at(min) {
        // The piece's box, a little loose so its edge texels are filled.
        let corners = piece.corners();
        let pad = 0.001;
        let pbox = corners.iter().fold((90.0f64, 180.0f64, -90.0f64, -180.0f64), |(a, b, c, d), &(lat, lon)| (a.min(lat), b.min(lon), c.max(lat), d.max(lon)));
        let pbox = (pbox.0 - pad, pbox.1 - pad, pbox.2 + pad, pbox.3 + pad);
        let areas: Vec<mahere_osm::Area> = sea
            .iter()
            .filter_map(|rings| {
                let clipped: Vec<Vec<(f32, f32)>> = rings.iter().map(|r| mahere_osm::landuse::clip_ring(r, pbox)).filter(|r| r.len() >= 4).map(|r| r.into_iter().map(|(lat, lon)| (lat as f32, lon as f32)).collect()).collect();
                (!clipped.is_empty()).then(|| mahere_osm::Area { class: mahere_osm::AreaClass::Water, rings: clipped, name: 0 })
            })
            .collect();
        if areas.is_empty() {
            continue;
        }
        pieces += 1;
        let (_, water) = mahere_tiles::bake_areas(&areas, base, min);
        for (key, cov) in water {
            if key.depth < min || key.ancestor(min) != piece || !sets[key.depth as usize].contains(&key) || cov.cov.iter().all(|&v| v == 0) {
                continue;
            }
            let merged = mahere_tiles::edit_vectors(&out.join(key.path()), |p| {
                p.water = Some(match p.water.take() {
                    Some(mut w) => {
                        for (a, &b) in w.cov.iter_mut().zip(&cov.cov) {
                            *a = (*a).max(b);
                        }
                        w
                    }
                    None => cov.clone(),
                });
            });
            match merged {
                Ok(true) => edited += 1,
                Ok(false) => {}
                Err(e) => eprintln!("{}: {e}", key.name()),
            }
        }
        eprintln!("piece {}: {} cells with sea so far, {:.1}s", piece.name(), edited, t.elapsed().as_secs_f32());
    }
    eprintln!("sea merged into {edited} cells over {pieces} pieces, {:.1}s", t.elapsed().as_secs_f32());
}
