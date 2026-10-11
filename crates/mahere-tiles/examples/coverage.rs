//! `coverage <area.vsf.zst> <cells dir> [depth]`: what the baked cells actually hold, not what the sources' outlines claim: every cell of the area at the depth (14 by default) decoded, its texels with no elevation and with no imagery counted, the worst cells listed with where they are.
use rayon::prelude::*;

static WET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let area = mahere_tiles::area::Area::decode(&std::fs::read(&args[1]).expect("area file")).expect("area document");
    let dir = std::path::PathBuf::from(&args[2]);
    let depth: u8 = args.get(3).and_then(|d| d.parse().ok()).unwrap_or(14);
    let keys = area.cells_at(depth);
    let tri = mahere_tiles::TRI as u64;
    // Per cell: (key, on disk, texels without elevation, texels without imagery, has an imagery plane).
    let rows: Vec<(mahere_tiles::CellKey, bool, u64, u64, bool)> = keys
        .par_iter()
        .map(|&k| {
            let Ok(bytes) = std::fs::read(dir.join(k.path())) else { return (k, false, tri, tri, false) };
            let Ok(p) = mahere_tiles::decode_cell(&bytes) else { return (k, false, tri, tri, false) };
            let no_dem = p.dem.as_ref().map_or(tri, |d| d.elev.iter().filter(|e| e.is_nan()).count() as u64);
            // Of those, how many the water layer marks as water: lidar gets no return off open water, so a gap there is the lake, not missing ground.
            let wet_gap = match (&p.dem, &p.water) {
                (Some(d), Some(w)) => (0..mahere_tiles::TRI).filter(|&i| d.elev[i].is_nan() && w.cov[i] > 0).count() as u64,
                // No terrain plane at all (a cell wholly on the lake): every texel is a gap.
                (None, Some(w)) => w.cov.iter().filter(|&&c| c > 0).count() as u64,
                _ => 0,
            };
            WET.fetch_add(wet_gap, std::sync::atomic::Ordering::Relaxed);
            let no_img = p.img.as_ref().map_or(tri, |im| (0..mahere_tiles::TRI).filter(|&i| im.red[i] == 0 && im.green[i] == 0 && im.blue[i] == 0).count() as u64);
            (k, true, no_dem, no_img, p.img.is_some())
        })
        .collect();
    let n = rows.len() as u64;
    let missing = rows.iter().filter(|r| !r.1).count();
    let (dem_gap, img_gap): (u64, u64) = rows.iter().fold((0, 0), |a, r| (a.0 + r.2, a.1 + r.3));
    let all = n * tri;
    println!("{} cells at depth {depth}: {} missing from the folder", n, missing);
    println!("texels without elevation: {:.3}%, of which {:.1}% are water", dem_gap as f64 * 100.0 / all as f64, WET.load(std::sync::atomic::Ordering::Relaxed) as f64 * 100.0 / dem_gap.max(1) as f64);
    println!("texels without imagery:   {:.3}%", img_gap as f64 * 100.0 / all as f64);
    let dem_cells = rows.iter().filter(|r| r.2 > tri / 100).count();
    let img_cells = rows.iter().filter(|r| r.3 > tri / 100).count();
    println!("cells more than 1% without elevation: {dem_cells}; without imagery: {img_cells}");
    // Where the gaps are: grouped by the depth-10 cell (about 7 km) they fall in, the biggest groups first, each with its centre and how much of it is missing.
    for (what, pick) in [("elevation", 2usize), ("imagery", 3usize)] {
        let mut groups: std::collections::BTreeMap<mahere_tiles::CellKey, (u64, u64)> = Default::default();
        for r in &rows {
            let gap = if pick == 2 { r.2 } else { r.3 };
            let g = groups.entry(r.0.ancestor(10.min(depth))).or_default();
            g.0 += gap;
            g.1 += tri;
        }
        let mut v: Vec<(mahere_tiles::CellKey, (u64, u64))> = groups.into_iter().filter(|(_, (g, _))| *g > 0).collect();
        v.sort_by_key(|(_, (g, _))| std::cmp::Reverse(*g));
        println!("{what} gaps by 7 km cell, largest first ({} cells have some):", v.len());
        for (k, (g, t)) in v.iter().take(10) {
            let c = k.corners();
            let (lat, lon) = ((c[0].0 + c[3].0) / 2.0, (c[0].1 + c[3].1) / 2.0);
            println!("  {lat:.3},{lon:.3}  {:.0}% of it missing, {:.1} km\u{b2}", *g as f64 * 100.0 / *t as f64, *g as f64 * (k.texel_m() / (1u64 << (depth - 10)) as f64).powi(2) * 0.866 / 2.0 / 1e6);
        }
    }
}
