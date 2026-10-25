//! `mahere-aprons <published dir> [--depth d] [--write]`: every published terrain cell's apron (the texels one step past each edge) checked against the neighbour it borders, found by where the texel lies on the globe (across a diamond's edge too), and with `--write` rewritten where it is off, the base relinked to the new terrain; the cells changed listed in `<published>/aprons-changed.txt` for the upload, terrain before base.
//! Why (Nick 2026-10-24, "you can see the grid"): a bake filled an apron from its neighbours only within one diamond and one batch, and the rest from the seabed grid, tens of metres off over Antarctica's ice; on the near-flat plateau the false step drew every contour along the tile's edge.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use mahere_coord::{Coord, uv_to_lat_lon};
use mahere_tiles::publish::{Layer, decode_layers, relinked, with_apron};
use mahere_tiles::{APRON, CellKey, TEX, apron_idx, tri_at, tri_idx, tri_off};
use rayon::prelude::*;

/// An edge texel's place among a cell's edge texels: the four sides' rows and columns, both halves.
fn edge_slot(tx: usize, ty: usize, half: usize) -> Option<usize> {
    let side = if tx == 0 {
        (0, ty)
    } else if tx == TEX - 1 {
        (1, ty)
    } else if ty == 0 {
        (2, tx)
    } else if ty == TEX - 1 {
        (3, tx)
    } else {
        return None;
    };
    Some((side.0 * 2 + half) * TEX + side.1)
}

fn edges_of(elev: &[f32]) -> Vec<f32> {
    let mut e = vec![f32::NAN; 4 * 2 * TEX];
    for i in 0..TEX {
        for half in 0..2 {
            for (tx, ty) in [(0, i), (TEX - 1, i), (i, 0), (i, TEX - 1)] {
                if let Some(s) = edge_slot(tx, ty, half) {
                    e[s] = elev[tri_idx(tx, ty, half)];
                }
            }
        }
    }
    e
}

/// The cell and texel a texel position of `key` lies in, at the same depth: (texel x, y, half), x and y possibly one past the cell's edge.
fn locate(key: CellKey, tx: i64, ty: i64, half: usize) -> (CellKey, usize, usize, usize) {
    let (u0, v0, size) = key.uv_rect();
    let off = tri_off(half);
    let (gx, gy) = (tx as f64 + off, ty as f64 + off);
    let (lat, lon) = uv_to_lat_lon(key.diamond(), u0 + gx / TEX as f64 * size, v0 + gy / TEX as f64 * size);
    let c = Coord::from_lat_lon(lat, lon);
    let nk = CellKey::containing(c, key.depth);
    let (u, v) = c.uv();
    let (nu0, nv0, ns) = nk.uv_rect();
    let full = (1u64 << 30) as f64;
    let fx = ((u as f64 / full - nu0) / ns * TEX as f64).clamp(0.0, TEX as f64 - 1e-9);
    let fy = ((v as f64 / full - nv0) / ns * TEX as f64).clamp(0.0, TEX as f64 - 1e-9);
    let (x, y, h) = tri_at(fx, fy);
    (nk, x.min(TEX - 1), y.min(TEX - 1), h)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(&a[1]);
    let only: Option<u8> = a.iter().position(|x| x == "--depth").and_then(|i| a.get(i + 1)).map(|v| v.parse().unwrap());
    let write = a.iter().any(|x| x == "--write");
    // The geometry first: an interior texel located from its own cell is itself.
    for key in [CellKey::containing(Coord::from_lat_lon(47.6, -122.3), 8), CellKey::containing(Coord::from_lat_lon(-85.0, 60.0), 6)] {
        for (tx, ty, half) in [(5usize, 7usize, 0usize), (100, 200, 1), (255, 0, 1), (0, 255, 0)] {
            let got = locate(key, tx as i64, ty as i64, half);
            assert_eq!(got, (key, tx, ty, half), "locating a texel of {} gave another", key.name());
        }
    }
    let dem_dir = root.join(Layer::Dem.dir());
    // --probe lat,lon,depth: one cell's apron against its neighbours, side by side.
    if let Some(spec) = a.iter().position(|x| x == "--probe").and_then(|i| a.get(i + 1)) {
        let v: Vec<f64> = spec.split(',').map(|x| x.parse().unwrap()).collect();
        let k = CellKey::containing(Coord::from_lat_lon(v[0], v[1]), v[2] as u8);
        let dem_of = |k: CellKey| std::fs::read(dem_dir.join(format!("{}.vsf", k.name()))).ok().and_then(|d| decode_layers(None, Some(&d), None, None).ok()?.dem);
        let me = dem_of(k).unwrap();
        for side in 0..4 {
            let mut lines = Vec::new();
            for (half, i) in [(0, 1), (1, 1), (0, 128), (1, 128), (0, 254), (1, 254)] {
                let (tx, ty) = match side { 0 => (-1, i as i64), 1 => (TEX as i64, i as i64), 2 => (i as i64, -1), _ => (i as i64, TEX as i64) };
                let (nk, x, y, h) = locate(k, tx, ty, half);
                let nv = dem_of(nk).map(|d| d.elev[tri_idx(x, y, h)]);
                lines.push(format!("h{half} i{i}: apron {:.1} -> {} ({x},{y},{h}) {:?}", me.apron[apron_idx(side, half, i)], if nk == k { "self".into() } else { format!("{:?}", nk.grid()) }, nv));
            }
            println!("side {side} (cell {:?} d{}):\n  {}", k.grid(), k.depth, lines.join("\n  "));
        }
        return;
    }
    let mut by_depth: HashMap<u8, Vec<CellKey>> = HashMap::new();
    for e in std::fs::read_dir(&dem_dir).unwrap().filter_map(|e| e.ok()) {
        let name = e.file_name().into_string().unwrap_or_default();
        if let Some(k) = name.strip_suffix(".vsf").and_then(CellKey::from_name) {
            if only.is_none_or(|d| d == k.depth) {
                by_depth.entry(k.depth).or_default().push(k);
            }
        }
    }
    let changed_list = Mutex::new(Vec::<CellKey>::new());
    let mut depths: Vec<u8> = by_depth.keys().copied().collect();
    depths.sort();
    for depth in depths {
        let keys = &by_depth[&depth];
        let t = std::time::Instant::now();
        let read_dem = |k: CellKey| std::fs::read(dem_dir.join(format!("{}.vsf", k.name()))).ok();
        // Every cell's edge texels.
        let edges: HashMap<CellKey, Vec<f32>> = keys
            .par_iter()
            .filter_map(|&k| {
                let doc = read_dem(k)?;
                let dem = decode_layers(None, Some(&doc), None, None).ok()?.dem?;
                Some((k, edges_of(&dem.elev)))
            })
            .collect();
        let (checked, off, written, worst_m) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), Mutex::new(0f32));
        keys.par_iter().for_each(|&k| {
            let Some(doc) = read_dem(k) else { return };
            let Some(dem) = decode_layers(None, Some(&doc), None, None).ok().and_then(|p| p.dem) else { return };
            let step = mahere_tiles::section_fields(&doc, "dem").and_then(|f| f.get("step").and_then(|v| v.first()).cloned()).and_then(|v| match v {
                vsf::VsfType::f5(x) => Some(x),
                _ => None,
            });
            let step = step.unwrap_or(0.05);
            let mut apron = dem.apron.clone();
            let mut worst = 0f32;
            let mut bad_side = false;
            for side in 0..4 {
                // A side is off when its apron disagrees with the neighbour by more than three times the ground's own change between neighbouring texels there, and by over a metre: a step the terrain does not have, which the contours trace (rugged ground's differences, a bake's sampling against a pyramid's means, stay as they are).
                let (mut across, mut natural, mut n) = (0f64, 0f64, 0usize);
                let mut side_apron: Vec<(usize, f32)> = Vec::new();
                for half in 0..2 {
                    for i in 0..TEX {
                        let (tx, ty) = match side {
                            0 => (-1, i as i64),
                            1 => (TEX as i64, i as i64),
                            2 => (i as i64, -1),
                            _ => (i as i64, TEX as i64),
                        };
                        let (nk, x, y, h) = locate(k, tx, ty, half);
                        if nk == k {
                            continue;
                        }
                        let Some(v) = edges.get(&nk).and_then(|e| edge_slot(x, y, h).map(|s| e[s])) else { continue };
                        if v.is_nan() {
                            continue;
                        }
                        let j = apron_idx(side, half, i);
                        let old = apron[j];
                        let (edge, inner) = match side {
                            0 => (dem.elev[tri_idx(0, i, half)], dem.elev[tri_idx(1, i, half)]),
                            1 => (dem.elev[tri_idx(TEX - 1, i, half)], dem.elev[tri_idx(TEX - 2, i, half)]),
                            2 => (dem.elev[tri_idx(i, 0, half)], dem.elev[tri_idx(i, 1, half)]),
                            _ => (dem.elev[tri_idx(i, TEX - 1, half)], dem.elev[tri_idx(i, TEX - 2, half)]),
                        };
                        if !old.is_nan() && !edge.is_nan() && !inner.is_nan() {
                            across += (old - v).abs() as f64;
                            natural += (edge - inner).abs() as f64;
                            n += 1;
                        } else if old.is_nan() {
                            across += 1e6;
                            n += 1;
                        }
                        side_apron.push((j, v));
                    }
                }
                let (across, natural) = (across / n.max(1) as f64, natural / n.max(1) as f64);
                if n > 0 && across > 1.0 && across > 3.0 * natural {
                    bad_side = true;
                    worst = worst.max(across.min(1e5) as f32);
                    for (j, v) in side_apron {
                        apron[j] = v;
                    }
                }
            }
            checked.fetch_add(1, Ordering::Relaxed);
            if !bad_side {
                return;
            }
            off.fetch_add(1, Ordering::Relaxed);
            {
                let mut w = worst_m.lock().unwrap();
                if worst.is_finite() {
                    *w = w.max(worst);
                }
            }
            if !write {
                return;
            }
            debug_assert_eq!(apron.len(), APRON);
            let base_path = root.join(Layer::Base.path(k));
            let (Ok(new_dem), Ok(base)) = (with_apron(&doc, &apron), std::fs::read(&base_path)) else {
                eprintln!("{}: could not rewrite", k.name());
                return;
            };
            let Ok(new_base) = relinked(&base, Layer::Dem, &new_dem) else {
                eprintln!("{}: its base names no terrain", k.name());
                return;
            };
            // Checked before anything is written: the same elevation to the bit, the new apron, a base naming the new terrain and otherwise the same planes.
            let new = decode_layers(None, Some(&new_dem), None, None).ok().and_then(|p| p.dem).expect("the rewritten terrain decodes");
            assert!(new.elev.iter().zip(&dem.elev).all(|(a, b)| a.to_bits() == b.to_bits()), "{}: the rewrite changed the elevation", k.name());
            // As quantised: a neighbour past what the cell's base and step reach is held at the reach (a coarse cell beside the deepest sea).
            let lo = mahere_tiles::section_fields(&doc, "dem").and_then(|f| f.get("base").and_then(|v| v.first()).cloned()).and_then(|v| match v {
                vsf::VsfType::f5(x) => Some(x),
                _ => None,
            });
            let lo = lo.unwrap_or(0.0);
            assert!(new.apron.iter().zip(&apron).all(|(a, b)| (a.is_nan() && b.is_nan()) || (a - b.clamp(lo, lo + 65534.0 * step)).abs() <= step), "{}: the apron did not take", k.name());
            assert!(mahere_tiles::publish::linked(&new_base, Layer::Dem, &new_dem), "{}: the base does not name the new terrain", k.name());
            let (was, now) = (decode_layers(Some(&base), None, None, None).unwrap(), decode_layers(Some(&new_base), None, None, None).unwrap());
            assert!(was.line.as_ref().map(|l| &l.class) == now.line.as_ref().map(|l| &l.class) && was.water.as_ref().map(|w| &w.cov) == now.water.as_ref().map(|w| &w.cov) && was.land.as_ref().map(|l| &l.class) == now.land.as_ref().map(|l| &l.class), "{}: the base's planes changed", k.name());
            // The terrain first, then the base naming it.
            let put = |p: PathBuf, d: &[u8]| {
                let part = p.with_extension(format!("part{}", std::process::id()));
                std::fs::write(&part, d).unwrap();
                std::fs::rename(&part, &p).unwrap();
            };
            // The originals kept beside, so the rewrite can be undone.
            for (l, d) in [(Layer::Dem, &doc), (Layer::Base, &base)] {
                let p = root.join("aprons-backup").join(l.path(k));
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                if !p.exists() {
                    std::fs::write(&p, d).unwrap();
                }
            }
            put(dem_dir.join(format!("{}.vsf", k.name())), &new_dem);
            put(base_path, &new_base);
            written.fetch_add(1, Ordering::Relaxed);
            changed_list.lock().unwrap().push(k);
        });
        eprintln!(
            "depth {depth}: {} cells, {} checked, {} with a side off its neighbour (worst side {:.1} m on average), {} rewritten, {:.0} s",
            keys.len(),
            checked.load(Ordering::Relaxed),
            off.load(Ordering::Relaxed),
            *worst_m.lock().unwrap(),
            written.load(Ordering::Relaxed),
            t.elapsed().as_secs_f32()
        );
    }
    if write {
        let list: Vec<String> = changed_list.into_inner().unwrap().iter().map(|k| k.name()).collect();
        std::fs::write(root.join("aprons-changed.txt"), list.join("\n") + "\n").unwrap();
        eprintln!("{} cells rewritten, listed in {}", list.len(), root.join("aprons-changed.txt").display());
    }
}
