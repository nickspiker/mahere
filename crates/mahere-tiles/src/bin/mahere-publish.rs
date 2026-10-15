//! `mahere-publish <baked dir> <published dir> [--min-depth n] [--max-depth n] [--check k]`: every baked cell in the directory (one level, `*.vsf.zst`) written as its published layers, `<published>/<layer>/<cell>.vsf` ([`mahere_tiles::publish`]), a cell already published and newer than its bake skipped, so a run can stop and go on.
//! `--check k` decodes every k-th cell both ways and stops on any difference; a summary of bytes in and out per layer at the end.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use mahere_tiles::CellKey;
use mahere_tiles::publish::{Layer, decode_layers, split};
use rayon::prelude::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).map(|v| v.parse::<u64>().unwrap());
    let (src, out) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]));
    let (min, max, check) = (arg("--min-depth").unwrap_or(0) as u8, arg("--max-depth").unwrap_or(255) as u8, arg("--check").unwrap_or(0) as usize);
    for l in Layer::ALL {
        std::fs::create_dir_all(out.join(l.dir())).unwrap();
    }
    let files: Vec<(CellKey, PathBuf)> = std::fs::read_dir(&src)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            if !name.ends_with(".vsf.zst") {
                return None;
            }
            let k = CellKey::from_name(&name)?;
            (k.depth >= min && k.depth <= max).then(|| (k, e.path()))
        })
        .collect();
    eprintln!("{} cells in {}", files.len(), src.display());
    let (done, skipped, failed) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let bytes_in = AtomicU64::new(0);
    let bytes_out: [AtomicU64; 4] = Default::default();
    let t = std::time::Instant::now();
    files.par_iter().enumerate().for_each(|(i, (key, path))| {
        let base_path = out.join(Layer::Base.path(*key));
        let newer = |p: &Path| p.metadata().and_then(|m| m.modified()).ok();
        if newer(&base_path).zip(newer(path)).is_some_and(|(o, s)| o >= s) {
            skipped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let bytes = std::fs::read(path).unwrap();
        bytes_in.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        let layers = match split(&bytes) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("{}: {e}", key.name());
                failed.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        if check > 0 && i % check == 0 {
            let get = |l: Layer| layers.get(&l).map(|v| v.as_slice());
            let new = decode_layers(get(Layer::Base), get(Layer::Dem), get(Layer::Img), get(Layer::Nir)).expect("published cell decodes");
            for l in [Layer::Dem, Layer::Img, Layer::Nir] {
                if let (Some(b), Some(d)) = (get(Layer::Base), get(l)) {
                    assert!(mahere_tiles::publish::linked(b, l, d), "{}: the base does not name its {}", key.name(), l.dir());
                }
            }
            let old = mahere_tiles::decode_cell(&bytes).expect("baked cell decodes");
            let diff = differs(&old, &new);
            assert!(diff.is_empty(), "{}: published cell differs from its bake in {}", key.name(), diff.join(", "));
        }
        // The image layers first and the base last, so a base newer than its bake means the whole cell is out.
        for (j, l) in Layer::ALL.iter().enumerate().rev() {
            let p = out.join(l.path(*key));
            match layers.get(l) {
                Some(d) => {
                    bytes_out[j].fetch_add(d.len() as u64, Ordering::Relaxed);
                    let part = p.with_extension(format!("part{}", std::process::id()));
                    std::fs::write(&part, d).unwrap();
                    std::fs::rename(&part, &p).unwrap();
                }
                None => {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 20000 == 0 {
            eprintln!("{n} cells, {:.0}s", t.elapsed().as_secs_f32());
        }
    });
    let gb = |b: u64| b as f64 / 1e9;
    let o: Vec<u64> = bytes_out.iter().map(|b| b.load(Ordering::Relaxed)).collect();
    eprintln!(
        "{} published, {} already, {} failed, {:.0}s: {:.2} GB baked -> base {:.2} + dem {:.2} + img {:.2} + nir {:.2} = {:.2} GB",
        done.load(Ordering::Relaxed),
        skipped.load(Ordering::Relaxed),
        failed.load(Ordering::Relaxed),
        t.elapsed().as_secs_f32(),
        gb(bytes_in.load(Ordering::Relaxed)),
        gb(o[0]),
        gb(o[1]),
        gb(o[2]),
        gb(o[3]),
        gb(o.iter().sum())
    );
    if failed.load(Ordering::Relaxed) > 0 {
        std::process::exit(1);
    }
}

/// The planes a renderer reads that differ between the bake and the published layers (terrain bit for bit, NaN included); empty when none do.
fn differs(a: &mahere_tiles::CellPlanes, b: &mahere_tiles::CellPlanes) -> Vec<&'static str> {
    let dem = match (&a.dem, &b.dem) {
        (Some(x), Some(y)) => x.elev.iter().zip(&y.elev).all(|(p, q)| p.to_bits() == q.to_bits()) && x.apron.iter().zip(&y.apron).all(|(p, q)| p.to_bits() == q.to_bits()),
        (None, None) => true,
        _ => false,
    };
    let img = match (&a.img, &b.img) {
        (Some(x), Some(y)) => x.red == y.red && x.green == y.green && x.blue == y.blue && x.nir == y.nir && x.tone == y.tone,
        (None, None) => true,
        _ => false,
    };
    let class = |x: &Option<mahere_tiles::ClassCell>, y: &Option<mahere_tiles::ClassCell>| match (x, y) {
        (Some(x), Some(y)) => x.class == y.class && x.cov == y.cov && x.mag == y.mag && x.uses == y.uses,
        (None, None) => true,
        _ => false,
    };
    let water = match (&a.water, &b.water) {
        (Some(x), Some(y)) => x.cov == y.cov,
        (None, None) => true,
        _ => false,
    };
    let names = match (&a.names, &b.names) {
        (Some(x), Some(y)) => x.line == y.line && x.area == y.area && x.entries.len() == y.entries.len(),
        (None, None) => true,
        _ => false,
    };
    let checks = [("dem", dem), ("img", img), ("water", water), ("names", names), ("line", class(&a.line, &b.line)), ("land", class(&a.land, &b.land)), ("own", class(&a.own, &b.own)), ("mvum", class(&a.mvum, &b.mvum)), ("geoid", a.geoid == b.geoid), ("route", a.route.is_some() == b.route.is_some())];
    checks.iter().filter(|(_, ok)| !ok).map(|(n, _)| *n).collect()
}
