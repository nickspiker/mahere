//! `mahere-crossings <baked dir> [--min-depth n] [--max-depth n]`: every baked cell in the directory (one level, `*.vsf`) with a route graph given its crossings (2026-11-26, [`mahere_tiles::route::Crossings`]): the edge-to-edge tables a route search jumps a cell by, for cells baked before the bake made them. A cell is written again only when its crossings change, so a publish after this picks up just those; run it again and nothing is written.
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use mahere_tiles::CellKey;
use rayon::prelude::*;
use vsf::{VsfBuilder, VsfSection};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).map(|v| v.parse::<u8>().unwrap());
    let src = PathBuf::from(&a[1]);
    let (min, max) = (arg("--min-depth").unwrap_or(0), arg("--max-depth").unwrap_or(255));
    let files: Vec<(CellKey, PathBuf)> = std::fs::read_dir(&src)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let k = CellKey::from_name(name.strip_suffix(".vsf").map(|_| name.as_str())?)?;
            (k.depth >= min && k.depth <= max).then(|| (k, e.path()))
        })
        .collect();
    eprintln!("{} cells in {}", files.len(), src.display());
    let (written, same, none, failed) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let added = AtomicU64::new(0);
    let t = std::time::Instant::now();
    files.par_iter().enumerate().for_each(|(i, (key, path))| {
        if i > 0 && i % 20_000 == 0 {
            eprintln!("{i} cells, {:.0} s", t.elapsed().as_secs_f64());
        }
        let bytes = std::fs::read(path).unwrap();
        let Some(route) = mahere_tiles::decode_route(&bytes).filter(|r| r.has_graph()) else {
            none.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let crossings = route.make_crossings(key.depth);
        if crossings == route.crossings {
            same.fetch_add(1, Ordering::Relaxed);
            return;
        }
        match with_crossings(&bytes, crossings.as_ref()) {
            Ok(doc) => {
                // Written beside and moved over, so a cell is never half written.
                let tmp = path.with_extension("vsf.crossing");
                std::fs::write(&tmp, &doc).unwrap();
                std::fs::rename(&tmp, path).unwrap();
                added.fetch_add((doc.len() as i64 - bytes.len() as i64).max(0) as u64, Ordering::Relaxed);
                written.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                eprintln!("{}: {e}", key.name());
                failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    });
    eprintln!("{} written, {} already right, {} without a graph, {} failed, {:.1} MB added, {:.0} s", written.into_inner(), same.into_inner(), none.into_inner(), failed.into_inner(), added.into_inner() as f64 / 1e6, t.elapsed().as_secs_f64());
}

/// The cell's document with its `crossing` section replaced by `crossings` (coded as the bake codes its sections), or taken out for None; every other section as it was.
fn with_crossings(doc: &[u8], crossings: Option<&mahere_tiles::route::Crossings>) -> Result<Vec<u8>, String> {
    let (header, end) = vsf::VsfHeader::decode(doc).map_err(|e| format!("{e:?}"))?;
    let sections = header.sections(doc, end).map_err(|e| format!("{e:?}"))?;
    let mut b = VsfBuilder::new();
    for s in sections.into_iter().filter(|s| s.name != "crossing") {
        let mut out = VsfSection::new(s.name.clone());
        for f in s.fields {
            out.add_field_multi(f.name, f.values);
        }
        b = b.add_section_direct(out);
    }
    if let Some(x) = crossings {
        let mut out = VsfSection::new("crossing");
        for f in x.section().fields {
            out.add_field_multi(f.name, mahere_tiles::ac::code_field(&f.values, None).unwrap_or(f.values));
        }
        b = b.add_section_direct(out);
    }
    b.build().map_err(|e| format!("{e:?}"))
}
