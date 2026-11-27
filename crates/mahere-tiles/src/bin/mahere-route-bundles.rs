//! `mahere-route-bundles <published dir> [--changed <list>]`: the route bundles (2026-11-27, [`mahere_tiles::publish::route_bundle`]) made from the published bases under `<published dir>/base`: every four siblings at [`mahere_tiles::publish::BUNDLED_FROM`] or finer, their routing in one document at their parent's name under `<published dir>/route`. A bundle is written only when its contents change, and the paths written (relative to the published dir, for `rclone copy --files-from`) go to `--changed`; run it again and nothing is written. A bundle whose cells have lost their routing is deleted, and named in the list as well so an upload notices it.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use mahere_tiles::CellKey;
use mahere_tiles::publish::{BUNDLED_FROM, Layer, route_bundle, same_contents};
use rayon::prelude::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(&a[1]);
    let changed_list = a.iter().position(|x| x == "--changed").and_then(|i| a.get(i + 1)).map(PathBuf::from);
    let base = root.join(Layer::Base.dir());
    std::fs::create_dir_all(root.join(Layer::Route.dir())).unwrap();
    // The bases bundled, by their parent: the bundle's name.
    let mut cells: Vec<(CellKey, CellKey)> = std::fs::read_dir(&base)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let k = CellKey::from_name(name.strip_suffix(".vsf")?)?;
            (k.depth >= BUNDLED_FROM).then(|| (k.parent(), k))
        })
        .collect();
    cells.sort_by_key(|(p, k)| (p.depth, p.prefix, k.prefix));
    let groups: Vec<&[(CellKey, CellKey)]> = cells.chunk_by(|x, y| x.0 == y.0).collect();
    eprintln!("{} bases at depth {BUNDLED_FROM} or finer, {} bundles to make", cells.len(), groups.len());
    let (written, same, empty, gone, failed) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let changed = Mutex::new(Vec::new());
    let t = std::time::Instant::now();
    groups.par_iter().enumerate().for_each(|(i, group)| {
        if i > 0 && i % 20_000 == 0 {
            eprintln!("{i} bundles, {:.0} s", t.elapsed().as_secs_f64());
        }
        let parent = group[0].0;
        let docs: Vec<(CellKey, Vec<u8>)> = group.iter().filter_map(|&(_, k)| Some((k, std::fs::read(base.join(Path::new(&Layer::Base.path(k)).file_name()?)).ok()?))).collect();
        let refs: Vec<(CellKey, &[u8])> = docs.iter().map(|(k, b)| (*k, b.as_slice())).collect();
        let rel = Layer::Route.path(parent);
        let out = root.join(&rel);
        match route_bundle(&refs) {
            Ok(Some(bundle)) => {
                if std::fs::read(&out).is_ok_and(|old| same_contents(&old, &bundle)) {
                    same.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                let tmp = out.with_extension("vsf.part");
                if std::fs::write(&tmp, &bundle).and_then(|_| std::fs::rename(&tmp, &out)).is_err() {
                    failed.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                written.fetch_add(1, Ordering::Relaxed);
                changed.lock().unwrap().push(rel);
            }
            Ok(None) => {
                empty.fetch_add(1, Ordering::Relaxed);
                if std::fs::remove_file(&out).is_ok() {
                    gone.fetch_add(1, Ordering::Relaxed);
                    changed.lock().unwrap().push(rel);
                }
            }
            Err(e) => {
                eprintln!("{}: {e}", parent.name());
                failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    });
    let changed = changed.into_inner().unwrap();
    if let Some(list) = changed_list {
        std::fs::write(&list, changed.iter().map(|p| format!("{p}\n")).collect::<String>()).unwrap();
    }
    eprintln!(
        "{} bundles written, {} as they were, {} with no routing ({} of them deleted), {} failed, in {:.0} s",
        written.into_inner(),
        same.into_inner(),
        empty.into_inner(),
        gone.into_inner(),
        failed.into_inner(),
        t.elapsed().as_secs_f64()
    );
}
