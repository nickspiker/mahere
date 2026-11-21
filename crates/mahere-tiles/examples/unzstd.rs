//! `unzstd <dir>...`: every baked cell in the directories still wrapped in zstd (`*.vsf.zst`, one level) written again as VSF whose sections code themselves (`<name>.vsf`, [`mahere_tiles::ac::code_document`]), each checked field for field against the old before the old file goes (Nick 2026-11-21: zstd was never approved). Runs on every core; a run can stop and go on.
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

fn main() {
    let dirs: Vec<std::path::PathBuf> = std::env::args().skip(1).map(Into::into).collect();
    if dirs.is_empty() {
        eprintln!("usage: unzstd <dir>...");
        std::process::exit(2);
    }
    for dir in dirs {
        let files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.to_string_lossy().ends_with(".vsf.zst"))
            .collect();
        let next = AtomicUsize::new(0);
        let (before, after, failed) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
        let t = std::time::Instant::now();
        std::thread::scope(|s| {
            for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(old) = files.get(i) else { break };
                    match recode(old) {
                        Ok((a, b)) => {
                            before.fetch_add(a, Ordering::Relaxed);
                            after.fetch_add(b, Ordering::Relaxed);
                        }
                        Err(e) => {
                            failed.fetch_add(1, Ordering::Relaxed);
                            eprintln!("{}: {e}", old.display());
                        }
                    }
                    if i & 0xFFF == 0xFFF {
                        eprintln!("{}: {} of {}", dir.display(), i + 1, files.len());
                    }
                });
            }
        });
        let mb = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / (1u64 << 20) as f64;
        println!("{}: {} cells, {:.0} MB in zstd, {:.0} MB coded in VSF, {} failed and left as they were, {:.0} s", dir.display(), files.len(), mb(&before), mb(&after), failed.load(Ordering::Relaxed), t.elapsed().as_secs_f64());
    }
}

/// One cell recoded beside its old file, checked, renamed into place, and the old file removed: the sizes before and after.
fn recode(old: &std::path::Path) -> Result<(u64, u64), String> {
    let z = std::fs::read(old).map_err(|e| e.to_string())?;
    let plain = zstd::decode_all(&z[..]).map_err(|e| e.to_string())?;
    let coded = mahere_tiles::ac::code_document(&plain)?;
    mahere_tiles::ac::same_fields(&plain, &coded)?;
    mahere_tiles::decode_cell(&coded)?;
    let new = old.with_extension("");
    let part = old.with_extension(format!("part{}", std::process::id()));
    std::fs::write(&part, &coded).map_err(|e| e.to_string())?;
    std::fs::rename(&part, &new).map_err(|e| e.to_string())?;
    std::fs::remove_file(old).map_err(|e| e.to_string())?;
    Ok((z.len() as u64, coded.len() as u64))
}
