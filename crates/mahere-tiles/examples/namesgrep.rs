//! namesgrep <cell dir> <text> [depth]: the names in a directory's cells that contain the text (any case), with their kind's word, reference and the cell; how many cells were read. Cells are read as baked (`.vsf.zst`) or as published (`.vsf`).
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let q = a[2].to_lowercase();
    let only: Option<u8> = a.get(3).and_then(|d| d.parse().ok());
    let (mut cells, mut hits) = (0, 0);
    for e in std::fs::read_dir(&a[1]).unwrap().flatten() {
        let p = e.path();
        let Ok(b) = std::fs::read(&p) else { continue };
        let plain = zstd::decode_all(&b[..]).unwrap_or(b);
        let Some(names) = mahere_tiles::decode_names(&plain) else { continue };
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let depth = mahere_tiles::CellKey::from_name(name.split('.').next().unwrap()).map(|k| k.depth);
        if only.is_some() && depth != only {
            continue;
        }
        cells += 1;
        for n in &names.entries {
            if n.name.to_lowercase().contains(&q) || n.reference.to_lowercase().contains(&q) {
                hits += 1;
                let word = mahere_tiles::names::POINT_NAMES.get(n.kind.saturating_sub(mahere_tiles::names::POINT) as usize).copied().unwrap_or("?");
                println!("{:?} {word:<12} {:<40} ref {:<20} {name}", depth, n.name, n.reference);
            }
        }
    }
    eprintln!("{hits} names in {cells} cells");
}
