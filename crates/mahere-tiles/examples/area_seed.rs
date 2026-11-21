//! `area_seed <area.vsf> <out dir> <baked dir>...`: the cells other areas baked under this area's depth-9 pieces, hard-linked into its bake directory before it is baked, so where the areas overlap the new bake merges over theirs (a coarser cell half in each keeps the other's half) instead of replacing it. A cell already in the out directory is left alone; a link costs no space, and the merge writes a new file beside it rather than into the linked one.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: area_seed <area.vsf> <out dir> <baked dir>...");
        std::process::exit(2);
    }
    let area = mahere_tiles::area::Area::decode(&std::fs::read(&a[1]).expect("area file")).expect("an area document");
    let mut pieces: Vec<mahere_tiles::CellKey> = area.pieces(9).into_iter().map(|p| p.0).collect();
    pieces.sort();
    let out = std::path::Path::new(&a[2]);
    std::fs::create_dir_all(out).expect("out dir");
    for dir in &a[3..] {
        let (mut linked, mut there) = (0usize, 0usize);
        for e in std::fs::read_dir(dir).expect("baked dir").flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let Some(key) = name.strip_suffix(".vsf").and_then(mahere_tiles::CellKey::from_name) else { continue };
            if key.depth < 9 || pieces.binary_search(&key.ancestor(9)).is_err() {
                continue;
            }
            let to = out.join(&name);
            if to.exists() {
                there += 1;
                continue;
            }
            std::fs::hard_link(e.path(), &to).unwrap_or_else(|err| panic!("{}: {err}", to.display()));
            linked += 1;
        }
        println!("{dir}: {linked} cells linked, {there} already there");
    }
}
