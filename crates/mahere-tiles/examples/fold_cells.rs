// fold_cells <over dir> <into dir> [--dem-loss m] [--img-loss levels]: write every cell of the first directory over the matching cell of the second, section by section as a bake does (the new wins inside its terrain footprint, the old stays outside it; a lossy section the new lacks is carried over byte for byte). The local bake over the global one: the 1 m rectangle on the 30 m globe, nothing knocked out around it.
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let loss = mahere_tiles::Loss { dem_m: value("--dem-loss").map(|v| v.parse().unwrap()).unwrap_or(0.2), img: value("--img-loss").map(|v| v.parse().unwrap()).unwrap_or(16) };
    let over = std::path::PathBuf::from(&args[0]);
    let into = std::path::PathBuf::from(&args[1]);
    let files: Vec<_> = std::fs::read_dir(&over).expect("over dir").flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "zst")).collect();
    let merged = AtomicUsize::new(0);
    files.par_iter().for_each(|path| {
        let name = path.file_name().unwrap();
        let bytes = std::fs::read(path).expect("read");
        let planes = mahere_tiles::decode_cell(&bytes).expect("decode");
        let dst = into.join(name);
        // Through the baker's own merge: the cell as a bake, written over whatever is there.
        let key = mahere_tiles::CellKey::from_name(name.to_str().unwrap()).expect("cell name");
        let cell = mahere_tiles::Cell { dem: planes.dem.map(|d| mahere_tiles::DemCell { elev: d.elev, apron: d.apron }), line: planes.line, land: planes.land, water: planes.water, img: planes.img };
        mahere_tiles::write_cell(&into, key, &cell, &loss).expect("write");
        if dst.exists() {
            merged.fetch_add(1, Ordering::Relaxed);
        }
    });
    eprintln!("{} cells written over {}, {} of them merged with a cell already there", files.len(), into.display(), merged.load(Ordering::Relaxed));
}
