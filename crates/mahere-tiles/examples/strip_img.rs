// strip_img <cell dir>: remove the imagery section from every cell that has one, leaving the other sections byte for byte (the terrain is lossy; it is never decoded here). For a bake to be redone under a new tone.
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    let dir = std::env::args().nth(1).expect("cell dir");
    let files: Vec<_> = std::fs::read_dir(&dir).expect("dir").flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "zst")).collect();
    let stripped = AtomicUsize::new(0);
    files.par_iter().for_each(|path| {
        let Ok(z) = std::fs::read(path) else { return };
        let Ok(plain) = zstd::decode_all(&z[..]) else { return };
        let Ok((header, end)) = vsf::VsfHeader::decode(&plain) else { return };
        let Ok(sections) = header.sections(&plain, end) else { return };
        if !sections.iter().any(|s| s.name == "img") {
            return;
        }
        let mut b = vsf::VsfBuilder::new();
        for s in sections.into_iter().filter(|s| s.name != "img") {
            b = b.add_section_direct(s);
        }
        let Ok(bytes) = b.build() else { return };
        let Ok(out) = zstd::encode_all(&bytes[..], 3) else { return };
        std::fs::write(path, out).expect("write");
        stripped.fetch_add(1, Ordering::Relaxed);
    });
    eprintln!("{} of {} cells had imagery; stripped", stripped.load(Ordering::Relaxed), files.len());
}
