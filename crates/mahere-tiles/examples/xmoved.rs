//! xmoved <old huffman_codes.bin> <published base dir> [max depth]: of the names in the cells, how many hold a character whose code is longer than 20 bits and is not ASCII (the characters VSF v11's reissued codebook moves, so v10 text holding them reads as other characters under v11), by script, with examples.
use rayon::prelude::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let book = std::fs::read(&a[1]).unwrap();
    let n = u32::from_le_bytes(book[8..12].try_into().unwrap()) as usize;
    let mut moved = vec![false; 0x110000];
    for i in 0..n {
        let o = 12 + i * 8;
        let cp = u32::from_le_bytes(book[o..o + 4].try_into().unwrap()) as usize;
        let len = (u32::from_le_bytes(book[o + 4..o + 8].try_into().unwrap()) >> 24) & 0x1F;
        if cp < moved.len() {
            moved[cp] = cp > 127 && len > 20;
        }
    }
    let max_depth: u8 = a.get(3).and_then(|d| d.parse().ok()).unwrap_or(14);
    let files: Vec<std::path::PathBuf> = std::fs::read_dir(&a[2]).unwrap().flatten().map(|e| e.path()).filter(|p| {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        mahere_tiles::CellKey::from_name(name.split('.').next().unwrap()).is_some_and(|k| k.depth <= max_depth)
    }).collect();
    let script = |c: char| -> &'static str {
        match c as u32 {
            0x0080..=0x024F => "Latin accented",
            0x0370..=0x03FF => "Greek",
            0x0400..=0x04FF => "Cyrillic",
            0x0590..=0x05FF => "Hebrew",
            0x0600..=0x06FF | 0x0750..=0x077F => "Arabic",
            0x0900..=0x0DFF => "Indic",
            0x0E00..=0x0E7F => "Thai",
            0x3040..=0x30FF => "Japanese kana",
            0x4E00..=0x9FFF | 0x3400..=0x4DBF => "CJK ideographs",
            0xAC00..=0xD7AF | 0x1100..=0x11FF => "Hangul",
            _ => "other",
        }
    };
    let per: Vec<(usize, usize, Vec<(&'static str, usize)>, Vec<String>)> = files.par_iter().map(|p| {
        let Ok(b) = std::fs::read(p) else { return (0, 0, Vec::new(), Vec::new()) };
        let plain = zstd::decode_all(&b[..]).unwrap_or(b);
        let Some(names) = mahere_tiles::decode_names(&plain) else { return (0, 0, Vec::new(), Vec::new()) };
        let (mut all, mut hit) = (0, 0);
        let mut scripts: Vec<(&'static str, usize)> = Vec::new();
        let mut ex = Vec::new();
        for e in &names.entries {
            for t in [&e.name, &e.reference] {
                if t.is_empty() {
                    continue;
                }
                all += 1;
                if let Some(c) = t.chars().find(|&c| moved[c as usize]) {
                    hit += 1;
                    let s = script(c);
                    match scripts.iter_mut().find(|x| x.0 == s) {
                        Some(x) => x.1 += 1,
                        None => scripts.push((s, 1)),
                    }
                    if ex.len() < 2 {
                        ex.push(t.clone());
                    }
                }
            }
        }
        (all, hit, scripts, ex)
    }).collect();
    let (mut all, mut hit) = (0, 0);
    let mut scripts: Vec<(&'static str, usize)> = Vec::new();
    let mut ex: Vec<String> = Vec::new();
    for (a, h, s, e) in per {
        all += a;
        hit += h;
        for (k, v) in s {
            match scripts.iter_mut().find(|x| x.0 == k) {
                Some(x) => x.1 += v,
                None => scripts.push((k, v)),
            }
        }
        if ex.len() < 24 {
            ex.extend(e);
        }
    }
    scripts.sort_by(|a, b| b.1.cmp(&a.1));
    println!("{} cells, {all} names, {hit} hold a moved character ({:.1}%)", files.len(), 100.0 * hit as f64 / all.max(1) as f64);
    for (k, v) in scripts {
        println!("  {k:<16} {v}");
    }
    println!("examples: {}", ex.join(" | "));
}
