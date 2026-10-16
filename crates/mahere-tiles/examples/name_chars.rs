//! `name_chars <cells dir>`: every distinct character in the cells' names, by script block, and the CJK ones written out (one line) to stdout's last line for a font subset.
fn main() {
    let dir = std::env::args().nth(1).expect("cells dir");
    let mut seen: std::collections::BTreeSet<char> = Default::default();
    let mut cells = 0usize;
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let p = e.path();
        if !p.to_string_lossy().ends_with(".vsf.zst") {
            continue;
        }
        let Ok(b) = std::fs::read(&p) else { continue };
        let Some(n) = mahere_tiles::decode_names(&b) else { continue };
        cells += 1;
        for named in &n.entries {
            seen.extend(named.name.chars());
        }
    }
    let cjk = |c: char| matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F | 0xF900..=0xFAFF | 0x20000..=0x3134F | 0x3000..=0x303F | 0xFF00..=0xFFEF);
    let han = seen.iter().filter(|&&c| matches!(c as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x3134F)).count();
    let kana = seen.iter().filter(|&&c| matches!(c as u32, 0x3040..=0x30FF)).count();
    let hangul = seen.iter().filter(|&&c| matches!(c as u32, 0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F)).count();
    eprintln!("{cells} cells, {} distinct characters; CJK: {han} han, {kana} kana, {hangul} hangul, {} in all", seen.len(), seen.iter().filter(|&&c| cjk(c)).count());
    println!("{}", seen.iter().filter(|&&c| cjk(c)).collect::<String>());
}
