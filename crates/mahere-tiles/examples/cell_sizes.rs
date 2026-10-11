//! Mean bytes per cell at each depth across a cell directory: the table an area's download estimate rests on.
fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "data/cells".into());
    let mut sum = [0u64; 32];
    let mut n = [0u64; 32];
    for e in std::fs::read_dir(&dir).expect("dir").flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(k) = mahere_tiles::CellKey::from_name(&name) else { continue };
        let b = e.metadata().map(|m| m.len()).unwrap_or(0);
        sum[k.depth as usize] += b;
        n[k.depth as usize] += 1;
    }
    for d in 0..32 {
        if n[d] > 0 {
            println!("depth {d:2}: {:7} cells, mean {:8} bytes, total {:6.1} MB", n[d], sum[d] / n[d], sum[d] as f64 / 1e6);
        }
    }
}
