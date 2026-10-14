//! `zstd_time <cell files...>`: how long today's cells take to unwrap and fully decode, for comparison with the range coder.
fn main() {
    let files: Vec<Vec<u8>> = std::env::args().skip(1).map(|f| std::fs::read(f).unwrap()).collect();
    let t = std::time::Instant::now();
    for z in &files {
        let _ = mahere_tiles::decode_cell(z).unwrap();
    }
    println!("{} cells decoded today's way in {:.1} ms each", files.len(), t.elapsed().as_secs_f64() * 1e3 / files.len() as f64);
}
