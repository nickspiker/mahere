//! `elev_outliers <area.vsf.zst> <cells dir> [floor m] [depth]`: texels whose elevation is below a floor (-500 m by default) in an area's cells, counted and located: a source's no-data value read as ground.
use rayon::prelude::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let area = mahere_tiles::area::Area::decode(&std::fs::read(&a[1]).unwrap()).unwrap();
    let dir = std::path::PathBuf::from(&a[2]);
    let floor: f32 = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(-500.0);
    let depth: u8 = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(14);
    // An optional fifth argument: only the cells under that cell (one piece).
    let under = a.get(5).and_then(|n| mahere_tiles::CellKey::from_name(n));
    let mut hits: Vec<(mahere_tiles::CellKey, usize, f32)> = area
        .cells_at(depth)
        .into_iter()
        .filter(|k| under.is_none_or(|p| k.ancestor(p.depth) == p))
        .collect::<Vec<_>>()
        .par_iter()
        .filter_map(|&k| {
            let p = mahere_tiles::decode_cell(&std::fs::read(dir.join(k.path())).ok()?).ok()?;
            let d = p.dem?;
            let bad: Vec<f32> = d.elev.iter().copied().filter(|e| !e.is_nan() && *e < floor).collect();
            (!bad.is_empty()).then(|| (k, bad.len(), bad.iter().copied().fold(f32::MAX, f32::min)))
        })
        .collect();
    hits.sort_by_key(|h| std::cmp::Reverse(h.1));
    println!("{} cells at depth {depth} with elevations below {floor} m, {} texels", hits.len(), hits.iter().map(|h| h.1).sum::<usize>());
    for (k, n, lo) in hits.iter().take(10) {
        let c = k.corners();
        println!("  {:.4},{:.4}  {n} texels, lowest {lo:.1} m", (c[0].0 + c[3].0) / 2.0, (c[0].1 + c[3].1) / 2.0);
    }
}
