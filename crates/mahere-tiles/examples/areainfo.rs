//! areainfo <area.vsf[.zst]> [published dir] [measured area.vsf[.zst]]: an area's name, detail, box and ground, its cells at each depth, and what its 1 m download would weigh: by the mean cell sizes `mean_cell_bytes` assumes, and, given a published directory and an area already baked, by that area's own bytes per cell at each depth and layer.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let area = mahere_tiles::area::Area::decode(&std::fs::read(&a[1]).unwrap()).expect("an area");
    let deep = area.cells_at(14);
    let (mut la0, mut lo0, mut la1, mut lo1) = (90.0f64, 180.0f64, -90.0f64, -180.0f64);
    for k in &deep {
        for (la, lo) in k.corners() {
            (la0, lo0, la1, lo1) = (la0.min(la), lo0.min(lo), la1.max(la), lo1.max(lo));
        }
    }
    let km2 = deep.len() as f64 * mahere_tiles::CellKey { depth: 14, prefix: 0 }.texel_m().powi(2) * (mahere_tiles::TEX * mahere_tiles::TEX) as f64 / 2.0 / 1e6 * 2.0;
    println!("{}: detail to depth {}, box {la0:.3},{lo0:.3} to {la1:.3},{lo1:.3}, about {km2:.0} km\u{b2} ({} depth-14 cells)", area.name, area.depth, deep.len());
    let counts: Vec<(u8, usize)> = (0..=14).map(|d| (d, area.count_at(d) as usize)).collect();
    println!("cells by depth: {}", counts.iter().filter(|c| c.0 >= 8).map(|(d, n)| format!("{d}:{n}")).collect::<Vec<_>>().join("  "));
    let (_, est) = area.estimate_to(14);
    println!("estimate from mean cell sizes: {:.2} GB", est as f64 / 1e9);
    if let (Some(dir), Some(m)) = (a.get(2), a.get(3)) {
        let measured = mahere_tiles::area::Area::decode(&std::fs::read(m).unwrap()).expect("a measured area");
        let mut total = 0.0;
        let mut by_layer = [0.0f64; 4];
        for d in 9..=14u8 {
            let cells = measured.cells_at(d);
            for (li, layer) in ["base", "dem", "img", "nir"].iter().enumerate() {
                let (mut n, mut bytes) = (0u64, 0u64);
                for k in &cells {
                    if let Ok(md) = std::fs::metadata(std::path::Path::new(dir).join(layer).join(format!("{}.vsf", k.name()))) {
                        n += 1;
                        bytes += md.len();
                    }
                }
                if n > 0 {
                    let mean = bytes as f64 / n as f64;
                    let ours = area.count_at(d) as f64 * mean;
                    by_layer[li] += ours;
                    total += ours;
                }
            }
        }
        println!("by {}'s bytes per cell: {:.2} GB (base {:.2}, terrain {:.2}, imagery {:.2}, infrared {:.2})", measured.name, total / 1e9, by_layer[0] / 1e9, by_layer[1] / 1e9, by_layer[2] / 1e9, by_layer[3] / 1e9);
    }
}
