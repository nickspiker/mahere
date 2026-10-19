//! `geoid_probe <published dir> <lat> <lon>`: the depth-0 cell's geoid about a point, the texels around it, centimetres (a check of what the geoid layer draws from).
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (lat, lon): (f64, f64) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let c = mahere_coord::Coord::from_lat_lon(lat, lon);
    let key = mahere_tiles::CellKey::containing(c, 0);
    let bytes = std::fs::read(std::path::Path::new(&a[1]).join("base").join(format!("{}.vsf", key.name()))).expect("the depth-0 base");
    let g = mahere_tiles::decode_geoid_only(&bytes).expect("a geoid plane");
    let (u, v) = c.uv();
    let (tx, ty) = (((u >> 22) & 255) as usize, ((v >> 22) & 255) as usize);
    println!("{} texel {tx},{ty} of {} values, u {u:#x}", key.name(), g.len());
    for y in ty.saturating_sub(1)..=(ty + 1).min(255) {
        let row: Vec<String> = (tx.saturating_sub(1)..=(tx + 1).min(255)).flat_map(|x| [0, 1].map(move |h| (x, h))).map(|(x, h)| format!("{:6}", g[y * 512 + 2 * x + h])).collect();
        println!("  row {y}: {}", row.join(" "));
    }
}
