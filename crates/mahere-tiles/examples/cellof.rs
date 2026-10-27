//! cellof <lat> <lon> <depth>: the name of the cell holding a place at a depth.
fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|x| x.parse().unwrap()).collect();
    println!("{}", mahere_tiles::CellKey::containing(mahere_coord::Coord::from_lat_lon(a[0], a[1]), a[2] as u8).name());
}
