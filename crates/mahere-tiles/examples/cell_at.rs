//! `cell_at <lat> <lon> <depth>`: the name of the cell there, as the bucket stores it.
fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|s| s.parse().expect("number")).collect();
    let k = mahere_tiles::CellKey::containing(mahere_coord::Coord::from_lat_lon(a[0], a[1]), a[2] as u8);
    println!("{}", k.path());
}
