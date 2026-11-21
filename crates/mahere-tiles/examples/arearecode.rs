//! arearecode <area file> <out.vsf>: an area document read (either form) and written as `Area::encode` writes it now, read back and checked plane for plane.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let old = std::fs::read(&a[1]).unwrap();
    let area = mahere_tiles::area::Area::decode(&old).expect("an area");
    let new = area.encode(true);
    std::fs::write(&a[2], &new).unwrap();
    let back = mahere_tiles::area::Area::decode(&new).expect("reads back");
    assert_eq!(back.name, area.name);
    assert_eq!(back.cells.len(), area.cells.len());
    for ((ka, ca), (kb, cb)) in area.cells.iter().zip(&back.cells) {
        assert_eq!(ka, kb);
        assert!(ca.cov == cb.cov, "a plane changed");
    }
    println!("{}: {} planes, {} bytes before, {} now, the same planes read back", area.name, area.cells.len(), old.len(), new.len());
}
