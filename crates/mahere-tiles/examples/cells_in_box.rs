// cells_in_box <lat0> <lon0> <lat1> <lon1> <from depth> <to depth>: the file names of every cell covering the box at each depth in the range, one a line, for a partial publish.
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n: Vec<f64> = a[..4].iter().map(|s| s.parse().unwrap()).collect();
    let (d0, d1): (u8, u8) = (a[4].parse().unwrap(), a[5].parse().unwrap());
    let (lat0, lon0, lat1, lon1) = (n[0].min(n[2]), n[1].min(n[3]), n[0].max(n[2]), n[1].max(n[3]));
    for d in d0..=d1 {
        for k in mahere_tiles::cells_covering(lat0, lon0, lat1, lon1, d) {
            println!("{}", k.path());
        }
    }
}
