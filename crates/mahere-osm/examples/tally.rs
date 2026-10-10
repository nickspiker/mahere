// tally <extract.osm.pbf>: the lines an extract yields, counted by class, with their weights' range; for checking what a bake will see.
use std::collections::BTreeMap;
fn main() {
    let path = std::env::args().nth(1).expect("extract");
    let f = mahere_osm::load_features(&path).expect("load");
    let mut by: BTreeMap<String, (usize, f32, f32)> = BTreeMap::new();
    for r in &f.roads {
        let e = by.entry(format!("{:?}", r.class)).or_insert((0, f32::MAX, f32::MIN));
        e.0 += 1;
        e.1 = e.1.min(r.weight);
        e.2 = e.2.max(r.weight);
    }
    for (c, (n, lo, hi)) in by {
        println!("{c}: {n} lines, weight {lo:.2}..{hi:.2}");
    }
}
