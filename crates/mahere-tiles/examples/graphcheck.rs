//! graphcheck <cell.vsf.zst>...: each cell's route section, its ways, nodes and corridors, and whether it carries the graph.
fn main() {
    for p in std::env::args().skip(1) {
        let b = std::fs::read(&p).unwrap();
        let plain = zstd::decode_all(&b[..]).unwrap_or(b);
        match mahere_tiles::decode_route(&plain) {
            Some(r) => println!("{p}: {} ways, {} nodes, {} corridors, graph {}", r.ways.len(), r.nodes.len(), r.corridors.len(), r.has_graph()),
            None => println!("{p}: no route section"),
        }
    }
}
