//! graphcheck <cell.vsf[.zst]>...: each cell's route section, its ways, nodes and corridors, and whether it carries the graph; with -v, each way's trixels (its own and stacked) and each corridor's marked trixels, so an unmarked corridor (a route's straight jump) shows.
fn main() {
    let verbose = std::env::args().any(|a| a == "-v");
    for p in std::env::args().skip(1).filter(|a| a != "-v") {
        let b = std::fs::read(&p).unwrap();
        let plain = zstd::decode_all(&b[..]).unwrap_or(b);
        let Some(r) = mahere_tiles::decode_route(&plain) else {
            println!("{p}: no route section");
            continue;
        };
        println!("{p}: {} ways, {} nodes, {} corridors, graph {}", r.ways.len(), r.nodes.len(), r.corridors.len(), r.has_graph());
        if !verbose {
            continue;
        }
        for (k, w) in r.ways.iter().enumerate() {
            let wi = k as u16 + 1;
            let own = r.way.iter().filter(|&&x| x == wi).count();
            let stacked = r.stack.iter().filter(|&&(_, x)| x == wi).count();
            let nodes = r.nodes.iter().filter(|n| n.1 == wi).count();
            let unmarked = (0..mahere_tiles::TRI).filter(|&i| r.ways_at(i).any(|x| x == wi) && r.corridor_at(i, wi) == 0 && r.node_of(i, wi).is_none()).count();
            println!("  way {wi} osm {} class {} {:?}: {own} own + {stacked} stacked trixels, {nodes} node trixels, {unmarked} neither node nor corridor", w.osm, w.class, w.name);
        }
        for (k, c) in r.corridors.iter().enumerate() {
            let id = k as u16 + 1;
            let marked = (0..mahere_tiles::TRI).filter(|&i| r.corridor_at(i, c.way) == id).count();
            if marked == 0 || verbose {
                println!("  corridor {id} way {} {} -> {} len {:.1} texels dir {}: {marked} trixels marked", c.way, c.a, c.b, c.len as f64 / 256.0, c.dir);
            }
        }
    }
}
