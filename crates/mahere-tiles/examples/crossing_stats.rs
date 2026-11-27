//! `crossing_stats <baked dir> [every k-th cell]`: what the route crossings cost (2026-11-26): per depth, the cells with a route graph, their edge nodes, the time to work out the tables and their bytes coded as a publish codes them.
use std::collections::BTreeMap;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let every: usize = a.get(2).map_or(1, |v| v.parse().unwrap());
    let mut files: Vec<_> = std::fs::read_dir(&a[1]).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().ends_with(".vsf")).collect();
    files.sort_by_key(|e| e.file_name());
    // depth -> (cells, ends, most ends, seconds, coded bytes, cell bytes)
    let mut by: BTreeMap<u8, (usize, usize, usize, f64, usize, usize)> = BTreeMap::new();
    for e in files.iter().step_by(every) {
        let name = e.file_name().into_string().unwrap();
        let Some(key) = mahere_tiles::CellKey::from_name(&name) else { continue };
        let bytes = std::fs::read(e.path()).unwrap();
        let Some(r) = mahere_tiles::decode_route(&bytes) else { continue };
        if !r.has_graph() {
            continue;
        }
        let t = std::time::Instant::now();
        let x = r.make_crossings(key.depth);
        let secs = t.elapsed().as_secs_f64();
        let coded = x.as_ref().map_or(0, |x| {
            let s = x.section();
            let mut out = vsf::VsfSection::new("crossing");
            for f in s.fields {
                out.add_field_multi(f.name, mahere_tiles::ac::code_field(&f.values, None).unwrap_or(f.values));
            }
            vsf::VsfBuilder::new().add_section_direct(out).build().unwrap().len()
        });
        let ends = x.as_ref().map_or(0, |x| x.ends.len());
        let s = by.entry(key.depth).or_default();
        s.0 += 1;
        s.1 += ends;
        s.2 = s.2.max(ends);
        s.3 += secs;
        s.4 += coded;
        s.5 += bytes.len();
    }
    for (d, s) in by {
        println!("depth {d}: {} cells, {:.1} ends on average (most {}), {:.2} ms each, tables {:.1} KB on average coded, {:.1}% of the cells' bytes", s.0, s.1 as f64 / s.0 as f64, s.2, s.3 * 1e3 / s.0 as f64, s.4 as f64 / s.0 as f64 / 1024., s.4 as f64 * 100. / s.5 as f64);
    }
}
