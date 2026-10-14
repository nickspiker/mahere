//! `section_sizes <cell files...>`: each section's share of the cells, raw and compressed on its own, to see what a fetch of only some sections would save.
fn main() {
    let mut total: std::collections::BTreeMap<String, (usize, usize, usize)> = Default::default();
    let mut whole = 0usize;
    let files: Vec<String> = std::env::args().skip(1).collect();
    for f in &files {
        let z = std::fs::read(f).unwrap();
        whole += z.len();
        let data = zstd::decode_all(&z[..]).unwrap();
        let (header, end) = vsf::VsfHeader::decode(&data).unwrap();
        for e in &header.fields {
            if e.size_bytes == 0 {
                continue;
            }
            let start = if e.offset_bytes != 0 { e.offset_bytes } else { end };
            let bytes = &data[start..(start + e.size_bytes).min(data.len())];
            let c = zstd::encode_all(bytes, 19).unwrap().len();
            let t = total.entry(e.name.clone()).or_default();
            t.0 += e.size_bytes;
            t.1 += c;
            t.2 += 1;
        }
    }
    let n = files.len().max(1);
    println!("{} cells, {:.0} KB each as stored", files.len(), whole as f64 / n as f64 / 1024.0);
    let csum: usize = total.values().map(|t| t.1).sum();
    for (name, (raw, c, k)) in &total {
        println!("  {name:6} in {k:4} cells  raw {:7.0} KB  zstd alone {:7.1} KB  {:4.1}% of the compressed", *raw as f64 / n as f64 / 1024.0, *c as f64 / n as f64 / 1024.0, *c as f64 * 100.0 / csum as f64);
    }
}
