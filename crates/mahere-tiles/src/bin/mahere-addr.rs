//! mahere-addr --tiger <dir> [--pbf <extract>]... --out <published dir>: the address documents (Nick 2026-10-27: "TIGER and all"), one per ZIP at `<out>/addr/<zip>.vsf`, from every TIGER/Line ADDRFEAT shapefile under the directory (each block's house numbers by side, with the ZIP on each side) and the addresses the OSM extracts place exactly; and `<out>/addr/zips.vsf`, every ZIP's box, merged with the ZIPs an earlier run wrote.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mahere_tiles::addr::{base, house_number, Point, Range, Side, ZipDoc, ZipIndex};
use rayon::prelude::*;

#[derive(Default)]
struct Builder {
    doc: ZipDoc,
    names: BTreeMap<String, u32>,
    bbox: Option<[f64; 4]>,
}

impl Builder {
    fn street(&mut self, name: &str) -> u32 {
        if let Some(&i) = self.names.get(name) {
            return i;
        }
        let i = self.doc.streets.len() as u32;
        self.doc.streets.push(name.to_string());
        self.names.insert(name.to_string(), i);
        i
    }
    fn grow(&mut self, lat: f64, lon: f64) {
        let b = self.bbox.get_or_insert([lat, lon, lat, lon]);
        *b = [b[0].min(lat), b[1].min(lon), b[2].max(lat), b[3].max(lon)];
    }
}

fn shapefiles(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            shapefiles(&p, out);
        } else if p.to_string_lossy().ends_with("_addrfeat.shp") {
            out.push(p);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let all = |k: &str| args.iter().enumerate().filter(|(_, a)| *a == k).filter_map(|(i, _)| args.get(i + 1).cloned()).collect::<Vec<_>>();
    let out = PathBuf::from(arg("--out").expect("--out"));
    let t = std::time::Instant::now();
    let mut files = Vec::new();
    shapefiles(Path::new(&arg("--tiger").expect("--tiger")), &mut files);
    files.sort();
    let edges: Vec<mahere_osm::addr::Edge> = files.par_iter().flat_map(|f| mahere_osm::addr::read_addrfeat(&f.to_string_lossy()).unwrap_or_else(|e| { eprintln!("{e}"); Vec::new() })).collect();
    eprintln!("{} TIGER files, {} edges, {:.1}s", files.len(), edges.len(), t.elapsed().as_secs_f32());

    // The ranges, by ZIP: a block whose two sides lie in one ZIP is one range there; sides in two ZIPs go each to its own.
    let mut zips: BTreeMap<u32, Builder> = BTreeMap::new();
    for e in &edges {
        let side = |s: Option<mahere_osm::addr::Side>| s.map(|s| (s.zip, Side { from: s.from, to: s.to, parity: s.parity }));
        let (l, r) = (side(e.left), side(e.right));
        let mut put = |zip: u32, left: Option<Side>, right: Option<Side>| {
            if zip == 0 {
                return;
            }
            let b = zips.entry(zip).or_default();
            let street = b.street(&e.street);
            for &(lat, lon) in &e.pts {
                b.grow(lat, lon);
            }
            b.doc.ranges.push(Range { street, left, right, pts: e.pts.clone() });
        };
        match (l, r) {
            (Some((zl, sl)), Some((zr, sr))) if zl == zr => put(zl, Some(sl), Some(sr)),
            _ => {
                if let Some((z, s)) = l {
                    put(z, Some(s), None);
                }
                if let Some((z, s)) = r {
                    put(z, None, Some(s));
                }
            }
        }
    }
    eprintln!("{} ZIPs from TIGER", zips.len());

    // The edges by a coarse grid (a fiftieth of a degree), for the ZIP of an OSM address that carries none: sorted, searched by binary search.
    let cell = |lat: f64, lon: f64| -> i64 { ((lat * 50.0).floor() as i64) << 32 | ((lon * 50.0).floor() as i64 & 0xFFFF_FFFF) };
    let mut grid: Vec<(i64, u32)> = Vec::new();
    for (i, e) in edges.iter().enumerate() {
        let mut cs: Vec<i64> = e.pts.iter().map(|&(a, o)| cell(a, o)).collect();
        cs.sort_unstable();
        cs.dedup();
        grid.extend(cs.into_iter().map(|c| (c, i as u32)));
    }
    grid.sort_unstable();
    let edge_zip = |e: &mahere_osm::addr::Edge| e.left.map(|s| s.zip).filter(|&z| z != 0).or(e.right.map(|s| s.zip).filter(|&z| z != 0));
    let nearest_zip = |lat: f64, lon: f64, street: &str| -> Option<u32> {
        let want = base(street);
        let k = lat.to_radians().cos();
        let mut best: Option<(bool, f64, u32)> = None;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let c = cell(lat + dy as f64 / 50.0, lon + dx as f64 / 50.0);
                let lo = grid.partition_point(|g| g.0 < c);
                for &(_, i) in grid[lo..].iter().take_while(|g| g.0 == c) {
                    let e = &edges[i as usize];
                    let Some(z) = edge_zip(e) else { continue };
                    let d = e.pts.iter().map(|&(a, o)| ((a - lat) * 111_320.0).powi(2) + ((o - lon) * 111_320.0 * k).powi(2)).fold(f64::MAX, f64::min).sqrt();
                    let same = base(&e.street) == want;
                    // The same street first, then the nearest.
                    if d < 2000.0 && best.is_none_or(|b| (same, -d) > (b.0, -b.1)) {
                        best = Some((same, d, z));
                    }
                }
            }
        }
        best.map(|b| b.2)
    };
    for pbf in all("--pbf") {
        let t = std::time::Instant::now();
        let pts = mahere_osm::addr::load_addresses(&pbf).expect("osm extract");
        let placed: Vec<(u32, u32, &mahere_osm::addr::AddrPoint)> = pts
            .par_iter()
            .filter_map(|p| {
                let n = house_number(&p.number)?;
                let z = if zips.contains_key(&p.zip) { p.zip } else { nearest_zip(p.lat, p.lon, &p.street)? };
                Some((z, n, p))
            })
            .collect();
        let mut kept = 0;
        for (z, n, p) in placed {
            let Some(b) = zips.get_mut(&z) else { continue };
            let street = b.street(&p.street);
            b.grow(p.lat, p.lon);
            b.doc.points.push(Point { street, number: n, lat: p.lat, lon: p.lon });
            kept += 1;
        }
        eprintln!("{pbf}: {} addresses, {kept} placed in a ZIP, {:.1}s", pts.len(), t.elapsed().as_secs_f32());
    }

    let dir = out.join("addr");
    std::fs::create_dir_all(&dir).unwrap();
    let mut index = std::fs::read(dir.join("zips.vsf")).ok().and_then(|b| ZipIndex::decode(&b)).unwrap_or_default();
    index.zips.retain(|(z, _)| !zips.contains_key(z));
    let bytes: Vec<(u32, [f64; 4], usize)> = zips
        .par_iter_mut()
        .map(|(&z, b)| {
            b.doc.zip = z;
            let enc = b.doc.encode();
            std::fs::write(dir.join(format!("{z:05}.vsf")), &enc).expect("write");
            (z, b.bbox.unwrap_or([0.0; 4]), enc.len())
        })
        .collect();
    let total: usize = bytes.iter().map(|b| b.2).sum();
    index.zips.extend(bytes.iter().map(|b| (b.0, b.1)));
    index.zips.sort_by_key(|z| z.0);
    std::fs::write(dir.join("zips.vsf"), index.encode()).expect("write index");
    eprintln!("{} ZIP documents, {:.1} MB, index of {} ZIPs, {:.1}s", bytes.len(), total as f64 / 1e6, index.zips.len(), t.elapsed().as_secs_f32());
}
