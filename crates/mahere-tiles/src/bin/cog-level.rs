//! Pull one overview level out of cloud-optimised GeoTIFFs over HTTP, without downloading the rest of the file: the header, then one ranged read of that level's tiles, rewritten as a small standalone tiled GeoTIFF with the geo tags of the full image scaled to the level. Resumable (finished files are skipped, partial ones written to `.part` and renamed). Usage:
//!   cog-level --base <url prefix> --keys <file of object keys> --level 2 --out <dir> [--jobs 8]
//! For the ESA WorldCover Sentinel-2 composite: base https://esa-worldcover-s2.s3.eu-central-1.amazonaws.com/rgbnir/2021/ and keys like N47/ESA_WorldCover_10m_2021_v200_N47W121_S2RGBNIR.tif; level 2 is 1/3000 degree (37 m), the source for depth 8.
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

/// Bytes [a, b) of a URL, retried with backoff through the odd DNS or connection hiccup.
fn fetch(agent: &ureq::Agent, url: &str, a: u64, b: u64) -> Result<Vec<u8>, String> {
    let mut last = String::new();
    for attempt in 0..8 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_secs(2u64 << attempt.min(6)));
        }
        match agent.get(url).set("Range", &format!("bytes={a}-{}", b - 1)).call() {
            Ok(resp) => {
                let mut buf = Vec::with_capacity((b - a) as usize);
                match resp.into_reader().take(b - a).read_to_end(&mut buf) {
                    Ok(_) if buf.len() as u64 == b - a => return Ok(buf),
                    Ok(_) => last = format!("short read {} of {}", buf.len(), b - a),
                    Err(e) => last = e.to_string(),
                }
            }
            Err(ureq::Error::Status(416, _)) => return Err(format!("{url}: range {a}..{b} not satisfiable")),
            Err(ureq::Error::Status(404, _)) => return Err(format!("{url}: not found")),
            Err(e) => last = e.to_string(),
        }
    }
    Err(format!("{url}: {last}"))
}

/// One IFD entry: its type and count, and its value bytes (inline or out of line, resolved).
#[derive(Clone)]
struct Entry {
    typ: u16,
    count: u32,
    bytes: Vec<u8>,
}

fn type_size(typ: u16) -> usize {
    match typ {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 => 8,
        _ => 1,
    }
}

impl Entry {
    fn u32s(&self) -> Vec<u32> {
        match self.typ {
            3 => self.bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect(),
            _ => self.bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        }
    }
    fn f64s(&self) -> Vec<f64> {
        self.bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    }
    fn longs(v: &[u32]) -> Entry {
        Entry { typ: 4, count: v.len() as u32, bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    fn doubles(v: &[f64]) -> Entry {
        Entry { typ: 12, count: v.len() as u32, bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
}

/// Every IFD of a little-endian classic TIFF, given enough of its head that the IFDs and their out-of-line values are inside; None asks for more of the head.
fn ifds(head: &[u8]) -> Result<Option<Vec<BTreeMap<u16, Entry>>>, String> {
    if head.len() < 8 || &head[..4] != b"II*\0" {
        return Err("not a little-endian classic TIFF".into());
    }
    let u16_at = |o: usize| u16::from_le_bytes([head[o], head[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([head[o], head[o + 1], head[o + 2], head[o + 3]]);
    let mut out = Vec::new();
    let mut off = u32_at(4) as usize;
    while off != 0 {
        if off + 2 > head.len() {
            return Ok(None);
        }
        let n = u16_at(off) as usize;
        if off + 6 + 12 * n > head.len() {
            return Ok(None);
        }
        let mut ifd = BTreeMap::new();
        for i in 0..n {
            let e = off + 2 + 12 * i;
            let (tag, typ, count) = (u16_at(e), u16_at(e + 2), u32_at(e + 4));
            let len = type_size(typ) * count as usize;
            let bytes = if len <= 4 {
                head[e + 8..e + 8 + len].to_vec()
            } else {
                let at = u32_at(e + 8) as usize;
                if at + len > head.len() {
                    return Ok(None);
                }
                head[at..at + len].to_vec()
            };
            ifd.insert(tag, Entry { typ, count, bytes });
        }
        out.push(ifd);
        off = u32_at(off + 2 + 12 * n) as usize;
    }
    Ok(Some(out))
}

/// A classic little-endian TIFF of one IFD followed by its tile data, the tile offsets pointing into it.
fn write_tiff(mut tags: BTreeMap<u16, Entry>, counts: &[u32], data: &[u8], rel: &[u32]) -> Vec<u8> {
    tags.insert(325, Entry::longs(counts));
    // Offsets are filled once the layout is known; reserve the entry now so the IFD size is right.
    tags.insert(324, Entry::longs(&vec![0; rel.len()]));
    let ifd_len = 2 + 12 * tags.len() + 4;
    let extra: usize = tags.values().map(|e| if e.bytes.len() > 4 { (e.bytes.len() + 1) & !1 } else { 0 }).sum();
    let data_at = 8 + ifd_len + extra;
    let offsets: Vec<u32> = rel.iter().map(|r| data_at as u32 + r).collect();
    tags.insert(324, Entry::longs(&offsets));
    let mut out = Vec::with_capacity(data_at + data.len());
    out.extend_from_slice(b"II*\0");
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&(tags.len() as u16).to_le_bytes());
    let mut blob = Vec::new();
    for (tag, e) in &tags {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&e.typ.to_le_bytes());
        out.extend_from_slice(&e.count.to_le_bytes());
        if e.bytes.len() <= 4 {
            let mut v = e.bytes.clone();
            v.resize(4, 0);
            out.extend_from_slice(&v);
        } else {
            out.extend_from_slice(&((8 + ifd_len + blob.len()) as u32).to_le_bytes());
            blob.extend_from_slice(&e.bytes);
            if blob.len() % 2 == 1 {
                blob.push(0);
            }
        }
    }
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&blob);
    debug_assert_eq!(out.len(), data_at);
    out.extend_from_slice(data);
    out
}

fn extract(agent: &ureq::Agent, url: &str, level: usize) -> Result<Vec<u8>, String> {
    let mut want = 65536u64;
    let all = loop {
        let head = fetch(agent, url, 0, want)?;
        match ifds(&head)? {
            Some(v) => break v,
            None if want < 1 << 24 => want *= 4,
            None => return Err(format!("{url}: IFDs not within the first 16 MB")),
        }
    };
    let full = &all[0];
    let lv = all.get(level).ok_or_else(|| format!("{url}: only {} levels", all.len()))?;
    let offs = lv.get(&324).ok_or("no TileOffsets")?.u32s();
    let counts = lv.get(&325).ok_or("no TileByteCounts")?.u32s();
    let lo = *offs.iter().min().unwrap() as u64;
    let hi = offs.iter().zip(&counts).map(|(&o, &c)| o as u64 + c as u64).max().unwrap();
    // A COG keeps each level's tiles together, so one read covers them; refuse anything that would drag in most of the file.
    if hi - lo > 4 * counts.iter().map(|&c| c as u64).sum::<u64>() {
        return Err(format!("{url}: level {level} tiles are scattered over {} bytes", hi - lo));
    }
    let data = fetch(agent, url, lo, hi)?;
    let rel: Vec<u32> = offs.iter().map(|&o| (o as u64 - lo) as u32).collect();
    let mut tags: BTreeMap<u16, Entry> = lv.iter().filter(|(t, _)| **t != 254).map(|(t, e)| (*t, e.clone())).collect();
    // The geo tags live on the full image only: the same tie point, the pixel scaled by the level's shrink.
    let (w0, h0) = (full[&256].u32s()[0] as f64, full[&257].u32s()[0] as f64);
    let (w, h) = (lv[&256].u32s()[0] as f64, lv[&257].u32s()[0] as f64);
    let scale = full.get(&33550).ok_or("no ModelPixelScale")?.f64s();
    tags.insert(33550, Entry::doubles(&[scale[0] * w0 / w, scale[1] * h0 / h, scale.get(2).copied().unwrap_or(0.0)]));
    for t in [33922u16, 34735, 34736, 34737, 42112] {
        if let Some(e) = full.get(&t) {
            tags.entry(t).or_insert_with(|| e.clone());
        }
    }
    // Four bands filed as greyscale (the WorldCover composite: red, green, blue, near-infrared) are refiled as RGB with one extra sample, which the decoder reads; the samples are untouched.
    if lv[&262].u32s()[0] == 1 && lv[&277].u32s()[0] == 4 {
        tags.insert(262, Entry { typ: 3, count: 1, bytes: 2u16.to_le_bytes().to_vec() });
        tags.insert(338, Entry { typ: 3, count: 1, bytes: 0u16.to_le_bytes().to_vec() });
    }
    Ok(write_tiff(tags, &counts, &data, &rel))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base = arg(&args, "--base").expect("--base");
    let keys = std::fs::read_to_string(arg(&args, "--keys").expect("--keys")).expect("keys file");
    let level: usize = arg(&args, "--level").map(|v| v.parse().unwrap()).unwrap_or(2);
    let out = arg(&args, "--out").expect("--out");
    let jobs: usize = arg(&args, "--jobs").map(|v| v.parse().unwrap()).unwrap_or(8);
    std::fs::create_dir_all(&out).expect("out dir");
    let keys: Vec<&str> = keys.lines().map(str::trim).filter(|k| k.ends_with(".tif")).collect();
    let agent = ureq::AgentBuilder::new().timeout_connect(std::time::Duration::from_secs(20)).timeout_read(std::time::Duration::from_secs(120)).build();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(jobs).build().unwrap();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let bytes = std::sync::atomic::AtomicU64::new(0);
    let failed = std::sync::Mutex::new(Vec::new());
    let t = std::time::Instant::now();
    let total = keys.len();
    pool.install(|| {
        keys.par_iter().for_each(|key| {
            let name = Path::new(key).file_name().unwrap().to_string_lossy().to_string();
            let dst = Path::new(&out).join(&name);
            if !dst.exists() {
                match extract(&agent, &format!("{base}{key}"), level) {
                    Ok(tif) => {
                        let part = dst.with_extension("tif.part");
                        std::fs::write(&part, &tif).expect("write");
                        std::fs::rename(&part, &dst).expect("rename");
                        bytes.fetch_add(tif.len() as u64, std::sync::atomic::Ordering::Relaxed);
                    }
                    Err(e) => {
                        eprintln!("FAILED {e}");
                        failed.lock().unwrap().push(key.to_string());
                    }
                }
            }
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n % 100 == 0 || n == total {
                let gb = bytes.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
                eprintln!("{n}/{total} files, {gb:.1} GB fetched, {:.0} MB/min", gb * 1000.0 / (t.elapsed().as_secs_f64() / 60.0));
            }
        });
    });
    let failed = failed.into_inner().unwrap();
    eprintln!("done: {} failed", failed.len());
    if !failed.is_empty() {
        std::fs::write(Path::new(&out).join(".failed"), failed.join("\n")).ok();
        std::process::exit(1);
    }
}
