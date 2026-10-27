//! Address documents (Nick 2026-10-27: geocoding, "TIGER and all"): one per ZIP code, `addr/<zip>.vsf` beside the cells, holding the ZIP's streets, their TIGER address ranges (each block's numbers by side and its line) and the addresses OSM places exactly; and one index of every ZIP's box, `addr/zips.vsf`, for an address typed without its ZIP. VSF then zstd, as an area is.
//!
//! A lookup asks a ZIP's document for a house number on a street: an exact address where one is placed, else the place along its block the number falls at, on its side of the street.

use vsf::types::Tensor;
use vsf::{VsfBuilder, VsfType};

pub use mahere_osm::addr::{base, house_number, normal};

/// One side of a block in a document: its first and last number, and which numbers it holds (1 odd, 2 even, 3 both).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Side {
    pub from: u32,
    pub to: u32,
    pub parity: u8,
}

/// A block of a street: its sides' numbers and its line, the numbers running from its first point to its last.
#[derive(Clone, Debug, PartialEq)]
pub struct Range {
    pub street: u32,
    pub left: Option<Side>,
    pub right: Option<Side>,
    pub pts: Vec<(f64, f64)>,
}

/// An address placed exactly.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub street: u32,
    pub number: u32,
    pub lat: f64,
    pub lon: f64,
}

/// A ZIP code's addresses.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZipDoc {
    pub zip: u32,
    /// The streets as they are named, each once.
    pub streets: Vec<String>,
    pub ranges: Vec<Range>,
    pub points: Vec<Point>,
}

/// Every ZIP's box: latitude and longitude, south-west then north-east.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZipIndex {
    pub zips: Vec<(u32, [f64; 4])>,
}

/// A place an address lookup found.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    /// The address as found: number, street as the map names it, ZIP.
    pub label: String,
    /// How it was found: an exact address, or a place along its block.
    pub detail: String,
    pub lat: f64,
    pub lon: f64,
    /// Lower is better: 0 the street's name exactly and the number placed, 1 the name and the number along its block, 2 and 3 the same on a street whose base name matched (another type: "Talon Trail" for "Talon Ln"), 4 the nearest number on the street.
    pub score: u8,
}

/// The document's path beside the cells.
pub fn zip_path(zip: u32) -> String {
    format!("addr/{zip:05}.vsf")
}

pub const INDEX_PATH: &str = "addr/zips.vsf";

fn micro(v: f64, off: f64) -> u32 {
    ((v + off) * 1e6).round() as u32
}

fn unmicro(v: u32, off: f64) -> f64 {
    v as f64 / 1e6 - off
}

fn build(sections: Vec<(&str, Vec<(String, VsfType)>)>) -> Vec<u8> {
    let mut b = VsfBuilder::new();
    for (name, fields) in sections {
        b = b.add_section(name, fields);
    }
    let raw = b.build().unwrap_or_default();
    zstd::encode_all(raw.as_slice(), 19).unwrap_or(raw)
}

/// A document's fields by section and name, each field's first value.
fn fields(bytes: &[u8]) -> Option<Vec<(String, String, VsfType)>> {
    let raw = zstd::decode_all(bytes).ok()?;
    let (header, end) = vsf::VsfHeader::decode(&raw).ok()?;
    let sections = header.sections(&raw, end).ok()?;
    let mut out = Vec::new();
    for s in sections {
        for f in s.fields {
            if let Some(v) = f.values.into_iter().next() {
                out.push((s.name.clone(), f.name.clone(), v));
            }
        }
    }
    Some(out)
}

fn u32s(v: &VsfType) -> Vec<u32> {
    match v {
        VsfType::t_u5(t) => t.data.clone(),
        VsfType::t_u4(t) => t.data.iter().map(|&x| x as u32).collect(),
        VsfType::t_u3(t) => t.data.iter().map(|&x| x as u32).collect(),
        _ => Vec::new(),
    }
}

impl ZipDoc {
    pub fn encode(&self) -> Vec<u8> {
        let side = |s: Option<Side>, f: fn(Side) -> u32| -> u32 { s.map_or(0, |s| f(s) + 1) };
        let r = &self.ranges;
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        for g in r {
            for &(a, o) in &g.pts {
                lat.push(micro(a, 90.0));
                lon.push(micro(o, 180.0));
            }
        }
        let t5 = |v: Vec<u32>| VsfType::t_u5(Tensor::new(vec![v.len()], v));
        build(vec![
            ("zip", vec![("code".to_string(), t5(vec![self.zip]))]),
            ("street", vec![("name".to_string(), VsfType::x(self.streets.join("\n")))]),
            (
                "range",
                vec![
                    ("street".to_string(), t5(r.iter().map(|g| g.street).collect())),
                    ("lfrom".to_string(), t5(r.iter().map(|g| side(g.left, |s| s.from)).collect())),
                    ("lto".to_string(), t5(r.iter().map(|g| side(g.left, |s| s.to)).collect())),
                    ("rfrom".to_string(), t5(r.iter().map(|g| side(g.right, |s| s.from)).collect())),
                    ("rto".to_string(), t5(r.iter().map(|g| side(g.right, |s| s.to)).collect())),
                    ("parity".to_string(), t5(r.iter().map(|g| g.left.map_or(0, |s| s.parity as u32) | g.right.map_or(0, |s| s.parity as u32) << 2).collect())),
                    ("nv".to_string(), t5(r.iter().map(|g| g.pts.len() as u32).collect())),
                    ("lat".to_string(), t5(lat)),
                    ("lon".to_string(), t5(lon)),
                ],
            ),
            (
                "point",
                vec![
                    ("street".to_string(), t5(self.points.iter().map(|p| p.street).collect())),
                    ("number".to_string(), t5(self.points.iter().map(|p| p.number).collect())),
                    ("lat".to_string(), t5(self.points.iter().map(|p| micro(p.lat, 90.0)).collect())),
                    ("lon".to_string(), t5(self.points.iter().map(|p| micro(p.lon, 180.0)).collect())),
                ],
            ),
        ])
    }

    pub fn decode(bytes: &[u8]) -> Option<ZipDoc> {
        let mut doc = ZipDoc::default();
        let mut r: [Vec<u32>; 9] = Default::default();
        let mut p: [Vec<u32>; 4] = Default::default();
        for (s, f, v) in fields(bytes)? {
            match (s.as_str(), f.as_str()) {
                ("zip", "code") => doc.zip = u32s(&v).first().copied().unwrap_or(0),
                ("street", "name") => {
                    if let VsfType::x(t) | VsfType::a(t) = v {
                        doc.streets = t.split('\n').map(str::to_string).collect();
                    }
                }
                ("range", name) => {
                    let i = ["street", "lfrom", "lto", "rfrom", "rto", "parity", "nv", "lat", "lon"].iter().position(|n| *n == name);
                    if let Some(i) = i {
                        r[i] = u32s(&v);
                    }
                }
                ("point", name) => {
                    let i = ["street", "number", "lat", "lon"].iter().position(|n| *n == name);
                    if let Some(i) = i {
                        p[i] = u32s(&v);
                    }
                }
                _ => {}
            }
        }
        let side = |from: u32, to: u32, parity: u32| -> Option<Side> { (from != 0 && to != 0).then_some(Side { from: from - 1, to: to - 1, parity: parity as u8 }) };
        let mut v = 0usize;
        for k in 0..r[0].len() {
            let nv = *r[6].get(k)? as usize;
            let pts = (v..v + nv).map(|j| Some((unmicro(*r[7].get(j)?, 90.0), unmicro(*r[8].get(j)?, 180.0)))).collect::<Option<Vec<_>>>()?;
            v += nv;
            let par = *r[5].get(k)?;
            doc.ranges.push(Range { street: r[0][k], left: side(*r[1].get(k)?, *r[2].get(k)?, par & 3), right: side(*r[3].get(k)?, *r[4].get(k)?, par >> 2), pts });
        }
        for k in 0..p[0].len() {
            doc.points.push(Point { street: p[0][k], number: *p[1].get(k)?, lat: unmicro(*p[2].get(k)?, 90.0), lon: unmicro(*p[3].get(k)?, 180.0) });
        }
        Some(doc)
    }

    /// The places a house number on a street could be, best first. `street` is what was typed after the number, city and state words and all: its longest leading run of words that names a street of this ZIP is the street.
    pub fn find(&self, number: u32, street: &str) -> Vec<Found> {
        let words: Vec<&str> = street.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()).collect();
        let normals: Vec<String> = self.streets.iter().map(|s| normal(s)).collect();
        let bases: Vec<String> = self.streets.iter().map(|s| base(s)).collect();
        // The longest leading run of words that matches a street's whole name, else one whose base matches.
        let mut matched: Vec<(u32, u8)> = Vec::new();
        for k in (1..=words.len()).rev() {
            let typed = words[..k].join(" ");
            let (n, b) = (normal(&typed), base(&typed));
            let exact: Vec<u32> = (0..self.streets.len()).filter(|&i| normals[i] == n).map(|i| i as u32).collect();
            if !exact.is_empty() {
                matched = exact.into_iter().map(|i| (i, 0)).collect();
                break;
            }
            let loose: Vec<u32> = (0..self.streets.len()).filter(|&i| !b.is_empty() && bases[i] == b).map(|i| i as u32).collect();
            if !loose.is_empty() && matched.is_empty() {
                matched = loose.into_iter().map(|i| (i, 2)).collect();
            }
        }
        let mut out = Vec::new();
        for &(s, loose) in &matched {
            let name = &self.streets[s as usize];
            let label = format!("{number} {name}, {:05}", self.zip);
            for p in self.points.iter().filter(|p| p.street == s && p.number == number) {
                out.push(Found { label: label.clone(), detail: "Address".into(), lat: p.lat, lon: p.lon, score: loose });
            }
            if out.iter().any(|f| f.score == loose) {
                continue;
            }
            for g in self.ranges.iter().filter(|g| g.street == s) {
                for (side, left) in [(g.left, true), (g.right, false)] {
                    let Some(sd) = side else { continue };
                    let (lo, hi) = (sd.from.min(sd.to), sd.from.max(sd.to));
                    let parity_ok = match sd.parity {
                        1 => number % 2 == 1,
                        2 => number % 2 == 0,
                        _ => true,
                    };
                    if number < lo || number > hi || !parity_ok {
                        continue;
                    }
                    let t = if sd.to == sd.from { 0.5 } else { (number as f64 - sd.from as f64) / (sd.to as f64 - sd.from as f64) };
                    let (lat, lon) = along(&g.pts, t, if left { 10.0 } else { -10.0 });
                    out.push(Found { label: label.clone(), detail: format!("Address \u{b7} along the block from {} to {}", sd.from, sd.to), lat, lon, score: loose + 1 });
                }
            }
        }
        if out.is_empty() {
            // The street is there and the number is not: its nearest number, said as such.
            let mut best: Option<(u32, f64, f64, u32)> = None;
            for &(s, _) in &matched {
                for g in self.ranges.iter().filter(|g| g.street == s) {
                    for (side, left) in [(g.left, true), (g.right, false)] {
                        let Some(sd) = side else { continue };
                        for (n, t) in [(sd.from, 0.0), (sd.to, 1.0)] {
                            let d = n.abs_diff(number);
                            if best.is_none_or(|b| d < b.0) {
                                let (lat, lon) = along(&g.pts, t, if left { 10.0 } else { -10.0 });
                                best = Some((d, lat, lon, n));
                            }
                        }
                    }
                }
                if let Some((d, lat, lon, n)) = best.filter(|b| b.0 > 0) {
                    let name = &self.streets[s as usize];
                    out.push(Found { label: format!("{n} {name}, {:05}", self.zip), detail: format!("No {number} on record \u{b7} {d} numbers away"), lat, lon, score: 4 });
                    break;
                }
            }
        }
        out.sort_by_key(|f| f.score);
        out.dedup_by(|a, b| a.label == b.label && (a.lat - b.lat).abs() < 1e-5 && (a.lon - b.lon).abs() < 1e-5);
        out
    }
}

/// The place a fraction `t` of the way along a line, `side` metres off it (positive to the left of its direction).
fn along(pts: &[(f64, f64)], t: f64, side: f64) -> (f64, f64) {
    let k = pts[0].0.to_radians().cos();
    let m = |a: (f64, f64), b: (f64, f64)| (((b.1 - a.1) * k * 111_320.0), ((b.0 - a.0) * 111_320.0));
    let lens: Vec<f64> = pts.windows(2).map(|w| {
        let (e, n) = m(w[0], w[1]);
        (e * e + n * n).sqrt()
    }).collect();
    let total: f64 = lens.iter().sum();
    let mut want = t * total;
    for (i, &l) in lens.iter().enumerate() {
        if want <= l || i + 1 == lens.len() {
            let f = if l > 0.0 { want / l } else { 0.0 };
            let (a, b) = (pts[i], pts[i + 1]);
            let (e, n) = m(a, b);
            let (lat, lon) = (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f);
            // Left of the heading is the heading turned a quarter anticlockwise.
            let (le, ln) = if l > 0.0 { (-n / l, e / l) } else { (0.0, 0.0) };
            return (lat + ln * side / 111_320.0, lon + le * side / (111_320.0 * k));
        }
        want -= l;
    }
    pts[0]
}

impl ZipIndex {
    pub fn encode(&self) -> Vec<u8> {
        let codes: Vec<u32> = self.zips.iter().map(|z| z.0).collect();
        let mut boxes = Vec::with_capacity(self.zips.len() * 4);
        for (_, b) in &self.zips {
            boxes.extend([micro(b[0], 90.0), micro(b[1], 180.0), micro(b[2], 90.0), micro(b[3], 180.0)]);
        }
        build(vec![(
            "zips",
            vec![("code".to_string(), VsfType::t_u5(Tensor::new(vec![codes.len()], codes))), ("box".to_string(), VsfType::t_u5(Tensor::new(vec![boxes.len()], boxes)))],
        )])
    }

    pub fn decode(bytes: &[u8]) -> Option<ZipIndex> {
        let (mut codes, mut boxes) = (Vec::new(), Vec::new());
        for (s, f, v) in fields(bytes)? {
            match (s.as_str(), f.as_str()) {
                ("zips", "code") => codes = u32s(&v),
                ("zips", "box") => boxes = u32s(&v),
                _ => {}
            }
        }
        let zips = codes.iter().enumerate().map(|(i, &c)| Some((c, [unmicro(*boxes.get(4 * i)?, 90.0), unmicro(*boxes.get(4 * i + 1)?, 180.0), unmicro(*boxes.get(4 * i + 2)?, 90.0), unmicro(*boxes.get(4 * i + 3)?, 180.0)]))).collect::<Option<Vec<_>>>()?;
        Some(ZipIndex { zips })
    }

    /// The ZIPs whose box lies within `reach` metres of a place, nearest first.
    pub fn near(&self, lat: f64, lon: f64, reach: f64) -> Vec<u32> {
        let k = lat.to_radians().cos();
        let mut d: Vec<(f64, u32)> = self
            .zips
            .iter()
            .map(|(z, b)| {
                let dy = if lat < b[0] { b[0] - lat } else if lat > b[2] { lat - b[2] } else { 0.0 };
                let dx = if lon < b[1] { b[1] - lon } else if lon > b[3] { lon - b[3] } else { 0.0 };
                (((dy * 111_320.0).powi(2) + (dx * 111_320.0 * k).powi(2)).sqrt(), *z)
            })
            .filter(|(d, _)| *d <= reach)
            .collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0));
        d.into_iter().map(|(_, z)| z).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn talon() -> ZipDoc {
        // Talon Ln in 59912 as TIGER has it: a block running east, odd numbers 599 to 665 on its right (south) side.
        ZipDoc {
            zip: 59912,
            streets: vec!["Talon Ln".into(), "Main St".into()],
            ranges: vec![Range { street: 0, left: Some(Side { from: 500, to: 656, parity: 2 }), right: Some(Side { from: 599, to: 665, parity: 1 }), pts: vec![(48.40, -114.20), (48.40, -114.19)] }],
            points: vec![Point { street: 1, number: 12, lat: 48.41, lon: -114.18 }],
        }
    }

    #[test]
    fn a_number_lands_along_its_block_on_its_side() {
        let doc = ZipDoc::decode(&talon().encode()).unwrap();
        // Positions round to a millionth of a degree, about ten centimetres.
        assert_eq!((doc.zip, &doc.streets, doc.ranges[0].right, doc.points[0].number), (59912, &talon().streets, talon().ranges[0].right, 12));
        assert!((doc.ranges[0].pts[1].1 - (-114.19)).abs() < 1e-6);
        let f = doc.find(647, "Talon Trail");
        assert_eq!(f[0].label, "647 Talon Ln, 59912");
        assert_eq!(f[0].score, 3, "the street's type differs, the number is along its block");
        // 647 is 48 of 66 numbers along, south of the line.
        let t = (647.0 - 599.0) / (665.0 - 599.0);
        assert!((f[0].lon - (-114.20 + 0.01 * t)).abs() < 1e-6, "{f:?}");
        assert!(f[0].lat < 48.40 && f[0].lat > 48.3998, "{f:?}");
        // An even number takes the north side.
        assert!(doc.find(600, "talon ln")[0].lat > 48.40);
        // City and state words after the street do not matter.
        assert_eq!(doc.find(647, "Talon Ln Columbia Falls MT")[0].score, 1);
        // A placed address is exact.
        let m = doc.find(12, "Main Street");
        assert_eq!((m[0].score, m[0].lat), (0, 48.41));
        // A number off the street's ranges gives its nearest, said so.
        let n = doc.find(900, "Talon Ln");
        assert_eq!(n[0].score, 4);
        assert!(n[0].label.starts_with("665 ") || n[0].label.starts_with("656 "), "{n:?}");
        assert!(doc.find(647, "Eagle Way").is_empty());
    }

    #[test]
    fn the_index_finds_the_zips_near_a_place() {
        let i = ZipIndex { zips: vec![(59912, [48.3, -114.3, 48.5, -114.0]), (98366, [47.4, -122.7, 47.6, -122.5])] };
        let i = ZipIndex::decode(&i.encode()).unwrap();
        assert_eq!(i.near(48.4, -114.2, 1000.0), vec![59912]);
        assert_eq!(i.near(47.0, -122.6, 50_000.0), vec![98366]);
        assert!(i.near(0.0, 0.0, 1000.0).is_empty());
    }
}
