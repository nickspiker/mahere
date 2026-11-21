//! An area's source lists for fetch-area.sh and bake-pieces.sh, written into data/areas/<area>/: dem-urls.txt (USGS 1 m lidar DEM tiles, and with --topobathy NOAA's CUDEM 1/9 arc-second topo-bathymetric tiles for the water), naip-urls.txt (the newest NAIP year that covers the box), and pieces.txt (each depth-9 piece of the shape with the files that touch it, lidar newest project first and the topo-bathy last so it fills only where lidar has no data).
//! Usage, from the repository root: `cargo run --release -p mahere-tiles --example area_sources -- <area> [--naip-year YYYY] [--topobathy <CUDEM directory URL>]`
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(area) = args.first() else {
        eprintln!("usage: area_sources <area> [--naip-year YYYY] [--topobathy <url>]");
        std::process::exit(2);
    };
    let arg = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let naip_year: Option<u32> = arg("--naip-year").map(|y| y.parse().expect("--naip-year: a year"));
    let topobathy = arg("--topobathy").map(|u| format!("{}/", u.trim_end_matches('/')));
    let root = format!("data/areas/{area}");
    // The area's document: VSF with its planes coded inside, or the zstd form areas had before 7.1.
    let doc = [format!("{root}/{area}.vsf"), format!("{root}/{area}.vsf.zst")].into_iter().find(|p| std::path::Path::new(p).exists()).unwrap_or_else(|| format!("{root}/{area}.vsf"));

    // The pieces and their boxes, from the area document.
    let a = mahere_tiles::area::Area::decode(&std::fs::read(&doc).unwrap_or_else(|e| panic!("{doc}: {e}"))).expect("an area document");
    let pieces = a.pieces(9);
    let lat0 = pieces.iter().map(|p| p.1.0).fold(f64::INFINITY, f64::min);
    let lon0 = pieces.iter().map(|p| p.1.1).fold(f64::INFINITY, f64::min);
    let lat1 = pieces.iter().map(|p| p.1.2).fold(f64::NEG_INFINITY, f64::max);
    let lon1 = pieces.iter().map(|p| p.1.3).fold(f64::NEG_INFINITY, f64::max);
    println!("{} pieces, box {lat0:.4},{lon0:.4} .. {lat1:.4},{lon1:.4}", pieces.len());
    let agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(120)).build();

    // Lidar: every 1 m DEM tile touching the box, its project's year deciding which wins where two cover a place.
    let mut dems: Vec<Source> = Vec::new();
    let mut offset = 0usize;
    loop {
        let q = format!("https://tnmaccess.nationalmap.gov/api/v1/products?datasets=Digital%20Elevation%20Model%20(DEM)%201%20meter&bbox={lon0:?},{lat0:?},{lon1:?},{lat1:?}&max=500&offset={offset}&outputFormat=JSON");
        let page = Json::parse(&body(agent.get(&q).call(), &q)).expect("TNM: JSON");
        let items = page.get("items").arr();
        for it in items {
            let url = it.get("downloadURL").str().to_string();
            let bb = it.get("boundingBox");
            let file = basename(&url).to_string();
            let year = years(&file).max().unwrap_or(0);
            dems.push(Source { box_: (bb.get("minX").num(), bb.get("minY").num(), bb.get("maxX").num(), bb.get("maxY").num()), url, file, year });
        }
        offset += items.len();
        if items.is_empty() || offset as f64 >= page.get("total").num() {
            break;
        }
    }
    // One file per name, the first listed: the same tile can be listed under two URLs.
    let mut seen: Vec<Source> = Vec::new();
    for d in dems {
        if !seen.iter().any(|s| s.file == d.file) {
            seen.push(d);
        }
    }
    let mut dems = seen;
    dems.sort_by(|a, b| b.year.cmp(&a.year).then_with(|| a.file.cmp(&b.file)));
    let mut projects: Vec<&str> = dems.iter().map(|d| d.url.rsplit("/Projects/").next().unwrap().split('/').next().unwrap()).collect();
    projects.sort();
    projects.dedup();
    println!("{} lidar tiles from {} projects", dems.len(), projects.len());

    // The topo-bathymetry: CUDEM tiles are a quarter degree, named by their north-west corner.
    let mut bathy: Vec<Source> = Vec::new();
    if let Some(dir) = &topobathy {
        let listing = body(agent.get(dir).call(), dir);
        let mut tiles = cudem_tiles(&listing);
        tiles.sort();
        tiles.dedup();
        for (file, nd, nf, wd, wf) in tiles {
            let north = nd as f64 + nf as f64 / 100.;
            let west = -(wd as f64 + wf as f64 / 100.);
            let b = (west, north - 0.25, west + 0.25, north);
            if b.0 <= lon1 && b.2 >= lon0 && b.1 <= lat1 && b.3 >= lat0 {
                bathy.push(Source { url: format!("{dir}{file}"), file, box_: b, year: 0 });
            }
        }
        // The newest version of a tile only: the greatest name among those that differ only in their `_<year>v<n>.tif` ending.
        let mut newest: Vec<Source> = Vec::new();
        for b in bathy {
            let key = version_key(&b.file).to_string();
            match newest.iter_mut().find(|n| version_key(&n.file) == key) {
                Some(n) if b.file > n.file => *n = b,
                Some(_) => {}
                None => newest.push(b),
            }
        }
        newest.sort_by(|a, b| a.file.cmp(&b.file));
        bathy = newest;
        println!("{} topo-bathymetry tiles", bathy.len());
    }

    // Imagery: the newest NAIP year with quads over the box (or the one asked for).
    let naip_items = |year: u32| -> Vec<Source> {
        let search = "https://planetarycomputer.microsoft.com/api/stac/v1/search";
        let req = format!("{{\"collections\": [\"naip\"], \"bbox\": [{lon0:?}, {lat0:?}, {lon1:?}, {lat1:?}], \"datetime\": \"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z\", \"limit\": 1000}}");
        let res = Json::parse(&body(agent.post(search).set("Content-Type", "application/json").send_string(&req), search)).expect("STAC: JSON");
        res.get("features")
            .arr()
            .iter()
            .map(|f| {
                let href = f.get("assets").get("image").get("href").str().to_string();
                let bb: Vec<f64> = f.get("bbox").arr().iter().map(Json::num).collect();
                Source { file: basename(&href).to_string(), url: href, box_: (bb[0], bb[1], bb[2], bb[3]), year }
            })
            .collect()
    };
    let years: Vec<u32> = match naip_year {
        Some(y) => vec![y],
        None => (2016..=2024).rev().collect(),
    };
    let mut naips = Vec::new();
    for y in years {
        naips = naip_items(y);
        if !naips.is_empty() {
            println!("{} NAIP {y} quads", naips.len());
            break;
        }
    }

    let dem_lines: String = dems.iter().chain(&bathy).map(|d| format!("{}\n", d.url)).collect();
    std::fs::write(format!("{root}/dem-urls.txt"), dem_lines).expect("write dem-urls.txt");
    let naip_lines: String = naips.iter().map(|n| format!("{}\n", n.url)).collect();
    std::fs::write(format!("{root}/naip-urls.txt"), naip_lines).expect("write naip-urls.txt");
    let mut lines = String::new();
    let mut empty = 0;
    for (key, b, _) in &pieces {
        let name = key.name();
        let pd: Vec<&str> = dems.iter().chain(&bathy).filter(|d| overlaps(d.box_, *b)).map(|d| d.file.as_str()).collect();
        let pn: Vec<&str> = naips.iter().filter(|n| overlaps(n.box_, *b)).map(|n| n.file.as_str()).collect();
        if pd.is_empty() || pn.is_empty() {
            empty += 1;
            println!("  {name}: {} dem, {} naip", pd.len(), pn.len());
            continue;
        }
        lines += &format!("{name} {} {}\n", pd.join(","), pn.join(","));
    }
    std::fs::write(format!("{root}/pieces.txt"), lines).expect("write pieces.txt");
    println!("wrote {root}/dem-urls.txt, naip-urls.txt, pieces.txt ({} pieces with sources)", pieces.len() - empty);
}

/// A source file: where it is fetched from, its name, its box (west, south, east, north), and for lidar its project's year.
struct Source {
    url: String,
    file: String,
    box_: (f64, f64, f64, f64),
    year: u32,
}

/// A source's box (west, south, east, north) against a piece's (lat0, lon0, lat1, lon1), a little loose.
fn overlaps(b: (f64, f64, f64, f64), p: (f64, f64, f64, f64)) -> bool {
    let pad = 0.002;
    b.0 <= p.3 + pad && b.2 >= p.1 - pad && b.1 <= p.2 + pad && b.3 >= p.0 - pad
}

fn basename(url: &str) -> &str {
    url.rsplit('/').next().unwrap()
}

/// A response's body, whole; a failed request ends the run.
fn body(r: Result<ureq::Response, ureq::Error>, url: &str) -> String {
    let mut s = String::new();
    r.unwrap_or_else(|e| panic!("{url}: {e}")).into_reader().read_to_string(&mut s).unwrap_or_else(|e| panic!("{url}: {e}"));
    s
}

/// The years in a file name: each run of 19dd or 20dd, scanned left to right without overlap.
fn years(name: &str) -> impl Iterator<Item = u32> + '_ {
    let b = name.as_bytes();
    let mut i = 0;
    std::iter::from_fn(move || {
        while i + 4 <= b.len() {
            let at = i;
            if (&b[at..at + 2] == b"19" || &b[at..at + 2] == b"20") && b[at + 2].is_ascii_digit() && b[at + 3].is_ascii_digit() {
                i += 4;
                return Some((b[at] - b'0') as u32 * 1000 + (b[at + 1] - b'0') as u32 * 100 + (b[at + 2] - b'0') as u32 * 10 + (b[at + 3] - b'0') as u32);
            }
            i += 1;
        }
        None
    })
}

/// The CUDEM tiles a directory listing names: `ncei19_n<deg>x<hundredths>_w<deg>x<hundredths>_<word>.tif`, with the name and its four numbers.
fn cudem_tiles(listing: &str) -> Vec<(String, u32, u32, u32, u32)> {
    let b = listing.as_bytes();
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let digits = |at: usize| -> Option<(u32, usize)> {
        let end = at + b[at..].iter().take_while(|c| c.is_ascii_digit()).count();
        (end > at).then(|| (listing[at..end].parse().unwrap(), end))
    };
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(off) = listing[i..].find("ncei19_n") {
        let start = i + off;
        i = start + 1;
        let tile = (|| {
            let (nd, at) = digits(start + 8)?;
            (b.get(at) == Some(&b'x')).then_some(())?;
            let (nf, at) = digits(at + 1)?;
            (listing[at..].starts_with("_w")).then_some(())?;
            let (wd, at) = digits(at + 2)?;
            (b.get(at) == Some(&b'x')).then_some(())?;
            let (wf, at) = digits(at + 1)?;
            (b.get(at) == Some(&b'_')).then_some(())?;
            // A word then `.tif`: a dot is no word character, so the shortest word followed by `.tif` is the whole run of word characters.
            let end = at + 1 + b[at + 1..].iter().take_while(|&&c| word(c)).count();
            (end > at + 1 && listing[end..].starts_with(".tif")).then_some(())?;
            Some((listing[start..end + 4].to_string(), nd, nf, wd, wf, end + 4))
        })();
        if let Some((file, nd, nf, wd, wf, end)) = tile {
            out.push((file, nd, nf, wd, wf));
            i = end;
        }
    }
    out
}

/// A CUDEM file's name without its `_<year>v<n>.tif` version ending, the same for every version of a tile.
fn version_key(file: &str) -> &str {
    // The version's digits hold no `v`, so its `v` is the stem's last.
    let Some(stem) = file.strip_suffix(".tif") else { return file };
    let Some(v) = stem.rfind('v') else { return file };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
    if v >= 5 && digits(&stem[v + 1..]) && digits(&stem[v - 4..v]) && stem.as_bytes()[v - 5] == b'_' { &file[..v - 5] } else { file }
}

/// A JSON value, read from a web API's answer. Objects keep their members in order and are searched in a line.
enum Json {
    /// true, false or null: nothing here reads them.
    Lit,
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn parse(s: &str) -> Result<Json, String> {
        let mut p = Parser { b: s.as_bytes(), i: 0 };
        let v = p.value()?;
        p.ws();
        if p.i != p.b.len() {
            return Err(format!("trailing bytes at {}", p.i));
        }
        Ok(v)
    }

    /// A member of an object; a missing member is a malformed answer, and ends the run.
    fn get(&self, key: &str) -> &Json {
        match self {
            Json::Obj(m) => &m.iter().find(|(k, _)| k == key).unwrap_or_else(|| panic!("no member {key}")).1,
            _ => panic!("{key}: not an object"),
        }
    }

    fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(a) => a,
            _ => panic!("not an array"),
        }
    }

    fn num(&self) -> f64 {
        match self {
            Json::Num(n) => *n,
            _ => panic!("not a number"),
        }
    }

    fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            _ => panic!("not a string"),
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at {}", c as char, self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.b.get(self.i) {
            Some(b'{') => {
                self.i += 1;
                let mut m = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.eat(b':')?;
                    m.push((k, self.value()?));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(m));
                        }
                        _ => return Err(format!("object at {}", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return Err(format!("array at {}", self.i)),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.b[self.i..].starts_with(b"true") => {
                self.i += 4;
                Ok(Json::Lit)
            }
            Some(b'f') if self.b[self.i..].starts_with(b"false") => {
                self.i += 5;
                Ok(Json::Lit)
            }
            Some(b'n') if self.b[self.i..].starts_with(b"null") => {
                self.i += 4;
                Ok(Json::Lit)
            }
            Some(_) => {
                let start = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.b[start..self.i]).unwrap();
                t.parse().map(Json::Num).map_err(|_| format!("number {t:?} at {start}"))
            }
            None => Err("unexpected end".into()),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.i) != Some(&b'"') {
            return Err(format!("expected a string at {}", self.i));
        }
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = *self.b.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|e| e.to_string()),
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            // A high surrogate takes the low one after it.
                            if (0xD800..0xDC00).contains(&cp) && self.b[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let lo = self.hex4()?;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            }
                            let ch = char::from_u32(cp).ok_or("bad \\u escape")?;
                            out.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes());
                        }
                        _ => return Err(format!("bad escape at {}", self.i)),
                    }
                }
                _ => out.push(c),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.b.get(self.i..self.i + 4).ok_or("short \\u escape")?;
        self.i += 4;
        u32::from_str_radix(std::str::from_utf8(h).map_err(|e| e.to_string())?, 16).map_err(|e| e.to_string())
    }
}
