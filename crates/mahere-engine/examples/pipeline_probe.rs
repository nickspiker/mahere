//! pipeline_probe <lat> <lon> <depth> [n]: n base cells around a point asked of the bucket's host over plain HTTP, one after another on one kept connection and then all at once pipelined on one (HTTP/1.1, RFC 9112 9.3.2), each response's arrival timed: whether the edge answers pipelined requests as they come or one after another (2026-11-27, cold routes on ferros).
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Instant;

use mahere_tiles::CellKey;

const HOST: &str = "brobdingnagian.holdmyoscilloscope.com";

/// One response off the stream: its status and body length.
fn response(r: &mut BufReader<TcpStream>) -> (u16, usize) {
    let mut line = String::new();
    r.read_line(&mut line).expect("status line");
    let status: u16 = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).expect("a status");
    let mut len = 0usize;
    loop {
        line.clear();
        r.read_line(&mut line).expect("a header");
        let l = line.trim_end();
        if l.is_empty() {
            break;
        }
        if let Some((k, v)) = l.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().expect("a length");
            }
        }
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).expect("the body");
    (status, len)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (lat, lon, depth): (f64, f64, u8) = (a[1].parse().expect("lat"), a[2].parse().expect("lon"), a[3].parse().expect("depth"));
    let n: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let key = CellKey::containing(mahere_coord::Coord::from_lat_lon(lat, lon), depth);
    let (cu, cv) = key.grid();
    let side = (n as f64).sqrt().ceil() as u64;
    let keys: Vec<CellKey> = (0..n as u64).map(|i| CellKey::from_grid(key.diamond(), depth, cu + i % side, cv + i / side)).collect();
    let get = |k: CellKey, salt: &str| format!("GET /mahere/cells/{}?e={}{salt} HTTP/1.1\r\nHost: {HOST}\r\n\r\n", mahere_tiles::publish::Layer::Base.path(k), mahere_engine::residency::CELL_EPOCH);
    for (what, pipelined, salt) in [("one at a time", false, ""), ("pipelined", true, "&p=1"), ("one at a time again (edge warm)", false, ""), ("pipelined again (edge warm)", true, "&p=1")] {
        let s = TcpStream::connect((HOST, 80)).expect("connect");
        s.set_nodelay(true).ok();
        let mut w = s.try_clone().expect("clone");
        let mut r = BufReader::new(s);
        // A first request so the connection is warm, as plain_http's kept ones are.
        w.write_all(format!("GET /mahere/cells/{}?e={} HTTP/1.1\r\nHost: {HOST}\r\n\r\n", mahere_tiles::publish::Layer::Base.path(key), mahere_engine::residency::CELL_EPOCH).as_bytes()).unwrap();
        response(&mut r);
        let t = Instant::now();
        let mut at = Vec::new();
        if pipelined {
            let all: String = keys.iter().map(|&k| get(k, salt)).collect();
            w.write_all(all.as_bytes()).unwrap();
            for _ in &keys {
                let (st, len) = response(&mut r);
                at.push((st, len, t.elapsed().as_millis()));
            }
        } else {
            for &k in &keys {
                w.write_all(get(k, salt).as_bytes()).unwrap();
                let (st, len) = response(&mut r);
                at.push((st, len, t.elapsed().as_millis()));
            }
        }
        let bytes: usize = at.iter().map(|x| x.1).sum();
        println!("{what}: {} responses, {} KB, all in {} ms; arrivals {:?}", at.len(), bytes >> 10, t.elapsed().as_millis(), at.iter().map(|x| x.2).collect::<Vec<_>>());
        let codes: Vec<u16> = at.iter().map(|x| x.0).collect();
        println!("  statuses {codes:?}");
    }
}
