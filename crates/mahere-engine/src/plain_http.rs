//! A plain HTTP/1.1 GET over a byte stream the platform dials, for a platform where ureq has no sockets: ferros, whose connections are the ip server's client TCP (mahere-ferros `net`), and which speaks no TLS, so the bucket is reached over plain HTTP (it answers port 80 as it answers 443).
//!
//! Nick 2026-11-11, on ferros: "we have tcp and http so let's get the full meal deal wired up with mahere".
//! The platform registers its [`Dial`] once at start ([`set_dial`]); [`get`] then sends a request on a connection kept from an earlier one to the same server, or a new one, reads the status line and the headers, and the body by its length, in chunks, or up to the close.
//! Everything here is pure over `Read` and `Write`, so the host's tests drive it with bytes in memory.
//!
//! Connections kept (ferros, 2026-11-14, the owner on the 8a: "the map seems to be loading quite slow ... cells come in quite a bit slower from cloudflare than on android").
//! Until then every GET was a connection of its own (`Connection: close`), four at a time: a cell paid a TCP handshake, the server's slow start and a close, and waited for one of the four.
//! Measured on the 8a over WiFi before (round trip 25 to 50 ms): a 2284-byte cell 811 ms in Mahere, its connection open 263 ms, the same cell 78 ms by httptest; 10 fetches a second at most.
//! Now up to [`CONNECTIONS`] at once, each kept after its response for the next request to the same server (HTTP/1.1 persistent connections, RFC 9112 9.3), as ureq keeps them for the Android build.

use std::io::{BufRead, BufReader, Read, Write};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A connection to read the response from and write the request to; closed when dropped.
pub trait Conn: Read + Write + Send {
    /// Let the connection go without a word to the server or the platform: one kept idle past [`KEEP`] (ferros: the ip server resets and frees a connection its app has not asked about for 32 s and gives its handle to the next CONNECT, so a call on a handle kept that long could reach another app's connection).
    fn abandon(&mut self) {}
}

/// How a platform opens a connection to a server.
pub trait Dial: Send + Sync {
    /// A connection to `host` on `port`.
    fn dial(&self, host: &str, port: u16) -> std::io::Result<Box<dyn Conn>>;

    /// What the platform has to say about its dials since it was last asked (its name lookups), for the network's summary line; nothing by default.
    fn said(&self) -> String {
        String::new()
    }
}

static DIAL: OnceLock<Arc<dyn Dial>> = OnceLock::new();

/// The platform's way to the network, once at start (a second call changes nothing).
pub fn set_dial(dial: Arc<dyn Dial>) {
    let _ = DIAL.set(dial);
}

/// The platform's way to the network, when it registered one.
pub fn dial() -> Option<Arc<dyn Dial>> {
    DIAL.get().cloned()
}

/// A response: its status, its `ETag` when it has one, its body, whether the connection may carry another request, and how long each part of the fetch took.
#[derive(Debug)]
pub struct Got {
    pub status: u16,
    pub etag: Option<String>,
    pub body: Vec<u8>,
    /// HTTP/1.1, no `Connection: close`, and a body that ends by its length or its chunks (not by the close).
    pub keep: bool,
    pub phases: Phases,
}

/// Where a fetch's time went, in microseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Phases {
    /// Waiting for one of the [`CONNECTIONS`].
    pub wait: u64,
    /// The dial (the platform's name lookup and its connect call); 0 on a kept connection.
    pub dial: u64,
    /// From the request sent to the response's first byte (on a new connection the TCP handshake too).
    pub first: u64,
    /// The rest of the response.
    pub body: u64,
    /// How long the connection had been kept idle (None: a new one).
    pub kept: Option<u64>,
    /// A kept connection gave no response and the request went again on a new one.
    pub retried: bool,
}

/// Connections at once, busy or kept idle: eight of the ip server's sixteen, the rest left to the device's own (attest's fgtw, the kist installer) and other apps; four until 2026-11-14.
/// Twenty loaders share them (residency `LOADERS`), each waiting for one as for a socket, a doorbell (a condition variable), not a poll.
pub const CONNECTIONS: usize = 1 << 3;

/// How long an idle connection is kept for a next request: 16 s.
/// WHY a bound: the ip server resets and frees a connection its app has not asked about for 32 s, and its handle goes to the next CONNECT of any app; a kept handle older than that could reach another app's connection. Half of it leaves the margin. No timer looks at the kept ones: whoever next asks for a connection lets the old ones go ([`Conn::abandon`]), and the ip server has freed them by then.
pub const KEEP: Duration = Duration::from_secs(1 << 4);

/// The network's measures, summed between two summary lines.
#[derive(Default)]
struct Sums {
    since: Option<Instant>,
    fetches: u64,
    kept: u64,
    fresh: u64,
    retried: u64,
    failed: u64,
    bytes: u64,
    wait_us: u64,
    first_kept_us: u64,
    first_fresh_us: u64,
    dial_us: u64,
    busy_most: usize,
}

/// The connections: how many are open (busy or idle), the idle ones, and the measures.
pub struct Pool {
    state: Mutex<PoolState>,
    wake: Condvar,
}

struct Idle {
    host: String,
    port: u16,
    conn: BufReader<Box<dyn Conn>>,
    since: Instant,
}

struct PoolState {
    open: usize,
    idle: Vec<Idle>,
    sums: Sums,
}

/// What [`Pool::take`] gives: a kept connection (with how long it was idle), or the right to dial a new one.
enum Took {
    Kept(BufReader<Box<dyn Conn>>, Duration),
    Dial,
}

impl Pool {
    pub const fn new() -> Pool {
        Pool { state: Mutex::new(PoolState { open: 0, idle: Vec::new(), sums: Sums { since: None, fetches: 0, kept: 0, fresh: 0, retried: 0, failed: 0, bytes: 0, wait_us: 0, first_kept_us: 0, first_fresh_us: 0, dial_us: 0, busy_most: 0 } }), wake: Condvar::new() }
    }

    /// A connection to `host:port`: a kept one when there is one and `reuse` allows it, else the right to dial (one of the [`CONNECTIONS`] taken), waiting while all are busy.
    /// An idle one kept past [`KEEP`] is let go first; an idle one gives its place when all are open (with `reuse` false, one for the same server too).
    fn take(&self, host: &str, port: u16, reuse: bool) -> Took {
        let mut s = self.state.lock().unwrap();
        loop {
            let now = Instant::now();
            let before = s.idle.len();
            s.idle.retain_mut(|c| {
                let fresh = now.duration_since(c.since) < KEEP;
                if !fresh {
                    c.conn.get_mut().abandon();
                }
                fresh
            });
            s.open -= before - s.idle.len();
            if let Some(i) = s.idle.iter().rposition(|c| reuse && c.host == host && c.port == port) {
                let c = s.idle.swap_remove(i);
                s.note_busy();
                return Took::Kept(c.conn, now.duration_since(c.since));
            }
            if s.open < CONNECTIONS {
                s.open += 1;
                s.note_busy();
                return Took::Dial;
            }
            if !s.idle.is_empty() {
                // All open and one idle for another server: it is closed and its place taken (the oldest, at the front).
                let c = s.idle.remove(0);
                s.note_busy();
                drop(s);
                drop(c);
                return Took::Dial;
            }
            s = self.wake.wait(s).unwrap();
        }
    }

    /// A connection's place given back (its connection closed or failed).
    fn give(&self) {
        self.state.lock().unwrap().open -= 1;
        self.wake.notify_all();
    }

    /// A connection kept for the next request to `host:port`.
    fn keep(&self, host: &str, port: u16, conn: BufReader<Box<dyn Conn>>) {
        self.state.lock().unwrap().idle.push(Idle { host: host.to_string(), port, conn, since: Instant::now() });
        self.wake.notify_all();
    }

    /// A fetch's measures added; every 16 s (the next fetch to end after them, no timer) the summary on the console: fetches and bytes a second, connections kept and new, the most busy at once, the waits.
    fn sum(&self, got: &Result<Got, String>, said: impl FnOnce() -> String) {
        let mut s = self.state.lock().unwrap();
        let idle_now = s.idle.len();
        let sums = &mut s.sums;
        let now = Instant::now();
        let since = *sums.since.get_or_insert(now);
        sums.fetches += 1;
        match got {
            Ok(g) => {
                let p = g.phases;
                sums.bytes += g.body.len() as u64;
                sums.wait_us += p.wait;
                sums.retried += p.retried as u64;
                if p.kept.is_some() {
                    sums.kept += 1;
                    sums.first_kept_us += p.first;
                } else {
                    sums.fresh += 1;
                    sums.first_fresh_us += p.first;
                    sums.dial_us += p.dial;
                }
            }
            Err(_) => sums.failed += 1,
        }
        let span = now.duration_since(since);
        if span < Duration::from_secs(1 << 4) {
            return;
        }
        let secs = span.as_secs_f64();
        let ms = |us: u64, n: u64| if n == 0 { 0 } else { us / n / 1000 };
        let line = format!(
            "mahere: the network, last {secs:.1} s: {} fetches ({:.1} a second, {} failed), {:.2} MB ({:.2} MB/s); {} on a kept connection (first byte {} ms on average), {} new (dial {} ms, first byte {} ms), {} sent again on a new one; the most busy at once {} of {CONNECTIONS}, {} idle now; waited for a connection {} ms on average{}",
            sums.fetches,
            sums.fetches as f64 / secs,
            sums.failed,
            sums.bytes as f64 / 1e6,
            sums.bytes as f64 / 1e6 / secs,
            sums.kept,
            ms(sums.first_kept_us, sums.kept),
            sums.fresh,
            ms(sums.dial_us, sums.fresh),
            ms(sums.first_fresh_us, sums.fresh),
            sums.retried,
            sums.busy_most,
            idle_now,
            ms(sums.wait_us, sums.fetches - sums.failed),
            {
                let d = said();
                if d.is_empty() { d } else { format!("; {d}") }
            },
        );
        s.sums = Sums::default();
        drop(s);
        eprintln!("{line}");
    }
}

impl PoolState {
    fn note_busy(&mut self) {
        let busy = self.open - self.idle.len();
        if busy > self.sums.busy_most {
            self.sums.busy_most = busy;
        }
    }
}

/// The device's one pool.
static POOL: Pool = Pool::new();

/// The most bytes the status line and headers may take (WHY a bound: they are the server's, read before anything says how long they are; a server streaming headers without end would grow them without end, and a few hundred bytes is usual).
const HEADERS_MAX: u64 = 1 << 13;
/// The most bytes one chunk-size line or trailer line may take (WHY: as above, a line is the server's and unbounded until its end comes).
const LINE_MAX: u64 = 1 << 12;

/// `http://host[:port]/path` split into (host, port, path and query). None for any other scheme (there is no TLS here), an empty host, a port that is not a number, or anything but printable ASCII.
pub fn split_url(url: &str) -> Option<(&str, u16, &str)> {
    if !url.bytes().all(|b| (0x21..0x7F).contains(&b)) {
        return None;
    }
    let rest = url.strip_prefix("http://")?;
    let slash = rest.find('/').unwrap_or(rest.len());
    let (authority, path) = rest.split_at(slash);
    let path = if path.is_empty() { "/" } else { path };
    let (host, port) = match authority.rfind(':') {
        Some(c) => (&authority[..c], authority[c + 1..].parse::<u16>().ok()?),
        None => (authority, 80),
    };
    if host.is_empty() || !host.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-') {
        return None;
    }
    Some((host, port, path))
}

/// GET `url` over the registered [`Dial`]: the response whatever its status (the caller reads 200, 304 and 404), or why there is none.
/// `if_none_match`: the tag of the copy the caller holds, so an unchanged one costs a 304.
pub fn get(url: &str, if_none_match: Option<&str>) -> Result<Got, String> {
    let dial = dial().ok_or("no network on this device")?;
    let got = get_with(&*dial, &POOL, url, if_none_match);
    POOL.sum(&got, || dial.said());
    got
}

/// [`get`] over `dial` and `pool` (the host's tests give their own).
fn get_with(dial: &dyn Dial, pool: &Pool, url: &str, if_none_match: Option<&str>) -> Result<Got, String> {
    let (host, port, path) = split_url(url).ok_or_else(|| format!("{url}: not a plain http URL"))?;
    let req = request(host, path, if_none_match);
    let t = Instant::now();
    let took = pool.take(host, port, true);
    let mut wait = t.elapsed();
    let mut retried = false;
    let (mut conn, mut kept, mut dialled) = match took {
        Took::Kept(c, idle) => (c, Some(idle), Duration::ZERO),
        Took::Dial => fresh(dial, pool, host, port)?,
    };
    loop {
        let t = Instant::now();
        // Sent, and its first byte waited for: on a kept connection a failure here is the server having closed it meanwhile (an idle connection is the server's to close, RFC 9112 9.6), and the request goes once more on a new one.
        let first = conn.get_mut().write_all(&req).and_then(|_| conn.fill_buf().map(|b| b.len()));
        let first = match first {
            Ok(n) if n > 0 => t.elapsed(),
            // WHY the retry is once and on a new connection: the server may have closed every idle one at once (its idle timeout), and a new one failing the same way is the network's failure, said to the caller.
            _ if kept.is_some() && !retried => {
                drop(conn);
                pool.give();
                retried = true;
                let w = Instant::now();
                let Took::Dial = pool.take(host, port, false) else { unreachable!("a take that may not reuse dials") };
                wait += w.elapsed();
                (conn, kept, dialled) = fresh(dial, pool, host, port)?;
                continue;
            }
            Ok(_) => {
                pool.give();
                return Err(format!("{host}: the connection closed before the response"));
            }
            Err(e) => {
                pool.give();
                return Err(format!("{host}: sending the request: {e}"));
            }
        };
        let t = Instant::now();
        let r = read_response(&mut conn);
        let body = t.elapsed();
        return match r {
            Ok(mut g) => {
                g.phases = Phases { wait: wait.as_micros() as u64, dial: dialled.as_micros() as u64, first: first.as_micros() as u64, body: body.as_micros() as u64, kept: kept.map(|d| d.as_micros() as u64), retried };
                // Kept only with nothing left over in its buffer (WHY: one request at a time, never pipelined, so a byte past the response is the server's mistake and would be read as the next response's start).
                if g.keep && conn.buffer().is_empty() {
                    pool.keep(host, port, conn);
                } else {
                    drop(conn);
                    pool.give();
                }
                Ok(g)
            }
            Err(e) => {
                drop(conn);
                pool.give();
                Err(format!("{host}: {e}"))
            }
        };
    }
}

/// A new connection to `host:port` with its place in the pool already taken (given back when the dial fails), and how long the dial took.
fn fresh(dial: &dyn Dial, pool: &Pool, host: &str, port: u16) -> Result<(BufReader<Box<dyn Conn>>, Option<Duration>, Duration), String> {
    let t = Instant::now();
    match dial.dial(host, port) {
        Ok(c) => Ok((BufReader::new(c), None, t.elapsed())),
        Err(e) => {
            pool.give();
            Err(format!("{host}: {e}"))
        }
    }
}

/// The request's bytes: HTTP/1.1, so the connection stays open for the next one unless the server says otherwise. A tag is sent only when it is printable ASCII (WHY: it is a stored copy of the server's header; a CR or LF in it would end the header early and write a header of its own).
fn request(host: &str, path: &str, if_none_match: Option<&str>) -> Vec<u8> {
    let mut r = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: mahere/{}\r\n", env!("CARGO_PKG_VERSION"));
    if let Some(t) = if_none_match.filter(|t| t.bytes().all(|b| (0x20..0x7F).contains(&b))) {
        r.push_str(&format!("If-None-Match: {t}\r\n"));
    }
    r.push_str("\r\n");
    r.into_bytes()
}

/// How a body ends.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Body {
    /// `Content-Length`: this many bytes.
    Length(u64),
    /// `Transfer-Encoding: chunked`.
    Chunked,
    /// Neither: everything up to the server's close.
    Close,
    /// None at all (304, 204, 1xx: RFC 9112 section 6.3).
    Empty,
}

/// One line of at most `max` bytes, without its CRLF; an error when the connection ends first or the line runs past `max`.
fn line(r: &mut impl BufRead, max: u64) -> Result<String, String> {
    let mut l = Vec::new();
    let n = r.by_ref().take(max).read_until(b'\n', &mut l).map_err(|e| e.to_string())?;
    if n == 0 {
        return Err("the connection closed before the response".into());
    }
    if l.last() != Some(&b'\n') {
        return Err(format!("a line of the response ran past {max} bytes"));
    }
    l.pop();
    if l.last() == Some(&b'\r') {
        l.pop();
    }
    String::from_utf8(l).map_err(|_| "a line of the response is not text".into())
}

/// The response from its first byte: the status line, the headers, the body.
pub fn read_response(r: &mut impl BufRead) -> Result<Got, String> {
    let mut used = 0u64;
    let status_line = line(r, HEADERS_MAX)?;
    used += status_line.len() as u64;
    let mut parts = status_line.split(' ');
    let version = parts.next().unwrap_or("");
    if !version.starts_with("HTTP/1.") {
        return Err(format!("not an HTTP/1 response: {status_line}"));
    }
    let status: u16 = parts.next().and_then(|s| s.parse().ok()).ok_or_else(|| format!("no status in: {status_line}"))?;
    let mut length = None;
    let mut chunked = false;
    let mut etag = None;
    // HTTP/1.1 keeps the connection unless the server says close; HTTP/1.0 closes it (RFC 9112 9.3; its keep-alive extension not taken up).
    let mut keep = version == "HTTP/1.1";
    loop {
        // WHY the subtraction cannot wrap: `used` stays at or under HEADERS_MAX (checked after each line).
        let h = line(r, HEADERS_MAX - used)?;
        used += h.len() as u64 + 2;
        if used > HEADERS_MAX {
            return Err(format!("headers past {HEADERS_MAX} bytes"));
        }
        if h.is_empty() {
            break;
        }
        let Some((name, value)) = h.split_once(':') else { continue };
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            length = Some(value.parse::<u64>().map_err(|_| format!("Content-Length: {value}"))?);
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            chunked = value.to_ascii_lowercase().split(',').any(|c| c.trim() == "chunked");
        } else if name.eq_ignore_ascii_case("etag") {
            etag = Some(value.to_string());
        } else if name.eq_ignore_ascii_case("connection") && value.to_ascii_lowercase().split(',').any(|c| c.trim() == "close") {
            keep = false;
        }
    }
    let body = if status == 304 || status == 204 || status < 200 {
        Body::Empty
    } else if chunked {
        Body::Chunked
    } else if let Some(n) = length {
        Body::Length(n)
    } else {
        Body::Close
    };
    let mut out = Vec::new();
    match body {
        Body::Empty => {}
        Body::Length(n) => {
            // The body's length known: its buffer made that size at once (ferros, 2026-11-12: `read_to_end` doubled its way up, a 17 MB font pack asking for 2^25 bytes beside the 2^24 it had, and Mahere on the Pro ended "memory allocation of 33554432 bytes failed").
            // WHY the bound: the length is the server's word, external input; a header past 2^26 (more than any document Mahere fetches, a font pack 17 MB, a cell 2) is read as before, growing as it comes, so a bad header cannot ask for an allocation of any size up front.
            if n <= 1 << 26 {
                out.reserve_exact(n as usize);
            }
            let got = r.by_ref().take(n).read_to_end(&mut out).map_err(|e| e.to_string())?;
            if got as u64 != n {
                return Err(format!("short body: {got} of {n} bytes"));
            }
        }
        Body::Close => {
            keep = false;
            r.read_to_end(&mut out).map_err(|e| e.to_string())?;
        }
        Body::Chunked => loop {
            let size_line = line(r, LINE_MAX)?;
            let digits = size_line.split(';').next().unwrap_or("").trim();
            let size = u64::from_str_radix(digits, 16).map_err(|_| format!("chunk size: {size_line}"))?;
            if size == 0 {
                // The trailer, up to its empty line.
                while !line(r, LINE_MAX)?.is_empty() {}
                break;
            }
            let got = r.by_ref().take(size).read_to_end(&mut out).map_err(|e| e.to_string())?;
            if got as u64 != size {
                return Err(format!("short chunk: {got} of {size} bytes"));
            }
            if !line(r, LINE_MAX)?.is_empty() {
                return Err("a chunk ran past its size".into());
            }
        },
    }
    Ok(Got { status, etag, body: out, keep, phases: Phases::default() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn read(bytes: &[u8]) -> Result<Got, String> {
        read_response(&mut BufReader::new(bytes))
    }

    #[test]
    fn urls() {
        assert_eq!(split_url("http://brobdingnagian.holdmyoscilloscope.com/mahere/cells/base/ab-_c.vsf?e=8&h=x"), Some(("brobdingnagian.holdmyoscilloscope.com", 80, "/mahere/cells/base/ab-_c.vsf?e=8&h=x")));
        assert_eq!(split_url("http://host:8080"), Some(("host", 8080, "/")));
        assert_eq!(split_url("https://host/x"), None);
        assert_eq!(split_url("http:///x"), None);
        assert_eq!(split_url("http://ho st/x"), None);
        assert_eq!(split_url("http://host:port/x"), None);
    }

    #[test]
    fn by_length() {
        let g = read(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nETag: \"abc\"\r\nConnection: close\r\n\r\nhello").unwrap();
        assert_eq!((g.status, g.etag.as_deref(), &g.body[..], g.keep), (200, Some("\"abc\""), &b"hello"[..], false));
        assert!(read(b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nhello").unwrap_err().contains("short body"));
        assert!(read(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: keep-alive\r\n\r\nhello").unwrap().keep);
    }

    #[test]
    fn chunked() {
        let g = read(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5;x=y\r\nhello\r\n1\r\n!\r\n0\r\nA: b\r\n\r\n").unwrap();
        assert_eq!((&g.body[..], g.keep), (&b"hello!"[..], true));
        assert!(read(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nhello\r\n0\r\n\r\n").is_err());
    }

    #[test]
    fn to_the_close_and_no_body() {
        let g = read(b"HTTP/1.1 200 OK\r\n\r\nall of it").unwrap();
        assert_eq!((&g.body[..], g.keep), (&b"all of it"[..], false));
        assert!(!read(b"HTTP/1.0 200 OK\r\nContent-Length: 1\r\n\r\nx").unwrap().keep);
        let g = read(b"HTTP/1.1 304 Not Modified\r\nETag: \"t\"\r\n\r\n").unwrap();
        assert_eq!((g.status, g.body.len(), g.keep), (304, 0, true));
        assert_eq!(read(b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot Found").unwrap().status, 404);
    }

    #[test]
    fn refused() {
        assert!(read(b"").is_err());
        assert!(read(b"SSH-2.0-x\r\n\r\n").is_err());
        let mut long = b"HTTP/1.1 200 OK\r\nX: ".to_vec();
        long.extend(std::iter::repeat_n(b'a', 1 << 14));
        long.extend_from_slice(b"\r\n\r\n");
        assert!(read(&long).is_err());
    }

    #[test]
    fn requests() {
        let r = String::from_utf8(request("h", "/p?e=8", Some("\"t\""))).unwrap();
        assert!(r.starts_with("GET /p?e=8 HTTP/1.1\r\nHost: h\r\n") && r.contains("If-None-Match: \"t\"\r\n") && r.ends_with("\r\n\r\n"));
        assert!(!r.contains("Connection: close"));
        assert!(!String::from_utf8(request("h", "/", Some("a\r\nX: y"))).unwrap().contains("If-None-Match"));
    }

    /// A connection that answers each request with the next of its canned responses, and nothing (the server's close) once they are used up; it counts the requests it was sent and whether it was let go without a close.
    struct Canned {
        responses: Vec<Vec<u8>>,
        at: std::io::Cursor<Vec<u8>>,
        sent: Arc<AtomicUsize>,
        abandoned: Arc<AtomicUsize>,
    }

    impl Read for Canned {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            self.at.read(b)
        }
    }

    impl Write for Canned {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            assert!(b.starts_with(b"GET /"));
            self.sent.fetch_add(1, Ordering::Relaxed);
            let next = if self.responses.is_empty() { Vec::new() } else { self.responses.remove(0) };
            self.at = std::io::Cursor::new(next);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Conn for Canned {
        fn abandon(&mut self) {
            self.abandoned.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A server whose every new connection carries `per_conn` responses; it counts its dials.
    struct Server {
        per_conn: Vec<Vec<u8>>,
        dials: AtomicUsize,
        sent: Arc<AtomicUsize>,
        abandoned: Arc<AtomicUsize>,
    }

    impl Server {
        fn new(per_conn: &[&[u8]]) -> Server {
            Server { per_conn: per_conn.iter().map(|r| r.to_vec()).collect(), dials: AtomicUsize::new(0), sent: Arc::default(), abandoned: Arc::default() }
        }
    }

    impl Dial for Server {
        fn dial(&self, host: &str, port: u16) -> std::io::Result<Box<dyn Conn>> {
            assert_eq!((host, port), ("example.org", 80));
            self.dials.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(Canned { responses: self.per_conn.clone(), at: std::io::Cursor::new(Vec::new()), sent: self.sent.clone(), abandoned: self.abandoned.clone() }))
        }
    }

    const OK3: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nvsf";

    /// Three GETs one after another on one connection, kept between them.
    #[test]
    fn kept_between_requests() {
        let (server, pool) = (Server::new(&[OK3, OK3, OK3]), Pool::new());
        for k in 0..3 {
            let g = get_with(&server, &pool, "http://example.org/mahere/cells/base/x.vsf?e=8", None).unwrap();
            assert_eq!((g.status, &g.body[..], g.phases.kept.is_some()), (200, &b"vsf"[..], k > 0));
        }
        assert_eq!(server.dials.load(Ordering::Relaxed), 1);
        assert_eq!(pool.state.lock().unwrap().open, 1);
    }

    /// A kept connection the server closed meanwhile: the request goes again on a new one, once.
    #[test]
    fn closed_while_kept() {
        let (server, pool) = (Server::new(&[OK3]), Pool::new());
        get_with(&server, &pool, "http://example.org/a", None).unwrap();
        let g = get_with(&server, &pool, "http://example.org/b", None).unwrap();
        assert!(g.phases.retried && g.phases.kept.is_none());
        assert_eq!((server.dials.load(Ordering::Relaxed), server.sent.load(Ordering::Relaxed)), (2, 3));
        assert_eq!(pool.state.lock().unwrap().open, 1);
    }

    /// `Connection: close` and a body to the close: never kept.
    #[test]
    fn closed_by_the_server() {
        let (server, pool) = (Server::new(&[b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nConnection: close\r\n\r\nx"]), Pool::new());
        get_with(&server, &pool, "http://example.org/a", None).unwrap();
        get_with(&server, &pool, "http://example.org/b", None).unwrap();
        assert_eq!(server.dials.load(Ordering::Relaxed), 2);
        assert_eq!(pool.state.lock().unwrap().open, 0);
    }

    /// One kept past KEEP is let go without a word (its handle may be another's by now) and a new one dialled.
    #[test]
    fn kept_too_long() {
        let (server, pool) = (Server::new(&[OK3, OK3]), Pool::new());
        get_with(&server, &pool, "http://example.org/a", None).unwrap();
        pool.state.lock().unwrap().idle[0].since -= KEEP;
        let g = get_with(&server, &pool, "http://example.org/b", None).unwrap();
        assert!(g.phases.kept.is_none() && !g.phases.retried);
        assert_eq!((server.dials.load(Ordering::Relaxed), server.abandoned.load(Ordering::Relaxed)), (2, 1));
        assert_eq!(pool.state.lock().unwrap().open, 1);
    }

    /// Never more than CONNECTIONS at once however many fetch, each kept and reused.
    #[test]
    fn at_most_connections() {
        // Each connection good for every request (one may carry them all).
        let (server, pool) = (Server::new(&[OK3; CONNECTIONS << 5]), Pool::new());
        std::thread::scope(|s| {
            for _ in 0..(CONNECTIONS << 2) {
                s.spawn(|| {
                    for _ in 0..8 {
                        get_with(&server, &pool, "http://example.org/a", None).unwrap();
                    }
                });
            }
        });
        let st = pool.state.lock().unwrap();
        assert!(st.open <= CONNECTIONS && server.dials.load(Ordering::Relaxed) <= CONNECTIONS);
        assert_eq!(server.sent.load(Ordering::Relaxed), CONNECTIONS << 5);
    }

    /// The whole GET over the registered dial.
    #[test]
    fn get_over_a_dial() {
        set_dial(Arc::new(Server::new(&[OK3])));
        let g = get("http://example.org/mahere/cells/base/x.vsf?e=8", None).unwrap();
        assert_eq!((g.status, &g.body[..]), (200, &b"vsf"[..]));
    }
}
