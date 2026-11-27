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
//! Measured after on the 8a (the same day): a cache check (304) 139 ms on a new connection, 54 ms on a kept one; zooming in on land 98 cells (30.6 MB) at 9.7 fetches and 3.0 MB/s in the busy spells (7.4 and 1.04 before), all eight busy at once, no failures. A fresh cell's first byte is now the bucket's (300 to 560 ms while Cloudflare fetches it from the origin).
//!
//! Pipelined (2026-11-27, Nick: "bundle and pipelining"): each connection its own thread, the requests for it written up to [`DEPTH`] ahead of the response being read, so the edge's quarter second to its first byte is waited for once for several cells. A cold 35 km route on the Pro had fetched 713 documents at about 8 a second, each waiting its turn for a connection and then a quarter second for its first byte.

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

/// Connections at once: sixteen. The ip server held sixteen for the whole device until 2026-11-27, when it came to hold as many as memory does (ferros 8089441c, each open connection 1.25 MB of its memory: twenty here); the edge answers a connection's requests in order, so cells it must fetch from the bucket come faster across more connections than deeper on fewer. Eight until 2026-11-27, four until 2026-11-14.
/// Twenty loaders and the route search's threads share them (residency `LOADERS`), each waiting for its response as for a socket, a doorbell (a channel), not a poll.
pub const CONNECTIONS: usize = 1 << 4;

/// Requests written on one connection before the first's response is read (HTTP/1.1 pipelining, RFC 9112 9.3.2; Nick 2026-11-27, "bundle and pipelining"): the edge's first byte, a quarter second on the Pro, is waited for once for several cells, not once for each. Measured from the desktop on a warm connection, sixteen cells the edge holds came in 405 ms pipelined against 1291 one at a time; ones it fetches from the bucket come as they would one at a time, the edge taking them in order.
pub const DEPTH: usize = 1 << 2;

/// How long an idle connection is kept for a next request: 16 s.
/// WHY a bound: the ip server resets and frees a connection its app has not asked about for 32 s, and its handle goes to the next CONNECT of any app; a kept handle older than that could reach another app's connection. Half of it leaves the margin. A connection's thread lets it go ([`Conn::abandon`]) when nothing has come for it for this long.
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
    /// The most requests on one connection at once, sent and not yet answered.
    deepest: usize,
}

/// A request waiting for its connection's thread: its bytes, where its response goes, when it was asked, and whether it has been sent once already on a connection that broke.
struct Ask {
    req: Vec<u8>,
    reply: std::sync::mpsc::Sender<Result<Got, String>>,
    asked: Instant,
    retried: bool,
}

/// A request sent and not yet answered: when it went, how long its connection had been idle before it (None: dialled for it), and the dial's time.
struct Sent {
    ask: Ask,
    at: Instant,
    kept: Option<Duration>,
    dial: Duration,
}

/// One connection's thread: the requests waiting for it, how many it holds (waiting and sent), and whether it has ended.
struct Pipe {
    host: String,
    port: u16,
    state: Mutex<PipeState>,
    wake: Condvar,
}

#[derive(Default)]
struct PipeState {
    waiting: std::collections::VecDeque<Ask>,
    held: usize,
    ended: bool,
}

/// The connections, a thread each, and the measures.
pub struct Pool {
    pipes: Mutex<Vec<Arc<Pipe>>>,
    wake: Condvar,
    sums: Mutex<Sums>,
    /// How long a connection is kept idle ([`KEEP`]; the host's tests give a short one).
    keep: Duration,
}

impl Pool {
    pub const fn new() -> Pool {
        Pool::keeping(KEEP)
    }

    const fn keeping(keep: Duration) -> Pool {
        Pool { pipes: Mutex::new(Vec::new()), wake: Condvar::new(), sums: Mutex::new(Sums { since: None, fetches: 0, kept: 0, fresh: 0, retried: 0, failed: 0, bytes: 0, wait_us: 0, first_kept_us: 0, first_fresh_us: 0, dial_us: 0, deepest: 0 }), keep }
    }

    /// `ask` onto a connection to `host:port`: an idle one to it, else a new one while there are under [`CONNECTIONS`], else the one to it holding the fewest (its thread writes it when fewer than [`DEPTH`] are unanswered). Wide before deep (the Pro, 2026-11-27: filling one connection to its depth before opening the next kept three of eight open, and the edge answers a connection's requests in their order, a cell it must fetch holding up the ones behind it). With every connection another server's, an idle one is ended for it, or the request waits for one to end. Queued under the pool's lock, on a connection seen not to have ended, so none takes a request as it goes.
    fn send(&'static self, dial: &Arc<dyn Dial>, host: &str, port: u16, ask: Ask) {
        let mut ask = Some(ask);
        let mut pipes = self.pipes.lock().unwrap();
        loop {
            pipes.retain(|p| !p.state.lock().unwrap().ended);
            let held = |p: &Arc<Pipe>| p.state.lock().unwrap().held;
            let least = pipes.iter().filter(|p| p.host == host && p.port == port).min_by_key(|p| held(p)).cloned();
            let pipe = match least {
                Some(p) if held(&p) == 0 || pipes.len() >= CONNECTIONS => p,
                _ if pipes.len() < CONNECTIONS => {
                    let p = Arc::new(Pipe { host: host.to_string(), port, state: Mutex::new(PipeState::default()), wake: Condvar::new() });
                    let (pipe, dial) = (p.clone(), dial.clone());
                    if let Err(e) = std::thread::Builder::new().name("http connection".into()).spawn(move || self.run(&*dial, &pipe)) {
                        let _ = ask.take().unwrap().reply.send(Err(format!("{host}: no thread for a connection: {e}")));
                        return;
                    }
                    pipes.push(p.clone());
                    p
                }
                _ => {
                    // Every connection another server's: an idle one ended for this one (its thread sees it and goes), or a wait for one to end.
                    if let Some(idle) = pipes.iter().find(|p| p.state.lock().unwrap().held == 0) {
                        idle.state.lock().unwrap().ended = true;
                        idle.wake.notify_all();
                    } else {
                        pipes = self.wake.wait(pipes).unwrap();
                    }
                    continue;
                }
            };
            let mut s = pipe.state.lock().unwrap();
            if s.ended {
                continue;
            }
            s.waiting.push_back(ask.take().unwrap());
            s.held += 1;
            pipe.wake.notify_all();
            return;
        }
    }

    /// A connection's thread: the requests waiting written, up to [`DEPTH`] unanswered, then the oldest's response read and handed back, until nothing has come for [`Pool::keep`]. A connection that breaks or that the server closes is dialled again and its unanswered requests sent on the new one, each once (a GET is safe to send again, RFC 9110 9.2.2); one that fails a second time is failed.
    fn run(&self, dial: &dyn Dial, pipe: &Pipe) {
        let (host, port) = (pipe.host.as_str(), pipe.port);
        let mut conn: Option<BufReader<Box<dyn Conn>>> = None;
        let mut sent: std::collections::VecDeque<Sent> = std::collections::VecDeque::new();
        let mut idle_since = Instant::now();
        loop {
            // The requests to send: what waits, up to DEPTH unanswered; with none sent and none waiting, a wait for one, or the end after `keep`.
            let mut to_send = Vec::new();
            {
                let mut s = pipe.state.lock().unwrap();
                while s.waiting.is_empty() && sent.is_empty() && !s.ended {
                    // WHY saturating: what is left of `keep`; past it the connection is let go, never a wait of less than nothing.
                    let left = self.keep.saturating_sub(idle_since.elapsed());
                    if left.is_zero() {
                        s.ended = true;
                        break;
                    }
                    s = pipe.wake.wait_timeout(s, left).unwrap().0;
                }
                if s.ended && sent.is_empty() {
                    // Ended (idle past `keep`, or its place given to another server, both only while it held nothing).
                    drop(s);
                    if let Some(mut c) = conn.take() {
                        c.get_mut().abandon();
                    }
                    self.wake.notify_all();
                    return;
                }
                while sent.len() + to_send.len() < DEPTH {
                    let Some(a) = s.waiting.pop_front() else { break };
                    to_send.push(a);
                }
            }
            // A connection for them: the one kept, or a new one.
            let (kept, mut dialled) = (conn.as_ref().map(|_| idle_since.elapsed()), Duration::ZERO);
            if conn.is_none() {
                let t = Instant::now();
                match dial.dial(host, port) {
                    Ok(c) => {
                        conn = Some(BufReader::new(c));
                        dialled = t.elapsed();
                    }
                    Err(e) => {
                        // No connection: everything this one holds fails, and it ends.
                        let e = format!("{host}: {e}");
                        let mut s = pipe.state.lock().unwrap();
                        s.ended = true;
                        s.held = 0;
                        let all: Vec<Ask> = to_send.into_iter().chain(sent.drain(..).map(|s| s.ask)).chain(s.waiting.drain(..)).collect();
                        drop(s);
                        for a in all {
                            let _ = a.reply.send(Err(e.clone()));
                        }
                        self.wake.notify_all();
                        return;
                    }
                }
            }
            let c = conn.as_mut().unwrap();
            let mut broken = false;
            for ask in to_send {
                broken = broken || c.get_mut().write_all(&ask.req).is_err();
                sent.push_back(Sent { ask, at: Instant::now(), kept, dial: dialled });
            }
            {
                let mut sums = self.sums.lock().unwrap();
                sums.deepest = sums.deepest.max(sent.len());
            }
            // The oldest's response, unless the connection broke on the way.
            let read = match broken {
                true => Err(format!("{host}: sending the request failed")),
                false => match c.fill_buf().map(|b| b.len()) {
                    Ok(n) if n > 0 => {
                        let (first, t) = (sent[0].at.elapsed(), Instant::now());
                        read_response(c).map(|g| (g, first, t.elapsed())).map_err(|e| format!("{host}: {e}"))
                    }
                    Ok(_) => Err(format!("{host}: the connection closed before the response")),
                    Err(e) => Err(format!("{host}: {e}")),
                },
            };
            match read {
                Ok((mut g, first, body)) => {
                    let s = sent.pop_front().unwrap();
                    g.phases = Phases { wait: s.at.duration_since(s.ask.asked).as_micros() as u64, dial: s.dial.as_micros() as u64, first: first.as_micros() as u64, body: body.as_micros() as u64, kept: s.kept.map(|d| d.as_micros() as u64), retried: s.ask.retried };
                    // Kept for the next only when the server keeps it; the requests sent after it on a connection it closes go again on a new one.
                    if !g.keep {
                        conn = None;
                        self.again(pipe, &mut sent);
                    }
                    let _ = s.ask.reply.send(Ok(g));
                    pipe.state.lock().unwrap().held -= 1;
                    self.wake.notify_all();
                    idle_since = Instant::now();
                }
                Err(e) => {
                    // The connection broke (a kept one the server closed meanwhile, RFC 9112 9.6, or the network): what it carried goes again on a new one, and a request already sent again fails, the network's failure said to its caller.
                    conn = None;
                    let (twice, once): (Vec<Sent>, Vec<Sent>) = sent.drain(..).partition(|s| s.ask.retried);
                    sent.extend(once);
                    self.again(pipe, &mut sent);
                    if !twice.is_empty() {
                        pipe.state.lock().unwrap().held -= twice.len();
                        for s in twice {
                            let _ = s.ask.reply.send(Err(e.clone()));
                        }
                        self.wake.notify_all();
                    }
                }
            }
        }
    }

    /// Requests sent on a connection that is gone, back to the front of the queue in their order, each marked as sent once already.
    fn again(&self, pipe: &Pipe, sent: &mut std::collections::VecDeque<Sent>) {
        let mut s = pipe.state.lock().unwrap();
        for mut a in sent.drain(..).rev().map(|s| s.ask) {
            a.retried = true;
            s.waiting.push_front(a);
        }
    }

    /// How many connections are open now.
    fn open(&self) -> usize {
        self.pipes.lock().unwrap().iter().filter(|p| !p.state.lock().unwrap().ended).count()
    }

    /// A fetch's measures added; every 16 s (the next fetch to end after them, no timer) the summary on the console: fetches and bytes a second, connections kept and new, the most requests on one at once, the waits.
    fn sum(&self, got: &Result<Got, String>, said: impl FnOnce() -> String) {
        let open = self.open();
        let mut sums = self.sums.lock().unwrap();
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
            "mahere: the network, last {secs:.1} s: {} fetches ({:.1} a second, {} failed), {:.2} MB ({:.2} MB/s); {} on a kept connection (first byte {} ms on average), {} new (dial {} ms, first byte {} ms), {} sent again on a new one; {open} of {CONNECTIONS} connections open, at most {} requests on one at once (of {DEPTH}); waited to be sent {} ms on average{}",
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
            sums.deepest,
            ms(sums.wait_us, sums.fetches - sums.failed),
            {
                let d = said();
                if d.is_empty() { d } else { format!("; {d}") }
            },
        );
        *sums = Sums::default();
        drop(sums);
        eprintln!("{line}");
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
    let got = get_with(&dial, &POOL, url, if_none_match);
    POOL.sum(&got, || dial.said());
    got
}

/// [`get`] over `dial` and `pool` (the host's tests give their own): the request queued on a connection, its response waited for.
fn get_with(dial: &Arc<dyn Dial>, pool: &'static Pool, url: &str, if_none_match: Option<&str>) -> Result<Got, String> {
    let (host, port, path) = split_url(url).ok_or_else(|| format!("{url}: not a plain http URL"))?;
    let (reply, answer) = std::sync::mpsc::channel();
    pool.send(dial, host, port, Ask { req: request(host, path, if_none_match), reply, asked: Instant::now(), retried: false });
    answer.recv().map_err(|_| format!("{host}: the connection's thread ended without an answer"))?
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

    /// A connection that answers each request with the next of its canned responses, queued behind any it has not been read for yet (a pipelined request is answered after the ones before it), and nothing more (the server's close) once they are used up; it counts the requests sent to it, the most sent and not yet read at once, and whether it was let go without a close.
    struct Canned {
        responses: Vec<Vec<u8>>,
        queued: std::collections::VecDeque<u8>,
        ends: std::collections::VecDeque<usize>,
        read: usize,
        sent: Arc<AtomicUsize>,
        deepest: Arc<AtomicUsize>,
        abandoned: Arc<AtomicUsize>,
    }

    impl Read for Canned {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            // A server takes a moment to answer, as one across a network does: requests queue behind it, and are pipelined.
            std::thread::sleep(Duration::from_millis(1));
            let n = b.len().min(self.queued.len());
            for x in b.iter_mut().take(n) {
                *x = self.queued.pop_front().unwrap();
            }
            self.read += n;
            while self.ends.front().is_some_and(|&e| e <= self.read) {
                self.ends.pop_front();
            }
            Ok(n)
        }
    }

    impl Write for Canned {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            assert!(b.starts_with(b"GET /"));
            self.sent.fetch_add(1, Ordering::Relaxed);
            if !self.responses.is_empty() {
                let next = self.responses.remove(0);
                self.queued.extend(&next);
                self.ends.push_back(self.read + self.queued.len());
            }
            self.deepest.fetch_max(self.ends.len(), Ordering::Relaxed);
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
        deepest: Arc<AtomicUsize>,
        abandoned: Arc<AtomicUsize>,
    }

    impl Server {
        fn new(per_conn: &[&[u8]]) -> Arc<Server> {
            Arc::new(Server { per_conn: per_conn.iter().map(|r| r.to_vec()).collect(), dials: AtomicUsize::new(0), sent: Arc::default(), deepest: Arc::default(), abandoned: Arc::default() })
        }
    }

    impl Dial for Server {
        fn dial(&self, host: &str, port: u16) -> std::io::Result<Box<dyn Conn>> {
            assert_eq!((host, port), ("example.org", 80));
            self.dials.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(Canned { responses: self.per_conn.clone(), queued: Default::default(), ends: Default::default(), read: 0, sent: self.sent.clone(), deepest: self.deepest.clone(), abandoned: self.abandoned.clone() }))
        }
    }

    const OK3: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nvsf";

    /// A pool for one test, alive for the test's threads (a connection's thread holds it).
    fn pool() -> &'static Pool {
        Box::leak(Box::new(Pool::new()))
    }

    fn get_on(server: &Arc<Server>, pool: &'static Pool, url: &str) -> Result<Got, String> {
        get_with(&(server.clone() as Arc<dyn Dial>), pool, url, None)
    }

    /// Three GETs one after another on one connection, kept between them.
    #[test]
    fn kept_between_requests() {
        let (server, pool) = (Server::new(&[OK3, OK3, OK3]), pool());
        for k in 0..3 {
            let g = get_on(&server, pool, "http://example.org/mahere/cells/base/x.vsf?e=8").unwrap();
            assert_eq!((g.status, &g.body[..], g.phases.kept.is_some()), (200, &b"vsf"[..], k > 0));
        }
        assert_eq!(server.dials.load(Ordering::Relaxed), 1);
        assert_eq!(pool.open(), 1);
    }

    /// A kept connection the server closed meanwhile: the request goes again on a new one, once.
    #[test]
    fn closed_while_kept() {
        let (server, pool) = (Server::new(&[OK3]), pool());
        get_on(&server, pool, "http://example.org/a").unwrap();
        let g = get_on(&server, pool, "http://example.org/b").unwrap();
        assert!(g.phases.retried && g.phases.kept.is_none());
        assert_eq!((server.dials.load(Ordering::Relaxed), server.sent.load(Ordering::Relaxed)), (2, 3));
    }

    /// A request that fails on its second connection too is failed, not sent a third time.
    #[test]
    fn failed_twice() {
        let (server, pool) = (Server::new(&[]), pool());
        assert!(get_on(&server, pool, "http://example.org/a").unwrap_err().contains("closed before the response"));
        assert_eq!((server.dials.load(Ordering::Relaxed), server.sent.load(Ordering::Relaxed)), (2, 2));
    }

    /// `Connection: close`: never kept, the next request on a new connection.
    #[test]
    fn closed_by_the_server() {
        let (server, pool) = (Server::new(&[b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nConnection: close\r\n\r\nx"]), pool());
        get_on(&server, pool, "http://example.org/a").unwrap();
        let g = get_on(&server, pool, "http://example.org/b").unwrap();
        assert!(g.phases.kept.is_none() && !g.phases.retried);
        assert_eq!(server.dials.load(Ordering::Relaxed), 2);
    }

    /// One idle past its keeping is let go without a word (its handle may be another's by now) and a new one dialled.
    #[test]
    fn kept_too_long() {
        let (server, pool) = (Server::new(&[OK3, OK3]), &*Box::leak(Box::new(Pool::keeping(Duration::from_millis(1 << 5)))));
        get_on(&server, pool, "http://example.org/a").unwrap();
        std::thread::sleep(Duration::from_millis(1 << 7));
        assert_eq!((pool.open(), server.abandoned.load(Ordering::Relaxed)), (0, 1));
        let g = get_on(&server, pool, "http://example.org/b").unwrap();
        assert!(g.phases.kept.is_none() && !g.phases.retried);
        assert_eq!(server.dials.load(Ordering::Relaxed), 2);
    }

    /// Never more than CONNECTIONS at once however many fetch, requests pipelined on them up to DEPTH, every one answered.
    #[test]
    fn at_most_connections_pipelined() {
        // Each connection good for every request (one may carry them all).
        let (server, pool) = (Server::new(&[OK3; CONNECTIONS << 5]), pool());
        std::thread::scope(|s| {
            for _ in 0..(CONNECTIONS << 2) {
                s.spawn(|| {
                    for _ in 0..8 {
                        assert_eq!(get_on(&server, pool, "http://example.org/a").unwrap().body, b"vsf");
                    }
                });
            }
        });
        assert!(pool.open() <= CONNECTIONS && server.dials.load(Ordering::Relaxed) <= CONNECTIONS);
        assert_eq!(server.sent.load(Ordering::Relaxed), CONNECTIONS << 5);
        let deepest = server.deepest.load(Ordering::Relaxed);
        assert!(deepest > 1 && deepest <= DEPTH, "requests pipelined, at most DEPTH deep: {deepest}");
    }

    /// The whole GET over the registered dial.
    #[test]
    fn get_over_a_dial() {
        set_dial(Server::new(&[OK3]));
        let g = get("http://example.org/mahere/cells/base/x.vsf?e=8", None).unwrap();
        assert_eq!((g.status, &g.body[..]), (200, &b"vsf"[..]));
    }
}
