//! A plain HTTP/1.1 GET over a byte stream the platform dials, for a platform where ureq has no sockets: ferros, whose connections are the ip server's client TCP (mahere-ferros `net`), and which speaks no TLS, so the bucket is reached over plain HTTP (it answers port 80 as it answers 443).
//!
//! Nick 2026-11-11, on ferros: "we have tcp and http so let's get the full meal deal wired up with mahere".
//! The platform registers its [`Dial`] once at start ([`set_dial`]); [`get`] then makes one request a connection (`Connection: close`, no connection kept: the ip server has two for the whole device), reads the status line and the headers, and the body by its length, in chunks, or up to the close.
//! Everything here is pure over `Read` and `Write`, so the host's tests drive it with bytes in memory.

use std::io::{BufRead, BufReader, Read, Write};
use std::sync::{Arc, OnceLock};

/// A connection to read the response from and write the request to; closed when dropped.
pub trait Conn: Read + Write + Send {}
impl<T: Read + Write + Send> Conn for T {}

/// How a platform opens a connection to a server.
pub trait Dial: Send + Sync {
    /// A connection to `host` on `port`.
    fn dial(&self, host: &str, port: u16) -> std::io::Result<Box<dyn Conn>>;
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

/// A response: its status, its `ETag` when it has one, and its body.
#[derive(Debug)]
pub struct Got {
    pub status: u16,
    pub etag: Option<String>,
    pub body: Vec<u8>,
}

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
    let (host, port, path) = split_url(url).ok_or_else(|| format!("{url}: not a plain http URL"))?;
    let mut conn = dial.dial(host, port).map_err(|e| format!("{host}: {e}"))?;
    conn.write_all(&request(host, path, if_none_match)).map_err(|e| format!("{host}: sending the request: {e}"))?;
    read_response(&mut BufReader::new(conn)).map_err(|e| format!("{host}: {e}"))
}

/// The request's bytes. A tag is sent only when it is printable ASCII (WHY: it is a stored copy of the server's header; a CR or LF in it would end the header early and write a header of its own).
fn request(host: &str, path: &str, if_none_match: Option<&str>) -> Vec<u8> {
    let mut r = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: mahere/{}\r\nConnection: close\r\n", env!("CARGO_PKG_VERSION"));
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
    if !parts.next().is_some_and(|v| v.starts_with("HTTP/1.")) {
        return Err(format!("not an HTTP/1 response: {status_line}"));
    }
    let status: u16 = parts.next().and_then(|s| s.parse().ok()).ok_or_else(|| format!("no status in: {status_line}"))?;
    let mut length = None;
    let mut chunked = false;
    let mut etag = None;
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
    Ok(Got { status, etag, body: out })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!((g.status, g.etag.as_deref(), &g.body[..]), (200, Some("\"abc\""), &b"hello"[..]));
        assert!(read(b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nhello").unwrap_err().contains("short body"));
    }

    #[test]
    fn chunked() {
        let g = read(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5;x=y\r\nhello\r\n1\r\n!\r\n0\r\nA: b\r\n\r\n").unwrap();
        assert_eq!(&g.body[..], b"hello!");
        assert!(read(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nhello\r\n0\r\n\r\n").is_err());
    }

    #[test]
    fn to_the_close_and_no_body() {
        assert_eq!(&read(b"HTTP/1.0 200 OK\r\n\r\nall of it").unwrap().body[..], b"all of it");
        let g = read(b"HTTP/1.1 304 Not Modified\r\nETag: \"t\"\r\n\r\n").unwrap();
        assert_eq!((g.status, g.body.len()), (304, 0));
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
        assert!(!String::from_utf8(request("h", "/", Some("a\r\nX: y"))).unwrap().contains("If-None-Match"));
    }

    /// The whole GET over a dial that answers from memory and keeps what was sent.
    #[test]
    fn get_over_a_dial() {
        struct Canned;
        struct Pipe(std::io::Cursor<Vec<u8>>);
        impl Read for Pipe {
            fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
                self.0.read(b)
            }
        }
        impl Write for Pipe {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                assert!(b.starts_with(b"GET /mahere/cells/base/x.vsf?e=8 HTTP/1.1\r\nHost: example.org\r\n"));
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl Dial for Canned {
            fn dial(&self, host: &str, port: u16) -> std::io::Result<Box<dyn Conn>> {
                assert_eq!((host, port), ("example.org", 80));
                Ok(Box::new(Pipe(std::io::Cursor::new(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nvsf".to_vec()))))
            }
        }
        set_dial(Arc::new(Canned));
        let g = get("http://example.org/mahere/cells/base/x.vsf?e=8", None).unwrap();
        assert_eq!((g.status, &g.body[..]), (200, &b"vsf"[..]));
    }
}
