//! Mahere's way to the network on ferros: connections over the ip server's client TCP (ferros payloads/ip, its `udp` endpoint: 3 INFO, 4 CONNECT, 5 SEND, 6 RECV, 7 CLOSE), the server's address from the ip server's resolver (10 RESOLVE), for mahere-engine's plain HTTP ([`Dial`]). Nick 2026-11-11: "we have tcp and http so let's get the full meal deal wired up with mahere".
//!
//! Every wait is a call the ip server holds until something comes (the wait byte of its ops 6 and 10), never a poll (Nick 2026-11-10: "No polling!"): a loader thread blocks in the call as it blocks on a socket elsewhere. The one deadline here is a real one: a connection that brings nothing for 32 s is given up (DNS's are the ip server's own).
//! The ip server's TCP since 2026-11-11 (Nick: "sounds like it needs re-engineered"): sixteen connections, a 1 MiB window each, receives of up to 256 KiB; Mahere takes four connections at once and reads each a big reply at a time.
//! Names since 2026-11-12: the device's one resolver, in the ip server, in place of the copy Mahere carried (its own questions, port and cache). What Mahere's cache did the device's does for every app: one lookup a name however many loaders ask (the others' calls held on it), the last good address answered at once while it is asked for again and kept when that lookup brings no answer, and asked for again after four failed connections in a row (the ip server counts its own connections).
//! Mahere holds one of those calls a host, its loaders waiting on it (`names.rs`): the ip server holds sixteen, the oldest giving way to a seventeenth, and Mahere's twenty loaders each holding their own pushed each other out, about 240 fetches failing "DNS still asking" in its first second (the 8a, 2026-11-12).

use crate::ferros::sys;
use crate::names::{Asked, Names};
use mahere_engine::plain_http::{Conn, Dial};
use std::io;
use std::sync::{Arc, Condvar, Mutex};

/// The ip server's INFO reply: 71 bytes (addresses, flags, the DNS servers).
const INFO_LEN: usize = 71;
/// How long a lookup's call is held at most: 16 s (the ip server asks AAAA then A, four tries a second apart each, 8 s at the worst).
const RESOLVE_WAIT_S: u8 = 1 << 4;
/// A RESOLVE answer: status, family, address (16), TTL left in seconds (u32 BE).
const RESOLVE_LEN: usize = 22;
/// How long a connection may bring nothing before it is given up: 32 s, the ip server's own idle reaping time (WHY a deadline: ip fails a connection only when what it sent goes unanswered, and a GET's body is all receiving; a server gone silent mid-body would hold a loader for ever).
const STALL_S: u8 = 1 << 5;
/// The connections Mahere holds at once: four of the ip server's sixteen, the rest left to the device's own (attest's fgtw, the kist installer) and other apps. The loaders wait their turn, a doorbell (a condition variable), not a poll.
const CONNECTIONS: usize = 1 << 2;
/// The most data one TCP receive asks for: 256 KiB, all one reply of the ip server carries (its RECV_MAX, the kernel's largest reply); the buffer holds the state byte ahead of it.
const RECV_MOST: usize = 1 << 18;
/// The most of a request one SEND carries: a call carries 4096 bytes (the kernel's limit), less the op and the handle.
const SEND_MAX: usize = (1 << 12) - 2;

/// The ip server at grant slot `slot`, the turn for a connection, the lookups under way (one a host), and the address said last for each host (a line on the console when it changes, not at every dial).
pub struct IpNet {
    slot: u64,
    turn: Arc<(Mutex<usize>, Condvar)>,
    names: Names,
    said: Mutex<Vec<(String, u8, [u8; 16])>>,
}

impl IpNet {
    /// The ip server at `slot`, when it answers there and holds receives; else why not (said on the console by the caller, and the map runs from its vault alone).
    pub fn open(slot: u64) -> Result<IpNet, String> {
        let mut info = [0u8; 128];
        let n = sys::call(slot, &[3], &mut info);
        if n == sys::ERR {
            return Err(format!("the ip server refused the call at slot {slot} (no network grant)"));
        }
        if n < INFO_LEN as u64 {
            return Err("the ip server's INFO is short".into());
        }
        Ok(IpNet { slot, turn: Arc::new((Mutex::new(0), Condvar::new())), names: Names::new(), said: Mutex::new(Vec::new()) })
    }

    fn call(&self, req: &[u8], reply: &mut [u8]) -> io::Result<usize> {
        let n = sys::call(self.slot, req, reply);
        if n == sys::ERR { Err(io::Error::other("the ip server refused the call")) } else { Ok(n as usize) }
    }

    /// `host`'s address from the ip server's resolver (op 10, either family: IPv6 first where it has a route, the device's preference): one call held for it however many loaders ask, the others waiting on it (`Names`), and asked again while the ip server answers "still asking", so a fetch waits for its address and never fails for want of one still coming. A new or changed address said on the console with its family and TTL.
    fn address(&self, host: &str) -> io::Result<(u8, [u8; 16])> {
        // WHY: a DNS name is at most 253 bytes (the ip server refuses a longer one too); not a name to ask for.
        if host.is_empty() || host.len() > 253 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "not a host name"));
        }
        self.names.address(host, || self.ask(host))
    }

    /// One RESOLVE call for `host`, held until its lookup ends (or `RESOLVE_WAIT_S`, or a seventeenth held call makes it give way: "still asking").
    fn ask(&self, host: &str) -> Asked {
        let t = std::time::Instant::now();
        let mut q = vec![10u8, 0, RESOLVE_WAIT_S];
        q.extend_from_slice(host.as_bytes());
        let mut r = [0u8; RESOLVE_LEN];
        let n = match self.call(&q, &mut r) {
            Ok(n) => n,
            Err(_) => return Asked::Failed(io::ErrorKind::Other, "the ip server refused the call"),
        };
        // Nobody present: the ip server answers an application nothing until someone attests (ferros AUTH.md "Radios while locked"); said as that, not as DNS finding nothing (2026-11-12: "DNS gave no address" all night on a locked Pro).
        if n == 0 {
            return Asked::Failed(io::ErrorKind::PermissionDenied, "nobody present (the ip server serves applications once someone attests)");
        }
        if n < RESOLVE_LEN {
            return Asked::Failed(io::ErrorKind::Other, "the ip server has no resolver (op 10)");
        }
        let (kind, why) = match r[0] {
            0 => {
                let (fam, a, ttl) = (r[1], r[2..18].try_into().unwrap(), u32::from_be_bytes(r[18..22].try_into().unwrap()));
                let mut said = self.said.lock().unwrap();
                if !said.iter().any(|(h, f, x)| h == host && *f == fam && *x == a) {
                    said.retain(|(h, _, _)| h != host);
                    said.push((host.to_string(), fam, a));
                    println!("mahere: {host} is {} for {ttl} s ({} ms)", shown(fam, &a), t.elapsed().as_millis());
                }
                return Asked::Found(fam, a);
            }
            1 => {
                println!("mahere: {host}: DNS still asking after {} ms; asked again", t.elapsed().as_millis());
                return Asked::Asking;
            }
            2 => (io::ErrorKind::NotFound, "DNS gave no address"),
            3 => (io::ErrorKind::InvalidInput, "not a host name"),
            4 => (io::ErrorKind::AddrNotAvailable, "no address or DNS server yet"),
            5 => (io::ErrorKind::TimedOut, "DNS gave no answer"),
            6 => (io::ErrorKind::Other, "the ip server's resolver is full"),
            _ => (io::ErrorKind::Other, "the ip server's resolver answered a status it does not define"),
        };
        println!("mahere: {host}: {why} ({} ms)", t.elapsed().as_millis());
        Asked::Failed(kind, why)
    }
}

/// An address as ferros shows numbers: IPv4 in decimal (A#), IPv6 in hex groups (G#).
fn shown(fam: u8, a: &[u8; 16]) -> String {
    if fam == 4 {
        format!("A#{}.{}.{}.{}", a[0], a[1], a[2], a[3])
    } else {
        let groups: Vec<String> = a.chunks(2).map(|g| format!("{:x}", u16::from_be_bytes([g[0], g[1]]))).collect();
        format!("G#{}", groups.join(":"))
    }
}

impl Dial for IpNet {
    fn dial(&self, host: &str, port: u16) -> io::Result<Box<dyn Conn>> {
        // The address before the turn: a dialer looking one up holds none of the four connections while it waits.
        let (fam, addr) = self.address(host)?;
        let turn = Turn::take(self.turn.clone());
        let mut req = vec![4u8, fam];
        req.extend_from_slice(&addr);
        req.extend_from_slice(&port.to_be_bytes());
        let mut r = [0u8; 16];
        let n = self.call(&req, &mut r)?;
        match (n, r[0]) {
            (2.., 0) => Ok(Box::new(IpConn { slot: self.slot, handle: r[1], rx: vec![0u8; 1 + RECV_MOST], at: 1, len: 1, state: 1, closed: false, _turn: turn })),
            (_, 2) => Err(io::Error::new(io::ErrorKind::AddrNotAvailable, "no address yet")),
            (_, 4) => Err(io::Error::other("the ip server has no free connection")),
            (_, 6) => Err(io::Error::other("the ip server has no memory for another connection")),
            _ => Err(io::Error::other("the ip server refused the connection")),
        }
    }
}

/// A turn at one of Mahere's connections, given back when dropped (with its connection).
struct Turn(Arc<(Mutex<usize>, Condvar)>);

impl Turn {
    fn take(t: Arc<(Mutex<usize>, Condvar)>) -> Turn {
        let mut held = t.0.lock().unwrap();
        while *held == CONNECTIONS {
            held = t.1.wait(held).unwrap();
        }
        *held += 1;
        drop(held);
        Turn(t)
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        *self.0.0.lock().unwrap() -= 1;
        self.0.1.notify_one();
    }
}

/// One connection on the ip server: its handle, the last reply (its state byte, then data read up to `at` of `len`), the state it said last, whether the server closed.
struct IpConn {
    slot: u64,
    handle: u8,
    rx: Vec<u8>,
    at: usize,
    len: usize,
    state: u8,
    closed: bool,
    /// Dropped after the connection's close (fields drop in order, after `drop`).
    _turn: Turn,
}

impl IpConn {
    fn call(&self, req: &[u8], reply: &mut [u8]) -> io::Result<usize> {
        let n = sys::call(self.slot, req, reply);
        if n == sys::ERR { Err(io::Error::other("the ip server refused the call")) } else { Ok(n as usize) }
    }
}

impl io::Read for IpConn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // Held until data, a change of the connection's state (it opened, the server closed, it failed), or STALL_S with nothing; the reply lands in `rx` whole, its data read from it until it is used up.
        let mut ask = [6, self.handle, STALL_S, 0, 0, 0, 0];
        ask[3..7].copy_from_slice(&(RECV_MOST as u32).to_be_bytes());
        while self.at == self.len {
            if self.closed {
                return Ok(0);
            }
            let n = sys::call(self.slot, &ask, &mut self.rx);
            if n == sys::ERR {
                return Err(io::Error::other("the ip server refused the call"));
            }
            // WHY the min: the length is the kernel's, never past the buffer it was given.
            let n = (n as usize).min(self.rx.len());
            if n == 0 {
                return Err(io::Error::other("the ip server answered nothing (nobody present)"));
            }
            let state = self.rx[0];
            match state {
                5 => return Err(io::Error::new(io::ErrorKind::NotConnected, "the connection is gone")),
                4 => {
                    // Reset, or what was sent went unanswered: the ip server counts it against the address (its name asked for again after four in a row).
                    return Err(io::Error::new(io::ErrorKind::ConnectionReset, "the connection failed (reset, or unanswered)"));
                }
                _ => {}
            }
            self.at = 1;
            self.len = n;
            if n == 1 {
                // The server closed and everything has been read.
                if state == 3 {
                    self.closed = true;
                    return Ok(0);
                }
                // The connection opened: wait again. The same state with nothing: the wait ran out.
                if state != self.state {
                    self.state = state;
                    continue;
                }
                return Err(io::Error::new(io::ErrorKind::TimedOut, format!("nothing for {STALL_S} s")));
            }
            self.state = state;
        }
        let k = (self.len - self.at).min(buf.len());
        buf[..k].copy_from_slice(&self.rx[self.at..self.at + k]);
        self.at += k;
        Ok(k)
    }
}

impl io::Write for IpConn {
    /// What the connection's send buffer took (the ip server sends it as the peer's window allows); none taken is an error, never a retry loop (a request goes onto a fresh connection whose 256 KiB buffer is empty).
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let part = &buf[..buf.len().min(SEND_MAX)];
        let mut req = Vec::with_capacity(2 + part.len());
        req.extend_from_slice(&[5, self.handle]);
        req.extend_from_slice(part);
        let mut r = [0u8; 16];
        let n = self.call(&req, &mut r)?;
        match (n, r[0]) {
            (3.., 0) => match u16::from_be_bytes([r[1], r[2]]) as usize {
                0 => Err(io::Error::new(io::ErrorKind::WriteZero, "the connection's send buffer is full")),
                took => Ok(took),
            },
            _ => Err(io::Error::new(io::ErrorKind::NotConnected, "not a connection (it failed or was closed)")),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Closed at the ip server (a FIN after what was sent; the server frees it once both sides have closed), then the turn is given back.
impl Drop for IpConn {
    fn drop(&mut self) {
        let mut r = [0u8; 4];
        let _ = self.call(&[7, self.handle], &mut r);
    }
}
