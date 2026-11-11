//! Mahere's way to the network on ferros: connections over the ip server's client TCP (ferros payloads/ip, its `udp` endpoint: 1 SEND, 2 RECV, 3 INFO, 4 CONNECT, 5 SEND, 6 RECV, 7 CLOSE), the server's address from the DNS server ip reports, for mahere-engine's plain HTTP ([`Dial`]). Nick 2026-11-11: "we have tcp and http so let's get the full meal deal wired up with mahere".
//!
//! Every wait is a receive the ip server holds until something comes (the wait byte of its ops 2 and 6), never a poll (Nick 2026-11-10: "No polling!"): a loader thread blocks in the call as it blocks on a socket elsewhere. The two deadlines are real ones: a DNS question lost on the way is asked again after a second, and a connection that brings nothing for 32 s is given up.
//! The ip server's TCP since 2026-11-11 (Nick: "sounds like it needs re-engineered"): sixteen connections, a 1 MiB window each, receives of up to 256 KiB; Mahere takes four connections at once and reads each a big reply at a time.

use crate::ferros::sys;
use mahere_engine::plain_http::{Conn, Dial};
use std::io;
use std::sync::{Arc, Condvar, Mutex};

/// The ip server's INFO reply: 71 bytes (addresses, flags, the DNS servers).
const INFO_LEN: usize = 71;
/// The port DNS questions leave from and their answers come to: the ip server hands a port's datagrams to whoever asks for that port, so each process has its own (httptest 53053, fgtw and the kist installer 53054).
const DNS_PORT: u16 = 53055;
/// How long one DNS question waits for its answer (WHY a deadline: a datagram lost on the way is never answered), and how many times it is asked.
const DNS_WAIT_S: u8 = 1;
const DNS_TRIES: usize = 1 << 2;
/// How long a connection may bring nothing before it is given up: 32 s, the ip server's own idle reaping time (WHY a deadline: ip fails a connection only when what it sent goes unanswered, and a GET's body is all receiving; a server gone silent mid-body would hold a loader for ever).
const STALL_S: u8 = 1 << 5;
/// The connections Mahere holds at once: four of the ip server's sixteen, the rest left to the device's own (attest's fgtw, the kist installer) and other apps. The loaders wait their turn, a doorbell (a condition variable), not a poll.
const CONNECTIONS: usize = 1 << 2;
/// The most data one TCP receive asks for: 256 KiB, all one reply of the ip server carries (its RECV_MAX, the kernel's largest reply); the buffer holds the state byte ahead of it.
const RECV_MOST: usize = 1 << 18;
/// A DNS answer's reply: the datagram's 21 bytes of family, address and ports, then a message of at most 1472 bytes.
const DNS_REPLY: usize = 1 << 11;
/// The most of a request one SEND carries: a call carries 4096 bytes (the kernel's limit), less the op and the handle.
const SEND_MAX: usize = (1 << 12) - 2;

/// The ip server at grant slot `slot`, the address last resolved, and the turn for a connection.
pub struct IpNet {
    slot: u64,
    found: Arc<Mutex<Option<(String, u8, [u8; 16])>>>,
    turn: Arc<(Mutex<usize>, Condvar)>,
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
        Ok(IpNet { slot, found: Arc::new(Mutex::new(None)), turn: Arc::new((Mutex::new(0), Condvar::new())) })
    }

    fn call(&self, req: &[u8], reply: &mut [u8]) -> io::Result<usize> {
        let n = sys::call(self.slot, req, reply);
        if n == sys::ERR { Err(io::Error::other("the ip server refused the call")) } else { Ok(n as usize) }
    }

    /// `host`'s address: the one found last when it is the same host, else asked of DNS (IPv6 first, the device's preference: ferros payloads/ip, owner 2026-10-29; then IPv4, also when the IPv6 question could not be asked), said on the console.
    fn address(&self, host: &str) -> io::Result<(u8, [u8; 16])> {
        let mut found = self.found.lock().unwrap();
        if let Some((h, fam, a)) = &*found {
            if h == host {
                return Ok((*fam, *a));
            }
        }
        let t = std::time::Instant::now();
        let mut info = [0u8; 128];
        let n = self.call(&[3], &mut info)?;
        if n < INFO_LEN {
            return Err(io::Error::other("the ip server's INFO is short"));
        }
        let flags = info[50];
        let mut dns4 = [0u8; 16];
        dns4[..4].copy_from_slice(&info[51..55]);
        let dns6: [u8; 16] = info[55..71].try_into().unwrap();
        let has4 = flags & 1 != 0 && dns4 != [0; 16];
        let mut got = None;
        if flags & 2 != 0 {
            let via = if dns6 != [0; 16] { Some((6, dns6)) } else if has4 { Some((4, dns4)) } else { None };
            if let Some((fam, server)) = via {
                got = self.query(fam, server, host, 28).ok().flatten().map(|a| (6u8, a));
            }
        }
        if got.is_none() && has4 {
            got = self.query(4, dns4, host, 1)?.map(|a| (4u8, a));
        }
        let Some((fam, a)) = got else {
            let why = if flags & 3 == 0 { "no address yet" } else { "DNS gave no address" };
            println!("mahere: {host}: {why} ({} ms)", t.elapsed().as_millis());
            return Err(io::Error::new(io::ErrorKind::NotFound, why));
        };
        println!("mahere: {host} is {} ({} ms)", shown(fam, &a), t.elapsed().as_millis());
        *found = Some((host.to_string(), fam, a));
        Ok((fam, a))
    }

    /// The first record of type `qtype` (1 A, 28 AAAA) for `host` from the DNS server `server` (family `fam`, IPv4 in its first 4 bytes): Some(address), None when the server says there is none or never answers.
    fn query(&self, fam: u8, server: [u8; 16], host: &str, qtype: u16) -> io::Result<Option<[u8; 16]>> {
        // WHY: a DNS name is at most 253 bytes in labels of 1 to 63; anything else is not a name to ask for.
        if host.is_empty() || host.len() > 253 || host.split('.').any(|l| l.is_empty() || l.len() > 63) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "not a host name"));
        }
        // The message's id from the kernel's entropy (the spawn's entropy grant): an answer for another question, or one forged without seeing ours, is passed over.
        let mut r = [0u8; 2];
        getrandom::getrandom(&mut r).map_err(|e| io::Error::other(format!("no entropy for a DNS id: {e}")))?;
        let id = u16::from_le_bytes(r);
        let mut q = vec![1u8, fam];
        q.extend_from_slice(&server);
        q.extend_from_slice(&53u16.to_be_bytes());
        q.extend_from_slice(&DNS_PORT.to_be_bytes());
        // The message: id, recursion desired, one question.
        q.extend_from_slice(&id.to_be_bytes());
        q.extend_from_slice(&[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
        for label in host.split('.') {
            q.push(label.len() as u8);
            q.extend_from_slice(label.as_bytes());
        }
        q.push(0);
        q.extend_from_slice(&qtype.to_be_bytes());
        q.extend_from_slice(&[0, 1]);
        let mut status = [0u8; 16];
        let mut out = vec![0u8; DNS_REPLY];
        let hold = [2, (DNS_PORT >> 8) as u8, DNS_PORT as u8, DNS_WAIT_S];
        for _ in 0..DNS_TRIES {
            // Sent (0), or the next hop still being resolved (1: the question is asked again after its wait); no address (2) or refused (3) end it.
            let n = self.call(&q, &mut status)?;
            if n >= 1 && status[0] >= 2 {
                return Err(io::Error::other(if status[0] == 2 { "no address yet" } else { "the ip server refused the question" }));
            }
            // Answers until ours, or the wait runs out (an empty reply). Another's answer on the port (an old try's) is passed over; each is held at most the wait. A datagram's reply is its 21 bytes of family, address and ports, then the message.
            loop {
                let n = self.call(&hold, &mut out)?;
                if n == 0 {
                    break;
                }
                if let Some(answer) = answer(&out[21..n], id, qtype) {
                    return Ok(answer);
                }
            }
        }
        Ok(None)
    }
}

/// The DNS message `m` read for message `id`: None when it is not that message's answer; Some(None) when it is and has no record of type `qtype` (none, or the name does not exist); Some(Some(address)) with the first one. Every read is checked (WHY: the message is the network's, any length and any content).
fn answer(m: &[u8], id: u16, qtype: u16) -> Option<Option<[u8; 16]>> {
    if m.len() < 12 || m[0..2] != id.to_be_bytes() || m[2] & 0x80 == 0 {
        return None;
    }
    if m[3] & 15 != 0 {
        return Some(None);
    }
    let count = |k: usize| u16::from_be_bytes([m[k], m[k + 1]]) as usize;
    let (questions, answers) = (count(4), count(6));
    // A name: labels to the zero, or a pointer that ends it.
    let skip = |mut i: usize| -> Option<usize> {
        loop {
            let b = *m.get(i)?;
            if b == 0 {
                return Some(i + 1);
            }
            if b & 0xC0 == 0xC0 {
                return Some(i + 2);
            }
            i += 1 + b as usize;
        }
    };
    let mut i = 12;
    for _ in 0..questions {
        i = skip(i)? + 4;
    }
    for _ in 0..answers {
        i = skip(i)?;
        let head = m.get(i..i + 10)?;
        let ty = u16::from_be_bytes([head[0], head[1]]);
        let len = u16::from_be_bytes([head[8], head[9]]) as usize;
        let data = m.get(i + 10..i + 10 + len)?;
        if ty == qtype && (len == 4 || len == 16) {
            let mut a = [0u8; 16];
            a[..len].copy_from_slice(data);
            return Some(Some(a));
        }
        i += 10 + len;
    }
    Some(None)
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
        let turn = Turn::take(self.turn.clone());
        let (fam, addr) = self.address(host)?;
        let mut req = vec![4u8, fam];
        req.extend_from_slice(&addr);
        req.extend_from_slice(&port.to_be_bytes());
        let mut r = [0u8; 16];
        let n = self.call(&req, &mut r)?;
        match (n, r[0]) {
            (2.., 0) => Ok(Box::new(IpConn { slot: self.slot, handle: r[1], rx: vec![0u8; 1 + RECV_MOST], at: 1, len: 1, state: 1, closed: false, found: self.found.clone(), _turn: turn })),
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
    found: Arc<Mutex<Option<(String, u8, [u8; 16])>>>,
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
                    // Reset, or what was sent went unanswered: the address is asked for again next time.
                    *self.found.lock().unwrap() = None;
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
