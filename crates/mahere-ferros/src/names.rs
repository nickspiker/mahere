//! The addresses Mahere's connections dial on ferros, found by DNS and shared (`net`, the ip server's client TCP). Mahere on ferros, 2026-11-11: a failed connection threw the address away, the next connect asked DNS again holding the lock the whole time, and every loader waited behind lookups of 1.2 to 9.8 s.
//!
//! A lookup runs outside the lock, one at a time (the answers come to one port); a dialer with no address for its host waits on it (a doorbell, `landed`), and one with an address gone stale keeps using it while it runs. An address is asked for again when its answer's TTL runs out or after [`FAILURES`] failed connections in a row, never for one. Pure over its lookup, so the host's tests drive it.

use std::io;
use std::sync::{Condvar, Mutex, MutexGuard};

/// What a lookup finds: the address's family (4 or 6), the address (IPv4 in its first 4 bytes), and the answer's TTL in seconds.
pub type Found = (u8, [u8; 16], u32);

/// Failed connections in a row to an address before it is asked for again (its answer's TTL asks sooner when it runs out first): one failure is the network's, not the address's.
pub const FAILURES: u32 = 1 << 2;

/// A host's address as DNS answered it: until when the answer holds (its TTL), and the connections to it that failed in a row since one worked.
struct Known {
    host: String,
    fam: u8,
    addr: [u8; 16],
    /// The answer's TTL, seconds.
    ttl: u32,
    until: std::time::Instant,
    failures: u32,
}

/// The addresses found, shared by every connection, and the one lookup under way.
pub struct Names {
    state: Mutex<NameState>,
    landed: Condvar,
}

#[derive(Default)]
struct NameState {
    known: Vec<Known>,
    /// The host a lookup is under way for.
    asking: Option<String>,
    /// Lookups finished, so a waiter knows the one it waited on is done; and the last one's failure, for its waiters to say.
    finished: u64,
    failed: Option<(String, String)>,
}

impl Names {
    pub fn new() -> Names {
        Names { state: Mutex::new(NameState::default()), landed: Condvar::new() }
    }

    /// A connection to `host` failed (reset, or what it sent went unanswered): counted, and past [`FAILURES`] in a row its address is asked for again (and used meanwhile).
    pub fn failed(&self, host: &str) {
        if let Some(k) = self.state.lock().unwrap().known.iter_mut().find(|k| k.host == host) {
            k.failures += 1;
        }
    }

    /// A connection to `host` brought something: its address works.
    pub fn worked(&self, host: &str) {
        if let Some(k) = self.state.lock().unwrap().known.iter_mut().find(|k| k.host == host) {
            k.failures = 0;
        }
    }

    /// `host`'s address: the one found when it still holds (inside its TTL, and fewer than [`FAILURES`] connections to it failed in a row); one that no longer holds is used while `lookup` asks for it again; with none, asked with `lookup`, or the lookup under way waited on. The lock is never held over a lookup.
    pub fn address(&self, host: &str, lookup: &dyn Fn(&str) -> io::Result<Found>) -> io::Result<(u8, [u8; 16])> {
        let mut s = self.state.lock().unwrap();
        let since = s.finished;
        loop {
            let now = std::time::Instant::now();
            match s.known.iter().find(|k| k.host == host).map(|k| (k.fam, k.addr, now < k.until && k.failures < FAILURES)) {
                Some((fam, a, true)) => return Ok((fam, a)),
                // Stale while a lookup runs: the last good address meanwhile.
                Some((fam, a, false)) if s.asking.is_some() => return Ok((fam, a)),
                Some((fam, a, false)) => {
                    let got = self.ask(s, host, lookup);
                    // Not found again: the last good address, until a lookup finds another.
                    return Ok(got.map_or((fam, a), |(fam, a, _)| (fam, a)));
                }
                None if s.asking.is_some() => s = self.landed.wait(s).unwrap(),
                // The lookup waited on was this host's and found nothing: its failure, not another lookup straight after it.
                None if s.finished != since && s.failed.as_ref().is_some_and(|(h, _)| h == host) => {
                    let why = s.failed.as_ref().map_or_else(String::new, |(_, e)| e.clone());
                    return Err(io::Error::new(io::ErrorKind::NotFound, why));
                }
                None => return self.ask(s, host, lookup).map(|(fam, a, _)| (fam, a)),
            }
        }
    }

    /// `host` asked with the lock let go for the lookup, then what it found kept: its address in place of the host's last, or, when it found none, the last kept on for another TTL (it worked before; a lookup that failed says nothing against it), and the failure for those waiting on it, woken.
    fn ask(&self, mut s: MutexGuard<'_, NameState>, host: &str, lookup: &dyn Fn(&str) -> io::Result<Found>) -> io::Result<Found> {
        s.asking = Some(host.to_string());
        drop(s);
        let got = lookup(host);
        let mut s = self.state.lock().unwrap();
        s.asking = None;
        s.finished += 1;
        let now = std::time::Instant::now();
        match &got {
            Ok((fam, addr, ttl)) => {
                s.known.retain(|k| k.host != host);
                s.known.push(Known { host: host.to_string(), fam: *fam, addr: *addr, ttl: *ttl, until: now + std::time::Duration::from_secs(*ttl as u64), failures: 0 });
                s.failed = None;
            }
            Err(e) => {
                if let Some(k) = s.known.iter_mut().find(|k| k.host == host) {
                    k.until = now + std::time::Duration::from_secs(k.ttl as u64);
                    k.failures = 0;
                }
                s.failed = Some((host.to_string(), e.to_string()));
            }
        }
        self.landed.notify_all();
        got
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    const A: [u8; 16] = [192, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const B: [u8; 16] = [192, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    /// Four dialers at once with no address: one lookup, the others wait on it and get its address.
    #[test]
    fn one_lookup_for_many_dialers() {
        let names = Arc::new(Names::new());
        let asked = Arc::new(AtomicUsize::new(0));
        let dialers: Vec<_> = (0..4)
            .map(|_| {
                let (names, asked) = (names.clone(), asked.clone());
                std::thread::spawn(move || {
                    names.address("h", &|_| {
                        asked.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(1 << 7));
                        Ok((4, A, 300))
                    })
                })
            })
            .collect();
        for d in dialers {
            assert_eq!(d.join().unwrap().unwrap(), (4, A));
        }
        assert_eq!(asked.load(Ordering::SeqCst), 1);
    }

    /// A failed lookup is said to those who waited on it, not asked again by each.
    #[test]
    fn a_failed_lookup_is_said_to_its_waiters() {
        let names = Arc::new(Names::new());
        let asked = Arc::new(AtomicUsize::new(0));
        let dialers: Vec<_> = (0..4)
            .map(|_| {
                let (names, asked) = (names.clone(), asked.clone());
                std::thread::spawn(move || {
                    names.address("h", &|_| {
                        asked.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(1 << 7));
                        Err(io::Error::new(io::ErrorKind::NotFound, "DNS gave no address"))
                    })
                })
            })
            .collect();
        for d in dialers {
            assert!(d.join().unwrap().is_err());
        }
        assert_eq!(asked.load(Ordering::SeqCst), 1);
    }

    /// One failed connection keeps the address; several in a row ask again, and while that lookup runs the others dial the last good address without waiting.
    #[test]
    fn failures_in_a_row_ask_again_without_holding_anyone() {
        let names = Arc::new(Names::new());
        assert_eq!(names.address("h", &|_| Ok((4, A, 300))).unwrap(), (4, A));
        names.failed("h");
        assert_eq!(names.address("h", &|_| panic!("asked again after one failure")).unwrap(), (4, A));
        for _ in 0..FAILURES {
            names.failed("h");
        }
        let slow = {
            let names = names.clone();
            std::thread::spawn(move || {
                names.address("h", &|_| {
                    std::thread::sleep(Duration::from_millis(1 << 8));
                    Ok((4, B, 300))
                })
            })
        };
        std::thread::sleep(Duration::from_millis(1 << 5));
        let t = std::time::Instant::now();
        assert_eq!(names.address("h", &|_| panic!("a second lookup at once")).unwrap(), (4, A), "the last good address meanwhile");
        assert!(t.elapsed() < Duration::from_millis(1 << 6), "not held behind the lookup");
        assert_eq!(slow.join().unwrap().unwrap(), (4, B));
        assert_eq!(names.address("h", &|_| panic!("the new address holds")).unwrap(), (4, B));
    }

    /// A refresh that finds nothing keeps the last good address, for another TTL.
    #[test]
    fn a_failed_refresh_keeps_the_address() {
        let names = Names::new();
        names.address("h", &|_| Ok((4, A, 0))).unwrap();
        assert_eq!(names.address("h", &|_| Err(io::Error::other("lost"))).unwrap(), (4, A));
        // The TTL of 0 asks every time; a host with a working address is never refused for a lookup that failed.
        assert_eq!(names.address("h", &|_| Err(io::Error::other("lost"))).unwrap(), (4, A));
        names.worked("h");
    }
}
