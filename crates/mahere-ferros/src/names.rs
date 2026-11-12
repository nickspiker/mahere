//! One lookup a host at a time, however many of Mahere's loaders ask for its address (2026-11-12, the 8a: about 240 fetches failed "DNS still asking" in Mahere's first second).
//! The ip server holds sixteen RESOLVE calls at once and answers the oldest "still asking" when a seventeenth comes; Mahere's twenty loaders each held one, so every new one pushed out another, whose fetch failed and whose loader asked again for its next cell.
//! Now the first loader to ask for a host holds the one call and the others wait on it (a condition variable, a doorbell, never a poll) and take its answer. "Still asking" is no answer: the call is made again, held again, until the lookup ends. Kept apart from the ip server's calls so the host tests drive it with a resolver of their own.

use std::io;
use std::sync::{Condvar, Mutex};

/// What one RESOLVE call answered.
pub enum Asked {
    /// The address: its family (4 or 6) and the address (IPv4 in the first four bytes).
    Found(u8, [u8; 16]),
    /// Status 1: the call's wait ran out, or a seventeenth held call made it give way; the lookup goes on in the ip server, so it is asked again.
    Asking,
    /// The lookup ended with no address, or nothing can be asked (no DNS server yet, nobody present): the fetch fails with this.
    Failed(io::ErrorKind, &'static str),
}

/// A lookup's outcome, kept for the loaders that waited on it.
type Outcome = Result<(u8, [u8; 16]), (io::ErrorKind, &'static str)>;

/// A host asked for: whether a loader holds its call now, how many of its lookups have ended, and the last one's outcome.
struct Host {
    name: String,
    asking: bool,
    ended: u64,
    last: Outcome,
}

/// The hosts asked for (Mahere's bucket and the few servers beside it: a list, searched), and the doorbell rung when a lookup ends.
pub struct Names {
    hosts: Mutex<Vec<Host>>,
    ended: Condvar,
}

impl Names {
    pub fn new() -> Names {
        Names { hosts: Mutex::new(Vec::new()), ended: Condvar::new() }
    }

    /// `host`'s address: `ask` (one held RESOLVE call) made by this caller when nobody is asking for it, made again while it answers "still asking"; else the lookup under way waited for and its outcome taken. Nothing is kept past the lookup: the next caller asks again, and the ip server's cache answers it at once while the address holds.
    pub fn address(&self, host: &str, ask: impl Fn() -> Asked) -> io::Result<(u8, [u8; 16])> {
        let mut hosts = self.hosts.lock().unwrap();
        // The list only grows, so a host's index holds while the lock is let go.
        let i = match hosts.iter().position(|h| h.name == host) {
            Some(i) => i,
            None => {
                hosts.push(Host { name: host.to_string(), asking: false, ended: 0, last: Err((io::ErrorKind::Other, "not asked yet")) });
                hosts.len() - 1
            }
        };
        if hosts[i].asking {
            let ended = hosts[i].ended;
            while hosts[i].ended == ended {
                hosts = self.ended.wait(hosts).unwrap();
            }
            return hosts[i].last.map_err(|(k, why)| io::Error::new(k, why));
        }
        hosts[i].asking = true;
        drop(hosts);
        let last = loop {
            match ask() {
                Asked::Found(fam, a) => break Ok((fam, a)),
                Asked::Asking => {}
                Asked::Failed(k, why) => break Err((k, why)),
            }
        };
        let mut hosts = self.hosts.lock().unwrap();
        let h = &mut hosts[i];
        (h.asking, h.ended, h.last) = (false, h.ended + 1, last);
        self.ended.notify_all();
        last.map_err(|(k, why)| io::Error::new(k, why))
    }
}
