//! The desktop's fleet sync (Nick 2026-10-22): a thread of its own joins the person's fleet, pulls Mahere's entries from the fleet state and pushes this device's changes (see mahere_store::sync). Pulls ride edges — attestation, the window coming to the front — never a timer; the UI applies what arrives in its tick.
//! Off unless MAHERE_SYNC is set, while the push into the live fleet state is new; MAHERE_SYNC=pull joins and pulls but only logs what it would push.

use mahere_store::sync::{Fleet, SettingEntry};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

enum Job {
    Pull,
    Push(Vec<SettingEntry>),
}

pub enum Arrived {
    /// The fleet's Mahere entries, after a pull.
    Entries(Vec<SettingEntry>),
}

pub struct FleetSync {
    tx: Sender<Job>,
    rx: Receiver<Arrived>,
    last_pull: Option<Instant>,
    /// A pull has landed: from here this device's changes go out (before, a fresh device's defaults would outrun the person's real preferences).
    pub pulled: bool,
}

/// Pulls closer together than this ride the last one.
const PULL_GAP: Duration = Duration::from_secs(30);

impl FleetSync {
    pub fn start() -> Option<FleetSync> {
        let mode = std::env::var("MAHERE_SYNC").ok()?;
        let read_only = mode == "pull";
        let (tx, jobs) = channel::<Job>();
        let (out, rx) = channel::<Arrived>();
        std::thread::Builder::new()
            .name("fleet".into())
            .spawn(move || {
                let mut fleet: Option<Fleet> = None;
                for job in jobs {
                    if fleet.is_none() {
                        match Fleet::join() {
                            Ok(f) => fleet = f,
                            Err(e) => eprintln!("fleet: join failed: {e}"),
                        }
                    }
                    let Some(f) = &fleet else {
                        eprintln!("fleet: no fleet key on this device (not attested, or not a member yet)");
                        continue;
                    };
                    match job {
                        Job::Pull => match f.pull() {
                            Ok(entries) => {
                                eprintln!("fleet: pulled {} Mahere entr{}", entries.len(), if entries.len() == 1 { "y" } else { "ies" });
                                let _ = out.send(Arrived::Entries(entries));
                            }
                            Err(e) => eprintln!("fleet: pull failed: {e}"),
                        },
                        Job::Push(entries) if read_only => {
                            for e in &entries {
                                eprintln!("fleet: would push {} ({}, updated {})", e.key, if e.tombstone { "deleted" } else { "set" }, e.updated);
                            }
                        }
                        Job::Push(entries) => {
                            let n = entries.len();
                            match f.push(entries) {
                                Ok(()) => eprintln!("fleet: pushed {n} entr{}", if n == 1 { "y" } else { "ies" }),
                                Err(e) => eprintln!("fleet: push failed: {e}"),
                            }
                        }
                    }
                }
            })
            .ok()?;
        Some(FleetSync { tx, rx, last_pull: None, pulled: false })
    }

    /// Ask for the fleet's entries, unless a pull went out moments ago.
    pub fn pull(&mut self) {
        if self.last_pull.is_some_and(|t| t.elapsed() < PULL_GAP) {
            return;
        }
        self.last_pull = Some(Instant::now());
        let _ = self.tx.send(Job::Pull);
    }

    pub fn push(&self, entries: Vec<SettingEntry>) {
        if !entries.is_empty() {
            let _ = self.tx.send(Job::Push(entries));
        }
    }

    pub fn take(&mut self) -> Vec<Arrived> {
        self.rx.try_iter().collect()
    }
}
