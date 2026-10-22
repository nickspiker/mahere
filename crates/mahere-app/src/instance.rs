//! One Mahere at a time (after Photon's): two copies on the same vault would race its rings and the session they both write.
//! The running copy holds an exclusive lock on a file beside the vault (the OS lets it go when the process dies, so a crash leaves nothing stale) and listens on the loopback for a later launch, whose only word is `show`: it brings the running window forward and exits.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;

/// Held for the life of the process: the lock, and the listener a later launch knocks on.
pub struct Instance {
    _lock: std::fs::File,
    listener: Option<TcpListener>,
}

/// Take the lock, or hand off to the copy that has it and exit. Without a config directory there is no vault to protect, and every launch runs.
pub fn claim() -> Option<Instance> {
    let dir = mahere_store::config_dir()?;
    let _ = std::fs::create_dir_all(&dir);
    let lock = std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(dir.join("instance.lock")).ok()?;
    if lock.try_lock().is_err() {
        if knock(&dir) {
            eprintln!("Mahere is already running; brought it forward.");
            std::process::exit(0);
        }
        eprintln!("Mahere is already running for {} and did not answer.", dir.display());
        std::process::exit(1);
    }
    // The port beside the lock, not in it: Windows keeps a locked file's bytes from other processes.
    let listener = TcpListener::bind(("127.0.0.1", 0)).ok();
    if let Some(port) = listener.as_ref().and_then(|l| l.local_addr().ok()).map(|a| a.port()) {
        let _ = std::fs::write(dir.join("instance.port"), port.to_string());
    }
    Some(Instance { _lock: lock, listener })
}

/// A later launch: tell the running copy to show itself. `false` when nobody answered.
fn knock(dir: &Path) -> bool {
    let Some(port) = std::fs::read_to_string(dir.join("instance.port")).ok().and_then(|p| p.trim().parse::<u16>().ok()) else {
        return false;
    };
    TcpStream::connect(("127.0.0.1", port)).and_then(|mut s| s.write_all(b"show\n")).is_ok()
}

impl Instance {
    /// Answer later launches on a thread of its own, each `show` woken through to the UI thread as `on_show`.
    pub fn serve(&mut self, on_show: impl Fn() + Send + 'static) {
        let Some(listener) = self.listener.take() else { return };
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                // A connect that never writes (a port scan) gives up here instead of holding the loop.
                let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(500)));
                let mut buf = [0u8; 16];
                let n = s.read(&mut buf).unwrap_or(0);
                if buf[..n].starts_with(b"show") {
                    on_show();
                }
            }
        });
    }
}
