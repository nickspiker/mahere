//! mahere's database: the kete/manifestus stack photon rides — a crash-proof
//! keyed object store (copy-on-write HAMT over mirrored VSF-sealed rings)
//! with per-key encryption. Three verbs: write, read, delete; a kill at any
//! byte leaves every entry committed-or-absent.
//!
//! mahere uses the device scope only (no identity layer): the vault opens
//! from a device secret minted on first run. Today it holds the session
//! (camera + sun, restored on launch) and recorded GPS tracks in chunked
//! entries; the dymaxion tile repack lands here next — `demcell|{prefix}`
//! and `featcell|{prefix}` keys are the planned tile store.

use std::sync::Arc;

pub use kete::{FlatStorage, StorageError};
use rand::RngCore;
use vsf::types::{EtType, Tensor};
use vsf::VsfType;

pub const APP: kete::App<'static> = kete::App { id: "mahere", dir: "mahere" };

/// Open THE mahere vault (shared registry underneath — concurrent opens
/// receive the same engine). `base`: Android passes filesDir; desktop passes
/// None (XDG config/data dirs).
pub fn open(base: Option<&str>) -> Result<Arc<FlatStorage>, String> {
    let secret_path = match base {
        Some(b) => {
            kete::set_vault_dirs_override(format!("{b}/vault"), format!("{b}/vault-shadow"));
            format!("{b}/device.key")
        }
        None => {
            let dir = dirs_config().ok_or("no config dir")?;
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            format!("{dir}/device.key")
        }
    };
    let secret = load_or_mint_secret(&secret_path)?;
    FlatStorage::open_device_shared(APP, secret).map_err(|e| format!("vault open: {e:?}"))
}

fn dirs_config() -> Option<String> {
    std::env::var("XDG_CONFIG_HOME")
        .ok()
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.config")))
        .map(|c| format!("{c}/mahere"))
}

/// The vault's value here is crash-proofness, not secrecy — but kete wants a
/// device secret, so mint 32 random bytes once and keep them beside the
/// rings.
fn load_or_mint_secret(path: &str) -> Result<[u8; 32], String> {
    if let Ok(bytes) = std::fs::read(path) {
        if bytes.len() == 32 {
            let mut s = [0u8; 32];
            s.copy_from_slice(&bytes);
            return Ok(s);
        }
    }
    let mut s = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut s);
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, s).map_err(|e| format!("{path}: {e}"))?;
    Ok(s)
}

// ==================== SESSION ====================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Session {
    pub lat: f64,
    pub lon: f64,
    pub ppd: f64,
    pub bearing: f64,
    pub sun_az: f64,
    pub sun_alt: f64,
}

// ==================== KEYS ====================

/// kete addresses an entry by hashing a logical key. Ours are VSF values —
/// the type tags are the domain separation (`d` names a domain, `u` a
/// depth or index, `wm` a world cell, `e` an instant) — flattened, hashed,
/// and spelled base64url for kete's string parameter. No delimiter and no
/// numeral ever appears in a key.
pub fn vault_key(parts: &[VsfType]) -> String {
    let mut bytes = Vec::new();
    for p in parts {
        bytes.extend(p.flatten());
    }
    mahere_tiles::base64url(blake3::hash(&bytes).as_bytes())
}

fn name(s: &str) -> VsfType {
    VsfType::d(s.to_string())
}

fn session_key() -> String {
    vault_key(&[name("session")])
}

/// Eagle Time now, as the `e` value tracks are keyed by.
fn et_now() -> i64 {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    vsf::types::eagle_time::from_unix_ns(d.as_secs() as i64, d.subsec_nanos())
}

fn track_chunk_key(start: i64, chunk: u32) -> String {
    vault_key(&[name("track"), VsfType::e(EtType::e6(start)), VsfType::u(chunk as usize, false)])
}

fn track_count_key(start: i64) -> String {
    vault_key(&[name("track"), VsfType::e(EtType::e6(start)), name("chunks")])
}

pub fn save_session(store: &FlatStorage, s: &Session) -> Result<(), StorageError> {
    let t = VsfType::t_f6(Tensor::new(
        vec![6],
        vec![s.lat, s.lon, s.ppd, s.bearing, s.sun_az, s.sun_alt],
    ));
    store.write_device(&session_key(), &t.flatten())
}

pub fn load_session(store: &FlatStorage) -> Option<Session> {
    let bytes = store.read_device(&session_key()).ok()??;
    let mut ptr = 0usize;
    let v = vsf::parse(&bytes, &mut ptr).ok()?;
    let d = tensor_f64(v)?;
    if d.len() != 6 {
        return None;
    }
    Some(Session { lat: d[0], lon: d[1], ppd: d[2], bearing: d[3], sun_az: d[4], sun_alt: d[5] })
}

/// Width-agnostic float-array read, per VSF doctrine.
fn tensor_f64(v: VsfType) -> Option<Vec<f64>> {
    Some(match v {
        VsfType::t_f5(t) => t.data.into_iter().map(|x| x as f64).collect(),
        VsfType::t_f6(t) => t.data,
        VsfType::v_f5(t) => t.data.into_iter().map(|x| x as f64).collect(),
        VsfType::v_f6(t) => t.data,
        _ => return None,
    })
}

// ==================== TRACKS ====================

/// Chunked GPS track recorder: (lat, lon, elevation m, unix seconds) per
/// fix, flushed every CHUNK fixes so a crash costs at most one chunk and a
/// day's hike never rewrites more than a small object per flush.
pub struct TrackRecorder {
    store: Arc<FlatStorage>,
    /// Eagle Time of the first fix's session — the track's identity.
    start: i64,
    buf: Vec<f64>,
    chunk: u32,
}

const CHUNK_FIXES: usize = 64;

impl TrackRecorder {
    pub fn new(store: Arc<FlatStorage>) -> TrackRecorder {
        TrackRecorder { store, start: et_now(), buf: Vec::new(), chunk: 0 }
    }

    pub fn on_fix(&mut self, lat: f64, lon: f64, elevation_m: f64) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.buf.extend_from_slice(&[lat, lon, elevation_m, ts]);
        if self.buf.len() >= CHUNK_FIXES * 4 {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        let key = track_chunk_key(self.start, self.chunk);
        let t = VsfType::t_f6(Tensor::new(vec![self.buf.len()], std::mem::take(&mut self.buf)));
        if self.store.write_device(&key, &t.flatten()).is_ok() {
            self.chunk += 1;
            // VSF all the way down: the counter is a u, not naked LE bytes.
            let _ = self.store.write_device(
                &track_count_key(self.start),
                &VsfType::u(self.chunk as usize, false).flatten(),
            );
        }
    }

    /// Chunks written so far (for readers listing a live session).
    pub fn chunks(&self) -> u32 {
        self.chunk
    }

    /// The track's identity: Eagle Time at recording start.
    pub fn start(&self) -> i64 {
        self.start
    }
}

/// One recorded chunk's fixes (lat, lon, elevation m, unix s) × n.
pub fn track_chunk(store: &FlatStorage, start: i64, chunk: u32) -> Option<Vec<f64>> {
    let bytes = store.read_device(&track_chunk_key(start, chunk)).ok()??;
    let mut ptr = 0usize;
    tensor_f64(vsf::parse(&bytes, &mut ptr).ok()?)
}


/// Chunk count for a recorded track session (width-agnostic VSF read).
pub fn track_chunks(store: &FlatStorage, start: i64) -> Option<u32> {
    let bytes = store.read_device(&track_count_key(start)).ok()??;
    let mut ptr = 0usize;
    let v = vsf::parse(&bytes, &mut ptr).ok()?;
    v.as_u64().and_then(|n| u32::try_from(n).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One serial test: set_vault_dirs_override is process-global, so
    /// parallel vault tests would race each other's directories.
    #[test]
    fn vault_round_trips_and_persists() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("mahere-store-{nanos}"));
        let base = dir.to_string_lossy().into_owned();
        std::fs::create_dir_all(&dir).unwrap();

        let store = open(Some(&base)).expect("open");
        let s = Session {
            lat: 46.2024,
            lon: -121.4909,
            ppd: 48_000.0,
            bearing: 0.7,
            sun_az: 315.0,
            sun_alt: 40.0,
        };
        save_session(&store, &s).unwrap();
        assert_eq!(load_session(&store), Some(s));

        // Recorder pattern: big chunk blobs interleaved with rapid counter
        // overwrites — the exact shape that looked flaky before the test
        // harness itself was fixed (stale reused dirs + global override).
        let mut rec = TrackRecorder::new(store.clone());
        let key = rec.start();
        for i in 0..200 {
            rec.on_fix(46.2 + i as f64 * 1e-5, -121.49, 2000.0 + i as f64);
        }
        rec.flush();
        assert_eq!(rec.chunks(), 4);
        assert_eq!(track_chunks(&store, key), Some(4));
        for c in 0..4u32 {
            assert!(track_chunk(&store, key, c).is_some(), "chunk {c} missing");
        }

        // Persistence across a real reopen: drop every handle, open again,
        // read everything back.
        drop(rec);
        drop(store);
        let store2 = open(Some(&base)).expect("reopen");
        assert_eq!(load_session(&store2), Some(s), "session must survive reopen");
        assert_eq!(track_chunks(&store2, key), Some(4), "track counter must survive reopen");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ==================== CELL CACHE ====================

/// The vault as the on-device cell tier: every cell fetched from the bucket
/// is written through here and served from here afterwards (offline
/// included). Keyed by the VSF values (`d` cell, `u` depth, `wm` cell) through [`vault_key`], so cells, session and tracks share the
/// vault with the type system as the namespace.
pub struct VaultCells(pub Arc<FlatStorage>);

fn cell_key(key: mahere_tiles::CellKey) -> String {
    vault_key(&[
        name("cell"),
        VsfType::u(key.depth as usize, false),
        VsfType::wm(vsf::types::WorldCell::from_raw(key.raw())),
    ])
}

impl mahere_engine::residency::CellStore for VaultCells {
    fn get(&self, key: mahere_tiles::CellKey) -> Option<Vec<u8>> {
        self.0.read_device(&cell_key(key)).ok().flatten()
    }
}

impl mahere_engine::residency::CellCache for VaultCells {
    fn put(&self, key: mahere_tiles::CellKey, bytes: &[u8]) {
        let _ = self.0.write_device(&cell_key(key), bytes);
    }
}
