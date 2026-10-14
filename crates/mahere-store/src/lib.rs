//! mahere's database: the kete/manifestus stack photon rides — a crash-proof keyed object store (copy-on-write HAMT over mirrored VSF-sealed rings) with per-key encryption. Three verbs: write, read, delete; a kill at any byte leaves every entry committed-or-absent.
//!
//! mahere uses the device scope only (no identity layer): the vault opens from a device secret minted on first run. Today it holds the session (camera + sun, restored on launch) and recorded GPS tracks in chunked entries; the dymaxion tile repack lands here next — `demcell|{prefix}`
//! and `featcell|{prefix}` keys are the planned tile store.

use std::sync::Arc;

pub use kete::{FlatStorage, StorageError};
use vsf::VsfBuilder;
use rand::RngCore;
use vsf::types::{EtType, Tensor};
use vsf::VsfType;

pub const APP: kete::App<'static> = kete::App { id: "mahere", dir: "mahere" };

/// Open THE mahere vault (shared registry underneath — concurrent opens receive the same engine). `base`: Android passes filesDir; desktop passes None (XDG config/data dirs).
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

/// The vault's value here is crash-proofness, not secrecy — but kete wants a device secret, so mint 32 random bytes once and keep them beside the rings.
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

/// kete addresses an entry by hashing a logical key. Ours are VSF values — the type tags are the domain separation (`d` names a domain, `u` a depth or index, `wm` a world cell, `e` an instant) — flattened, hashed, and spelled base64url for kete's string parameter. No delimiter and no numeral ever appears in a key.
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

/// The app's settings: the cache budget in bytes, the layer mask as bits, and the mode flags, as one tensor of integers in the vault under `d` settings.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Settings {
    pub cache_budget: u64,
    pub layer_bits: u64,
    pub real_sun: bool,
    pub real_light: bool,
    pub follow_heading: bool,
    pub lock_to_fix: bool,
    /// Index into the engine's themes.
    pub theme: u64,
    /// Highlights compressed rather than linear; stored as its absence (flag 8 = linear) so settings from before it load compressed.
    pub compressed: bool,
    /// The UI's scale, fluor's RU multiplier, kept to a thousandth; 1 when the settings predate it.
    pub ui_ru: f32,
    /// What a touch on the map does: 0 navigates, 1 the pen, 2 the selection; 0 when the settings predate it.
    pub tool: u8,
    /// The pen's colour and opacity, RGBA; None when the settings predate it, and the map keeps its default.
    pub pen_rgba: Option<[u8; 4]>,
    /// How a tap routes there: 0 it doesn't, 1 car, 2 bike, 3 foot; 0 when the settings predate it.
    pub travel: u8,
    /// Release updates install themselves; on when the settings predate it, as Photon's are.
    pub auto_update: bool,
    /// A track is being recorded, the screen on or off.
    pub record: bool,
}

fn settings_key() -> String {
    vault_key(&[name("settings")])
}

pub fn save_settings(store: &FlatStorage, s: &Settings) -> Result<(), StorageError> {
    let flags = (s.real_sun as u64) | (s.follow_heading as u64) << 1 | (s.lock_to_fix as u64) << 2 | (!s.compressed as u64) << 3 | (s.real_light as u64) << 4;
    // The pen's colour is stored one more than its packed RGBA, so 0 says there is none.
    let pen = s.pen_rgba.map_or(0, |c| u32::from_be_bytes(c) as u64 + 1);
    let t = VsfType::t_u6(Tensor::new(vec![10], vec![s.cache_budget, s.layer_bits, flags, s.theme, (s.ui_ru * 1000.0).round() as u64, s.tool as u64, pen, s.travel as u64, !s.auto_update as u64, s.record as u64]));
    store.write_device(&settings_key(), &t.flatten())
}

pub fn load_settings(store: &FlatStorage) -> Option<Settings> {
    let bytes = store.read_device(&settings_key()).ok()??;
    let mut ptr = 0usize;
    let d = match vsf::parse(&bytes, &mut ptr).ok()? {
        VsfType::t_u6(t) => t.data,
        VsfType::v_u6(t) => t.data,
        _ => return None,
    };
    if d.len() < 3 {
        return None;
    }
    Some(Settings { cache_budget: d[0], layer_bits: d[1], real_sun: d[2] & 1 != 0, follow_heading: d[2] & 2 != 0, lock_to_fix: d[2] & 4 != 0, theme: d.get(3).copied().unwrap_or(0), compressed: d[2] & 8 == 0, real_light: d[2] & 16 != 0, ui_ru: d.get(4).map_or(1.0, |&v| v as f32 / 1000.0), tool: d.get(5).map_or(0, |&v| v.min(2) as u8), pen_rgba: d.get(6).filter(|&&v| v != 0).map(|&v| ((v - 1) as u32).to_be_bytes()), travel: d.get(7).map_or(0, |&v| v.min(3) as u8), auto_update: d.get(8).is_none_or(|&v| v == 0), record: d.get(9).is_some_and(|&v| v != 0) })
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

/// One GPS fix as the receiver measured it, nothing derived (Nick 2026-10-13: never store the map's height, never throw away the GPS's): the place as a dymaxion coordinate, when (unix milliseconds), the height above the WGS84 ellipsoid, the horizontal and vertical accuracy, the speed and the bearing; None where the receiver gave none. Elevation above sea level is worked out when the track is shown: the map's terrain where the fix is on the ground, else the height less the geoid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fix {
    pub coord: u64,
    pub unix_ms: i64,
    pub height_m: Option<f32>,
    pub accuracy_m: f32,
    pub v_accuracy_m: Option<f32>,
    pub speed_ms: Option<f32>,
    pub bearing_deg: Option<f32>,
}

/// Chunked track recorder: fixes flushed every CHUNK fixes as one VSF document of columns (coordinates, times, heights in centimetres, accuracies, speeds, bearings), zstd-compressed, so a crash costs at most one chunk and a day's hike never rewrites more than a small object per flush.
pub struct TrackRecorder {
    store: Arc<FlatStorage>,
    /// Eagle Time of the recording's start: the track's identity.
    start: i64,
    buf: Vec<Fix>,
    chunk: u32,
}

const CHUNK_FIXES: usize = 64;

/// An absent column value in its 16-bit field.
const NONE16: u16 = u16::MAX;

impl TrackRecorder {
    pub fn new(store: Arc<FlatStorage>) -> TrackRecorder {
        TrackRecorder { store, start: et_now(), buf: Vec::new(), chunk: 0 }
    }

    pub fn on_gps(&mut self, fix: Fix) {
        self.buf.push(fix);
        if self.buf.len() >= CHUNK_FIXES {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        let fixes = std::mem::take(&mut self.buf);
        let n = fixes.len();
        let cm16 = |v: Option<f32>| v.map_or(NONE16, |m| (m * 100.0).round().clamp(0.0, 65534.0) as u16);
        let doc = VsfBuilder::new()
            .add_section(
                "fixes",
                vec![
                    ("coord".to_string(), VsfType::t_u6(Tensor::new(vec![n], fixes.iter().map(|f| f.coord).collect()))),
                    ("ms".to_string(), VsfType::t_i6(Tensor::new(vec![n], fixes.iter().map(|f| f.unix_ms).collect()))),
                    ("height".to_string(), VsfType::t_i5(Tensor::new(vec![n], fixes.iter().map(|f| f.height_m.map_or(i32::MIN, |h| (h * 100.0).round() as i32)).collect()))),
                    ("accuracy".to_string(), VsfType::t_u4(Tensor::new(vec![n], fixes.iter().map(|f| cm16(Some(f.accuracy_m))).collect()))),
                    ("vaccuracy".to_string(), VsfType::t_u4(Tensor::new(vec![n], fixes.iter().map(|f| cm16(f.v_accuracy_m)).collect()))),
                    ("speed".to_string(), VsfType::t_u4(Tensor::new(vec![n], fixes.iter().map(|f| cm16(f.speed_ms)).collect()))),
                    ("bearing".to_string(), VsfType::t_u4(Tensor::new(vec![n], fixes.iter().map(|f| f.bearing_deg.map_or(NONE16, |b| (b.rem_euclid(360.0) * 100.0).round() as u16 % 36000)).collect()))),
                ],
            )
            .build();
        let Ok(doc) = doc else { return };
        let Ok(z) = zstd::encode_all(&doc[..], 3) else { return };
        if self.store.write_device(&track_chunk_key(self.start, self.chunk), &z).is_ok() {
            self.chunk += 1;
            // VSF all the way down: the counter is a u, not naked LE bytes.
            let _ = self.store.write_device(&track_count_key(self.start), &VsfType::u(self.chunk as usize, false).flatten());
        }
    }

    /// A new track from here on: what is held of the last one written, the next fix the first of a track of its own.
    pub fn begin(&mut self) {
        self.flush();
        self.start = et_now();
        self.chunk = 0;
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

/// One recorded chunk's fixes. A chunk from before (latitude, longitude, the map's elevation, unix seconds, as f64s) reads with the place and time only: its height was the map's, not the receiver's.
pub fn track_chunk(store: &FlatStorage, start: i64, chunk: u32) -> Option<Vec<Fix>> {
    let bytes = store.read_device(&track_chunk_key(start, chunk)).ok()??;
    if bytes.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
        let plain = zstd::decode_all(&bytes[..]).ok()?;
        let (header, end) = vsf::VsfHeader::decode(&plain).ok()?;
        let s = header.sections(&plain, end).ok()?.into_iter().find(|s| s.name == "fixes")?;
        // Columns read whatever way VSF hands them back (a one-dimensional tensor parses as a vector).
        let col = |name: &str| s.fields.iter().find(|f| f.name == name).and_then(|f| f.values.first().cloned());
        let u64s = |v: Option<VsfType>| match v {
            Some(VsfType::t_u6(t)) => Some(t.data),
            Some(VsfType::v_u6(t)) => Some(t.data),
            _ => None,
        };
        let i64s = |v: Option<VsfType>| match v {
            Some(VsfType::t_i6(t)) => Some(t.data),
            Some(VsfType::v_i6(t)) => Some(t.data),
            _ => None,
        };
        let i32s = |v: Option<VsfType>| match v {
            Some(VsfType::t_i5(t)) => Some(t.data),
            Some(VsfType::v_i5(t)) => Some(t.data),
            _ => None,
        };
        let u16s = |v: Option<VsfType>| match v {
            Some(VsfType::t_u4(t)) => Some(t.data),
            Some(VsfType::v_u4(t)) => Some(t.data),
            _ => None,
        };
        let (c, t, hh, a, va, sp, b) = (u64s(col("coord"))?, i64s(col("ms"))?, i32s(col("height"))?, u16s(col("accuracy"))?, u16s(col("vaccuracy"))?, u16s(col("speed"))?, u16s(col("bearing"))?);
        let n = c.len();
        if [t.len(), hh.len(), a.len(), va.len(), sp.len(), b.len()].iter().any(|&l| l != n) {
            return None;
        }
        let m16 = |v: u16| (v != NONE16).then_some(v as f32 / 100.0);
        return Some((0..n).map(|i| Fix { coord: c[i], unix_ms: t[i], height_m: (hh[i] != i32::MIN).then_some(hh[i] as f32 / 100.0), accuracy_m: a[i] as f32 / 100.0, v_accuracy_m: m16(va[i]), speed_ms: m16(sp[i]), bearing_deg: m16(b[i]) }).collect());
    }
    let mut ptr = 0usize;
    let old = tensor_f64(vsf::parse(&bytes, &mut ptr).ok()?)?;
    Some(old.chunks_exact(4).map(|f| Fix { coord: mahere_coord::Coord::from_lat_lon(f[0], f[1]).raw(), unix_ms: (f[3] * 1000.0) as i64, height_m: None, accuracy_m: 0.0, v_accuracy_m: None, speed_ms: None, bearing_deg: None }).collect())
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

    /// One serial test: set_vault_dirs_override is process-global, so parallel vault tests would race each other's directories.
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

        // The settings keep the tool and the pen's colour, a fully transparent black pen included.
        for pen in [Some([230, 40, 40, 255]), Some([0, 0, 0, 0]), None] {
            let set = Settings { cache_budget: 1 << 30, layer_bits: 0b1011, real_sun: true, compressed: true, ui_ru: 1.25, tool: 1, pen_rgba: pen, travel: 2, auto_update: true, record: true, ..Default::default() };
            save_settings(&store, &set).unwrap();
            assert_eq!(load_settings(&store), Some(set));
        }

        // Recorder pattern: big chunk blobs interleaved with rapid counter overwrites — the exact shape that looked flaky before the test harness itself was fixed (stale reused dirs + global override).
        let mut rec = TrackRecorder::new(store.clone());
        let key = rec.start();
        for i in 0..200 {
            rec.on_gps(Fix { coord: mahere_coord::Coord::from_lat_lon(46.2 + i as f64 * 1e-5, -121.49).raw(), unix_ms: 1_791_900_000_000 + i * 1000, height_m: (i % 2 == 0).then_some(2000.0 + i as f32), accuracy_m: 3.5, v_accuracy_m: Some(6.0), speed_ms: None, bearing_deg: Some(359.99) });
        }
        rec.flush();
        assert_eq!(rec.chunks(), 4);
        assert_eq!(track_chunks(&store, key), Some(4));
        for c in 0..4u32 {
            assert!(track_chunk(&store, key, c).is_some(), "chunk {c} missing");
        }
        // A fix reads back as it was measured: heights where given and absent where not, to the centimetre.
        let first = track_chunk(&store, key, 0).unwrap();
        assert_eq!(first.len(), 64);
        assert_eq!(first[2].height_m, Some(2002.0));
        assert_eq!(first[3].height_m, None);
        assert_eq!((first[1].unix_ms, first[1].accuracy_m, first[1].v_accuracy_m, first[1].speed_ms, first[1].bearing_deg), (1_791_900_001_000, 3.5, Some(6.0), None, Some(359.99)));

        // Persistence across a real reopen: drop every handle, open again, read everything back.
        drop(rec);
        drop(store);
        let store2 = open(Some(&base)).expect("reopen");
        assert_eq!(load_session(&store2), Some(s), "session must survive reopen");
        assert_eq!(track_chunks(&store2, key), Some(4), "track counter must survive reopen");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ==================== CELL CACHE ====================

/// The vault as the on-device cell tier: every cell fetched from the bucket is written through here and served from here afterwards (offline included). Keyed by the VSF values (`d` cell, `u` depth, `wm` cell) through [`vault_key`], so cells, session and tracks share the vault with the type system as the namespace.
///
/// Every cached cell is dated: an index document (`d` cells, `d` index) records each cell's last use as Eagle Time and its size, so the cache can be purged oldest-first to a byte budget ([`VaultCells::purge_to`]). The index is rewritten after every so many new cells and on demand ([`VaultCells::flush`], the pause moment).
pub struct VaultCells {
    store: Arc<FlatStorage>,
    index: std::sync::Mutex<CellIndex>,
    /// Cells an area keeps on the device: never purged, and not counted against the budget.
    pinned: std::sync::Mutex<std::collections::HashSet<(u8, u64)>>,
}

#[derive(Default)]
struct CellIndex {
    /// (depth, raw cell) → (last use, Eagle Time; bytes).
    entries: std::collections::HashMap<(u8, u64), (i64, u32)>,
    /// New cells since the last flush.
    unsaved: u32,
}

fn cells_index_key() -> String {
    vault_key(&[name("cells"), name("index")])
}

impl VaultCells {
    pub fn new(store: Arc<FlatStorage>) -> VaultCells {
        let mut index = CellIndex::default();
        if let Ok(Some(bytes)) = store.read_device(&cells_index_key()) {
            index.entries = decode_cell_index(&bytes);
        }
        VaultCells { store, index: std::sync::Mutex::new(index), pinned: std::sync::Mutex::new(Default::default()) }
    }

    fn touch(&self, key: mahere_tiles::CellKey, bytes: usize) {
        let mut ix = self.index.lock().unwrap();
        let fresh = ix.entries.insert((key.depth, key.raw()), (et_now(), bytes as u32)).is_none();
        if fresh {
            ix.unsaved += 1;
        }
        if ix.unsaved >= 64 {
            drop(ix);
            self.flush();
        }
    }

    /// Write the index document.
    pub fn flush(&self) {
        let (bytes, n) = {
            let mut ix = self.index.lock().unwrap();
            ix.unsaved = 0;
            (encode_cell_index(&ix.entries), ix.entries.len())
        };
        if let Some(b) = bytes {
            let _ = self.store.write_device(&cells_index_key(), &b);
        }
        let _ = n;
    }

    /// Bytes of cells the index knows about, the pinned ones included.
    pub fn cached_bytes(&self) -> u64 {
        self.index.lock().unwrap().entries.values().map(|&(_, b)| b as u64).sum()
    }

    /// Bytes of pinned cells on the device.
    pub fn pinned_bytes(&self) -> u64 {
        let pinned = self.pinned.lock().unwrap();
        self.index.lock().unwrap().entries.iter().filter(|(k, _)| pinned.contains(k)).map(|(_, &(_, b))| b as u64).sum()
    }

    /// The cells the areas keep: the whole set, replacing the last.
    pub fn set_pinned(&self, keys: impl IntoIterator<Item = mahere_tiles::CellKey>) {
        *self.pinned.lock().unwrap() = keys.into_iter().map(|k| (k.depth, k.raw())).collect();
    }

    /// Delete the least recently used unpinned cells until they fit `budget` bytes; returns how many went. The pinned cells stay whatever the budget.
    pub fn purge_to(&self, budget: u64) -> usize {
        let victims: Vec<(u8, u64)> = {
            let ix = self.index.lock().unwrap();
            let pinned = self.pinned.lock().unwrap();
            let mut total: u64 = ix.entries.iter().filter(|(k, _)| !pinned.contains(k)).map(|(_, &(_, b))| b as u64).sum();
            let mut by_age: Vec<(&(u8, u64), &(i64, u32))> = ix.entries.iter().filter(|(k, _)| !pinned.contains(k)).collect();
            by_age.sort_by_key(|(_, v)| v.0);
            let mut out = Vec::new();
            for (k, &(_, b)) in by_age {
                if total <= budget {
                    break;
                }
                total -= b as u64;
                out.push(*k);
            }
            out
        };
        for &(depth, raw) in &victims {
            let key = mahere_tiles::CellKey { depth, prefix: raw >> (60 - 2 * depth as u32) };
            let _ = self.store.delete_device(&cell_key(key));
            let _ = self.store.delete_device(&tag_key(key));
            self.index.lock().unwrap().entries.remove(&(depth, raw));
        }
        if !victims.is_empty() {
            self.flush();
        }
        victims.len()
    }
}

fn encode_cell_index(entries: &std::collections::HashMap<(u8, u64), (i64, u32)>) -> Option<Vec<u8>> {
    let mut depth = Vec::with_capacity(entries.len());
    let mut cell = Vec::with_capacity(entries.len());
    let mut seen = Vec::with_capacity(entries.len());
    let mut bytes = Vec::with_capacity(entries.len());
    for (&(d, raw), &(t, b)) in entries {
        depth.push(d);
        cell.push(raw);
        seen.push(t);
        bytes.push(b);
    }
    let n = entries.len();
    VsfBuilder::new()
        .add_section(
            "cells",
            vec![
                ("depth".to_string(), VsfType::t_u3(Tensor::new(vec![n], depth))),
                ("cell".to_string(), VsfType::t_u6(Tensor::new(vec![n], cell))),
                ("seen".to_string(), VsfType::t_i6(Tensor::new(vec![n], seen))),
                ("bytes".to_string(), VsfType::t_u5(Tensor::new(vec![n], bytes))),
            ],
        )
        .build()
        .ok()
}

fn decode_cell_index(data: &[u8]) -> std::collections::HashMap<(u8, u64), (i64, u32)> {
    let mut out = std::collections::HashMap::new();
    let Ok((header, end)) = vsf::VsfHeader::decode(data) else { return out };
    let Ok(sections) = header.sections(data, end) else { return out };
    for s in sections {
        if s.name != "cells" {
            continue;
        }
        let mut depth: Vec<u8> = Vec::new();
        let mut cell: Vec<u64> = Vec::new();
        let mut seen: Vec<i64> = Vec::new();
        let mut bytes: Vec<u32> = Vec::new();
        for f in s.fields {
            // A one-dimensional tensor comes back as a vector; both spellings are the same data.
            match (f.name.as_str(), f.values.into_iter().next()) {
                ("depth", Some(VsfType::t_u3(t))) => depth = t.data,
                ("depth", Some(VsfType::v_u3(t))) => depth = t.data,
                ("cell", Some(VsfType::t_u6(t))) => cell = t.data,
                ("cell", Some(VsfType::v_u6(t))) => cell = t.data,
                ("seen", Some(VsfType::t_i6(t))) => seen = t.data,
                ("seen", Some(VsfType::v_i6(t))) => seen = t.data,
                ("bytes", Some(VsfType::t_u5(t))) => bytes = t.data,
                ("bytes", Some(VsfType::v_u5(t))) => bytes = t.data,
                _ => {}
            }
        }
        let n = depth.len().min(cell.len()).min(seen.len()).min(bytes.len());
        for i in 0..n {
            out.insert((depth[i], cell[i]), (seen[i], bytes[i]));
        }
    }
    out
}

/// Where a kept cell's source tag (the bucket's ETag) lives, beside the cell.
fn tag_key(key: mahere_tiles::CellKey) -> String {
    vault_key(&[
        name("celltag"),
        VsfType::u(mahere_engine::residency::CELL_EPOCH as usize, false),
        VsfType::u(key.depth as usize, false),
        VsfType::wm(vsf::types::WorldCell::from_raw(key.raw())),
    ])
}

fn cell_key(key: mahere_tiles::CellKey) -> String {
    vault_key(&[
        name("cell"),
        VsfType::u(mahere_engine::residency::CELL_EPOCH as usize, false),
        VsfType::u(key.depth as usize, false),
        VsfType::wm(vsf::types::WorldCell::from_raw(key.raw())),
    ])
}

// ==================== USER CELLS ====================

/// The user's own cells (ink and selection) in the vault, keyed `d` user, `u` depth, `wm` cell, with an index document (`d` user, `d` index) naming every one, so they all load at start.
pub struct VaultUser {
    store: Arc<FlatStorage>,
    index: std::sync::Mutex<std::collections::HashMap<(u8, u64), (i64, u32)>>,
}

fn user_key(key: mahere_tiles::CellKey) -> String {
    vault_key(&[name("user"), VsfType::u(key.depth as usize, false), VsfType::wm(vsf::types::WorldCell::from_raw(key.raw()))])
}

fn user_index_key() -> String {
    vault_key(&[name("user"), name("index")])
}

impl VaultUser {
    pub fn new(store: Arc<FlatStorage>) -> VaultUser {
        let index = store.read_device(&user_index_key()).ok().flatten().map(|b| decode_cell_index(&b)).unwrap_or_default();
        VaultUser { store, index: std::sync::Mutex::new(index) }
    }

    fn flush(&self) {
        let bytes = encode_cell_index(&self.index.lock().unwrap());
        if let Some(b) = bytes {
            let _ = self.store.write_device(&user_index_key(), &b);
        }
    }
}

impl mahere_engine::ink::UserStore for VaultUser {
    fn put(&self, key: mahere_tiles::CellKey, bytes: &[u8]) {
        if self.store.write_device(&user_key(key), bytes).is_ok() {
            self.index.lock().unwrap().insert((key.depth, key.raw()), (et_now(), bytes.len() as u32));
            self.flush();
        }
    }

    fn delete(&self, key: mahere_tiles::CellKey) {
        let _ = self.store.write_device(&user_key(key), &[]);
        self.index.lock().unwrap().remove(&(key.depth, key.raw()));
        self.flush();
    }

    fn load_all(&self) -> Vec<(mahere_tiles::CellKey, Vec<u8>)> {
        let keys: Vec<(u8, u64)> = self.index.lock().unwrap().keys().copied().collect();
        keys.into_iter()
            .filter_map(|(depth, raw)| {
                let key = mahere_tiles::CellKey { depth, prefix: raw >> (60 - 2 * depth as u32) };
                let bytes = self.store.read_device(&user_key(key)).ok().flatten()?;
                (!bytes.is_empty()).then_some((key, bytes))
            })
            .collect()
    }
}

// ==================== AREAS ====================

/// The user's areas in the vault: one document each under (`d` area, `x` name), an index document (`d` area, `d` index) naming them.
pub struct VaultAreas {
    store: Arc<FlatStorage>,
    names: std::sync::Mutex<Vec<String>>,
}

fn area_key(area_name: &str) -> String {
    vault_key(&[name("area"), VsfType::x(area_name.to_string())])
}

fn area_index_key() -> String {
    vault_key(&[name("area"), name("index")])
}

impl VaultAreas {
    pub fn new(store: Arc<FlatStorage>) -> VaultAreas {
        let names = store.read_device(&area_index_key()).ok().flatten().map(|b| decode_names(&b)).unwrap_or_default();
        VaultAreas { store, names: std::sync::Mutex::new(names) }
    }

    fn flush(&self) {
        let names = self.names.lock().unwrap().clone();
        if let Some(b) = encode_names(&names) {
            let _ = self.store.write_device(&area_index_key(), &b);
        }
    }
}

fn encode_names(names: &[String]) -> Option<Vec<u8>> {
    VsfBuilder::new().add_section("areas", names.iter().enumerate().map(|(i, n)| (format!("a{i}"), VsfType::x(n.clone()))).collect()).build().ok()
}

fn decode_names(data: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let Ok((header, end)) = vsf::VsfHeader::decode(data) else { return out };
    let Ok(sections) = header.sections(data, end) else { return out };
    for s in sections {
        if s.name != "areas" {
            continue;
        }
        for f in s.fields {
            if let Some(VsfType::x(n) | VsfType::a(n)) = f.values.into_iter().next() {
                out.push(n);
            }
        }
    }
    out
}

impl mahere_engine::areas::AreaStore for VaultAreas {
    fn put(&self, area_name: &str, bytes: &[u8]) {
        if self.store.write_device(&area_key(area_name), bytes).is_ok() {
            let mut names = self.names.lock().unwrap();
            if !names.iter().any(|n| n == area_name) {
                names.push(area_name.to_string());
            }
            drop(names);
            self.flush();
        }
    }

    fn delete(&self, area_name: &str) {
        let _ = self.store.write_device(&area_key(area_name), &[]);
        self.names.lock().unwrap().retain(|n| n != area_name);
        self.flush();
    }

    fn load_all(&self) -> Vec<Vec<u8>> {
        let names = self.names.lock().unwrap().clone();
        names.iter().filter_map(|n| self.store.read_device(&area_key(n)).ok().flatten()).filter(|b| !b.is_empty()).collect()
    }
}

impl mahere_engine::residency::CellStore for VaultCells {
    fn get(&self, key: mahere_tiles::CellKey) -> mahere_engine::residency::Fetch {
        use mahere_engine::residency::Fetch;
        match self.store.read_device(&cell_key(key)) {
            Ok(Some(b)) => {
                self.touch(key, b.len());
                Fetch::Bytes(b)
            }
            Ok(None) => Fetch::Absent,
            Err(e) => Fetch::Failed(format!("{e:?}")),
        }
    }
}

impl mahere_engine::residency::CellCache for VaultCells {
    fn put(&self, key: mahere_tiles::CellKey, bytes: &[u8]) {
        if self.store.write_device(&cell_key(key), bytes).is_ok() {
            self.touch(key, bytes.len());
        }
    }

    fn tag(&self, key: mahere_tiles::CellKey) -> Option<String> {
        self.store.read_device(&tag_key(key)).ok().flatten().and_then(|b| String::from_utf8(b).ok())
    }

    fn set_tag(&self, key: mahere_tiles::CellKey, tag: &str) {
        let _ = self.store.write_device(&tag_key(key), tag.as_bytes());
    }
}

#[cfg(test)]
mod index_tests {
    use super::*;

    #[test]
    fn cell_index_round_trips() {
        let mut entries = std::collections::HashMap::new();
        entries.insert((13u8, 0x3000_0000_0000_0000u64), (et_now(), 48_123u32));
        entries.insert((9u8, 0x7000_0000_0000_0000u64), (et_now() - 5, 1_024u32));
        let bytes = encode_cell_index(&entries).unwrap();
        assert_eq!(decode_cell_index(&bytes), entries);
    }
}

// ==================== THEMES ====================

/// Themes on disk: one VSF document per theme in the vault under (`d` theme, `x` name), colours authored in VSF RGB at gamma 2 as Photon authors its palette and kept that way: the renderers convert to the display at their one encode. The built-ins are written on first launch so a user can edit them in place; an index document (`d` theme, `d` index) lists the names.
pub mod themes {
    use super::{name, vault_key};
    use kete::FlatStorage;
    use mahere_engine::LayerMask;
    use mahere_engine::theme::Theme;
    use vsf::types::Tensor;
    use vsf::{VsfBuilder, VsfType};

    fn theme_key(theme_name: &str) -> String {
        vault_key(&[name("theme"), VsfType::x(theme_name.to_string())])
    }

    fn index_key() -> String {
        vault_key(&[name("theme"), name("index")])
    }

    fn rgb_t(c: [u8; 4]) -> VsfType {
        VsfType::t_u3(Tensor::new(vec![4], c.to_vec()))
    }

    fn table_t<const N: usize>(t: &[[u8; 4]; N]) -> VsfType {
        VsfType::t_u3(Tensor::new(vec![N, 4], t.iter().flatten().copied().collect()))
    }

    pub fn encode(t: &Theme) -> Option<Vec<u8>> {
        let stops: Vec<f32> = t.hypso.iter().map(|(m, _)| *m).collect();
        let stop_rgb: Vec<u8> = t.hypso.iter().flat_map(|(_, c)| c.to_vec()).collect();
        VsfBuilder::new()
            .add_section("theme", vec![("name".to_string(), VsfType::x(t.name.clone())), ("layers".to_string(), VsfType::u(t.layers.bits() as usize, false)), ("revision".to_string(), VsfType::u(t.revision as usize, false)), ("edited".to_string(), VsfType::u(t.edited as usize, false))])
            .add_section("hypso", vec![("metres".to_string(), VsfType::t_f5(Tensor::new(vec![stops.len()], stops))), ("rgb".to_string(), VsfType::t_u3(Tensor::new(vec![t.hypso.len(), 3], stop_rgb)))])
            .add_section("tables", vec![("land".to_string(), table_t(&t.land)), ("line".to_string(), table_t(&t.line)), ("ownership".to_string(), table_t(&t.own)), ("vehicleuse".to_string(), table_t(&t.mvum))])
            .add_section(
                "inks",
                vec![
                    ("sea".to_string(), rgb_t(t.sea)),
                    ("flat".to_string(), rgb_t(t.flat)),
                    ("background".to_string(), rgb_t(t.bg)),
                    ("noterrain".to_string(), rgb_t(t.no_dem)),
                    ("water".to_string(), rgb_t(t.water)),
                    ("contour".to_string(), rgb_t(t.contour)),
                    ("index".to_string(), rgb_t(t.contour_index)),
                    ("wilderness".to_string(), rgb_t(t.wild)),
                ],
            )
            .add_section("light", {
                let mut f = vec![("sun".to_string(), VsfType::t_f5(Tensor::new(vec![3], t.sun.to_vec()))), ("sky".to_string(), VsfType::t_f5(Tensor::new(vec![3], t.sky.to_vec())))];
                // The theme's own light, absolute: nine coefficients for each of red, green and blue.
                if let Some(p) = &t.probe {
                    f.push(("probe".to_string(), VsfType::t_f5(Tensor::new(vec![3, 9], p.l.iter().flatten().copied().collect()))));
                }
                f
            })
            .add_section("labels", vec![("colours".to_string(), table_t(&t.label)), ("look".to_string(), VsfType::t_f5(Tensor::new(vec![4], t.label_look.to_vec()))), ("amount".to_string(), VsfType::t_f5(Tensor::new(vec![2], t.label_amount.to_vec())))])
            .build()
            .ok()
    }

    fn bytes_of(v: &VsfType) -> Option<Vec<u8>> {
        match v {
            VsfType::t_u3(t) => Some(t.data.clone()),
            VsfType::v_u3(t) => Some(t.data.clone()),
            _ => None,
        }
    }

    fn floats_of(v: &VsfType) -> Option<Vec<f32>> {
        match v {
            VsfType::t_f5(t) => Some(t.data.clone()),
            VsfType::v_f5(t) => Some(t.data.clone()),
            VsfType::t_f6(t) => Some(t.data.iter().map(|&x| x as f32).collect()),
            VsfType::v_f6(t) => Some(t.data.iter().map(|&x| x as f32).collect()),
            _ => None,
        }
    }

    /// A colour with its opacity; a colour saved before opacities (three bytes) takes `alpha`.
    fn rgba(v: &VsfType, alpha: u8) -> Option<[u8; 4]> {
        let b = bytes_of(v)?;
        match b.len() {
            4 => Some([b[0], b[1], b[2], b[3]]),
            3 => Some([b[0], b[1], b[2], alpha]),
            _ => None,
        }
    }

    fn table<const N: usize>(v: &VsfType, alpha: u8) -> Option<[[u8; 4]; N]> {
        let b = bytes_of(v)?;
        if b.len() == N * 4 {
            return Some(std::array::from_fn(|i| [b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]));
        }
        if b.len() == N * 3 {
            // Saved before opacities: the empty class clear, the rest at the default.
            return Some(std::array::from_fn(|i| [b[3 * i], b[3 * i + 1], b[3 * i + 2], if i == 0 { 0 } else { alpha }]));
        }
        None
    }

    pub fn decode(data: &[u8]) -> Option<Theme> {
        let (header, end) = vsf::VsfHeader::decode(data).ok()?;
        let sections = header.sections(data, end).ok()?;
        let mut fields: std::collections::HashMap<(String, String), VsfType> = std::collections::HashMap::new();
        for s in sections {
            for f in s.fields {
                if let Some(v) = f.values.into_iter().next() {
                    fields.insert((s.name.clone(), f.name), v);
                }
            }
        }
        let get = |s: &str, f: &str| fields.get(&(s.to_string(), f.to_string()));
        let theme_name = match get("theme", "name")? {
            VsfType::x(s) | VsfType::a(s) => s.clone(),
            _ => return None,
        };
        let layers = match get("theme", "layers")? {
            VsfType::u(b, _) => LayerMask::from_bits(*b as u32),
            VsfType::u3(b) => LayerMask::from_bits(*b as u32),
            VsfType::u4(b) => LayerMask::from_bits(*b as u32),
            VsfType::u5(b) => LayerMask::from_bits(*b),
            VsfType::u6(b) => LayerMask::from_bits(*b as u32),
            _ => LayerMask::default(),
        };
        let metres = floats_of(get("hypso", "metres")?)?;
        let rgb = bytes_of(get("hypso", "rgb")?)?;
        if metres.len() != 5 || rgb.len() != 15 {
            return None;
        }
        let hypso: [(f32, [u8; 3]); 5] = std::array::from_fn(|i| (metres[i], [rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]]));
        let revision = match get("theme", "revision") {
            Some(VsfType::u(r, _)) => *r as u32,
            Some(VsfType::u3(r)) => *r as u32,
            _ => 0,
        };
        let edited = match get("theme", "edited") {
            Some(VsfType::u(r, _)) => *r != 0,
            Some(VsfType::u3(r)) => *r != 0,
            // A theme saved before the flag with a name no built-in has is the user's own.
            _ => !mahere_engine::theme::BUILTIN_NAMES.contains(&theme_name.as_str()),
        };
        // Contour opacities from a theme saved before they lived in the colours.
        let contour_alpha = get("inks", "contouralpha").and_then(floats_of).filter(|a| a.len() == 2).map_or([0.85, 0.85], |a| [a[0], a[1]]);
        let sun = floats_of(get("light", "sun")?)?;
        let sky = floats_of(get("light", "sky")?)?;
        if sun.len() != 3 || sky.len() != 3 {
            return None;
        }
        Some(Theme {
            name: theme_name,
            revision,
            edited,
            hypso,
            sea: rgba(get("inks", "sea")?, 255)?,
            flat: rgba(get("inks", "flat")?, 255)?,
            bg: rgba(get("inks", "background")?, 255)?,
            no_dem: rgba(get("inks", "noterrain")?, 255)?,
            land: table(get("tables", "land")?, 217)?,
            line: table(get("tables", "line")?, 255)?,
            water: rgba(get("inks", "water")?, 255)?,
            contour: rgba(get("inks", "contour")?, (contour_alpha[0] * 255.0).round() as u8)?,
            contour_index: rgba(get("inks", "index")?, (contour_alpha[1] * 255.0).round() as u8)?,
            // Saved before the overlays: the shipped colours.
            own: get("tables", "ownership").and_then(|v| table(v, 200)).unwrap_or(mahere_engine::theme::OWN),
            wild: get("inks", "wilderness").and_then(|v| rgba(v, 230)).unwrap_or(mahere_engine::theme::WILD),
            mvum: get("tables", "vehicleuse").and_then(|v| table(v, 220)).unwrap_or(mahere_engine::theme::MVUM),
            sun: [sun[0], sun[1], sun[2]],
            sky: [sky[0], sky[1], sky[2]],
            probe: get("light", "probe").and_then(floats_of).filter(|v| v.len() == 27).map(|v| mahere_engine::sh::Sh9 { l: std::array::from_fn(|c| std::array::from_fn(|k| v[c * 9 + k])) }),
            layers,
            // Saved before labels had a look of their own: the palette for its ground (set below) and the shipped look.
            label: get("labels", "colours").and_then(|v| table(v, 255)).unwrap_or([[0; 4]; 11]),
            label_look: get("labels", "look").and_then(floats_of).filter(|v| v.len() == 4).map_or(mahere_engine::theme::LABEL_LOOK, |v| [v[0], v[1], v[2], v[3]]),
            label_amount: get("labels", "amount").and_then(floats_of).filter(|v| v.len() == 2).map_or(mahere_engine::theme::LABEL_AMOUNT, |v| [v[0], v[1]]),
        })
        .map(|mut t| {
            if get("labels", "colours").is_none() {
                t.label = t.default_labels();
            }
            t
        })
    }

    fn write_index(store: &FlatStorage, names: &[String]) {
        let joined: Vec<VsfType> = names.iter().map(|n| VsfType::x(n.clone())).collect();
        let mut section = vsf::VsfSection::new("themes");
        section.add_field_multi("names", joined);
        if let Ok(b) = VsfBuilder::new().add_section_direct(section).build() {
            let _ = store.write_device(&index_key(), &b);
        }
    }

    pub fn names(store: &FlatStorage) -> Vec<String> {
        let Ok(Some(bytes)) = store.read_device(&index_key()) else { return Vec::new() };
        let Ok((header, end)) = vsf::VsfHeader::decode(&bytes) else { return Vec::new() };
        let Ok(sections) = header.sections(&bytes, end) else { return Vec::new() };
        let mut out = Vec::new();
        for s in sections {
            for f in s.fields {
                if f.name == "names" {
                    for v in f.values {
                        if let VsfType::x(n) | VsfType::a(n) = v {
                            out.push(n);
                        }
                    }
                }
            }
        }
        out
    }

    pub fn save(store: &FlatStorage, t: &Theme) {
        if let Some(b) = encode(t) {
            let _ = store.write_device(&theme_key(&t.name), &b);
            let mut list = names(store);
            if !list.iter().any(|n| *n == t.name) {
                list.push(t.name.clone());
                write_index(store, &list);
            }
        }
    }

    pub fn load(store: &FlatStorage, theme_name: &str) -> Option<Theme> {
        let bytes = store.read_device(&theme_key(theme_name)).ok()??;
        decode(&bytes)
    }

    /// The built-ins into the vault where they are missing or an older revision, then every theme the vault holds, in index order, as authored (VSF RGB): the renderers convert to the display at their one encode.
    pub fn load_all(store: &FlatStorage) -> Vec<Theme> {
        for t in mahere_engine::theme::builtin() {
            // A newer shipped revision replaces the stored one unless the user has edited it; theirs stays until they reset.
            if load(store, &t.name).is_none_or(|old| old.revision < t.revision && !old.edited) {
                save(store, &t);
            }
        }
        names(store).iter().filter_map(|n| load(store, n)).collect()
    }

    /// Remove a theme document and its index entry.
    pub fn delete(store: &FlatStorage, theme_name: &str) {
        let _ = store.delete_device(&theme_key(theme_name));
        let list: Vec<String> = names(store).into_iter().filter(|n| n != theme_name).collect();
        write_index(store, &list);
    }
}

#[cfg(test)]
mod theme_tests {
    #[test]
    fn a_theme_survives_the_document() {
        let t = mahere_engine::theme::night();
        let bytes = super::themes::encode(&t).unwrap();
        let back = super::themes::decode(&bytes).unwrap();
        assert_eq!(back.name, "Night");
        assert_eq!(back.line, t.line);
        assert_eq!(back.hypso, t.hypso);
        assert_eq!(back.layers.bits(), t.layers.bits());
        assert_eq!(back.contour_index, t.contour_index);
    }

    #[test]
    fn every_builtin_survives_with_its_light_and_overlays() {
        for t in mahere_engine::theme::builtin() {
            let back = super::themes::decode(&super::themes::encode(&t).unwrap()).unwrap();
            assert_eq!(back.probe, t.probe, "{}", t.name);
            assert!(back.probe.is_some(), "{} ships a light", t.name);
            assert_eq!((back.own, back.wild, back.mvum, back.land, back.flat), (t.own, t.wild, t.mvum, t.land, t.flat), "{}", t.name);
        }
        // A theme without its own light keeps none.
        let mut t = mahere_engine::theme::trail();
        t.probe = None;
        assert_eq!(super::themes::decode(&super::themes::encode(&t).unwrap()).unwrap().probe, None);
    }
}
