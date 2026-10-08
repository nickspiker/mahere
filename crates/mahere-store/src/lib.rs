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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Settings {
    pub cache_budget: u64,
    pub layer_bits: u64,
    pub real_sun: bool,
    pub follow_heading: bool,
    pub lock_to_fix: bool,
    /// Index into the engine's themes.
    pub theme: u64,
    /// The highlight rail on the display; stored as its absence (flag 8 = straight) so settings from before it load rolled.
    pub rolloff: bool,
}

fn settings_key() -> String {
    vault_key(&[name("settings")])
}

pub fn save_settings(store: &FlatStorage, s: &Settings) -> Result<(), StorageError> {
    let flags = (s.real_sun as u64) | (s.follow_heading as u64) << 1 | (s.lock_to_fix as u64) << 2 | (!s.rolloff as u64) << 3;
    let t = VsfType::t_u6(Tensor::new(vec![4], vec![s.cache_budget, s.layer_bits, flags, s.theme]));
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
    Some(Settings { cache_budget: d[0], layer_bits: d[1], real_sun: d[2] & 1 != 0, follow_heading: d[2] & 2 != 0, lock_to_fix: d[2] & 4 != 0, theme: d.get(3).copied().unwrap_or(0), rolloff: d[2] & 8 == 0 })
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

/// Chunked GPS track recorder: (lat, lon, elevation m, unix seconds) per fix, flushed every CHUNK fixes so a crash costs at most one chunk and a day's hike never rewrites more than a small object per flush.
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

        // Recorder pattern: big chunk blobs interleaved with rapid counter overwrites — the exact shape that looked flaky before the test harness itself was fixed (stale reused dirs + global override).
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
        VaultCells { store, index: std::sync::Mutex::new(index) }
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

    /// Bytes of cells the index knows about.
    pub fn cached_bytes(&self) -> u64 {
        self.index.lock().unwrap().entries.values().map(|&(_, b)| b as u64).sum()
    }

    /// Delete the least recently used cells until the cache fits `budget` bytes; returns how many went.
    pub fn purge_to(&self, budget: u64) -> usize {
        let victims: Vec<(u8, u64)> = {
            let ix = self.index.lock().unwrap();
            let mut total: u64 = ix.entries.values().map(|&(_, b)| b as u64).sum();
            let mut by_age: Vec<(&(u8, u64), &(i64, u32))> = ix.entries.iter().collect();
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

fn cell_key(key: mahere_tiles::CellKey) -> String {
    vault_key(&[
        name("cell"),
        VsfType::u(mahere_engine::residency::CELL_EPOCH as usize, false),
        VsfType::u(key.depth as usize, false),
        VsfType::wm(vsf::types::WorldCell::from_raw(key.raw())),
    ])
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

    fn rgb_t(c: [u8; 3]) -> VsfType {
        VsfType::t_u3(Tensor::new(vec![3], c.to_vec()))
    }

    fn table_t<const N: usize>(t: &[[u8; 3]; N]) -> VsfType {
        VsfType::t_u3(Tensor::new(vec![N, 3], t.iter().flatten().copied().collect()))
    }

    pub fn encode(t: &Theme) -> Option<Vec<u8>> {
        let stops: Vec<f32> = t.hypso.iter().map(|(m, _)| *m).collect();
        let stop_rgb: Vec<u8> = t.hypso.iter().flat_map(|(_, c)| c.to_vec()).collect();
        VsfBuilder::new()
            .add_section("theme", vec![("name".to_string(), VsfType::x(t.name.clone())), ("layers".to_string(), VsfType::u(t.layers.bits() as usize, false)), ("revision".to_string(), VsfType::u(t.revision as usize, false))])
            .add_section("hypso", vec![("metres".to_string(), VsfType::t_f5(Tensor::new(vec![stops.len()], stops))), ("rgb".to_string(), VsfType::t_u3(Tensor::new(vec![t.hypso.len(), 3], stop_rgb)))])
            .add_section("tables", vec![("land".to_string(), table_t(&t.land)), ("line".to_string(), table_t(&t.line))])
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
                    ("contouralpha".to_string(), VsfType::t_f5(Tensor::new(vec![2], t.contour_alpha.to_vec()))),
                ],
            )
            .add_section("light", vec![("sun".to_string(), VsfType::t_f5(Tensor::new(vec![3], t.sun.to_vec()))), ("sky".to_string(), VsfType::t_f5(Tensor::new(vec![3], t.sky.to_vec())))])
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

    fn rgb3(v: &VsfType) -> Option<[u8; 3]> {
        let b = bytes_of(v)?;
        (b.len() == 3).then(|| [b[0], b[1], b[2]])
    }

    fn table<const N: usize>(v: &VsfType) -> Option<[[u8; 3]; N]> {
        let b = bytes_of(v)?;
        if b.len() != N * 3 {
            return None;
        }
        Some(std::array::from_fn(|i| [b[3 * i], b[3 * i + 1], b[3 * i + 2]]))
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
        let contour_alpha = get("inks", "contouralpha").and_then(floats_of).filter(|a| a.len() == 2).map_or([0.85, 0.85], |a| [a[0], a[1]]);
        let sun = floats_of(get("light", "sun")?)?;
        let sky = floats_of(get("light", "sky")?)?;
        if sun.len() != 3 || sky.len() != 3 {
            return None;
        }
        Some(Theme {
            name: theme_name,
            revision,
            contour_alpha,
            hypso,
            sea: rgb3(get("inks", "sea")?)?,
            flat: rgb3(get("inks", "flat")?)?,
            bg: rgb3(get("inks", "background")?)?,
            no_dem: rgb3(get("inks", "noterrain")?)?,
            land: table(get("tables", "land")?)?,
            line: table(get("tables", "line")?)?,
            water: rgb3(get("inks", "water")?)?,
            contour: rgb3(get("inks", "contour")?)?,
            contour_index: rgb3(get("inks", "index")?)?,
            sun: [sun[0], sun[1], sun[2]],
            sky: [sky[0], sky[1], sky[2]],
            layers,
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
            if load(store, &t.name).is_none_or(|old| old.revision < t.revision) {
                save(store, &t);
            }
        }
        names(store).iter().filter_map(|n| load(store, n)).collect()
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
}
