//! The cell cache on the device as plain files (Nick 2026-10-18): one file per cell and layer, as the bucket serves it, and an index document of when each was last used and its size, so the cache is purged oldest-first to a budget and an area's cells are kept whatever the budget.
//! Cells were vault entries until 0.222; the vault opened by walking an index of every one (8 to 63 s on the phone, the map black meanwhile) and wrote each as its own durable commit. Cells are public and fetched again at will, a layer checked against the hash its base names, so a file that is torn or lost costs a download and nothing more; the vault keeps the user's own data.
//! Layout: `<dir>/e<epoch>/<layer>/<depth>/<ancestor four levels up>/<cell>.vsf`, the bucket's ETag beside it as `<cell>.tag`, the index at `<dir>/e<epoch>/index.vsf`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use mahere_engine::residency::{CELL_EPOCH, CellCache, CellStore, Fetch};
use mahere_tiles::CellKey;
use mahere_tiles::publish::Layer;

use crate::{decode_cell_index, encode_cell_index, et_now};

/// New cells between writes of the index: a cell the index has not heard of when the app dies is a file the budget does not count until the index is made again from the files.
const FLUSH_EVERY: u32 = 256;

/// The device's cell files.
pub struct CellFiles {
    root: PathBuf,
    index: Mutex<Index>,
    /// Cells an area keeps on the device: never purged, and not counted against the budget.
    pinned: Mutex<HashSet<(u8, u64)>>,
}

#[derive(Default)]
struct Index {
    /// (depth with the layer in its top bits, raw cell) → (last use, Eagle Time; bytes).
    entries: HashMap<(u8, u64), (i64, u32)>,
    /// New cells since the last write of the index.
    unsaved: u32,
}

/// A layer's code: in the index key's depth byte (depths stop at 30) and in the vault keys of old.
pub(crate) fn layer_code(layer: Layer) -> u8 {
    match layer {
        Layer::Base => 0,
        Layer::Img => 1,
        Layer::Nir => 2,
        Layer::Dem => 3,
    }
}

fn layer_dir(code: u8) -> &'static str {
    match code {
        0 => "base",
        1 => "img",
        2 => "nir",
        _ => "dem",
    }
}

/// A file written whole or not at all: beside its place under a name of its own, then renamed over it.
fn write_whole(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static PART: AtomicU64 = AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = path.with_extension(format!("part{}-{}", std::process::id(), PART.fetch_add(1, Ordering::Relaxed)));
    std::fs::write(&part, bytes)?;
    std::fs::rename(&part, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&part);
    })
}

impl CellFiles {
    /// The cache under `dir`, its index read; with no index (a first start, or one lost), made again from the files there.
    pub fn open(dir: &Path) -> Arc<CellFiles> {
        let root = dir.join(format!("e{CELL_EPOCH}"));
        let _ = std::fs::create_dir_all(&root);
        let mut index = Index::default();
        match std::fs::read(root.join("index.vsf")) {
            Ok(bytes) => index.entries = decode_cell_index(&bytes),
            Err(_) => index.entries = scan(&root),
        }
        Arc::new(CellFiles { root, index: Mutex::new(index), pinned: Mutex::new(HashSet::new()) })
    }

    fn ix(key: CellKey, code: u8) -> (u8, u64) {
        (key.depth | code << 6, key.raw())
    }

    fn path(&self, key: CellKey, code: u8) -> PathBuf {
        let group = key.ancestor(key.depth.saturating_sub(4));
        self.root.join(layer_dir(code)).join(format!("{:02}", key.depth)).join(group.name()).join(format!("{}.vsf", key.name()))
    }

    fn touch(&self, key: CellKey, code: u8, bytes: usize, seen: i64) {
        let mut ix = self.index.lock().unwrap();
        if ix.entries.insert(Self::ix(key, code), (seen, bytes as u32)).is_none() {
            ix.unsaved += 1;
        }
        if ix.unsaved >= FLUSH_EVERY {
            drop(ix);
            self.flush();
        }
    }

    /// Write the index document.
    pub fn flush(&self) {
        let bytes = {
            let mut ix = self.index.lock().unwrap();
            ix.unsaved = 0;
            encode_cell_index(&ix.entries)
        };
        if let Some(b) = bytes {
            if let Err(e) = write_whole(&self.root.join("index.vsf"), &b) {
                eprintln!("cells: index not written: {e}");
            }
        }
    }

    fn get_layer(&self, key: CellKey, code: u8) -> Fetch {
        match std::fs::read(self.path(key, code)) {
            Ok(b) => {
                self.touch(key, code, b.len(), et_now());
                Fetch::Bytes(b)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Fetch::Absent,
            Err(e) => Fetch::Failed(e.to_string()),
        }
    }

    fn put_layer(&self, key: CellKey, code: u8, bytes: &[u8], seen: i64) {
        match write_whole(&self.path(key, code), bytes) {
            Ok(()) => self.touch(key, code, bytes.len(), seen),
            Err(e) => eprintln!("cells: {} {} not kept: {e}", key.name(), layer_dir(code)),
        }
    }

    fn tag_of(&self, key: CellKey, code: u8) -> Option<String> {
        std::fs::read_to_string(self.path(key, code).with_extension("tag")).ok()
    }

    fn set_tag_of(&self, key: CellKey, code: u8, tag: &str) {
        let _ = write_whole(&self.path(key, code).with_extension("tag"), tag.as_bytes());
    }

    /// Bytes of cells the index knows about, the pinned ones included.
    pub fn cached_bytes(&self) -> u64 {
        self.index.lock().unwrap().entries.values().map(|&(_, b)| b as u64).sum()
    }

    /// Bytes of pinned cells on the device.
    pub fn pinned_bytes(&self) -> u64 {
        let pinned = self.pinned.lock().unwrap();
        self.index.lock().unwrap().entries.iter().filter(|(k, _)| pinned.contains(&(k.0 & 63, k.1))).map(|(_, &(_, b))| b as u64).sum()
    }

    /// The cells the areas keep: the whole set, replacing the last.
    pub fn set_pinned(&self, keys: impl IntoIterator<Item = CellKey>) {
        *self.pinned.lock().unwrap() = keys.into_iter().map(|k| (k.depth, k.raw())).collect();
    }

    /// Delete the least recently used unpinned cells until they fit `budget` bytes; returns how many went. The pinned cells stay whatever the budget.
    pub fn purge_to(&self, budget: u64) -> usize {
        let victims: Vec<(u8, u64)> = {
            let ix = self.index.lock().unwrap();
            let pinned = self.pinned.lock().unwrap();
            let unpinned = |k: &(u8, u64)| !pinned.contains(&(k.0 & 63, k.1));
            let mut total: u64 = ix.entries.iter().filter(|(k, _)| unpinned(k)).map(|(_, &(_, b))| b as u64).sum();
            let mut by_age: Vec<(&(u8, u64), &(i64, u32))> = ix.entries.iter().filter(|(k, _)| unpinned(k)).collect();
            // The image layers go first (a map without its photo still reads), then the terrain, the base last, oldest first within each.
            let keep = |code: u8| match code {
                0 => 2,
                3 => 1,
                _ => 0,
            };
            by_age.sort_by_key(|(k, v)| (keep(k.0 >> 6), v.0));
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
        for &(packed, raw) in &victims {
            let (depth, code) = (packed & 63, packed >> 6);
            let key = CellKey { depth, prefix: raw >> (60 - 2 * depth as u32) };
            let path = self.path(key, code);
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(path.with_extension("tag"));
            self.index.lock().unwrap().entries.remove(&(packed, raw));
        }
        if !victims.is_empty() {
            self.flush();
        }
        victims.len()
    }

    /// Every cell file here, handed to `f` with its layer code, bytes and tag — the move into the vault. Returns how many.
    pub(crate) fn drain_into(&self, mut f: impl FnMut(CellKey, u8, &[u8], Option<&str>)) -> usize {
        let mut n = 0;
        for &(packed, raw) in scan(&self.root).keys() {
            let (depth, code) = (packed & 63, packed >> 6);
            let key = CellKey { depth, prefix: raw >> (60 - 2 * depth as u32) };
            if let Ok(bytes) = std::fs::read(self.path(key, code)) {
                f(key, code, &bytes, self.tag_of(key, code).as_deref());
                n += 1;
            }
        }
        n
    }

    /// One of a published cell's layers as a cache of its own over the same files, index and pins.
    pub fn layer(self: &Arc<Self>, layer: Layer) -> Arc<CellFileLayer> {
        Arc::new(CellFileLayer { cells: self.clone(), code: layer_code(layer) })
    }
}

/// The index made again from the files themselves, each dated now: a first start finds nothing, one whose index was lost keeps what it had.
fn scan(root: &Path) -> HashMap<(u8, u64), (i64, u32)> {
    let mut out = HashMap::new();
    let now = et_now();
    for code in 0..4u8 {
        let Ok(depths) = std::fs::read_dir(root.join(layer_dir(code))) else { continue };
        for d in depths.flatten() {
            let Ok(groups) = std::fs::read_dir(d.path()) else { continue };
            for g in groups.flatten() {
                let Ok(files) = std::fs::read_dir(g.path()) else { continue };
                for f in files.flatten() {
                    let name = f.file_name();
                    let Some(name) = name.to_str().and_then(|n| n.strip_suffix(".vsf")) else { continue };
                    if let (Some(key), Ok(meta)) = (CellKey::from_name(name), f.metadata()) {
                        out.insert(CellFiles::ix(key, code), (now, meta.len() as u32));
                    }
                }
            }
        }
    }
    out
}

/// A published cell's layer (the base, terrain, imagery or infrared) in the files.
pub struct CellFileLayer {
    cells: Arc<CellFiles>,
    code: u8,
}

impl CellStore for CellFiles {
    fn get(&self, key: CellKey) -> Fetch {
        self.get_layer(key, 0)
    }
}

impl CellCache for CellFiles {
    fn put(&self, key: CellKey, bytes: &[u8]) {
        self.put_layer(key, 0, bytes, et_now());
    }

    fn tag(&self, key: CellKey) -> Option<String> {
        self.tag_of(key, 0)
    }

    fn set_tag(&self, key: CellKey, tag: &str) {
        self.set_tag_of(key, 0, tag);
    }
}

impl CellStore for CellFileLayer {
    fn get(&self, key: CellKey) -> Fetch {
        self.cells.get_layer(key, self.code)
    }
}

impl CellCache for CellFileLayer {
    fn put(&self, key: CellKey, bytes: &[u8]) {
        self.cells.put_layer(key, self.code, bytes, et_now());
    }

    fn tag(&self, key: CellKey) -> Option<String> {
        self.cells.tag_of(key, self.code)
    }

    fn set_tag(&self, key: CellKey, tag: &str) {
        self.cells.set_tag_of(key, self.code, tag);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mahere-cells-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// A cell and its layers are kept, read back, tagged, counted, and found again by a cache opened over the same files, with or without its index.
    #[test]
    fn cells_survive_a_reopen_and_a_lost_index() {
        let dir = scratch("reopen");
        let key = CellKey { depth: 14, prefix: 12345 };
        {
            let c = CellFiles::open(&dir);
            c.put(key, b"base");
            c.layer(Layer::Dem).put(key, b"terrain!");
            c.layer(Layer::Dem).set_tag(key, "\"etag\"");
            assert!(matches!(c.get(key), Fetch::Bytes(b) if b == b"base"));
            assert!(matches!(c.layer(Layer::Img).get(key), Fetch::Absent));
            assert_eq!(c.layer(Layer::Dem).tag(key).as_deref(), Some("\"etag\""));
            assert_eq!(c.cached_bytes(), 12);
            c.flush();
        }
        let c = CellFiles::open(&dir);
        assert_eq!(c.cached_bytes(), 12, "the index read back");
        assert!(matches!(c.layer(Layer::Dem).get(key), Fetch::Bytes(b) if b == b"terrain!"));
        drop(c);
        std::fs::remove_file(dir.join(format!("e{CELL_EPOCH}")).join("index.vsf")).unwrap();
        let c = CellFiles::open(&dir);
        assert_eq!(c.cached_bytes(), 12, "the index made again from the files");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The purge takes the oldest unpinned cells, imagery before terrain before bases, and never a pinned one.
    #[test]
    fn the_purge_keeps_pins_and_bases_longest() {
        let dir = scratch("purge");
        let c = CellFiles::open(&dir);
        let (a, b, p) = (CellKey { depth: 12, prefix: 1 }, CellKey { depth: 12, prefix: 2 }, CellKey { depth: 12, prefix: 3 });
        c.put(a, &[0; 100]);
        c.layer(Layer::Img).put(b, &[0; 100]);
        c.put(p, &[0; 100]);
        c.set_pinned([p]);
        assert_eq!(c.purge_to(100), 1);
        assert!(matches!(c.layer(Layer::Img).get(b), Fetch::Absent), "the imagery went first");
        assert!(matches!(c.get(a), Fetch::Bytes(_)));
        assert_eq!(c.purge_to(0), 1);
        assert!(matches!(c.get(p), Fetch::Bytes(_)), "a pinned cell stays whatever the budget");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
