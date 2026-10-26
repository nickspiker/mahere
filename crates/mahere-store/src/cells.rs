//! The cell cache on the device, in the vault (Nick 2026-10-25: everything in the vault, which is ferros' file system and has to be fast enough to hold it). Each cell's layer is a bulk vault entry under its VSF key (the keys cells had before 0.223), its ETag beside it, and an index entry records when each was last used and its size, so the cache is purged oldest-first to a budget and an area's cells are kept whatever the budget.
//! Cells left the vault on 2026-10-18 because it opened by walking every entry (8 to 63 s on the phone) and wrote each cell as a durable commit of its own; kete now opens lazily and writes in batches (manifestus 2026-10-25: 1.1 ms a 40 KB cell at 128 a batch, the open 30 ms at any size). The plain files they went to named cells by their bucket names, which a Mac's or Windows's file system takes letter case out of: two cells named alike but for case were one file there, each showing the other's roads and names.
//! Writes from the loaders queue here and go to the vault a burst at a time ([`kete::FlatStorage::write_device_bulk`], kept out of kete's read cache); reads see a queued or landing write at once.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use kete::FlatStorage;
use mahere_engine::residency::{CELL_EPOCH, CellCache, CellStore, Fetch};
use mahere_tiles::CellKey;
use mahere_tiles::publish::Layer;

use crate::{cell_key, cells_index_key, decode_cell_index, encode_cell_index, et_now, tag_key};

/// New cells between writes of the index: a cell the index has not heard of when the app dies is one the budget does not count.
const FLUSH_EVERY: u32 = 256;

/// How long the writer lets a burst of puts gather before it writes them as one batch.
const GATHER: Duration = Duration::from_millis(40);

/// A layer's code: in the index key's depth byte (depths stop at 30) and in the vault keys.
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

#[derive(Default)]
struct Index {
    /// (depth with the layer in its top bits, raw cell) → (last use, Eagle Time; bytes).
    entries: HashMap<(u8, u64), (i64, u32)>,
    /// New cells since the last write of the index.
    unsaved: u32,
}

/// The writes on their way to the vault: queued, and the batch being written (an Arc, so a reader looks in it without holding up the write).
#[derive(Default)]
struct Writes {
    queue: Mutex<Vec<(String, Vec<u8>)>>,
    landing: Mutex<Arc<Vec<(String, Vec<u8>)>>>,
    wake: Condvar,
}

impl Writes {
    /// A write queued, replacing one to the same key still queued.
    fn push(&self, key: String, bytes: Vec<u8>) {
        let mut q = self.queue.lock().unwrap();
        match q.iter_mut().find(|(k, _)| *k == key) {
            Some(e) => e.1 = bytes,
            None => q.push((key, bytes)),
        }
        self.wake.notify_one();
    }

    /// The newest bytes queued or landing for `key`.
    fn find(&self, key: &str) -> Option<Vec<u8>> {
        if let Some((_, b)) = self.queue.lock().unwrap().iter().rev().find(|(k, _)| k == key) {
            return Some(b.clone());
        }
        let landing = self.landing.lock().unwrap().clone();
        landing.iter().rev().find(|(k, _)| k == key).map(|(_, b)| b.clone())
    }
}

/// The device's cells, in the vault.
pub struct VaultCells {
    vault: Arc<FlatStorage>,
    index: Mutex<Index>,
    /// Cells an area keeps on the device: never purged, and not counted against the budget.
    pinned: Mutex<HashSet<(u8, u64)>>,
    writes: Arc<Writes>,
}

impl VaultCells {
    /// The cell cache in `vault`, its index read, its writer started.
    pub fn open(vault: Arc<FlatStorage>) -> Arc<VaultCells> {
        let entries = vault.read_device(&cells_index_key()).ok().flatten().map(|b| decode_cell_index(&b)).unwrap_or_default();
        let writes = Arc::new(Writes::default());
        let (w, v) = (writes.clone(), vault.clone());
        if let Err(e) = std::thread::Builder::new().name("cells into the vault".into()).spawn(move || writer(&v, &w)) {
            eprintln!("cells: the vault's writer did not start ({e}), cells are not kept");
        }
        Arc::new(VaultCells { vault, index: Mutex::new(Index { entries, unsaved: 0 }), pinned: Mutex::new(HashSet::new()), writes })
    }

    fn ix(key: CellKey, code: u8) -> (u8, u64) {
        (key.depth | code << 6, key.raw())
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

    /// Write the index (queued with the cells).
    pub fn flush(&self) {
        let bytes = {
            let mut ix = self.index.lock().unwrap();
            ix.unsaved = 0;
            encode_cell_index(&ix.entries)
        };
        if let Some(b) = bytes {
            self.writes.push(cells_index_key(), b);
        }
    }

    fn get_layer(&self, key: CellKey, code: u8) -> Fetch {
        let k = cell_key(key, code);
        if let Some(b) = self.writes.find(&k) {
            return Fetch::Bytes(b);
        }
        match self.vault.read_device_bulk(&k) {
            Ok(Some(b)) => {
                self.touch(key, code, b.len(), et_now());
                Fetch::Bytes(b)
            }
            Ok(None) => Fetch::Absent,
            Err(e) => Fetch::Failed(format!("{e:?}")),
        }
    }

    fn put_layer(&self, key: CellKey, code: u8, bytes: &[u8]) {
        self.writes.push(cell_key(key, code), bytes.to_vec());
        self.touch(key, code, bytes.len(), et_now());
    }

    fn tag_of(&self, key: CellKey, code: u8) -> Option<String> {
        let k = tag_key(key, code);
        let b = match self.writes.find(&k) {
            Some(b) => b,
            None => self.vault.read_device_bulk(&k).ok()??,
        };
        String::from_utf8(b).ok()
    }

    fn set_tag_of(&self, key: CellKey, code: u8, tag: &str) {
        self.writes.push(tag_key(key, code), tag.as_bytes().to_vec());
    }

    /// Whether a cell's layer is kept here.
    pub fn has(&self, key: CellKey, code: u8) -> bool {
        self.index.lock().unwrap().entries.contains_key(&Self::ix(key, code))
    }

    /// The cells kept for one layer, any depth: what a search over the device reads (the base holds the names).
    pub fn keys(&self, layer: Layer) -> Vec<CellKey> {
        let code = layer_code(layer);
        self.index.lock().unwrap().entries.keys().filter(|(packed, _)| packed >> 6 == code).map(|&(packed, raw)| {
            let depth = packed & 63;
            CellKey { depth, prefix: raw >> (60 - 2 * depth as u32) }
        }).collect()
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

    /// Delete the least recently used unpinned cells until they fit `budget` bytes, one vault batch for the lot; returns how many went. The pinned cells stay whatever the budget.
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
        if victims.is_empty() {
            return 0;
        }
        let mut gone = Vec::with_capacity(victims.len() * 2);
        for &(packed, raw) in &victims {
            let (depth, code) = (packed & 63, packed >> 6);
            let key = CellKey { depth, prefix: raw >> (60 - 2 * depth as u32) };
            gone.push(cell_key(key, code));
            gone.push(tag_key(key, code));
        }
        if let Err(e) = self.vault.delete_device_batch(&gone) {
            eprintln!("cells: the purge's deletes failed, kept for the next: {e:?}");
            return 0;
        }
        {
            let mut ix = self.index.lock().unwrap();
            for k in &victims {
                ix.entries.remove(k);
            }
        }
        self.flush();
        victims.len()
    }

    /// One of a published cell's layers as a cache of its own over the same vault, index and pins.
    pub fn layer(self: &Arc<Self>, layer: Layer) -> Arc<VaultCellLayer> {
        Arc::new(VaultCellLayer { cells: self.clone(), code: layer_code(layer) })
    }

    /// The cells a Mahere from 2026-10-18 to 2026-10-25 kept as plain files under `dir`, moved into the vault on a thread of their own, the files deleted as they go: nothing fetched again, the areas' cells kept. The tree named by bucket names (`e<epoch>`) is dropped instead on a Mac or Windows, where two cells named alike but for case had shared one file and no copy there can be trusted. Stopped part way, the next start goes on with what is left.
    pub fn take_files(self: &Arc<Self>, dir: &Path) {
        let roots: Vec<(std::path::PathBuf, bool)> = [(dir.join(format!("e{CELL_EPOCH}-hex")), true), (dir.join(format!("e{CELL_EPOCH}")), false)].into_iter().filter(|(p, _)| p.is_dir()).collect();
        let dropped = dir.join(format!("e{CELL_EPOCH}.dropped"));
        if roots.is_empty() && !dropped.is_dir() {
            return;
        }
        let cells = self.clone();
        let spawned = std::thread::Builder::new().name("cell files into the vault".into()).spawn(move || {
            let _ = std::fs::remove_dir_all(&dropped);
            for (root, hex) in roots {
                if !hex && cfg!(any(target_os = "macos", target_os = "windows")) {
                    eprintln!("cells: the files kept under case-blind names are dropped and fetched again as they are needed");
                    let _ = std::fs::remove_dir_all(&root);
                    continue;
                }
                cells.take_tree(&root, hex);
            }
        });
        if let Err(e) = spawned {
            eprintln!("cells: the move of the cell files did not start: {e}");
        }
    }

    /// [`Self::take_files`] for one tree: its files read a few dozen at a time, written to the vault as one batch, then deleted.
    fn take_tree(&self, root: &Path, hex: bool) {
        let t = std::time::Instant::now();
        let mut files: Vec<(CellKey, u8, std::path::PathBuf)> = Vec::new();
        for code in 0..4u8 {
            let Ok(depths) = std::fs::read_dir(root.join(layer_dir(code))) else { continue };
            for d in depths.flatten() {
                let Ok(groups) = std::fs::read_dir(d.path()) else { continue };
                for g in groups.flatten() {
                    let Ok(entries) = std::fs::read_dir(g.path()) else { continue };
                    for f in entries.flatten() {
                        let name = f.file_name();
                        let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".vsf")) else { continue };
                        let key = if hex { CellKey::from_file_name(stem) } else { CellKey::from_name(stem) };
                        if let Some(key) = key {
                            files.push((key, code, f.path()));
                        }
                    }
                }
            }
        }
        let mut moved = 0usize;
        for chunk in files.chunks(32) {
            let mut batch: Vec<(String, Vec<u8>)> = Vec::with_capacity(chunk.len() * 2);
            let mut sizes = Vec::with_capacity(chunk.len());
            for (key, code, path) in chunk {
                let Ok(bytes) = std::fs::read(path) else { continue };
                sizes.push((*key, *code, bytes.len()));
                batch.push((cell_key(*key, *code), bytes));
                if let Ok(tag) = std::fs::read(path.with_extension("tag")) {
                    batch.push((tag_key(*key, *code), tag));
                }
            }
            if let Err(e) = self.vault.write_device_bulk(&batch) {
                eprintln!("cells: the move of the cell files stopped ({e:?}), the next start goes on");
                self.flush();
                return;
            }
            let now = et_now();
            for (key, code, len) in sizes {
                self.touch(key, code, len, now);
            }
            for (_, _, path) in chunk {
                let _ = std::fs::remove_file(path);
                let _ = std::fs::remove_file(path.with_extension("tag"));
            }
            moved += chunk.len();
        }
        self.flush();
        let _ = std::fs::remove_dir_all(root);
        eprintln!("cells: {moved} cell files moved into the vault in {:.1} s", t.elapsed().as_secs_f32());
    }
}

/// The writer: a burst of queued writes gathered, then written as one vault batch, while reads find them in `landing`.
fn writer(vault: &FlatStorage, writes: &Writes) {
    loop {
        {
            let mut q = writes.queue.lock().unwrap();
            while q.is_empty() {
                q = writes.wake.wait(q).unwrap();
            }
        }
        std::thread::sleep(GATHER);
        let batch = Arc::new(std::mem::take(&mut *writes.queue.lock().unwrap()));
        *writes.landing.lock().unwrap() = batch.clone();
        if let Err(e) = vault.write_device_bulk(&batch) {
            eprintln!("cells: {} writes to the vault failed, fetched again when wanted: {e:?}", batch.len());
        }
        *writes.landing.lock().unwrap() = Arc::new(Vec::new());
    }
}

/// A published cell's layer (the base, terrain, imagery or infrared) in the vault.
pub struct VaultCellLayer {
    cells: Arc<VaultCells>,
    code: u8,
}

impl CellStore for VaultCells {
    fn get(&self, key: CellKey) -> Fetch {
        self.get_layer(key, 0)
    }
}

impl CellCache for VaultCells {
    fn put(&self, key: CellKey, bytes: &[u8]) {
        self.put_layer(key, 0, bytes);
    }

    fn tag(&self, key: CellKey) -> Option<String> {
        self.tag_of(key, 0)
    }

    fn set_tag(&self, key: CellKey, tag: &str) {
        self.set_tag_of(key, 0, tag);
    }
}

impl CellStore for VaultCellLayer {
    fn get(&self, key: CellKey) -> Fetch {
        self.cells.get_layer(key, self.code)
    }
}

impl CellCache for VaultCellLayer {
    fn put(&self, key: CellKey, bytes: &[u8]) {
        self.cells.put_layer(key, self.code, bytes);
    }

    fn tag(&self, key: CellKey) -> Option<String> {
        self.cells.tag_of(key, self.code)
    }

    fn set_tag(&self, key: CellKey, tag: &str) {
        self.cells.set_tag_of(key, self.code, tag);
    }
}
