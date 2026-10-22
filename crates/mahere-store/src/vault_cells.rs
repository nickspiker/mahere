//! The cell cache in the vault (Nick 2026-10-22): every cell fetched from the bucket is kept as a device-scope vault entry of the CACHE class — sealed like everything the vault holds (which cells a device has is where its person has been), evicted by the vault's own reap as the tract laps, never grown for, never synced. The cells an area keeps are written durable instead, and stay whatever the budget.
//! It replaced the plain cell files of 0.223 (cells.rs), which left the cache readable to anyone holding the disk. Those files are brought in on a thread of their own the first time, then removed.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use kete::FlatStorage;
use mahere_engine::residency::{CellCache, CellStore, Fetch};
use mahere_tiles::CellKey;
use mahere_tiles::publish::Layer;

use crate::cells::layer_code;
use crate::{cell_key, tag_key};

/// The device's cells, in the vault.
pub struct VaultCells {
    vault: Arc<FlatStorage>,
    /// Cells an area keeps: written durable, so the reap moves them instead of evicting them.
    pinned: Mutex<HashSet<(u8, u64)>>,
}

impl VaultCells {
    pub fn open(vault: Arc<FlatStorage>) -> Arc<VaultCells> {
        Arc::new(VaultCells { vault, pinned: Mutex::new(HashSet::new()) })
    }

    fn pinned(&self, key: CellKey) -> bool {
        self.pinned.lock().unwrap().contains(&(key.depth, key.raw()))
    }

    fn get_layer(&self, key: CellKey, code: u8) -> Fetch {
        match self.vault.read_device(&cell_key(key, code)) {
            Ok(Some(b)) if !b.is_empty() => Fetch::Bytes(b),
            Ok(_) => Fetch::Absent,
            Err(e) => Fetch::Failed(format!("{e:?}")),
        }
    }

    fn put_layer(&self, key: CellKey, code: u8, bytes: &[u8]) {
        let k = cell_key(key, code);
        let r = if self.pinned(key) { self.vault.write_device(&k, bytes) } else { self.vault.write_device_cache(&k, bytes) };
        if let Err(e) = r {
            eprintln!("cells: {} not kept: {e:?}", key.name());
        }
    }

    fn tag_of(&self, key: CellKey, code: u8) -> Option<String> {
        self.vault.read_device(&tag_key(key, code)).ok().flatten().and_then(|b| String::from_utf8(b).ok())
    }

    fn set_tag_of(&self, key: CellKey, code: u8, tag: &str) {
        let k = tag_key(key, code);
        let _ = if self.pinned(key) { self.vault.write_device(&k, tag.as_bytes()) } else { self.vault.write_device_cache(&k, tag.as_bytes()) };
    }

    /// Bytes the cached cells hold (the vault's cache class).
    pub fn cached_bytes(&self) -> u64 {
        self.vault.vault_stats().map_or(0, |s| s.cache_blocks * s.block_bytes)
    }

    /// The cells the areas keep: the whole set, replacing the last. A cell newly kept is written again durable, one no longer kept is written again as cache — once, here, never on the paths that read.
    pub fn set_pinned(&self, keys: impl IntoIterator<Item = CellKey>) {
        let new: HashSet<(u8, u64)> = keys.into_iter().map(|k| (k.depth, k.raw())).collect();
        let old = std::mem::replace(&mut *self.pinned.lock().unwrap(), new.clone());
        let key_of = |&(depth, raw): &(u8, u64)| CellKey { depth, prefix: raw >> (60 - 2 * depth as u32) };
        for k in new.difference(&old).map(key_of) {
            for code in 0..4u8 {
                if let Ok(Some(b)) = self.vault.read_device(&cell_key(k, code)) {
                    let _ = self.vault.write_device(&cell_key(k, code), &b);
                }
            }
        }
        for k in old.difference(&new).map(key_of) {
            for code in 0..4u8 {
                if let Ok(Some(b)) = self.vault.read_device(&cell_key(k, code)) {
                    let _ = self.vault.write_device_cache(&cell_key(k, code), &b);
                }
            }
        }
    }

    /// The cache's allowance in bytes: the vault evicts the oldest cells past it while idle. The kept (pinned) cells do not count against it.
    pub fn purge_to(&self, budget: u64) {
        self.vault.set_cache_budget(budget);
    }

    /// Nothing to write: every cell is in the vault the moment it is kept.
    pub fn flush(&self) {}

    /// One of a published cell's layers as a cache of its own over the same vault and pins.
    pub fn layer(self: &Arc<Self>, layer: Layer) -> Arc<VaultCellLayer> {
        Arc::new(VaultCellLayer { cells: self.clone(), code: layer_code(layer) })
    }

    /// The plain cell files of 0.223 into the vault, on a thread of their own, then removed: the old directory, if any.
    pub fn adopt_files(self: &Arc<Self>, dir: std::path::PathBuf) {
        if !dir.is_dir() {
            return;
        }
        let me = self.clone();
        let _ = std::thread::Builder::new().name("cells into the vault".into()).spawn(move || {
            let files = crate::CellFiles::open(&dir);
            let n = files.drain_into(|key, code, bytes, tag| {
                me.put_layer(key, code, bytes);
                if let Some(t) = tag {
                    me.set_tag_of(key, code, t);
                }
            });
            drop(files);
            let _ = std::fs::remove_dir_all(&dir);
            eprintln!("cells: {n} cell files brought into the vault, the files removed");
        });
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
