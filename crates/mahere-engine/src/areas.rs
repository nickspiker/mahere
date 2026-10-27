//! Areas: a selection's shape saved under a name, downloaded to a chosen detail, pinned in the cache, shared as a file (Nick 2026-10-10: "snapshot and share the shape").
//!
//! The shape itself is `mahere_tiles::area::Area`. This is what a host keeps: the list, the store they persist in, the download in progress for one, and the estimate the panel shows.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use mahere_tiles::CellKey;
pub use mahere_tiles::area::{AREA_MAX_DEPTH, AREA_MIN_DEPTH, Area};
use rustc_hash::FxHashSet;

use crate::ink::UserLayer;
use crate::residency::{CellStore, Fetch};

/// Where areas live between runs: the vault on the phone, a directory on a desktop. One document each, by name.
pub trait AreaStore: Send + Sync {
    fn put(&self, name: &str, bytes: &[u8]);
    fn delete(&self, name: &str);
    fn load_all(&self) -> Vec<Vec<u8>>;
}

/// A download's counters, shared with its thread.
#[derive(Default)]
pub struct Progress {
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub bytes: AtomicU64,
    pub absent: AtomicU64,
    pub failed: AtomicU64,
    pub cancel: AtomicBool,
    pub finished: AtomicBool,
}

/// One area as a host holds it: the shape, its estimate at its depth, and a download if one runs.
pub struct AreaEntry {
    pub area: Area,
    /// (cells, bytes) down to the area's depth, from the mean cell sizes.
    pub estimate: (u64, u64),
    pub download: Option<Arc<Progress>>,
}

/// What the panel shows of an area.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AreaView {
    pub name: String,
    pub depth: u8,
    pub cells: u64,
    pub est_bytes: u64,
    /// Bytes found by the last completed download, if one completed on this device.
    pub fetched: Option<u64>,
    /// A download in progress: cells done, cells in all, bytes so far.
    pub progress: Option<(u64, u64, u64)>,
}

/// Every area, and whether there is a selection to save as a new one.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AreasView {
    pub areas: Vec<AreaView>,
    pub has_selection: bool,
}

pub struct Areas {
    pub list: Vec<AreaEntry>,
    store: Option<Arc<dyn AreaStore>>,
    cells: Arc<dyn CellStore>,
    /// The other layers' stores, when cells come published in layers: an area's download takes its terrain and imagery too, so both are there offline.
    extra: Vec<Arc<dyn CellStore>>,
    /// Where address documents come from: an area's download takes the ZIPs it covers, so an address there is found offline (Nick 2026-10-27).
    docs: Option<Arc<dyn crate::geocode::DocStore>>,
}

/// Eagle Time now.
fn et_now() -> i64 {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    vsf::types::eagle_time::from_unix_ns(d.as_secs() as i64, d.subsec_nanos())
}

impl Areas {
    pub fn new(cells: Arc<dyn CellStore>) -> Areas {
        Areas { list: Vec::new(), store: None, cells, extra: Vec::new(), docs: None }
    }

    /// Downloads take these layers (the terrain, the imagery) as well as the base.
    pub fn set_layer_stores(&mut self, stores: Vec<Arc<dyn CellStore>>) {
        self.extra = stores;
    }

    /// The store, and everything in it loaded.
    pub fn set_store(&mut self, store: Arc<dyn AreaStore>) {
        for bytes in store.load_all() {
            if let Some(a) = Area::decode(&bytes) {
                self.push(a);
            }
        }
        self.list.sort_by_key(|e| e.area.created);
        self.store = Some(store);
    }

    fn push(&mut self, area: Area) -> usize {
        let estimate = area.estimate_to(area.depth);
        self.list.push(AreaEntry { area, estimate, download: None });
        self.list.len() - 1
    }

    fn save(&self, i: usize) {
        if let (Some(store), Some(e)) = (&self.store, self.list.get(i)) {
            store.put(&e.area.name, &e.area.encode(false));
        }
    }

    /// A name no area has yet: the world words of the shape's centre, numbered if taken.
    fn fresh_name(&self, area: &Area) -> String {
        let base = area
            .centre()
            .map(|(lat, lon)| vsf::types::WorldCoord::from_lat_lon(lat, lon).to_words().split_whitespace().take(2).collect::<Vec<_>>().join(" "))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "area".into());
        let taken = |n: &str| self.list.iter().any(|e| e.area.name == n);
        if !taken(&base) {
            return base;
        }
        (2..).map(|n| format!("{base} {n}")).find(|n| !taken(n)).unwrap()
    }

    /// The selection as a new area at `depth`: the planes made disjoint, the name from the place. None when there is no selection.
    pub fn save_selection(&mut self, user: &UserLayer, depth: u8) -> Option<usize> {
        let planes = mahere_tiles::area::disjoint_planes(user.drawn_selection());
        if planes.is_empty() {
            return None;
        }
        let mut area = Area { name: String::new(), created: et_now(), depth: depth.clamp(AREA_MIN_DEPTH, AREA_MAX_DEPTH), cells: planes, fetched: None };
        area.name = self.fresh_name(&area);
        let i = self.push(area);
        self.save(i);
        Some(i)
    }

    /// A shared area document taken in: kept under its own name (numbered if taken).
    pub fn import(&mut self, bytes: &[u8]) -> Option<usize> {
        let mut area = Area::decode(bytes)?;
        area.fetched = None;
        if area.name.is_empty() || self.list.iter().any(|e| e.area.name == area.name) {
            area.name = self.fresh_name(&area);
        }
        let i = self.push(area);
        self.save(i);
        Some(i)
    }

    pub fn set_depth(&mut self, i: usize, depth: u8) {
        let Some(e) = self.list.get_mut(i) else { return };
        let depth = depth.clamp(AREA_MIN_DEPTH, AREA_MAX_DEPTH);
        if e.area.depth == depth {
            return;
        }
        e.area.depth = depth;
        e.estimate = e.area.estimate_to(depth);
        self.save(i);
    }

    pub fn delete(&mut self, i: usize) {
        if i >= self.list.len() {
            return;
        }
        self.cancel(i);
        let e = self.list.remove(i);
        if let Some(store) = &self.store {
            store.delete(&e.area.name);
        }
    }

    /// The document to share: the shape without this device's record.
    pub fn share_bytes(&self, i: usize) -> Option<(String, Vec<u8>)> {
        self.list.get(i).map(|e| (e.area.name.clone(), e.area.encode(true)))
    }

    /// Fetch every cell of the area down to its depth, on threads of their own, through the same store the view reads (so the vault keeps them). A refresh forgets what the store remembered as absent, so cells baked since are found, and checks every kept cell against the bucket, so cells baked again are taken anew.
    pub fn download(&mut self, i: usize, refresh: bool) {
        let Some(e) = self.list.get_mut(i) else { return };
        if e.download.as_ref().is_some_and(|p| !p.finished.load(Ordering::Relaxed)) {
            return;
        }
        if refresh {
            self.cells.forget_absent();
        }
        let progress = Arc::new(Progress::default());
        let cells: Vec<CellKey> = e.area.all_cells_to(e.area.depth);
        progress.total.store(cells.len() as u64, Ordering::Relaxed);
        e.download = Some(progress.clone());
        let (store, extra) = (self.cells.clone(), self.extra.clone());
        // The ZIPs whose boxes meet the area's, from the index: their documents fetched with the cells.
        if let Some(docs) = self.docs.clone() {
            let corners: Vec<(f64, f64)> = e.area.cells.iter().flat_map(|(k, _)| k.corners()).collect();
            let (la0, lo0, la1, lo1) = corners.iter().fold((90.0f64, 180.0f64, -90.0f64, -180.0f64), |b, &(a, o)| (b.0.min(a), b.1.min(o), b.2.max(a), b.3.max(o)));
            std::thread::spawn(move || {
                let Some(index) = docs.doc(mahere_tiles::addr::INDEX_PATH).and_then(|b| mahere_tiles::addr::ZipIndex::decode(&b)) else { return };
                let zips: Vec<u32> = index.zips.iter().filter(|(_, b)| b[0] <= la1 && b[2] >= la0 && b[1] <= lo1 && b[3] >= lo0).map(|z| z.0).collect();
                for z in &zips {
                    let _ = docs.doc(&mahere_tiles::addr::zip_path(*z));
                }
                eprintln!("areas: {} ZIP documents on the device for the area", zips.len());
            });
        }
        std::thread::spawn(move || fetch_all(store, extra, cells, progress, refresh));
    }

    /// Where address documents come from, for an area's download to take its ZIPs.
    pub fn set_docs(&mut self, docs: Arc<dyn crate::geocode::DocStore>) {
        self.docs = Some(docs);
    }

    pub fn cancel(&mut self, i: usize) {
        if let Some(p) = self.list.get(i).and_then(|e| e.download.as_ref()) {
            p.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Finished downloads folded into their areas: the bytes found recorded and saved. Returns whether anything changed for the panel.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        for i in 0..self.list.len() {
            let finished = self.list[i].download.as_ref().is_some_and(|p| p.finished.load(Ordering::Relaxed));
            if finished {
                let p = self.list[i].download.take().unwrap();
                if !p.cancel.load(Ordering::Relaxed) {
                    self.list[i].area.fetched = Some((et_now(), p.bytes.load(Ordering::Relaxed)));
                    self.save(i);
                }
                changed = true;
            }
        }
        changed
    }

    pub fn downloading(&self) -> bool {
        self.list.iter().any(|e| e.download.is_some())
    }

    /// Every cell any area wants, down to its depth: what the cache must keep.
    pub fn pinned(&self) -> FxHashSet<CellKey> {
        let mut set = FxHashSet::default();
        for e in &self.list {
            set.extend(e.area.all_cells_to(e.area.depth));
        }
        set
    }

    pub fn view(&self, has_selection: bool) -> AreasView {
        let areas = self
            .list
            .iter()
            .map(|e| AreaView {
                name: e.area.name.clone(),
                depth: e.area.depth,
                cells: e.estimate.0,
                est_bytes: e.estimate.1,
                fetched: e.area.fetched.map(|(_, b)| b),
                progress: e.download.as_ref().map(|p| (p.done.load(Ordering::Relaxed), p.total.load(Ordering::Relaxed), p.bytes.load(Ordering::Relaxed))),
            })
            .collect();
        AreasView { areas, has_selection }
    }
}

/// The fetch itself: depth by depth so a cell found absent spares its descendants, four cells at a time within a depth.
fn fetch_all(store: Arc<dyn CellStore>, extra: Vec<Arc<dyn CellStore>>, cells: Vec<CellKey>, progress: Arc<Progress>, refresh: bool) {
    let absent = std::sync::Mutex::new(FxHashSet::<CellKey>::default());
    let max_depth = cells.iter().map(|k| k.depth).max().unwrap_or(0);
    'depths: for d in 0..=max_depth {
        let level: Vec<CellKey> = cells.iter().copied().filter(|k| k.depth == d).collect();
        if level.is_empty() {
            continue;
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    loop {
                        if progress.cancel.load(Ordering::Relaxed) {
                            return;
                        }
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&key) = level.get(i) else { return };
                        // Under a cell found absent nothing exists.
                        let under_absent = {
                            let ab = absent.lock().unwrap();
                            (0..key.depth).any(|a| ab.contains(&key.ancestor(a)))
                        };
                        if under_absent {
                            absent.lock().unwrap().insert(key);
                            progress.absent.fetch_add(1, Ordering::Relaxed);
                        } else {
                            // Download again checks every kept cell against the bucket and takes the newer copy (Nick 2026-10-13).
                            match if refresh { store.refresh(key) } else { store.get(key) } {
                                Fetch::Bytes(b) => {
                                    progress.bytes.fetch_add(b.len() as u64, Ordering::Relaxed);
                                    // The cell's other layers beside it, absent or not.
                                    for i in &extra {
                                        match if refresh { i.refresh(key) } else { i.get(key) } {
                                            Fetch::Bytes(b) => {
                                                progress.bytes.fetch_add(b.len() as u64, Ordering::Relaxed);
                                            }
                                            Fetch::Failed(_) => {
                                                progress.failed.fetch_add(1, Ordering::Relaxed);
                                            }
                                            Fetch::Absent => {}
                                        }
                                    }
                                }
                                Fetch::Absent => {
                                    absent.lock().unwrap().insert(key);
                                    progress.absent.fetch_add(1, Ordering::Relaxed);
                                }
                                Fetch::Failed(_) => {
                                    progress.failed.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        progress.done.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });
        if progress.cancel.load(Ordering::Relaxed) {
            break 'depths;
        }
    }
    progress.finished.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Camera;
    use std::sync::Mutex;

    /// A store that counts what it was asked for and has every cell at depth 10 and above, nothing below.
    struct Counting(Mutex<Vec<CellKey>>);

    impl CellStore for Counting {
        fn get(&self, key: CellKey) -> Fetch {
            self.0.lock().unwrap().push(key);
            if key.depth <= 10 { Fetch::Bytes(vec![0; 100]) } else { Fetch::Absent }
        }
    }

    #[derive(Default)]
    struct MemStore(Mutex<std::collections::HashMap<String, Vec<u8>>>);

    impl AreaStore for MemStore {
        fn put(&self, name: &str, bytes: &[u8]) {
            self.0.lock().unwrap().insert(name.to_string(), bytes.to_vec());
        }
        fn delete(&self, name: &str) {
            self.0.lock().unwrap().remove(name);
        }
        fn load_all(&self) -> Vec<Vec<u8>> {
            self.0.lock().unwrap().values().cloned().collect()
        }
    }

    fn selected_layer() -> UserLayer {
        let mut u = UserLayer::default();
        let cam = Camera::new(46.2, -122.19, 24000.0, 0.0);
        u.select(&cam, 800, 600, 11, &[(200.0, 200.0), (400.0, 200.0), (400.0, 400.0), (200.0, 400.0)], true);
        u
    }

    #[test]
    fn a_selection_becomes_an_area_that_downloads_and_spares_what_is_absent() {
        let counting = Arc::new(Counting(Mutex::new(Vec::new())));
        let store = Arc::new(MemStore::default());
        let mut areas = Areas::new(counting.clone());
        areas.set_store(store.clone());
        let i = areas.save_selection(&selected_layer(), 12).unwrap();
        assert!(!areas.list[i].area.name.is_empty());
        assert!(areas.list[i].estimate.0 > 12, "cells down to depth 12: {:?}", areas.list[i].estimate);
        assert_eq!(store.load_all().len(), 1, "saved");
        areas.download(i, false);
        while !areas.poll() {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let fetched = areas.list[i].area.fetched.unwrap().1;
        assert!(fetched > 0);
        let asked = counting.0.lock().unwrap().clone();
        assert!(asked.iter().all(|k| k.depth <= 11), "nothing under an absent cell was asked for: {:?}", asked.iter().filter(|k| k.depth > 11).count());
        assert!(asked.iter().any(|k| k.depth == 11), "the first absent level was probed");
        // The pinned set is the whole pyramid down to the depth.
        assert_eq!(areas.pinned().len() as u64, areas.list[i].estimate.0);
        // Round trip through the store.
        let mut again = Areas::new(counting.clone());
        again.set_store(store.clone());
        assert_eq!(again.list.len(), 1);
        assert_eq!(again.list[0].area.fetched.map(|f| f.1), Some(fetched));
        // Shared bytes come back as a second area under a numbered name.
        let (_, bytes) = again.share_bytes(0).unwrap();
        let j = again.import(&bytes).unwrap();
        assert_ne!(again.list[j].area.name, again.list[0].area.name);
        assert_eq!(again.list[j].area.fetched, None);
        again.delete(0);
        assert_eq!(store.load_all().len(), 1);
    }
}
