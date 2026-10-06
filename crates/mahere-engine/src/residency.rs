//! Clipmap residency: which cells are decoded and resident, and the loader
//! thread that feeds them. Zero locks anywhere near the pixel loop — the
//! main thread sends generation-stamped want-lists, the loader reads /
//! decompresses / decodes / repacks off-thread, and the main thread drains
//! a channel of finished planes at frame start.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use mahere_tiles::{CellKey, TRI, decode_cell_fields, plane_f32_mem, plane_u8_mem};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Where cell bytes come from, addressed by (layer, cell) — a string path
/// exists only where a filesystem or URL demands one.
pub trait CellStore: Send + Sync + 'static {
    fn get(&self, layer: Layer, key: CellKey) -> Option<Vec<u8>>;
}

pub struct DirStore(pub PathBuf);

impl CellStore for DirStore {
    fn get(&self, layer: Layer, key: CellKey) -> Option<Vec<u8>> {
        std::fs::read(self.0.join(key.path(layer.name()))).ok()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Layer {
    Dem,
    Line,
}

impl Layer {
    pub fn name(self) -> &'static str {
        match self {
            Layer::Dem => "dem",
            Layer::Line => "line",
        }
    }
}

/// Decoded dem cell, packed one u64 per triangle texel for the hot loop
/// (memory order `((ty << 8 | tx) << 1) | half`):
/// `[elev_q u16 | nx i16 | ny i16 | nz i16]`, elev_q = (elev + 500) * 4
/// clamped (0.25 m steps), 0xFFFF = no data. snorm16 normals (i16, not u8:
/// u8 bands on gentle slopes, exactly where hillshade banding shows).
pub struct DemPacked {
    pub texel: Box<[u64]>,
}

pub const ELEV_NODATA: u16 = 0xFFFF;

pub struct LinePlanes {
    pub class: Box<[u8]>,
    pub cov: Box<[u8]>,
}

pub enum Planes {
    Dem(DemPacked),
    Line(LinePlanes),
    /// Loader confirmed the file does not exist.
    Absent,
}

pub struct Loaded {
    pub layer: Layer,
    pub key: CellKey,
    pub planes: Planes,
}

struct WantList {
    generation: u64,
    /// Pre-sorted by priority (cell distance from view center).
    list: Vec<(Layer, CellKey)>,
}

pub enum Entry {
    Dem(DemPacked),
    Line(LinePlanes),
    Absent,
}

/// Resident cells for one layer: page table over decoded planes, LRU by
/// frame stamp. Plain maps — a few hundred entries, scanned only on insert
/// pressure, probed per BLOCK (not per pixel) during render.
pub struct Pool {
    pub map: FxHashMap<(u8, u64), Entry>,
    stamp: FxHashMap<(u8, u64), u64>,
    cap: usize,
}

impl Pool {
    #[cfg(test)]
    pub fn new_for_tests() -> Pool {
        Pool::new(64)
    }

    fn new(cap: usize) -> Pool {
        Pool { map: FxHashMap::default(), stamp: FxHashMap::default(), cap }
    }

    pub fn touch(&mut self, depth: u8, prefix: u64, frame: u64) {
        self.stamp.insert((depth, prefix), frame);
    }

    fn insert(&mut self, key: CellKey, e: Entry, frame: u64, desired: &FxHashSet<(Layer, u8, u64)>, layer: Layer) {
        if self.map.len() >= self.cap {
            // Evict the stalest resident not currently desired.
            let victim = self
                .map
                .keys()
                .filter(|&&(d, p)| !desired.contains(&(layer, d, p)))
                .min_by_key(|&&(d, p)| self.stamp.get(&(d, p)).copied().unwrap_or(0))
                .copied();
            if let Some(v) = victim {
                self.map.remove(&v);
                self.stamp.remove(&v);
            }
        }
        self.stamp.insert((key.depth, key.prefix), frame);
        self.map.insert((key.depth, key.prefix), e);
    }
}

pub struct Residency {
    want_tx: Sender<WantList>,
    done_rx: Receiver<Loaded>,
    pub dem: Pool,
    pub line: Pool,
    pending: FxHashSet<(Layer, u8, u64)>,
    pub desired: FxHashSet<(Layer, u8, u64)>,
    generation: u64,
    pub frame: u64,
}

impl Residency {
    pub fn new(store: Arc<dyn CellStore>) -> Residency {
        let (want_tx, want_rx) = channel::<WantList>();
        let (done_tx, done_rx) = channel::<Loaded>();
        std::thread::spawn(move || loader_thread(store, want_rx, done_tx));
        Residency {
            want_tx,
            done_rx,
            dem: Pool::new(192),
            line: Pool::new(224),
            pending: FxHashSet::default(),
            desired: FxHashSet::default(),
            generation: 0,
            frame: 0,
        }
    }

    /// Drain finished cells into the pools. Returns how many arrived.
    pub fn drain(&mut self) -> usize {
        let mut n = 0;
        while let Ok(l) = self.done_rx.try_recv() {
            self.pending.remove(&(l.layer, l.key.depth, l.key.prefix));
            let entry = match l.planes {
                Planes::Dem(p) => Entry::Dem(p),
                Planes::Line(p) => Entry::Line(p),
                Planes::Absent => Entry::Absent,
            };
            match l.layer {
                Layer::Dem => self.dem.insert(l.key, entry, self.frame, &self.desired, Layer::Dem),
                Layer::Line => self.line.insert(l.key, entry, self.frame, &self.desired, Layer::Line),
            }
            n += 1;
        }
        n
    }

    /// Declare the frame's desired set — the cells this view needs at its
    /// depth plus the parents it falls back through. Everything else is
    /// dropped now (out of view or zoom mismatch: gone, re-fetched if it
    /// comes back), in-flight requests outside it are forgotten, and only
    /// what's missing is requested, nearest-first.
    pub fn want(&mut self, mut list: Vec<(Layer, CellKey)>, center: (u64, u64)) {
        self.desired = list.iter().map(|&(l, k)| (l, k.depth, k.prefix)).collect();
        let desired = &self.desired;
        self.dem.map.retain(|&(d, p), _| desired.contains(&(Layer::Dem, d, p)));
        self.dem.stamp.retain(|&(d, p), _| desired.contains(&(Layer::Dem, d, p)));
        self.line.map.retain(|&(d, p), _| desired.contains(&(Layer::Line, d, p)));
        self.line.stamp.retain(|&(d, p), _| desired.contains(&(Layer::Line, d, p)));
        self.pending.retain(|id| desired.contains(id));
        list.retain(|&(l, k)| {
            let id = (l, k.depth, k.prefix);
            !self.pending.contains(&id)
                && !match l {
                    Layer::Dem => self.dem.map.contains_key(&(k.depth, k.prefix)),
                    Layer::Line => self.line.map.contains_key(&(k.depth, k.prefix)),
                }
        });
        if list.is_empty() {
            return;
        }
        list.sort_by_key(|&(_, k)| {
            let (cu, cv) = k.grid();
            // Chebyshev distance in this depth's grid, normalized by shifting
            // the center (given at depth 30-ish precision) down.
            let sh = 30 - k.depth as u32;
            let (ku, kv) = (center.0 >> sh, center.1 >> sh);
            (cu.abs_diff(ku)).max(cv.abs_diff(kv))
        });
        for &(l, k) in &list {
            self.pending.insert((l, k.depth, k.prefix));
        }
        self.generation += 1;
        let _ = self.want_tx.send(WantList { generation: self.generation, list });
    }

    pub fn converged(&self) -> bool {
        self.pending.is_empty()
    }
}

/// The newest want-list is the only one that matters: an older list's
/// leftovers are cells the view no longer needs (the main side forgets
/// them as pending too, so they're re-requested if they come back).
/// Cells decode in parallel a small chunk at a time so the nearest-first
/// order still holds and a newer list preempts within a few cells.
fn loader_thread(store: Arc<dyn CellStore>, want_rx: Receiver<WantList>, done_tx: Sender<Loaded>) {
    let chunk = rayon::current_num_threads().clamp(2, 8);
    let mut current: Option<WantList> = None;
    loop {
        if current.is_none() {
            match want_rx.recv() {
                Ok(w) => current = Some(w),
                Err(_) => return,
            }
        }
        while let Ok(w) = want_rx.try_recv() {
            current = Some(w);
        }
        let Some(w) = current.take() else { continue };
        let mut list = w.list;
        while !list.is_empty() {
            let n = list.len().min(chunk);
            let batch: Vec<(Layer, CellKey)> = list.drain(..n).collect();
            let loaded: Vec<Loaded> = batch.par_iter().map(|&(l, k)| load_cell(&*store, l, k)).collect();
            for l in loaded {
                if done_tx.send(l).is_err() {
                    return;
                }
            }
            if let Ok(newer) = want_rx.try_recv() {
                current = Some(newer);
                break;
            }
        }
    }
}

fn load_cell(store: &dyn CellStore, layer: Layer, key: CellKey) -> Loaded {
    let Some(bytes) = store.get(layer, key) else {
        return Loaded { layer, key, planes: Planes::Absent };
    };
    let Ok(fields) = decode_cell_fields(&bytes) else {
        return Loaded { layer, key, planes: Planes::Absent };
    };
    let planes = match layer {
        Layer::Line => {
            match (fields.get("class").and_then(plane_u8_mem), fields.get("cov").and_then(plane_u8_mem)) {
                (Some(class), Some(cov)) => Planes::Line(LinePlanes {
                    class: class.into_boxed_slice(),
                    cov: cov.into_boxed_slice(),
                }),
                _ => Planes::Absent,
            }
        }
        Layer::Dem => {
            let (Some(elev), Some(ge), Some(gn)) = (
                fields.get("elev").and_then(plane_f32_mem),
                fields.get("ge").and_then(plane_f32_mem),
                fields.get("gn").and_then(plane_f32_mem),
            ) else {
                return Loaded { layer, key, planes: Planes::Absent };
            };
            let mut texel = vec![0u64; TRI].into_boxed_slice();
            for i in 0..TRI {
                let e = elev[i];
                let eq: u16 = if e.is_nan() {
                    ELEV_NODATA
                } else {
                    ((e + 500.0) * 4.0).clamp(0.0, 65534.0) as u16
                };
                let inv = 1.0 / (1.0 + ge[i] * ge[i] + gn[i] * gn[i]).sqrt();
                let nx = (-ge[i] * inv * 32767.0) as i16;
                let ny = (-gn[i] * inv * 32767.0) as i16;
                let nz = (inv * 32767.0) as i16;
                texel[i] = (eq as u64)
                    | ((nx as u16 as u64) << 16)
                    | ((ny as u16 as u64) << 32)
                    | ((nz as u16 as u64) << 48);
            }
            Planes::Dem(DemPacked { texel })
        }
    };
    Loaded { layer, key, planes }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A store with nothing in it: every request resolves to Absent, which
    /// is enough to exercise the residency policy end to end.
    struct Empty;
    impl CellStore for Empty {
        fn get(&self, _layer: Layer, _key: CellKey) -> Option<Vec<u8>> {
            None
        }
    }

    fn settle(r: &mut Residency) {
        for _ in 0..500 {
            r.drain();
            if r.converged() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("loader never converged");
    }

    #[test]
    fn view_only_residency() {
        let mut r = Residency::new(Arc::new(Empty));
        let a = CellKey { depth: 8, prefix: 0x3_0000 };
        let b = CellKey { depth: 8, prefix: 0x3_0001 };
        r.want(vec![(Layer::Dem, a)], (0, 0));
        settle(&mut r);
        assert!(r.dem.map.contains_key(&(8, a.prefix)));
        // A new view that no longer needs `a`: it's dropped at once, `b` is
        // requested, and nothing stays pending for the old view.
        r.want(vec![(Layer::Dem, b)], (0, 0));
        assert!(!r.dem.map.contains_key(&(8, a.prefix)));
        assert!(r.pending.contains(&(Layer::Dem, 8, b.prefix)));
        settle(&mut r);
        assert!(r.dem.map.contains_key(&(8, b.prefix)));
        // Re-wanting a resident cell requests nothing.
        r.want(vec![(Layer::Dem, b)], (0, 0));
        assert!(r.converged());
    }
}

// ==================== TIERED STORE: VAULT CACHE OVER THE BUCKET ====================

/// Where the baked cells live publicly. The path under it is exactly a
/// cell's `CellKey::path`, so the bake directory and the bucket are the
/// same thing.
pub const DEFAULT_CELLS_URL: &str = "https://brobdingnagian.holdmyoscilloscope.com/mahere/cells";

/// A store that can also keep what it's given (the vault).
pub trait CellCache: CellStore {
    fn put(&self, layer: Layer, key: CellKey, bytes: &[u8]);
}

/// The far tier. `Ok(None)` is a definite absence (the cell was never
/// baked); `Err` is a failure to find out (offline, timeout), which must
/// not be remembered as absence.
pub trait RemoteStore: Send + Sync + 'static {
    fn fetch(&self, layer: Layer, key: CellKey) -> Result<Option<Vec<u8>>, String>;
}

/// Cache first, then the bucket, writing hits through. Definite misses
/// are remembered for the process (most of the world isn't baked yet);
/// failures are not, so a cell that couldn't be fetched is tried again
/// the next time the view wants it.
pub struct TieredStore {
    cache: Arc<dyn CellCache>,
    remote: Arc<dyn RemoteStore>,
    missing: std::sync::Mutex<FxHashSet<(Layer, u8, u64)>>,
}

impl TieredStore {
    pub fn new(cache: Arc<dyn CellCache>, remote: Arc<dyn RemoteStore>) -> TieredStore {
        TieredStore { cache, remote, missing: std::sync::Mutex::new(FxHashSet::default()) }
    }
}

impl CellStore for TieredStore {
    fn get(&self, layer: Layer, key: CellKey) -> Option<Vec<u8>> {
        if let Some(b) = self.cache.get(layer, key) {
            return Some(b);
        }
        let id = (layer, key.depth, key.prefix);
        if self.missing.lock().unwrap().contains(&id) {
            return None;
        }
        match self.remote.fetch(layer, key) {
            Ok(Some(b)) => {
                self.cache.put(layer, key, &b);
                Some(b)
            }
            Ok(None) => {
                self.missing.lock().unwrap().insert(id);
                None
            }
            Err(e) => {
                eprintln!("cell fetch {}: {e}", key.path(layer.name()));
                None
            }
        }
    }
}

/// The bucket over HTTPS. 404 is absence; anything else is a failure.
pub struct HttpStore {
    base: String,
    agent: ureq::Agent,
}

impl HttpStore {
    pub fn new(base: &str) -> HttpStore {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(10))
            .timeout_read(std::time::Duration::from_secs(30))
            .build();
        HttpStore { base: base.trim_end_matches('/').to_string(), agent }
    }
}

impl RemoteStore for HttpStore {
    fn fetch(&self, layer: Layer, key: CellKey) -> Result<Option<Vec<u8>>, String> {
        let url = format!("{}/{}", self.base, key.path(layer.name()));
        match self.agent.get(&url).call() {
            Ok(resp) => {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf).map_err(|e| e.to_string())?;
                Ok(Some(buf))
            }
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// The bucket with no cache at all (a desktop without a vault).
impl CellStore for HttpStore {
    fn get(&self, layer: Layer, key: CellKey) -> Option<Vec<u8>> {
        self.fetch(layer, key).ok().flatten()
    }
}
