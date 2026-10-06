//! Clipmap residency: which cells are decoded and resident, and the loader thread that feeds them. Zero locks anywhere near the pixel loop — the main thread sends want-lists, the loader reads / decompresses / decodes / repacks off-thread, and the main thread drains a channel of finished planes at frame start.
//!
//! One object per cell carries every layer, so residency is one pool keyed by cell; an entry holds whichever planes the cell had at its depth.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use mahere_tiles::{CellKey, ClassCell, CovCell, ImgCell, decode_cell};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// What a store found for a cell: bytes, a definite absence (never baked), or a failure to find out (offline, timeout, a bad read) — which must not be remembered as absence.
pub enum Fetch {
    Bytes(Vec<u8>),
    Absent,
    Failed(String),
}

/// Where cell bytes come from, addressed by cell — a string path exists only where a filesystem or URL demands one.
pub trait CellStore: Send + Sync + 'static {
    fn get(&self, key: CellKey) -> Fetch;
}

pub struct DirStore(pub PathBuf);

impl CellStore for DirStore {
    fn get(&self, key: CellKey) -> Fetch {
        match std::fs::read(self.0.join(key.path())) {
            Ok(b) => Fetch::Bytes(b),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Fetch::Absent,
            Err(e) => Fetch::Failed(e.to_string()),
        }
    }
}

/// Decoded dem planes packed one u64 per triangle texel for the hot loop (memory order `((ty << 8 | tx) << 1) | half`):
/// `[elev_q u16 | nx i16 | ny i16 | nz i16]`, elev_q = (elev + 500) * 4 (0.25 m steps), 0xFFFF = no data.
pub struct DemPacked {
    pub texel: Box<[u64]>,
}

/// A resident cell: whichever layers it carried. All `None` = the loader confirmed the object does not exist (absent), which still ends probing.
#[derive(Default)]
pub struct Entry {
    pub dem: Option<DemPacked>,
    pub line: Option<ClassCell>,
    pub land: Option<ClassCell>,
    pub water: Option<CovCell>,
    pub img: Option<ImgCell>,
}

impl Entry {
    pub fn is_absent(&self) -> bool {
        self.dem.is_none() && self.line.is_none() && self.land.is_none() && self.water.is_none() && self.img.is_none()
    }
    pub fn has_vec(&self) -> bool {
        self.line.is_some() || self.land.is_some() || self.water.is_some()
    }
}

pub struct Loaded {
    pub key: CellKey,
    /// `None` = the fetch failed; the cell is not resident and will be asked for again after a short backoff.
    pub entry: Option<Entry>,
}

struct WantList {
    /// Pre-sorted by priority (cell distance from view center).
    list: Vec<CellKey>,
}

/// Resident cells: page table over decoded planes. Plain map — a few hundred entries, probed per BLOCK (not per pixel) during render.
#[derive(Default)]
pub struct Pool {
    pub map: FxHashMap<CellKey, Entry>,
}

pub struct Residency {
    want_tx: Sender<WantList>,
    done_rx: Receiver<Loaded>,
    pub pool: Pool,
    pending: FxHashSet<CellKey>,
    /// Cells whose last fetch failed, with when: not re-requested until RETRY_AFTER has passed, so an offline phone doesn't hammer timeouts every frame.
    failed: FxHashMap<CellKey, std::time::Instant>,
    pub desired: FxHashSet<CellKey>,
    pub frame: u64,
}

const RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(3);

impl Residency {
    pub fn new(store: Arc<dyn CellStore>) -> Residency {
        let (want_tx, want_rx) = channel::<WantList>();
        let (done_tx, done_rx) = channel::<Loaded>();
        std::thread::spawn(move || loader_thread(store, want_rx, done_tx));
        Residency {
            want_tx,
            done_rx,
            pool: Pool::default(),
            pending: FxHashSet::default(),
            failed: FxHashMap::default(),
            desired: FxHashSet::default(),
            frame: 0,
        }
    }

    /// Drain finished cells into the pool. Returns how many arrived.
    pub fn drain(&mut self) -> usize {
        let mut n = 0;
        while let Ok(l) = self.done_rx.try_recv() {
            self.pending.remove(&l.key);
            match l.entry {
                Some(e) => {
                    self.pool.map.insert(l.key, e);
                    n += 1;
                }
                None => {
                    self.failed.insert(l.key, std::time::Instant::now());
                }
            }
        }
        n
    }

    /// Declare the frame's desired set — the cells this view needs at its depths plus the parents it falls back through. Everything else is dropped now (out of view or zoom mismatch: gone, re-fetched if it comes back), in-flight requests outside it are forgotten, and only what's missing is requested, nearest-first.
    pub fn want(&mut self, mut list: Vec<CellKey>, center: (u64, u64)) {
        self.desired = list.iter().copied().collect();
        let desired = &self.desired;
        self.pool.map.retain(|k, _| desired.contains(k));
        self.pending.retain(|k| desired.contains(k));
        let now = std::time::Instant::now();
        self.failed.retain(|k, t| desired.contains(k) && now.duration_since(*t) < RETRY_AFTER);
        list.retain(|k| !self.pending.contains(k) && !self.pool.map.contains_key(k) && !self.failed.contains_key(k));
        if list.is_empty() {
            return;
        }
        list.sort_by_key(|k| {
            let (cu, cv) = k.grid();
            // Chebyshev distance in this depth's grid, normalized by shifting the center (given at depth 30-ish precision) down.
            let sh = 30 - k.depth as u32;
            let (ku, kv) = (center.0 >> sh, center.1 >> sh);
            (cu.abs_diff(ku)).max(cv.abs_diff(kv))
        });
        for &k in &list {
            self.pending.insert(k);
        }
        let _ = self.want_tx.send(WantList { list });
    }

    pub fn converged(&self) -> bool {
        self.pending.is_empty()
    }
}

/// The newest want-list is the only one that matters: an older list's leftovers are cells the view no longer needs (the main side forgets them as pending too, so they're re-requested if they come back).
/// Cells decode in parallel a small chunk at a time so the nearest-first order still holds and a newer list preempts within a few cells.
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
            let batch: Vec<CellKey> = list.drain(..n).collect();
            let loaded: Vec<Loaded> = batch.par_iter().map(|&k| load_cell(&*store, k)).collect();
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

fn load_cell(store: &dyn CellStore, key: CellKey) -> Loaded {
    let bytes = match store.get(key) {
        Fetch::Bytes(b) => b,
        Fetch::Absent => return Loaded { key, entry: Some(Entry::default()) },
        Fetch::Failed(e) => {
            eprintln!("cell {} fetch failed: {e}", key.path());
            return Loaded { key, entry: None };
        }
    };
    let planes = match decode_cell(&bytes) {
        Ok(p) => p,
        Err(e) => {
            // A corrupt object: treat as a failure so it's retried (the vault may have a bad copy) rather than remembered as absent.
            eprintln!("cell {} decode failed ({} bytes): {e}", key.path(), bytes.len());
            return Loaded { key, entry: None };
        }
    };
    let dem = planes.dem.map(|d| DemPacked { texel: d.pack_texels(key).into_boxed_slice() });
    if std::env::var_os("MAHERE_TRACE").is_some() || cfg!(target_os = "android") {
        eprintln!("cell {} d{} loaded: dem={} line={} land={} water={} ({} bytes)", key.name(), key.depth, dem.is_some(), planes.line.is_some(), planes.land.is_some(), planes.water.is_some(), bytes.len());
    }
    Loaded { key, entry: Some(Entry { dem, line: planes.line, land: planes.land, water: planes.water, img: planes.img }) }
}

// ==================== TIERED STORE: VAULT CACHE OVER THE BUCKET ====================

/// Where the baked cells live publicly. The path under it is exactly a cell's `CellKey::path`, so the bake directory and the bucket are the same thing.
pub const DEFAULT_CELLS_URL: &str = "https://brobdingnagian.holdmyoscilloscope.com/mahere/cells";

/// Cell format epoch: bumped whenever the encoding changes incompatibly. It rides on the fetch URL as a query (so the CDN edge, which caches a key for hours, sees a new key) and in the vault key (so a cached cell of an older format is never read back as this one). Old clients keep fetching the old objects they understand.
pub const CELL_EPOCH: u32 = 2;

/// A store that can also keep what it's given (the vault).
pub trait CellCache: CellStore {
    fn put(&self, key: CellKey, bytes: &[u8]);
}

/// The far tier: the same outcomes as a store, over the network.
pub trait RemoteStore: Send + Sync + 'static {
    fn fetch(&self, key: CellKey) -> Fetch;
}

/// Cache first, then the bucket, writing hits through. Definite misses are remembered for the process (most of the world isn't baked yet); failures are not, so a cell that couldn't be fetched is tried again the next time the view wants it.
pub struct TieredStore {
    cache: Arc<dyn CellCache>,
    remote: Arc<dyn RemoteStore>,
    missing: std::sync::Mutex<FxHashSet<CellKey>>,
}

impl TieredStore {
    pub fn new(cache: Arc<dyn CellCache>, remote: Arc<dyn RemoteStore>) -> TieredStore {
        TieredStore { cache, remote, missing: std::sync::Mutex::new(FxHashSet::default()) }
    }
}

impl CellStore for TieredStore {
    fn get(&self, key: CellKey) -> Fetch {
        match self.cache.get(key) {
            Fetch::Bytes(b) => return Fetch::Bytes(b),
            Fetch::Failed(e) => eprintln!("vault read {} failed: {e}", key.path()),
            Fetch::Absent => {}
        }
        if self.missing.lock().unwrap().contains(&key) {
            return Fetch::Absent;
        }
        match self.remote.fetch(key) {
            Fetch::Bytes(b) => {
                self.cache.put(key, &b);
                Fetch::Bytes(b)
            }
            Fetch::Absent => {
                self.missing.lock().unwrap().insert(key);
                Fetch::Absent
            }
            Fetch::Failed(e) => Fetch::Failed(e),
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
    fn fetch(&self, key: CellKey) -> Fetch {
        let url = format!("{}/{}?e={}", self.base, key.path(), CELL_EPOCH);
        match self.agent.get(&url).call() {
            Ok(resp) => {
                let expect: Option<usize> = resp.header("Content-Length").and_then(|v| v.parse().ok());
                let mut buf = Vec::new();
                if let Err(e) = std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf) {
                    return Fetch::Failed(e.to_string());
                }
                if let Some(n) = expect {
                    if n != buf.len() {
                        return Fetch::Failed(format!("short body: {} of {n} bytes", buf.len()));
                    }
                }
                Fetch::Bytes(buf)
            }
            Err(ureq::Error::Status(404, _)) => Fetch::Absent,
            Err(e) => Fetch::Failed(e.to_string()),
        }
    }
}

/// The bucket with no cache at all (a desktop without a vault).
impl CellStore for HttpStore {
    fn get(&self, key: CellKey) -> Fetch {
        self.fetch(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A store with nothing in it: every request resolves to absent, which is enough to exercise the residency policy end to end.
    struct Empty;
    impl CellStore for Empty {
        fn get(&self, _key: CellKey) -> Fetch {
            Fetch::Absent
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
        r.want(vec![a], (0, 0));
        settle(&mut r);
        assert!(r.pool.map.contains_key(&a));
        // A new view that no longer needs `a`: it's dropped at once, `b` is requested, and nothing stays pending for the old view.
        r.want(vec![b], (0, 0));
        assert!(!r.pool.map.contains_key(&a));
        assert!(r.pending.contains(&b));
        settle(&mut r);
        assert!(r.pool.map.contains_key(&b));
        // Re-wanting a resident cell requests nothing.
        r.want(vec![b], (0, 0));
        assert!(r.converged());
    }
}
