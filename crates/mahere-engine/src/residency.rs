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

/// The elevation plane for a GPU: the cell's own quantisation (`base` + k·`step`, k a u16, 0xFFFF no data) laid out as one 516×258 image — the 256×256×2 texels plus the one-texel apron on every side, so a shader derives edge normals from the neighbour's data exactly as the CPU does. Column `2·(tx+1)+half`, row `ty+1`.
pub struct DemQ {
    pub base: f32,
    pub step: f32,
    pub tex: Box<[u16]>,
}

pub const DEMQ_W: usize = 2 * (mahere_tiles::TEX + 2);
pub const DEMQ_H: usize = mahere_tiles::TEX + 2;

impl DemQ {
    pub fn from_planes(d: &mahere_tiles::DemPlanes) -> DemQ {
        use mahere_tiles::{APRON, TEX, apron_idx, tri_idx};
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for &e in d.elev.iter().chain(d.apron.iter()) {
            if !e.is_nan() {
                lo = lo.min(e);
                hi = hi.max(e);
            }
        }
        if lo == f32::MAX {
            lo = 0.0;
            hi = 0.0;
        }
        let step = ((hi - lo) / 65000.0).max(0.05);
        let q = |e: f32| -> u16 { if e.is_nan() { 0xFFFF } else { (((e - lo) / step).round() as i32).clamp(0, 65534) as u16 } };
        let mut tex = vec![0xFFFFu16; DEMQ_W * DEMQ_H].into_boxed_slice();
        for ty in 0..TEX {
            for tx in 0..TEX {
                for half in 0..2 {
                    tex[(ty + 1) * DEMQ_W + 2 * (tx + 1) + half] = q(d.elev[tri_idx(tx, ty, half)]);
                }
            }
        }
        debug_assert_eq!(d.apron.len(), APRON);
        for half in 0..2 {
            for i in 0..TEX {
                tex[(i + 1) * DEMQ_W + half] = q(d.apron[apron_idx(0, half, i)]);
                tex[(i + 1) * DEMQ_W + 2 * (TEX + 1) + half] = q(d.apron[apron_idx(1, half, i)]);
                tex[2 * (i + 1) + half] = q(d.apron[apron_idx(2, half, i)]);
                tex[(TEX + 1) * DEMQ_W + 2 * (i + 1) + half] = q(d.apron[apron_idx(3, half, i)]);
            }
        }
        DemQ { base: lo, step, tex }
    }
}

/// A resident cell: whichever layers it carried. All `None` = the loader confirmed the object does not exist (absent), which still ends probing.
pub const PRESENT_DEM: u8 = 1;
pub const PRESENT_LINE: u8 = 2;
pub const PRESENT_LAND: u8 = 4;
pub const PRESENT_WATER: u8 = 8;
pub const PRESENT_IMG: u8 = 16;

#[derive(Default)]
pub struct Entry {
    /// Which planes the cell carried when it loaded (PRESENT_ bits): the truth about the cell even after a GPU host has released a plane's CPU copy.
    pub present: u8,
    pub dem: Option<DemPacked>,
    /// The same elevation as `dem`, in the layout a GPU uploads; kept beside the packed texels so either renderer can run.
    pub dem_q: Option<DemQ>,
    pub line: Option<ClassCell>,
    pub land: Option<ClassCell>,
    pub water: Option<CovCell>,
    pub img: Option<ImgCell>,
}

impl Entry {
    /// Quantised elevation (0.25 m steps from -500 m, ELEV_NODATA for none) at a memory-order texel, from whichever dem form is held.
    pub fn elev_q_at(&self, i: usize) -> Option<u16> {
        if let Some(p) = &self.dem {
            return Some((p.texel[i] & 0xFFFF) as u16);
        }
        let d = self.dem_q.as_ref()?;
        let (tx, ty, half) = ((i >> 1) & (mahere_tiles::TEX - 1), i >> 9, i & 1);
        let q = d.tex[(ty + 1) * DEMQ_W + 2 * (tx + 1) + half];
        Some(if q == 0xFFFF { mahere_tiles::ELEV_NODATA } else { mahere_tiles::quantize_elev(d.base + q as f32 * d.step) })
    }
    pub fn has_dem(&self) -> bool {
        self.present & PRESENT_DEM != 0
    }
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
    /// Whether the loader packs the CPU raster's u64 texels (normals included) for each dem; a GPU-only host turns it off and saves a megabyte and a few milliseconds per cell.
    pack_cpu: Arc<std::sync::atomic::AtomicBool>,
    want_tx: Sender<WantList>,
    done_rx: Receiver<Loaded>,
    pub pool: Pool,
    pending: FxHashSet<CellKey>,
    /// Cells whose last fetch failed, with when: not re-requested until RETRY_AFTER has passed, so an offline phone doesn't hammer timeouts every frame.
    failed: FxHashMap<CellKey, std::time::Instant>,
    pub desired: FxHashSet<CellKey>,
    pub frame: u64,
    /// Bumped whenever the pool's membership changes, so a planner can reuse its last plan while nothing moved.
    pub pool_version: u64,
    /// The last missing list sent to the loader, so an unchanged one is not sent again.
    last_sent: Vec<CellKey>,
    last_send_at: std::time::Instant,
}

const RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(3);

/// Cells at this depth and above are never evicted.
pub const PIN_DEPTH: u8 = 8;

impl Residency {
    pub fn new(store: Arc<dyn CellStore>) -> Residency {
        let (want_tx, want_rx) = channel::<WantList>();
        let (done_tx, done_rx) = channel::<Loaded>();
        let pack_cpu = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let pack = pack_cpu.clone();
        std::thread::spawn(move || loader_thread(store, want_rx, done_tx, pack));
        Residency {
            pack_cpu,
            want_tx,
            done_rx,
            pool: Pool::default(),
            pending: FxHashSet::default(),
            failed: FxHashMap::default(),
            desired: FxHashSet::default(),
            frame: 0,
            pool_version: 0,
            last_sent: Vec::new(),
            last_send_at: std::time::Instant::now(),
        }
    }

    pub fn set_pack_cpu(&self, pack: bool) {
        self.pack_cpu.store(pack, std::sync::atomic::Ordering::Relaxed);
    }

    /// Drain finished cells into the pool. Returns how many arrived.
    pub fn drain(&mut self) -> usize {
        let mut n = 0;
        while let Ok(l) = self.done_rx.try_recv() {
            self.pending.remove(&l.key);
            self.last_sent.retain(|k| *k != l.key);
            match l.entry {
                Some(e) => {
                    self.pool.map.insert(l.key, e);
                    self.pool_version += 1;
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
        // The coarse levels stay resident wherever the view goes: a zoom out always has a frame to show while finer cells arrive, and they are few and small.
        let before = self.pool.map.len();
        self.pool.map.retain(|k, _| desired.contains(k) || k.depth <= PIN_DEPTH);
        if self.pool.map.len() != before {
            self.pool_version += 1;
        }
        self.pending.retain(|k| desired.contains(k));
        let now = std::time::Instant::now();
        self.failed.retain(|k, t| desired.contains(k) && now.duration_since(*t) < RETRY_AFTER);
        // Everything still missing goes every time, pending or not: the loader keeps only the newest list, so a cell dropped from an older one would otherwise stay pending forever and never arrive (the holes Nick saw once the orientation sensor made every frame a new list). Nothing is sent while the missing set is unchanged.
        list.retain(|k| !self.pool.map.contains_key(k) && !self.failed.contains_key(k));
        list.sort_by_key(|k| {
            let (cu, cv) = k.grid();
            // Chebyshev distance in this depth's grid, normalized by shifting the center (given at depth 30-ish precision) down.
            let sh = 30 - k.depth as u32;
            let (ku, kv) = (center.0 >> sh, center.1 >> sh);
            (cu.abs_diff(ku)).max(cv.abs_diff(kv))
        });
        // An unchanged list is still re-sent once a second: a cell the loader skipped as just-done can otherwise be waited on forever after a quick zoom out and back.
        if list == self.last_sent && self.last_send_at.elapsed() < std::time::Duration::from_secs(1) {
            return;
        }
        self.last_send_at = std::time::Instant::now();
        self.pending.clear();
        for &k in &list {
            self.pending.insert(k);
        }
        self.last_sent = list.clone();
        if !list.is_empty() {
            let _ = self.want_tx.send(WantList { list });
        }
    }

    /// Cells asked for and not yet arrived.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn converged(&self) -> bool {
        self.pending.is_empty()
    }
}

/// The newest want-list is the only one that matters: an older list's leftovers are cells the view no longer needs (the main side forgets them as pending too, so they're re-requested if they come back).
/// Cells decode in parallel a small chunk at a time so the nearest-first order still holds and a newer list preempts within a few cells.
fn loader_thread(store: Arc<dyn CellStore>, want_rx: Receiver<WantList>, done_tx: Sender<Loaded>, pack_cpu: Arc<std::sync::atomic::AtomicBool>) {
    // Decoding runs in its own pool: on the global one a frame's lattice waited behind cells mid-decode, and a plan took fifty milliseconds while the view streamed.
    let threads = (rayon::current_num_threads() / 2).clamp(2, 4);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).thread_name(|i| format!("mahere-load-{i}")).build().expect("loader pool");
    let chunk = threads * 2;
    let mut current: Option<WantList> = None;
    // Cells finished in the last moment: a newer list arrives before the main side has drained them, and would load them twice.
    let mut done: FxHashMap<CellKey, std::time::Instant> = FxHashMap::default();
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
        let now = std::time::Instant::now();
        done.retain(|_, t| now.duration_since(*t) < std::time::Duration::from_millis(800));
        list.retain(|k| !done.contains_key(k));
        while !list.is_empty() {
            let n = list.len().min(chunk);
            let batch: Vec<CellKey> = list.drain(..n).collect();
            let pack = pack_cpu.load(std::sync::atomic::Ordering::Relaxed);
            let loaded: Vec<Loaded> = pool.install(|| batch.par_iter().map(|&k| load_cell(&*store, k, pack)).collect());
            let t = std::time::Instant::now();
            for l in loaded {
                done.insert(l.key, t);
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

fn load_cell(store: &dyn CellStore, key: CellKey, pack_cpu: bool) -> Loaded {
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
    let dem_q = planes.dem.as_ref().map(DemQ::from_planes);
    let dem = if pack_cpu { planes.dem.map(|d| DemPacked { texel: d.pack_texels(key).into_boxed_slice() }) } else { None };
    if std::env::var_os("MAHERE_TRACE").is_some() || cfg!(target_os = "android") {
        eprintln!("cell {} d{} loaded: dem={} line={} land={} water={} ({} bytes)", key.name(), key.depth, dem.is_some(), planes.line.is_some(), planes.land.is_some(), planes.water.is_some(), bytes.len());
    }
    let present = (dem_q.is_some() as u8) * PRESENT_DEM
        | (planes.line.is_some() as u8) * PRESENT_LINE
        | (planes.land.is_some() as u8) * PRESENT_LAND
        | (planes.water.is_some() as u8) * PRESENT_WATER
        | (planes.img.is_some() as u8) * PRESENT_IMG;
    Loaded { key, entry: Some(Entry { present, dem, dem_q, line: planes.line, land: planes.land, water: planes.water, img: planes.img }) }
}

// ==================== TIERED STORE: VAULT CACHE OVER THE BUCKET ====================

/// Where the baked cells live publicly. The path under it is exactly a cell's `CellKey::path`, so the bake directory and the bucket are the same thing.
pub const DEFAULT_CELLS_URL: &str = "https://brobdingnagian.holdmyoscilloscope.com/mahere/cells";

/// Cell format epoch: bumped whenever the encoding changes incompatibly. It rides on the fetch URL as a query (so the CDN edge, which caches a key for hours, sees a new key) and in the vault key (so a cached cell of an older format is never read back as this one). Old clients keep fetching the old objects they understand.
pub const CELL_EPOCH: u32 = 4;

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
        // Deeper than PIN_DEPTH: the coarse levels are never evicted.
        let a = CellKey { depth: 12, prefix: 0x3_0000 };
        let b = CellKey { depth: 12, prefix: 0x3_0001 };
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
