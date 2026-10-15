//! Clipmap residency: which cells are decoded and resident, and the loader thread that feeds them. Zero locks anywhere near the pixel loop — the main thread sends want-lists, the loader reads / decompresses / decodes / repacks off-thread, and the main thread drains a channel of finished planes at frame start.
//!
//! One object per cell carries every layer, so residency is one pool keyed by cell; an entry holds whichever planes the cell had at its depth.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use mahere_tiles::{CellKey, ClassCell, CovCell, ImgCell};
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
    /// Forget every absence remembered, so cells baked since the last look are asked for again (an area's refresh).
    fn forget_absent(&self) {}
    /// The cell as the source has it now, checked past any copy kept: an area's "Download again".
    fn refresh(&self, key: CellKey) -> Fetch {
        self.get(key)
    }
    /// The cell as the source has it, asked for by the name its base gives it (its provenance hash), so no copy kept on the way (the vault, the bucket's edge cache) can answer with an older one.
    fn refresh_named(&self, key: CellKey, name: &str) -> Fetch {
        let _ = name;
        self.refresh(key)
    }
    /// Cells whose kept copy was found out of date and replaced since the last call, for residency to load again.
    fn take_changed(&self) -> Vec<CellKey> {
        Vec::new()
    }
    /// The cell from what is on the device alone, never the network: for anything on the drawing thread (labels, a tap), which must not wait on a connection (the hang of 2026-10-13: a label layout fetched cells from the bucket mid-frame).
    fn get_local(&self, key: CellKey) -> Fetch {
        self.get(key)
    }
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

/// One layer of a published cells directory (`<root>/<layer>/<cell>.vsf`), as the bucket has it: an empty file is a cell with nothing in that layer.
pub struct LayerDir(pub PathBuf, pub mahere_tiles::publish::Layer);

impl CellStore for LayerDir {
    fn get(&self, key: CellKey) -> Fetch {
        match std::fs::read(self.0.join(self.1.path(key))) {
            Ok(b) => Fetch::Bytes(b),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Fetch::Absent,
            Err(e) => Fetch::Failed(e.to_string()),
        }
    }
}

/// Decoded dem planes packed one u64 per triangle texel for the hot loop (memory order `((ty << 8 | tx) << 1) | half`):
/// `[elev_q u16 | nx i16 | ny i16 | nz i16]`, elev_q = (elev + 11000) * 2 (half-metre steps), 0xFFFF = no data.
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
pub const PRESENT_OWN: u8 = 32;
pub const PRESENT_MVUM: u8 = 64;

#[derive(Default)]
pub struct Entry {
    /// Which planes the cell carried when it loaded (PRESENT_ bits): the truth about the cell even after a GPU host has released a plane's CPU copy.
    pub present: u8,
    /// The largest magnitude per line class in the cell's line plane (index = class id), so a view can scale every line against what it shows.
    pub line_mag_max: [u8; 32],
    /// The cell's elevation span, quantised (ELEV_NODATA, 0 when it has none): the contour interval is fit to the union over the view.
    pub elev_lo: u16,
    pub elev_hi: u16,
    pub dem: Option<DemPacked>,
    /// The same elevation as `dem`, in the layout a GPU uploads; kept beside the packed texels so either renderer can run.
    pub dem_q: Option<DemQ>,
    pub line: Option<ClassCell>,
    pub land: Option<ClassCell>,
    pub water: Option<CovCell>,
    pub img: Option<ImgCell>,
    /// Land ownership and motor vehicle use, where an area's overlay pass put them.
    pub own: Option<ClassCell>,
    pub mvum: Option<ClassCell>,
    /// The layers loaded with the base ([`WANT_DEM`], [`WANT_IMG`], [`WANT_NIR`]; all of them for a cell baked whole).
    pub layers: u8,
    /// The cell exists (its base was there, if empty): a cell with only terrain or imagery is not absent.
    pub exists: bool,
}

impl Entry {
    /// Quantised elevation (half-metre steps from -11000 m, ELEV_NODATA for none) at a memory-order texel, from whichever dem form is held.
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
        !self.exists && self.dem.is_none() && self.line.is_none() && self.land.is_none() && self.water.is_none() && self.img.is_none() && self.own.is_none() && self.mvum.is_none()
    }
    pub fn has_vec(&self) -> bool {
        self.line.is_some() || self.land.is_some() || self.water.is_some() || self.own.is_some() || self.mvum.is_some()
    }
}

pub struct Loaded {
    pub key: CellKey,
    /// `None` = the fetch failed; the cell is not resident and will be asked for again after a short backoff.
    pub entry: Option<Entry>,
}

struct WantList {
    /// Pre-sorted by priority: every missing cell's base first (Nick 2026-10-14: the whole scene in lines, then the terrain filling in), then resident cells' layers, each nearest the view's centre first. Each with the layers to load beside its base.
    list: Vec<(CellKey, u8)>,
    /// The layers the view shows, fetched into the store beside each base as it is fetched (Nick 2026-10-15), so the terrain's round trip runs with the base's and the fill after the lines reads the device.
    warm: u8,
    /// Cells around a settled view, fetched into the store and decoded for nothing (Nick 2026-10-15: make the area being looked at hot), so a pan or a zoom in reads the device instead of waiting on the bucket.
    prefetch: Vec<CellKey>,
}

/// What a loader does with a cell.
#[derive(Clone, Copy)]
enum Work {
    /// Load and decode it with these layers beside its base, for the pool.
    Load(CellKey, u8),
    /// Fetch it into the store and decode nothing: its base too or not, and these layers.
    Fetch(CellKey, bool, u8),
}

/// The loaders' work: the cells to load, the layers to fetch beside their bases, and what is loading or just loaded.
#[derive(Default)]
struct Queues {
    work: std::collections::VecDeque<Work>,
    warm: std::collections::VecDeque<Work>,
    busy: FxHashSet<(CellKey, u8)>,
    /// Cells finished in the last moment: a newer list arrives before the main side has drained them, and would load them twice.
    done: FxHashMap<(CellKey, u8), std::time::Instant>,
    closed: bool,
}

/// The most cells fetched around a settled view: about 3 MB of base and terrain, 5 MB with imagery.
const PREFETCH_MAX: usize = 48;

/// Chebyshev distance from the view's centre in a cell's own grid (the centre given at depth-30 precision).
fn near(k: &CellKey, center: (u64, u64)) -> u64 {
    let (cu, cv) = k.grid();
    let sh = 30 - k.depth as u32;
    cu.abs_diff(center.0 >> sh).max(cv.abs_diff(center.1 >> sh))
}

/// Resident cells: page table over decoded planes. Plain map — a few hundred entries, probed per BLOCK (not per pixel) during render.
#[derive(Default)]
pub struct Pool {
    pub map: FxHashMap<CellKey, Entry>,
}

/// The layers a published cell comes in besides its base (Nick 2026-10-14: imagery and infrared only when a view shows them, and the terrain a layer of its own): each from a store of its own.
#[derive(Clone)]
pub struct LayerStores {
    pub dem: Arc<dyn CellStore>,
    pub img: Arc<dyn CellStore>,
    pub nir: Arc<dyn CellStore>,
}

/// Which layers the view wants loaded with each cell's base.
pub const WANT_IMG: u8 = 1;
pub const WANT_NIR: u8 = 2;
pub const WANT_DEM: u8 = 4;

pub struct Residency {
    /// The store the loader reads, for reading one cell's planes back on demand (a tap's details on a host whose GPU took the CPU copies).
    store: Arc<dyn CellStore>,
    /// The other layers' stores, when cells come published in layers.
    layers: Option<LayerStores>,
    /// The layers the view wants beside each base ([`WANT_DEM`], [`WANT_IMG`], [`WANT_NIR`]).
    want_layers: u8,
    /// Cells in view, there, still without a layer the view wants, at the last want: the view is not settled while any are (a host that stops drawing once settled would never ask for them).
    fill_left: usize,
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
    last_sent: Vec<(CellKey, u8)>,
    last_send_at: std::time::Instant,
    /// Cells the store replaced with a newer copy, for whatever keeps something read from them to forget it.
    pub reloaded: Vec<CellKey>,
    /// Cells already prefetched, so a settled view sends each once.
    prefetched: FxHashSet<CellKey>,
    /// The view the ring was last looked for around (a fingerprint of its cells), so a still view does not look every frame.
    prefetch_view: u64,
}

const RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(3);

/// Cells at this depth and above are never evicted.
pub const PIN_DEPTH: u8 = 4;

impl Residency {
    pub fn new(store: Arc<dyn CellStore>) -> Residency {
        Residency::layered(store, None)
    }

    /// Cells from a base store, with the image layers from their own stores when they come published in layers.
    pub fn layered(store: Arc<dyn CellStore>, layers: Option<LayerStores>) -> Residency {
        let (want_tx, want_rx) = channel::<WantList>();
        let (done_tx, done_rx) = channel::<Loaded>();
        let pack_cpu = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let pack = pack_cpu.clone();
        let read_store = store.clone();
        let l = layers.clone();
        std::thread::spawn(move || loader_thread(store, l, want_rx, done_tx, pack));
        Residency {
            store: read_store,
            layers,
            want_layers: WANT_DEM,
            fill_left: 0,
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
            reloaded: Vec::new(),
            prefetched: FxHashSet::default(),
            prefetch_view: 0,
        }
    }

    /// The layers the view shows ([`WANT_DEM`], [`WANT_IMG`], [`WANT_NIR`]): one newly wanted is loaded for every resident cell, each kept on show until its fuller copy lands.
    pub fn set_imagery(&mut self, want: u8) {
        if want != self.want_layers {
            self.want_layers = want;
            self.last_sent.clear();
            // Unsettled until the next want has looked for cells without the new layers.
            self.fill_left = usize::MAX;
        }
    }

    /// The image layers' stores, when cells come in layers.
    pub fn layer_stores(&self) -> Option<LayerStores> {
        self.layers.clone()
    }

    pub fn set_pack_cpu(&self, pack: bool) {
        self.pack_cpu.store(pack, std::sync::atomic::Ordering::Relaxed);
    }

    /// Drain finished cells into the pool. Returns how many arrived. A resident cell its store found changed is dropped, so the view asks for it again and gets the new copy.
    pub fn drain(&mut self) -> usize {
        let mut n = 0;
        for k in self.store.take_changed() {
            self.reloaded.push(k);
            if self.pool.map.remove(&k).is_some() {
                self.pool_version += 1;
                n += 1;
            }
        }
        while let Ok(l) = self.done_rx.try_recv() {
            self.pending.remove(&l.key);
            self.last_sent.retain(|(k, _)| *k != l.key);
            match l.entry {
                // A base arriving after the same cell with its layers (an older request overtaken) leaves the fuller copy on show.
                Some(e) if self.pool.map.get(&l.key).is_some_and(|old| old.layers & !e.layers != 0 && e.layers & !old.layers == 0) => {}
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
        // Resident cells (there, base and all) without a layer the view now shows: asked for again with it, after every missing base.
        let want = self.want_layers;
        let mut fill: Vec<CellKey> = if self.layers.is_some() {
            list.iter().copied().filter(|k| !self.failed.contains_key(k) && self.pool.map.get(k).is_some_and(|e| e.exists && e.layers & want != want)).collect()
        } else {
            Vec::new()
        };
        list.retain(|k| !self.pool.map.contains_key(k) && !self.failed.contains_key(k));
        // Below a cell the loader confirmed absent nothing exists: its descendants are marked absent here and never asked for, so zooming past the finest data does not ask the bucket for a screen of cells that cannot be there (Nick 2026-10-10).
        let mut under_absent = Vec::new();
        list.retain(|k| {
            if self.under_absent(*k) {
                under_absent.push(*k);
                return false;
            }
            true
        });
        if !under_absent.is_empty() {
            for k in under_absent {
                self.pool.map.insert(k, Entry::default());
            }
            self.pool_version += 1;
        }
        list.sort_by_key(|k| near(k, center));
        fill.sort_by_key(|k| near(k, center));
        self.fill_left = fill.len();
        // Cells baked whole carry everything in one read; cells in layers come base first, the layers after.
        let first = if self.layers.is_some() { 0 } else { want };
        let list: Vec<(CellKey, u8)> = list.into_iter().map(|k| (k, first)).chain(fill.into_iter().map(|k| (k, want))).collect();
        let warm = if self.layers.is_some() { want } else { 0 };
        // Nothing missing: the cells around the view, into the store.
        let view = self.desired.iter().fold(self.desired.len() as u64, |h, k| h.wrapping_add((k.prefix ^ ((k.depth as u64) << 58)).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        if list.is_empty() && self.layers.is_some() && view != self.prefetch_view {
            self.prefetch_view = view;
            let prefetch = self.prefetch_set(center);
            if !prefetch.is_empty() {
                if self.prefetched.len() > 20_000 {
                    self.prefetched.clear();
                }
                self.prefetched.extend(prefetch.iter().copied());
                self.last_sent.clear();
                let _ = self.want_tx.send(WantList { list: Vec::new(), warm, prefetch });
                return;
            }
        }
        // An unchanged list is still re-sent once a second: a cell the loader skipped as just-done can otherwise be waited on forever after a quick zoom out and back.
        if list == self.last_sent && self.last_send_at.elapsed() < std::time::Duration::from_secs(1) {
            return;
        }
        self.last_send_at = std::time::Instant::now();
        self.pending.clear();
        for &(k, _) in &list {
            self.pending.insert(k);
        }
        self.last_sent = list.clone();
        if !list.is_empty() {
            let _ = self.want_tx.send(WantList { list, warm, prefetch: Vec::new() });
        }
    }

    /// Whether a cell lies under one the loader confirmed absent, where nothing exists.
    fn under_absent(&self, k: CellKey) -> bool {
        (0..k.depth).any(|d| self.pool.map.get(&k.ancestor(d)).is_some_and(|e| e.present == 0 && e.is_absent()))
    }

    /// The cells to fetch around a settled view: the ring one cell out at each depth it shows, and the children of the cells at its centre (the next zoom in), nearest first, none resident, prefetched already, or under an absent cell.
    fn prefetch_set(&self, center: (u64, u64)) -> Vec<CellKey> {
        let Some(finest) = self.desired.iter().map(|k| k.depth).max() else { return Vec::new() };
        let mut out: FxHashSet<CellKey> = FxHashSet::default();
        for k in self.desired.iter().filter(|k| k.depth > PIN_DEPTH) {
            let (cu, cv) = k.grid();
            let n = 1i64 << k.depth;
            for (du, dv) in [(-1i64, -1i64), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
                let (u, v) = (cu as i64 + du, cv as i64 + dv);
                if (0..n).contains(&u) && (0..n).contains(&v) {
                    out.insert(CellKey::from_grid(k.diamond(), k.depth, u as u64, v as u64));
                }
            }
            let there = self.pool.map.get(k).is_some_and(|e| e.exists || e.present != 0);
            if k.depth == finest && there && near(k, center) <= 1 {
                out.extend((0..4).map(|q| k.child(q)));
            }
        }
        let mut list: Vec<CellKey> = out.into_iter().filter(|k| !self.desired.contains(k) && !self.pool.map.contains_key(k) && !self.prefetched.contains(k) && !self.under_absent(*k)).collect();
        list.sort_by_key(|k| (near(k, center), k.depth));
        list.truncate(PREFETCH_MAX);
        list
    }

    /// A resident cell's planes, decoded again from its store (the vault on a phone, so no network): for reading what a GPU host no longer keeps on the CPU.
    pub fn read_planes(&self, key: CellKey) -> Option<mahere_tiles::CellPlanes> {
        // Resident with something in it (not a cell known to be absent), and read from the device only.
        if self.pool.map.get(&key).is_none_or(|e| e.present == 0) {
            return None;
        }
        let base = match self.store.get_local(key) {
            Fetch::Bytes(b) => b,
            _ => Vec::new(),
        };
        // The terrain from its own layer when cells come in layers, so a tap's elevation and slope are there.
        let dem = match self.layers.as_ref().map(|l| l.dem.get_local(key)) {
            Some(Fetch::Bytes(b)) => b,
            _ => Vec::new(),
        };
        mahere_tiles::publish::decode_layers(non_empty(&base), non_empty(&dem), None, None).ok()
    }

    /// The store cells come from, for work that reads cells the view has not asked for (a route search).
    pub fn store(&self) -> Arc<dyn CellStore> {
        self.store.clone()
    }

    /// A resident cell's names alone, read again from its store; None when it has none or is not resident.
    pub fn read_names(&self, key: CellKey) -> Option<mahere_tiles::NameCell> {
        if self.pool.map.get(&key).is_none_or(|e| e.present == 0) {
            return None;
        }
        match self.store.get_local(key) {
            Fetch::Bytes(b) => mahere_tiles::decode_names(&b),
            _ => None,
        }
    }

    /// Cells asked for and not yet arrived.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn converged(&self) -> bool {
        self.pending.is_empty() && self.fill_left == 0
    }
}

/// The newest want-list is the only one that matters: an older list's leftovers are cells the view no longer needs (the main side forgets them as pending too, so they're re-requested if they come back).
/// Each loader takes the next cell as it finishes one and sends it on at once, so a slow cell holds up only itself (in batches, a whole batch waited on its slowest fetch); a newer list replaces what is queued.
fn loader_thread(store: Arc<dyn CellStore>, layers: Option<LayerStores>, want_rx: Receiver<WantList>, done_tx: Sender<Loaded>, pack_cpu: Arc<std::sync::atomic::AtomicBool>) {
    // Fetching is waiting on the network far more than decoding (a cell is a round trip of a hundred milliseconds or more to the bucket, Nick 2026-10-15: tiles slow though small), so cells from the bucket load a dozen at a time, with eight more fetching the layers beside the bases; from the device, as many as there are cores to decode them.
    let (loaders, warmers) = if layers.is_some() { (12, 8) } else { ((rayon::current_num_threads() / 2).clamp(2, 4), 0) };
    // Decoding runs in its own pool: on the global one a frame's lattice waited behind cells mid-decode, and a plan took fifty milliseconds while the view streamed.
    let pool = rayon::ThreadPoolBuilder::new().num_threads(loaders + warmers).thread_name(|i| format!("mahere-load-{i}")).build().expect("loader pool");
    let shared = Arc::new((std::sync::Mutex::new(Queues::default()), std::sync::Condvar::new()));
    for i in 0..loaders + warmers {
        let (store, layers, done_tx, pack_cpu, shared) = (store.clone(), layers.clone(), done_tx.clone(), pack_cpu.clone(), shared.clone());
        pool.spawn(move || loader(i >= loaders, &*store, layers.as_ref(), &done_tx, &pack_cpu, &shared));
    }
    let (lock, wake) = &*shared;
    while let Ok(mut w) = want_rx.recv() {
        while let Ok(newer) = want_rx.try_recv() {
            w = newer;
        }
        let mut q = lock.lock().unwrap();
        let now = std::time::Instant::now();
        q.done.retain(|_, t| now.duration_since(*t) < std::time::Duration::from_millis(800));
        let Queues { work, warm, busy, done, .. } = &mut *q;
        work.clear();
        warm.clear();
        for &(k, want) in &w.list {
            if done.contains_key(&(k, want)) || busy.contains(&(k, want)) {
                continue;
            }
            work.push_back(Work::Load(k, want));
            if want == 0 && w.warm != 0 {
                warm.push_back(Work::Fetch(k, false, w.warm));
            }
        }
        work.extend(w.prefetch.iter().map(|&k| Work::Fetch(k, true, w.warm)));
        wake.notify_all();
    }
    lock.lock().unwrap().closed = true;
    wake.notify_all();
}

/// One loader: the next cell from its queue (the loads, or the layers beside them), until the residency is gone.
fn loader(warm_side: bool, store: &dyn CellStore, layers: Option<&LayerStores>, done_tx: &Sender<Loaded>, pack_cpu: &std::sync::atomic::AtomicBool, shared: &(std::sync::Mutex<Queues>, std::sync::Condvar)) {
    let (lock, wake) = shared;
    loop {
        let job = {
            let mut q = lock.lock().unwrap();
            loop {
                if q.closed {
                    return;
                }
                let next = if warm_side { q.warm.pop_front() } else { q.work.pop_front() };
                if let Some(j) = next {
                    if let Work::Load(k, want) = j {
                        q.busy.insert((k, want));
                    }
                    break j;
                }
                q = wake.wait(q).unwrap();
            }
        };
        match job {
            Work::Load(k, want) => {
                let l = load_cell(store, layers, want, k, pack_cpu.load(std::sync::atomic::Ordering::Relaxed));
                {
                    let mut q = lock.lock().unwrap();
                    q.busy.remove(&(k, want));
                    q.done.insert((k, want), std::time::Instant::now());
                }
                if done_tx.send(l).is_err() {
                    return;
                }
            }
            Work::Fetch(k, base, want) => fetch_only(store, layers, k, base, want),
        }
    }
}

/// A cell fetched into its stores and decoded for nothing: its base (when asked, and then nothing more if it has none), and the layers wanted.
fn fetch_only(store: &dyn CellStore, layers: Option<&LayerStores>, key: CellKey, base: bool, want: u8) {
    if base && !matches!(store.get(key), Fetch::Bytes(_)) {
        return;
    }
    let Some(l) = layers else { return };
    for (bit, s) in [(WANT_DEM, &l.dem), (WANT_IMG, &l.img), (WANT_NIR, &l.nir)] {
        if want & bit != 0 {
            let _ = s.get(key);
        }
    }
}

/// One layer's bytes: Some(empty) for none (absent, or a published cell with no base), None for a failure to try again.
fn layer_bytes(store: &dyn CellStore, key: CellKey, what: &str) -> Option<Vec<u8>> {
    match store.get(key) {
        Fetch::Bytes(b) => Some(b),
        Fetch::Absent => Some(Vec::new()),
        Fetch::Failed(e) => {
            eprintln!("cell {} {what} fetch failed: {e}", key.name());
            None
        }
    }
}

fn non_empty(b: &[u8]) -> Option<&[u8]> {
    (!b.is_empty()).then_some(b)
}

fn load_cell(store: &dyn CellStore, layers: Option<&LayerStores>, want: u8, key: CellKey, pack_cpu: bool) -> Loaded {
    // No base, no cell (a published cell always has one, if empty); a failure is tried again.
    let bytes = match store.get(key) {
        Fetch::Bytes(b) => b,
        Fetch::Absent => return Loaded { key, entry: Some(Entry::default()) },
        Fetch::Failed(e) => {
            eprintln!("cell {} fetch failed: {e}", key.name());
            return Loaded { key, entry: None };
        }
    };
    let layers_loaded = if layers.is_some() { want } else { u8::MAX };
    // The layers the view shows, beside the base.
    let mut img_bytes: (Vec<u8>, Vec<u8>) = Default::default();
    let mut dem_bytes: Vec<u8> = Vec::new();
    if let Some(l) = layers {
        use mahere_tiles::publish::Layer;
        // A layer the base does not name (a copy kept from an older publish, in the vault or the bucket's edge cache) is fetched again by the name the base gives it, past every copy; still not the one named, the cell shows without it rather than mixing publishes.
        let get = |store: &dyn CellStore, layer: Layer| -> Option<Vec<u8>> {
            let b = layer_bytes(store, key, layer.dir())?;
            if b.is_empty() || bytes.is_empty() || mahere_tiles::publish::linked(&bytes, layer, &b) {
                return Some(b);
            }
            let fresh = match mahere_tiles::publish::link_name(&bytes, layer) {
                Some(name) => store.refresh_named(key, &name),
                None => store.refresh(key),
            };
            match fresh {
                Fetch::Bytes(fresh) if mahere_tiles::publish::linked(&bytes, layer, &fresh) => Some(fresh),
                Fetch::Failed(_) => None,
                _ => {
                    eprintln!("cell {} {}: not the layer its base names", key.name(), layer.dir());
                    Some(Vec::new())
                }
            }
        };
        if want & WANT_DEM != 0 {
            let Some(b) = get(&*l.dem, Layer::Dem) else { return Loaded { key, entry: None } };
            dem_bytes = b;
        }
        if want & WANT_IMG != 0 {
            let Some(b) = get(&*l.img, Layer::Img) else { return Loaded { key, entry: None } };
            img_bytes.0 = b;
        }
        if want & WANT_NIR != 0 {
            let Some(b) = get(&*l.nir, Layer::Nir) else { return Loaded { key, entry: None } };
            img_bytes.1 = b;
        }
    }
    if bytes.is_empty() && dem_bytes.is_empty() && img_bytes.0.is_empty() && img_bytes.1.is_empty() {
        return Loaded { key, entry: Some(Entry { layers: layers_loaded, exists: true, ..Entry::default() }) };
    }
    let planes = match mahere_tiles::publish::decode_layers(non_empty(&bytes), non_empty(&dem_bytes), non_empty(&img_bytes.0), non_empty(&img_bytes.1)) {
        Ok(p) => p,
        Err(e) => {
            // A corrupt object: treat as a failure so it's retried (the vault may have a bad copy) rather than remembered as absent.
            eprintln!("cell {} decode failed ({} bytes): {e}", key.name(), bytes.len());
            return Loaded { key, entry: None };
        }
    };
    let (elev_lo, elev_hi) = planes.dem.as_ref().map_or((mahere_tiles::ELEV_NODATA, 0), |d| {
        d.elev.iter().filter(|e| !e.is_nan()).fold((mahere_tiles::ELEV_NODATA, 0u16), |(lo, hi), &e| {
            let q = mahere_tiles::quantize_elev(e);
            (lo.min(q), hi.max(q))
        })
    });
    let dem_q = planes.dem.as_ref().map(DemQ::from_planes);
    let dem = if pack_cpu { planes.dem.map(|d| DemPacked { texel: d.pack_texels(key).into_boxed_slice() }) } else { None };
    if std::env::var_os("MAHERE_TRACE").is_some() || cfg!(target_os = "android") {
        eprintln!("cell {} d{} loaded: dem={} line={} land={} water={} ({} bytes)", key.name(), key.depth, dem.is_some(), planes.line.is_some(), planes.land.is_some(), planes.water.is_some(), bytes.len());
    }
    let present = (dem_q.is_some() as u8) * PRESENT_DEM
        | (planes.line.is_some() as u8) * PRESENT_LINE
        | (planes.land.is_some() as u8) * PRESENT_LAND
        | (planes.water.is_some() as u8) * PRESENT_WATER
        | (planes.img.is_some() as u8) * PRESENT_IMG
        | (planes.own.is_some() as u8) * PRESENT_OWN
        | (planes.mvum.is_some() as u8) * PRESENT_MVUM;
    let mut line_mag_max = [0u8; 32];
    if let Some(l) = &planes.line {
        for i in 0..mahere_tiles::TRI {
            if l.cov[i] != 0 {
                let c = (l.class[i] as usize).min(31);
                line_mag_max[c] = line_mag_max[c].max(l.mag_at(i));
            }
        }
    }
    // The renderers unroll imagery through one table, the agreed tone's; a cell stored in any other is brought to it here, once.
    let mut img = planes.img;
    if let Some(im) = &mut img {
        im.retone(mahere_tiles::tone::IMG_TAG);
    }
    Loaded { key, entry: Some(Entry { present, line_mag_max, elev_lo, elev_hi, dem, dem_q, line: planes.line, land: planes.land, water: planes.water, img, own: planes.own, mvum: planes.mvum, layers: layers_loaded, exists: true }) }
}

// ==================== TIERED STORE: VAULT CACHE OVER THE BUCKET ====================

/// Where the baked cells live publicly. The path under it is exactly a cell's `CellKey::path`, so the bake directory and the bucket are the same thing.
pub const DEFAULT_CELLS_URL: &str = "https://brobdingnagian.holdmyoscilloscope.com/mahere/cells";

/// Cell format epoch: bumped whenever the encoding changes incompatibly. It rides on the fetch URL as a query (so the CDN edge, which caches a key for hours, sees a new key) and in the vault key (so a cached cell of an older format is never read back as this one). Old clients keep fetching the old objects they understand.
pub const CELL_EPOCH: u32 = 8;

/// A store that can also keep what it's given (the vault), and the source's tag for each copy (its ETag), so a copy can be checked without fetching it again.
pub trait CellCache: CellStore {
    fn put(&self, key: CellKey, bytes: &[u8]);
    fn tag(&self, _key: CellKey) -> Option<String> {
        None
    }
    fn set_tag(&self, _key: CellKey, _tag: &str) {}
}

/// What checking a kept copy against the source found.
pub enum Checked {
    /// The copy is the source's.
    Same,
    /// The source has other bytes, with their tag.
    Changed(Vec<u8>, Option<String>),
    /// The source no longer has the cell.
    Gone,
    Failed(String),
}

/// The far tier: the same outcomes as a store, over the network.
pub trait RemoteStore: Send + Sync + 'static {
    fn fetch(&self, key: CellKey) -> Fetch;
    /// The cell at an address its name makes, which no cache between has seen under an older copy.
    fn fetch_named(&self, key: CellKey, name: &str) -> Fetch {
        let _ = name;
        self.fetch(key)
    }
    /// The cell unless the source still has the copy with this tag; with no tag, the cell and its tag.
    fn check(&self, key: CellKey, tag: Option<&str>) -> Checked {
        let _ = tag;
        match self.fetch(key) {
            Fetch::Bytes(b) => Checked::Changed(b, None),
            Fetch::Absent => Checked::Gone,
            Fetch::Failed(e) => Checked::Failed(e),
        }
    }
}

/// Cache first, then the bucket, writing hits through. Definite misses are remembered for the process (most of the world isn't baked yet); failures are not, so a cell that couldn't be fetched is tried again the next time the view wants it.
/// A kept copy is served at once and checked against the bucket behind it, once a session (Nick 2026-10-13: a rebake must reach a phone that already has the cell): unchanged costs a 304, changed is kept and handed to residency to load again.
pub struct TieredStore {
    cache: Arc<dyn CellCache>,
    remote: Arc<dyn RemoteStore>,
    missing: std::sync::Mutex<FxHashSet<CellKey>>,
    checks: Arc<Checks>,
    /// Cells being fetched now, so a second asker (a layer fetched beside its base, then asked for by the fill) waits on the first fetch instead of making its own.
    inflight: std::sync::Mutex<FxHashSet<CellKey>>,
    landed: std::sync::Condvar,
}

/// The copies waiting to be checked, those checked this session, and those found changed.
#[derive(Default)]
struct Checks {
    queue: std::sync::Mutex<std::collections::VecDeque<CellKey>>,
    wake: std::sync::Condvar,
    seen: std::sync::Mutex<FxHashSet<CellKey>>,
    changed: std::sync::Mutex<Vec<CellKey>>,
}

/// Threads checking kept copies against the bucket.
const CHECKERS: usize = 2;

/// Check one kept copy against the source: the same, nothing to do; changed, kept with its tag and reported; gone or failed, left as it is.
fn check_copy(cache: &dyn CellCache, remote: &dyn RemoteStore, key: CellKey) -> Option<Vec<u8>> {
    match remote.check(key, cache.tag(key).as_deref()) {
        Checked::Same => None,
        Checked::Changed(b, tag) => {
            let same = matches!(cache.get(key), Fetch::Bytes(old) if old == b);
            if let Some(t) = &tag {
                cache.set_tag(key, t);
            }
            if same {
                return None;
            }
            cache.put(key, &b);
            Some(b)
        }
        Checked::Gone | Checked::Failed(_) => None,
    }
}

impl TieredStore {
    pub fn new(cache: Arc<dyn CellCache>, remote: Arc<dyn RemoteStore>) -> TieredStore {
        let checks = Arc::new(Checks::default());
        for _ in 0..CHECKERS {
            let (cache, remote, checks) = (cache.clone(), remote.clone(), checks.clone());
            std::thread::spawn(move || loop {
                let key = {
                    let mut q = checks.queue.lock().unwrap();
                    loop {
                        if let Some(k) = q.pop_front() {
                            break k;
                        }
                        q = checks.wake.wait(q).unwrap();
                    }
                };
                if check_copy(cache.as_ref(), remote.as_ref(), key).is_some() {
                    checks.changed.lock().unwrap().push(key);
                }
            });
        }
        TieredStore { cache, remote, missing: std::sync::Mutex::new(FxHashSet::default()), checks, inflight: Default::default(), landed: Default::default() }
    }

    /// Fetch from the bucket and keep it with its tag; a cell already being fetched is waited for and read from what that fetch kept (fetched again only if it failed).
    fn fetch_and_keep(&self, key: CellKey) -> Fetch {
        loop {
            let mut f = self.inflight.lock().unwrap();
            if f.insert(key) {
                break;
            }
            while f.contains(&key) {
                f = self.landed.wait(f).unwrap();
            }
            drop(f);
            if let Fetch::Bytes(b) = self.cache.get(key) {
                return Fetch::Bytes(b);
            }
            if self.missing.lock().unwrap().contains(&key) {
                return Fetch::Absent;
            }
        }
        let got = self.fetch_now(key);
        self.inflight.lock().unwrap().remove(&key);
        self.landed.notify_all();
        got
    }

    fn fetch_now(&self, key: CellKey) -> Fetch {
        match self.remote.check(key, None) {
            Checked::Changed(b, tag) => {
                self.cache.put(key, &b);
                if let Some(t) = &tag {
                    self.cache.set_tag(key, t);
                }
                self.checks.seen.lock().unwrap().insert(key);
                Fetch::Bytes(b)
            }
            Checked::Same => self.cache.get(key),
            Checked::Gone => {
                self.missing.lock().unwrap().insert(key);
                Fetch::Absent
            }
            Checked::Failed(e) => Fetch::Failed(e),
        }
    }
}

impl CellStore for TieredStore {
    fn forget_absent(&self) {
        self.missing.lock().unwrap().clear();
    }

    fn get(&self, key: CellKey) -> Fetch {
        match self.cache.get(key) {
            Fetch::Bytes(b) => {
                // Served now; checked against the bucket behind it, once a session.
                if self.checks.seen.lock().unwrap().insert(key) {
                    self.checks.queue.lock().unwrap().push_back(key);
                    self.checks.wake.notify_one();
                }
                return Fetch::Bytes(b);
            }
            Fetch::Failed(e) => eprintln!("vault read {} failed: {e}", key.path()),
            Fetch::Absent => {}
        }
        if self.missing.lock().unwrap().contains(&key) {
            return Fetch::Absent;
        }
        self.fetch_and_keep(key)
    }

    fn refresh(&self, key: CellKey) -> Fetch {
        if !matches!(self.cache.get(key), Fetch::Bytes(_)) {
            self.missing.lock().unwrap().remove(&key);
            return self.fetch_and_keep(key);
        }
        self.checks.seen.lock().unwrap().insert(key);
        if check_copy(self.cache.as_ref(), self.remote.as_ref(), key).is_some() {
            self.checks.changed.lock().unwrap().push(key);
        }
        self.cache.get(key)
    }

    fn refresh_named(&self, key: CellKey, name: &str) -> Fetch {
        match self.remote.fetch_named(key, name) {
            Fetch::Bytes(b) => {
                self.cache.put(key, &b);
                self.checks.seen.lock().unwrap().insert(key);
                self.missing.lock().unwrap().remove(&key);
                Fetch::Bytes(b)
            }
            other => other,
        }
    }

    fn take_changed(&self) -> Vec<CellKey> {
        std::mem::take(&mut *self.checks.changed.lock().unwrap())
    }

    fn get_local(&self, key: CellKey) -> Fetch {
        self.cache.get(key)
    }
}

/// The bucket over HTTPS. 404 is absence; anything else is a failure.
pub struct HttpStore {
    base: String,
    agent: ureq::Agent,
    /// The published layer this store fetches (`<base>/<layer>/<cell>.vsf`); None for a baked directory served as it is (`<base>/<cell>.vsf.zst`).
    layer: Option<mahere_tiles::publish::Layer>,
}

impl HttpStore {
    pub fn new(base: &str) -> HttpStore {
        // A connection kept for every loader (ureq keeps one a host by default, so eleven of twelve loaders paid a TCP and TLS handshake on every cell, about 160 ms each).
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(10))
            .timeout_read(std::time::Duration::from_secs(30))
            .max_idle_connections_per_host(32)
            .build();
        HttpStore { base: base.trim_end_matches('/').to_string(), agent, layer: None }
    }

    /// One published layer of the cells under `base`.
    pub fn layer(base: &str, layer: mahere_tiles::publish::Layer) -> HttpStore {
        HttpStore { layer: Some(layer), ..HttpStore::new(base) }
    }

    fn url(&self, key: CellKey) -> String {
        match self.layer {
            Some(l) => format!("{}/{}?e={}", self.base, l.path(key), CELL_EPOCH),
            None => format!("{}/{}?e={}", self.base, key.path(), CELL_EPOCH),
        }
    }
}

impl RemoteStore for HttpStore {
    fn check(&self, key: CellKey, tag: Option<&str>) -> Checked {
        let url = self.url(key);
        let mut req = self.agent.get(&url);
        if let Some(t) = tag {
            req = req.set("If-None-Match", t);
        }
        match req.call() {
            Ok(resp) if resp.status() == 304 => Checked::Same,
            Ok(resp) => {
                let etag = resp.header("ETag").map(str::to_string);
                let expect: Option<usize> = resp.header("Content-Length").and_then(|v| v.parse().ok());
                let mut buf = Vec::new();
                if let Err(e) = std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf) {
                    return Checked::Failed(e.to_string());
                }
                if expect.is_some_and(|n| n != buf.len()) {
                    return Checked::Failed(format!("short body: {} of {:?} bytes", buf.len(), expect));
                }
                Checked::Changed(buf, etag)
            }
            Err(ureq::Error::Status(404, _)) => Checked::Gone,
            Err(e) => Checked::Failed(e.to_string()),
        }
    }

    fn fetch(&self, key: CellKey) -> Fetch {
        self.fetch_url(&self.url(key))
    }

    fn fetch_named(&self, key: CellKey, name: &str) -> Fetch {
        self.fetch_url(&format!("{}&h={name}", self.url(key)))
    }
}

impl HttpStore {
    fn fetch_url(&self, url: &str) -> Fetch {
        match self.agent.get(url).call() {
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

    fn refresh_named(&self, key: CellKey, name: &str) -> Fetch {
        self.fetch_named(key, name)
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

    /// A kept copy is served at once, checked behind it, and replaced and reported when the source has changed; an unchanged one stays quiet.
    #[test]
    fn a_kept_copy_is_checked_and_replaced_when_the_source_changed() {
        use std::sync::Mutex;
        #[derive(Default)]
        struct Kept(Mutex<FxHashMap<CellKey, (Vec<u8>, Option<String>)>>);
        impl CellStore for Kept {
            fn get(&self, key: CellKey) -> Fetch {
                self.0.lock().unwrap().get(&key).map_or(Fetch::Absent, |(b, _)| Fetch::Bytes(b.clone()))
            }
        }
        impl CellCache for Kept {
            fn put(&self, key: CellKey, bytes: &[u8]) {
                self.0.lock().unwrap().entry(key).or_default().0 = bytes.to_vec();
            }
            fn tag(&self, key: CellKey) -> Option<String> {
                self.0.lock().unwrap().get(&key).and_then(|(_, t)| t.clone())
            }
            fn set_tag(&self, key: CellKey, tag: &str) {
                self.0.lock().unwrap().entry(key).or_default().1 = Some(tag.to_string());
            }
        }
        struct Source(FxHashMap<CellKey, (Vec<u8>, String)>);
        impl RemoteStore for Source {
            fn fetch(&self, key: CellKey) -> Fetch {
                self.0.get(&key).map_or(Fetch::Absent, |(b, _)| Fetch::Bytes(b.clone()))
            }
            fn check(&self, key: CellKey, tag: Option<&str>) -> Checked {
                match self.0.get(&key) {
                    Some((_, t)) if Some(t.as_str()) == tag => Checked::Same,
                    Some((b, t)) => Checked::Changed(b.clone(), Some(t.clone())),
                    None => Checked::Gone,
                }
            }
        }
        let (old, same) = (CellKey { depth: 9, prefix: 1 }, CellKey { depth: 9, prefix: 2 });
        let kept = Arc::new(Kept::default());
        kept.put(old, b"old");
        kept.set_tag(old, "v1");
        kept.put(same, b"same");
        kept.set_tag(same, "s1");
        let source = Arc::new(Source([(old, (b"new".to_vec(), "v2".to_string())), (same, (b"same".to_vec(), "s1".to_string()))].into_iter().collect()));
        let store = TieredStore::new(kept.clone(), source);
        assert!(matches!(store.get(old), Fetch::Bytes(b) if b == b"old"), "the kept copy is served at once");
        assert!(matches!(store.get(same), Fetch::Bytes(_)));
        let mut changed = Vec::new();
        for _ in 0..500 {
            changed.extend(store.take_changed());
            if !changed.is_empty() && kept.tag(old).as_deref() == Some("v2") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(changed, vec![old], "only the changed cell is reported");
        assert!(matches!(store.get(old), Fetch::Bytes(b) if b == b"new"));
        assert_eq!(kept.tag(old).as_deref(), Some("v2"));
    }

    /// A layer asked for by its name passes every copy on the way: the bucket's edge still serving the old bytes at the plain address, the named address has the new, and the vault keeps them.
    #[test]
    fn a_named_refresh_passes_a_stale_edge() {
        use std::sync::Mutex;
        #[derive(Default)]
        struct Kept(Mutex<FxHashMap<CellKey, Vec<u8>>>);
        impl CellStore for Kept {
            fn get(&self, key: CellKey) -> Fetch {
                self.0.lock().unwrap().get(&key).map_or(Fetch::Absent, |b| Fetch::Bytes(b.clone()))
            }
        }
        impl CellCache for Kept {
            fn put(&self, key: CellKey, bytes: &[u8]) {
                self.0.lock().unwrap().insert(key, bytes.to_vec());
            }
        }
        struct Edge;
        impl RemoteStore for Edge {
            fn fetch(&self, _key: CellKey) -> Fetch {
                Fetch::Bytes(b"stale".to_vec())
            }
            fn fetch_named(&self, _key: CellKey, name: &str) -> Fetch {
                Fetch::Bytes(if name == "abc" { b"fresh".to_vec() } else { b"stale".to_vec() })
            }
        }
        let key = CellKey { depth: 9, prefix: 1 };
        let kept = Arc::new(Kept::default());
        kept.put(key, b"stale");
        let store = TieredStore::new(kept.clone(), Arc::new(Edge));
        assert!(matches!(store.refresh(key), Fetch::Bytes(b) if b == b"stale"), "the plain address is what the edge kept");
        assert!(matches!(store.refresh_named(key, "abc"), Fetch::Bytes(b) if b == b"fresh"));
        assert!(matches!(store.get(key), Fetch::Bytes(b) if b == b"fresh"), "the vault keeps what the name fetched");
    }

    /// A settled view's ring is fetched into the store once, decoded for nothing; nothing is fetched under a cell known absent, nor a layer for a cell with no base.
    #[test]
    fn a_settled_view_prefetches_its_ring_once() {
        use std::sync::Mutex;
        #[derive(Default)]
        struct Counting(Mutex<Vec<CellKey>>);
        impl CellStore for Counting {
            fn get(&self, key: CellKey) -> Fetch {
                self.0.lock().unwrap().push(key);
                Fetch::Absent
            }
        }
        let base = Arc::new(Counting::default());
        let dem = Arc::new(Counting::default());
        let layers = LayerStores { dem: dem.clone(), img: Arc::new(Empty), nir: Arc::new(Empty) };
        let mut r = Residency::layered(base.clone(), Some(layers));
        let a = CellKey::from_grid(3, 12, 100, 100);
        r.want(vec![a], (0, 0));
        settle(&mut r);
        // Settled: the next want sends the ring.
        r.want(vec![a], (0, 0));
        for _ in 0..500 {
            if base.0.lock().unwrap().len() >= 9 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        r.want(vec![a], (0, 0));
        std::thread::sleep(std::time::Duration::from_millis(20));
        let got = base.0.lock().unwrap().clone();
        assert_eq!(got.len(), 9, "the cell, then its eight neighbours once: {got:?}");
        assert!(got[1..].iter().all(|k| k.depth == 12 && *k != a && near(k, (a.grid().0 << 18, a.grid().1 << 18)) <= 1));
        assert!(!got.iter().any(|k| k.depth == 13), "no children of an absent cell");
        // The base fetched as the view loaded had its terrain fetched beside it; the ring's absent bases fetched no terrain.
        assert_eq!(dem.0.lock().unwrap().as_slice(), &[a]);
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
