//! What labels are made from (Nick 2026-10-13: "flood fill and some maths and counting"): every named texel in view, at the depth the map draws its lines, put on the screen. The layout and the lettering are the panel's, which has the text.
use std::collections::HashMap;
use std::sync::Arc;

use mahere_coord::Coord;
use mahere_tiles::{CellKey, NameCell, Named, TEX};

use crate::MapCore;

/// The named texels in view, in screen pixels: each named line's texels, each named area's texels sampled about every four pixels, and each named point.
#[derive(Default)]
pub struct NamedTexels {
    pub lines: Vec<(Named, Vec<[f32; 2]>)>,
    pub areas: Vec<(Named, Vec<[f32; 2]>)>,
    pub points: Vec<(Named, [f32; 2])>,
    /// How far apart the line texels are on the screen, pixels.
    pub texel_px: f32,
    /// How far apart the area samples are, pixels.
    pub area_step_px: f32,
}

/// A river's kind (a line's kind is its class plus one) and a named lake's (an area's is its class above AREA).
const RIVER: u8 = mahere_tiles::osm::RoadClass::Waterway as u8 + 1;
const LAKE: u8 = mahere_tiles::names::AREA + mahere_tiles::osm::AreaClass::Water as u8;
const _: () = assert!(RIVER == 12 && LAKE == mahere_tiles::names::AREA + 12, "the panel letters waterways as 12 and lakes as AREA + 12");

/// The most cells' names kept between gathers; past it the cache starts again.
const NAMES_KEPT: usize = 4096;

/// How far past the screen's edge a texel still counts, so a label running off the edge is laid out whole.
const MARGIN: f32 = 64.0;

/// A feature's identity across cells: the same kind, name and ref in two cells is one thing.
fn identity(n: &Named) -> (u8, String, String) {
    (n.kind, n.name.clone(), n.reference.clone())
}

/// What a label layout needs from the map, taken on the drawing thread and read on another (Nick 2026-10-13: the layout must never hold up a frame): the camera, the screen, the depth the lines are drawn at, which cells are resident with something in them, where their bytes are, and which were replaced by a newer copy since the last job.
pub struct NameJob {
    pub cam: crate::Camera,
    pub w: usize,
    pub h: usize,
    pub depth: u8,
    resident: rustc_hash::FxHashSet<CellKey>,
    store: Arc<dyn crate::residency::CellStore>,
    pub reloaded: Vec<CellKey>,
}

/// A worker's cache of cells' names; None for a cell with none.
pub type NamesCache = rustc_hash::FxHashMap<CellKey, Option<Arc<NameCell>>>;

impl MapCore {
    /// The inputs of a label layout for the view, for another thread to gather from.
    pub fn name_job(&mut self, w: usize, h: usize) -> NameJob {
        let resident = self.res.pool.map.iter().filter(|(_, e)| e.present != 0).map(|(k, _)| *k).collect();
        NameJob { cam: self.cam, w, h, depth: self.vec_depth, resident, store: self.res.store(), reloaded: std::mem::take(&mut self.names_reloaded) }
    }

    /// Every named texel in view, gathered here and now (a host that letters on its drawing thread, or a check).
    pub fn gather_names(&mut self, w: usize, h: usize) -> NamedTexels {
        let job = self.name_job(w, h);
        let mut cache = std::mem::take(&mut self.names_cache);
        let out = gather(&job, &mut cache);
        self.names_cache = cache;
        out
    }
}

/// A cell's names, from the cache or read from the device (never the network); a cell not resident is not looked at.
fn cell_names(job: &NameJob, cache: &mut NamesCache, key: CellKey) -> Option<Arc<NameCell>> {
    if !job.resident.contains(&key) {
        return None;
    }
    if let Some(n) = cache.get(&key) {
        return n.clone();
    }
    if cache.len() >= NAMES_KEPT {
        cache.clear();
    }
    let n = match job.store.get_local(key) {
        crate::residency::Fetch::Bytes(b) => mahere_tiles::decode_names(&b).map(Arc::new),
        _ => None,
    };
    cache.insert(key, n.clone());
    n
}

/// Every named texel in view at the depth the lines are drawn, a cell missing there read from its nearest resident ancestor within three levels. Runs on any thread.
pub fn gather(job: &NameJob, cache: &mut NamesCache) -> NamedTexels {
    for k in &job.reloaded {
        cache.remove(k);
    }
    let (w, h, depth) = (job.w, job.h, job.depth);
    let mut keys: Vec<CellKey> = Vec::new();
    let step = 32usize;
    for gy in (0..=h + step).step_by(step) {
        for gx in (0..=w + step).step_by(step) {
            let (x, y) = (gx.min(w) as f64, gy.min(h) as f64);
            if !job.cam.on_globe(x, y, w, h) {
                continue;
            }
            let (lat, lon) = job.cam.screen_to_geo(x, y, w, h);
            let k = CellKey::containing(Coord::from_lat_lon(lat, lon), depth);
            if !keys.contains(&k) {
                keys.push(k);
            }
        }
    }
    let mut cells: Vec<(CellKey, Arc<NameCell>)> = Vec::new();
    for k in keys {
        for up in 0..=3u8 {
            if k.depth < up {
                break;
            }
            let a = k.ancestor(k.depth - up);
            if let Some(n) = cell_names(job, cache, a) {
                if !cells.iter().any(|(c, _)| *c == a) {
                    cells.push((a, n));
                }
                break;
            }
            if job.resident.contains(&a) {
                // Resident and nameless: nothing named here at this depth.
                break;
            }
        }
    }
        let (fw, fh) = (w as f32, h as f32);
        let on_screen = |p: [f32; 2]| p[0] >= -MARGIN && p[1] >= -MARGIN && p[0] <= fw + MARGIN && p[1] <= fh + MARGIN;
        let mut lines: HashMap<(u8, String, String), (Named, Vec<[f32; 2]>)> = HashMap::new();
        let mut areas: HashMap<(u8, String, String), (Named, Vec<[f32; 2]>)> = HashMap::new();
        let mut out = NamedTexels::default();
        let (mut tpx_sum, mut tpx_n) = (0.0f32, 0u32);
        let mut area_step = 0.0f32;
        for (key, names) in cells {
            // The cell's square on the screen from its corners: a texel's place is the bilinear blend, near enough over one cell.
            let c = key.corners().map(|(lat, lon)| {
                let (x, y) = job.cam.geo_to_screen(lat, lon, w, h);
                [x as f32, y as f32]
            });
            let place = |u: f32, v: f32| -> [f32; 2] {
                let mix = |k: usize| c[0][k] * (1.0 - u) * (1.0 - v) + c[1][k] * u * (1.0 - v) + c[2][k] * (1.0 - u) * v + c[3][k] * u * v;
                [mix(0), mix(1)]
            };
            let texel_px = ((c[1][0] - c[0][0]).hypot(c[1][1] - c[0][1]) / TEX as f32).max(1e-3);
            if key.depth == depth {
                tpx_sum += texel_px;
                tpx_n += 1;
            }
            let centroid = |i: usize| -> [f32; 2] {
                let (half, tx, ty) = (i & 1, (i >> 1) & (TEX - 1), i >> 9);
                let off = if half == 0 { 1.0 / 3.0 } else { 2.0 / 3.0 };
                place((tx as f32 + off) / TEX as f32, (ty as f32 + off) / TEX as f32)
            };
            for (i, &e) in names.line.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                let n = &names.entries[e as usize - 1];
                // A river's centreline runs on through the lakes it feeds; inside a named lake it is the lake's water, not the river's, so its name is not lettered there (Nick 2026-10-15: river labels showed in the lake).
                if n.kind == RIVER && names.area_at(i).is_some_and(|a| a.kind == LAKE) {
                    continue;
                }
                let p = centroid(i);
                if !on_screen(p) {
                    continue;
                }
                if n.is_point() {
                    out.points.push((n.clone(), p));
                } else {
                    lines.entry(identity(n)).or_insert_with(|| (n.clone(), Vec::new())).1.push(p);
                }
            }
            if !names.area.is_empty() {
                // About four pixels between samples: an area's shape, not every texel of it.
                let k = ((4.0 / texel_px).round() as usize).clamp(1, TEX);
                area_step = area_step.max(k as f32 * texel_px);
                for ty in (0..TEX).step_by(k) {
                    for tx in (0..TEX).step_by(k) {
                        let i = mahere_tiles::tri_idx(tx, ty, 0);
                        let e = names.area[i];
                        if e == 0 {
                            continue;
                        }
                        let p = centroid(i);
                        if on_screen(p) {
                            let n = &names.entries[e as usize - 1];
                            areas.entry(identity(n)).or_insert_with(|| (n.clone(), Vec::new())).1.push(p);
                        }
                    }
                }
            }
        }
        out.lines = lines.into_values().collect();
        out.areas = areas.into_values().collect();
        out.texel_px = if tpx_n > 0 { tpx_sum / tpx_n as f32 } else { 1.0 };
        out.area_step_px = area_step.max(1.0);
        out
}
