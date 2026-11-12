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
    /// The least and the most rank of every point met on the screen, by kind (lo > hi for a kind not met): the points kept are the best of each patch ([`POINTS_A_PATCH`]), and a layout that sizes a kind by where each stands among its kind in view still sees them all.
    pub rank_span: Vec<(u32, u32)>,
}

/// A river's kind (a line's kind is its class plus one) and a named lake's (an area's is its class above AREA).
const RIVER: u8 = mahere_tiles::osm::RoadClass::Waterway as u8 + 1;
const LAKE: u8 = mahere_tiles::names::AREA + mahere_tiles::osm::AreaClass::Water as u8;
const _: () = assert!(RIVER == 12 && LAKE == mahere_tiles::names::AREA + 12, "the panel letters waterways as 12 and lakes as AREA + 12");

/// The most cells' names kept between gathers; past it the cache starts again (a backstop: the cache keeps only resident cells anyway).
const NAMES_KEPT: usize = 4096;

/// How far past the screen's edge a texel still counts, so a label running off the edge is laid out whole.
const MARGIN: f32 = 64.0;

/// A feature's identity across cells: the same kind, name and ref in two cells is one thing (borrowed from the cell's table, so a texel met costs no copy of its name).
fn identity(n: &Named) -> (u8, &str, &str) {
    (n.kind, n.name.as_str(), n.reference.as_str())
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

/// A cell's names as the lettering and the search read them: its table, and only the texels that are named (Mahere on ferros, 2026-11-11: the decoded planes were two u16 a texel, half a megabyte a cell kept for every cell in view, nearly all of it zeros).
#[derive(Default)]
pub struct CellNames {
    pub entries: Vec<Named>,
    /// The named line and point texels: (texel, entry + 1), in texel order.
    pub line: Vec<(u32, u16)>,
    /// The named area texels as runs in texel order: (first texel, texels, entry + 1).
    pub area: Vec<(u32, u32, u16)>,
}

impl CellNames {
    /// The named texels of a decoded cell's planes (a plane not in the cell is empty).
    pub fn from_cell(n: NameCell) -> CellNames {
        let line = n.line.iter().enumerate().filter(|(_, e)| **e != 0).map(|(i, &e)| (i as u32, e)).collect();
        let mut area: Vec<(u32, u32, u16)> = Vec::new();
        for (i, &e) in n.area.iter().enumerate() {
            if e == 0 {
                continue;
            }
            match area.last_mut() {
                Some(r) if r.2 == e && (r.0 + r.1) as usize == i => r.1 += 1,
                _ => area.push((i as u32, 1, e)),
            }
        }
        CellNames { entries: n.entries, line, area }
    }

    /// The area a texel lies in.
    pub fn area_at(&self, i: usize) -> Option<&Named> {
        // The last run starting at or before the texel; none (k = 0) when every run starts past it.
        let k = self.area.partition_point(|r| r.0 as usize <= i);
        let r = self.area[k.checked_sub(1)?];
        (i < (r.0 + r.1) as usize).then(|| &self.entries[r.2 as usize - 1])
    }

    /// Every named area texel: (texel, entry + 1).
    pub fn area_texels(&self) -> impl Iterator<Item = (usize, u16)> + '_ {
        self.area.iter().flat_map(|&(s, n, e)| (s as usize..(s + n) as usize).map(move |i| (i, e)))
    }

    /// What it holds in memory, near enough: the table with its text, and the named texels.
    pub fn bytes(&self) -> usize {
        self.entries.iter().map(|n| std::mem::size_of::<Named>() + n.name.capacity() + n.reference.capacity()).sum::<usize>() + self.line.capacity() * std::mem::size_of::<(u32, u16)>() + self.area.capacity() * std::mem::size_of::<(u32, u32, u16)>()
    }
}

/// A worker's cache of cells' names; None for a cell with none.
pub type NamesCache = rustc_hash::FxHashMap<CellKey, Option<Arc<CellNames>>>;

/// A cell's names decoded from its bytes, kept as [`CellNames`].
pub fn decode_names(bytes: &[u8]) -> Option<CellNames> {
    mahere_tiles::decode_names(bytes).map(CellNames::from_cell)
}

/// What a cache of names holds, in bytes.
pub fn cache_bytes(cache: &NamesCache) -> usize {
    cache.values().map(|n| n.as_ref().map_or(0, |n| n.bytes())).sum()
}

/// A cache of names brought under `budget` bytes, the cells farthest from `(lat, lon)` let go first.
pub fn trim_cache(cache: &mut NamesCache, budget: usize, lat: f64, lon: f64) {
    let mut held = cache_bytes(cache);
    if held <= budget {
        return;
    }
    let mut by_far: Vec<(f64, CellKey, usize)> = cache
        .iter()
        .map(|(k, n)| {
            // A corner stands for the cell: near enough to order cells by how far they are.
            let (clat, clon) = k.corners()[0];
            let d = (clat - lat).powi(2) + ((clon - lon + 540.0).rem_euclid(360.0) - 180.0).powi(2);
            (d, *k, n.as_ref().map_or(0, |n| n.bytes()))
        })
        .collect();
    by_far.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, k, bytes) in by_far {
        if held <= budget {
            break;
        }
        cache.remove(&k);
        held -= bytes;
    }
}

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
        let out = gather(&job, &mut cache, &|_| Some(0));
        self.names_cache = cache;
        out
    }
}

/// A cell's names, from the cache or read from the device (never the network); a cell not resident is not looked at.
fn cell_names(job: &NameJob, cache: &mut NamesCache, key: CellKey) -> Option<Arc<CellNames>> {
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
        crate::residency::Fetch::Bytes(b) => decode_names(&b).map(Arc::new),
        _ => None,
    };
    cache.insert(key, n.clone());
    n
}

/// How near two copies of one named point must land to be one label, pixels.
const POINT_MERGE_PX: f32 = 96.0;

/// The most points of one tier gathered in one patch of the screen (a 128th of its span square, about 14 pixels on the Pro), the best by rank: more than the patch could letter, and a globe's points are tens of thousands, not most of a million. Measured at the Pro's globe, three runs each: 115 to 121 labels placed with it, 122 to 127 without (the layout's order varies run to run with its hash maps); four a tier placed 115 to 122, and one cap over every tier 104 to 109 (a patch's towns crowded out its summits).
const POINTS_A_PATCH: usize = 1 << 3;

/// The screen's span, the harmonic mean of its sides.
fn job_span(w: usize, h: usize) -> f32 {
    // WHY the max: a job made before the shell's first resize has a 0 x 0 screen, and 0 / 0 is no span.
    (2 * w * h) as f32 / (w + h).max(1) as f32
}

/// Every named texel in view at the depth the lines are drawn, a cell missing there read from its nearest resident ancestor within three levels, of the names `tier` gives a tier (the lettering's density cut: a name it would never letter is not gathered), and of the points only the best few of each patch of the screen by tier, then rank ([`POINTS_A_PATCH`]). Mahere on ferros, 2026-11-11: a globe's 750 thousand points and 77 thousand lines were gathered for 150 labels, most of a gigabyte on the desktop. Runs on any thread.
pub fn gather(job: &NameJob, cache: &mut NamesCache, tier: &dyn Fn(&Named) -> Option<u8>) -> NamedTexels {
    for k in &job.reloaded {
        cache.remove(k);
    }
    // Only cells still resident keep their names: every cell a session passed through stayed until the 4096th (2026-10-23: a Pixel 3a killed for memory zoomed in at the south pole, the names of every cell seen on the way held).
    cache.retain(|k, _| job.resident.contains(k));
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
    let mut cells: Vec<(CellKey, Arc<CellNames>)> = Vec::new();
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
    // By identity, borrowed from the cells' tables: a name is copied once, for the result, never for each texel met.
    let mut lines: HashMap<(u8, &str, &str), (&Named, Vec<[f32; 2]>)> = HashMap::new();
    let mut areas: HashMap<(u8, &str, &str), (&Named, Vec<[f32; 2]>)> = HashMap::new();
    // One label a point (Nick 2026-10-16: "Madrid Madrid, Paris Paris? yeahnah"): the same place met in a parent cell and its child, or on two texels, lands a few pixels apart; copies of one point within a few label heights merge, and places sharing a name far apart stay apart. A city's limits lettered as an area are another label and stay.
    // The copies of one point are found by its identity first, then by distance among those alone (2026-10-18: compared against every point kept, a globe's tens of thousands of points took minutes, and the lettering never caught up with the view); merged as they are met, so the copies are never held.
    // The points by patch of the screen, the best few of each.
    // WHY the max: a 0 x 0 screen has no span, and a patch of none would make the count of patches infinite.
    let patch = (job_span(w, h) / (1 << 7) as f32).max(1.);
    let across = ((fw + 2. * MARGIN) / patch) as usize + 1;
    let mut patches: Vec<Vec<(u8, u32, &Named, [f32; 2])>> = (0..across * (((fh + 2. * MARGIN) / patch) as usize + 1)).map(|_| Vec::new()).collect();
    let mut placed: HashMap<(u8, &str, &str), Vec<[f32; 2]>> = HashMap::new();
    let mut rank_span = vec![(u32::MAX, 0u32); 1 << 8];
    let (mut tpx_sum, mut tpx_n) = (0.0f32, 0u32);
    let mut area_step = 0.0f32;
    for (key, names) in &cells {
        let tiers: Vec<Option<u8>> = names.entries.iter().map(tier).collect();
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
        for &(i, e) in &names.line {
            let Some(t) = tiers[e as usize - 1] else { continue };
            let (i, n) = (i as usize, &names.entries[e as usize - 1]);
            // A river's centreline runs on through the lakes it feeds; inside a named lake it is the lake's water, not the river's, so its name is not lettered there (Nick 2026-10-15: river labels showed in the lake).
            if n.kind == RIVER && names.area_at(i).is_some_and(|a| a.kind == LAKE) {
                continue;
            }
            let p = centroid(i);
            if !on_screen(p) {
                continue;
            }
            if n.is_point() {
                let s = &mut rank_span[n.kind as usize];
                *s = (s.0.min(n.rank), s.1.max(n.rank));
                if !placed.get(&identity(n)).is_some_and(|at| at.iter().any(|q| (q[0] - p[0]).hypot(q[1] - p[1]) < POINT_MERGE_PX)) {
                    let b = &mut patches[((p[1] + MARGIN) / patch) as usize * across + ((p[0] + MARGIN) / patch) as usize];
                    // Tier first, then rank (the layout's order).
                    let better = |x: (u8, u32), y: (u8, u32)| x.0 < y.0 || (x.0 == y.0 && x.1 > y.1);
                    // Each tier its own few (the bits above the lowest): a patch's towns that cannot be lettered must not crowd out its summits, which can.
                    let same = |x: u8| x >> 1 == t >> 1;
                    if b.iter().filter(|e| same(e.0)).count() < POINTS_A_PATCH {
                        b.push((t, n.rank, n, p));
                        placed.entry(identity(n)).or_default().push(p);
                    } else {
                        // The tier full in this patch: the point takes the place of its tier's worst, if it is better.
                        let worst = (0..b.len()).filter(|&x| same(b[x].0)).fold(usize::MAX, |w, x| if w == usize::MAX || better((b[w].0, b[w].1), (b[x].0, b[x].1)) { x } else { w });
                        if better((t, n.rank), (b[worst].0, b[worst].1)) {
                            // The point let go no longer stands for its copies nearby: one met later may take its place.
                            let (_, _, gone, q) = std::mem::replace(&mut b[worst], (t, n.rank, n, p));
                            if let Some(at) = placed.get_mut(&identity(gone)) {
                                at.retain(|r| *r != q);
                            }
                            placed.entry(identity(n)).or_default().push(p);
                        }
                    }
                }
            } else {
                lines.entry(identity(n)).or_insert_with(|| (n, Vec::new())).1.push(p);
            }
        }
        if !names.area.is_empty() {
            // About four pixels between samples: an area's shape, not every texel of it.
            let k = ((4.0 / texel_px).round() as usize).clamp(1, TEX);
            area_step = area_step.max(k as f32 * texel_px);
            for ty in (0..TEX).step_by(k) {
                for tx in (0..TEX).step_by(k) {
                    let i = mahere_tiles::tri_idx(tx, ty, 0);
                    let Some(n) = names.area_at(i) else { continue };
                    if tier(n).is_none() {
                        continue;
                    }
                    let p = centroid(i);
                    if on_screen(p) {
                        areas.entry(identity(n)).or_insert_with(|| (n, Vec::new())).1.push(p);
                    }
                }
            }
        }
    }
    NamedTexels {
        lines: lines.into_values().map(|(n, pts)| (n.clone(), pts)).collect(),
        areas: areas.into_values().map(|(n, pts)| (n.clone(), pts)).collect(),
        points: patches.into_iter().flatten().map(|(_, _, n, p)| (n.clone(), p)).collect(),
        texel_px: if tpx_n > 0 { tpx_sum / tpx_n as f32 } else { 1.0 },
        area_step_px: area_step.max(1.0),
        rank_span,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cell's names keep only their named texels, the areas as runs, and read back as the planes did.
    #[test]
    fn cell_names_keep_the_named_texels() {
        let named = |kind: u8, name: &str| Named { kind, rank: 0, name: name.to_string(), reference: String::new() };
        let mut line = vec![0u16; mahere_tiles::TRI];
        let mut area = vec![0u16; mahere_tiles::TRI];
        line[7] = 1;
        line[300] = 2;
        area[10..20].iter_mut().for_each(|e| *e = 2);
        area[20..25].iter_mut().for_each(|e| *e = 1);
        area[40] = 2;
        let cell = NameCell { line: line.clone(), area: area.clone(), entries: vec![named(1, "Road"), named(mahere_tiles::names::AREA, "Lake")] };
        let n = CellNames::from_cell(cell);
        assert_eq!(n.line, vec![(7, 1), (300, 2)]);
        assert_eq!(n.area, vec![(10, 10, 2), (20, 5, 1), (40, 1, 2)]);
        for i in 0..64 {
            assert_eq!(n.area_at(i).map(|e| e.name.as_str()), (area[i] != 0).then(|| n.entries[area[i] as usize - 1].name.as_str()), "texel {i}");
        }
        let texels: Vec<(usize, u16)> = n.area_texels().collect();
        assert_eq!(texels, area.iter().enumerate().filter(|(_, e)| **e != 0).map(|(i, &e)| (i, e)).collect::<Vec<_>>());
        // A cell without planes has no named texels.
        assert!(CellNames::from_cell(NameCell { line: Vec::new(), area: Vec::new(), entries: Vec::new() }).area_at(0).is_none());
    }
}
