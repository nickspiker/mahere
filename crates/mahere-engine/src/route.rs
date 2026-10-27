//! Routes by flood fill (Nick 2026-10-13): A* over the trixels the ways' centrelines were baked into, at the base depth. A route moves along a way from a trixel to any that touches it, changes ways only where they truly meet (a join; a bridge stacked over a road never meets it), and keeps to a one-way's direction. Driving costs each trixel its length at the posted limit or the class's speed; cycling and walking at their pace. The time along the found path is worked again from the 1 m elevation: Tobler's hiking function on foot, a cost per metre climbed on a bike.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use mahere_coord::{uv_to_lat_lon, Coord};
use mahere_tiles::route::RouteCell;
use mahere_tiles::{tri_at, tri_idx, use_bits, CellKey, TEX};

use crate::residency::{CellStore, Fetch};

/// The depth routes are found at: the areas' finest.
pub const ROUTE_DEPTH: u8 = 14;

/// The coarsest depth the route pyramid goes to: the global bake's.
pub const ROUTE_MIN: u8 = 8;

/// How far from a trip's ends a depth's maze is walked, metres: beyond it the search climbs to the coarser depth, whose major roads carry the middle of the trip. Doubles a depth up; unbounded at the coarsest.
fn reach(depth: u8) -> f64 {
    if depth <= ROUTE_MIN { f64::MAX } else { 2000.0 * (1u32 << (ROUTE_DEPTH - depth)) as f64 }
}

/// The side of a diamond's trixel grid at a depth.
fn side(depth: u8) -> u32 {
    (1u32 << depth) * TEX as u32
}

/// Metres between two places, flat over the span a trip covers.
fn crow(a: (f64, f64), b: (f64, f64)) -> f64 {
    let k = ((a.0 + b.0) * 0.5).to_radians().cos();
    (((a.0 - b.0) * 111_320.0).powi(2) + ((a.1 - b.1) * 111_320.0 * k).powi(2)).sqrt()
}

/// The most states a search looks at before giving up.
const MAX_STATES: usize = 6_000_000;

/// How far from a way a start or an end may be, trixels (about 340 m at depth 14).
const SNAP_REACH: i64 = 200;

/// How far a start or an end looks for a way at a coarser depth, metres: the fine maze is not everywhere, and a trip from open country starts on the nearest major road.
const SNAP_REACH_M: f64 = 2000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Travel {
    Car,
    Bike,
    Foot,
}

/// A ferry's line class id (`RoadClass::Ferry` + 1).
const FERRY: u8 = mahere_tiles::osm::RoadClass::Ferry as u8 + 1;

impl Travel {
    pub fn word(self) -> &'static str {
        match self {
            Travel::Car => "Car",
            Travel::Bike => "Bike",
            Travel::Foot => "Foot",
        }
    }

    /// The fastest this way of travelling goes, m/s: the search's estimate of what is left may never be more than the truth.
    fn top_speed(self) -> f64 {
        match self {
            Travel::Car => 130.0 / 3.6,
            Travel::Bike => 30.0 / 3.6,
            Travel::Foot => 6.0 / 3.6,
        }
    }

    /// The speed on a way, m/s; None where this way of travelling may not go.
    fn speed(self, w: &mahere_tiles::route::Way) -> Option<f64> {
        let unpaved = w.uses & use_bits::UNPAVED != 0;
        // Trail difficulty (sac_scale, 1 T1 .. 6 T6) and roughness (1 gravel .. 3 high clearance) from the grade byte (2026-10-16).
        let (difficulty, rough) = (w.grade & 15, w.grade >> 4);
        // A ferry carries every way of travelling at its own pace.
        if w.class == FERRY {
            return Some(if w.speed_kmh > 0 { w.speed_kmh as f64 } else { 20.0 } / 3.6);
        }
        let kmh = match self {
            Travel::Car => {
                if w.uses & use_bits::MOTOR == 0 {
                    return None;
                }
                // A road for high clearance only where nothing else goes.
                let rough_factor = [1.0, 0.8, 0.5, 0.15][rough.min(3) as usize];
                let class = match w.class {
                    1 => 105.0,
                    2 => 90.0,
                    3 => 70.0,
                    4 => 60.0,
                    5 => 50.0,
                    6 => 40.0,
                    _ => 20.0,
                };
                let posted = if w.speed_kmh > 0 { w.speed_kmh as f64 } else { class };
                // A gated, private or seasonal way only where nothing else goes.
                let penalty = if w.uses & use_bits::RESTRICTED != 0 { 0.2 } else { 1.0 };
                posted * if unpaved { 0.6 } else { 1.0 } * penalty * rough_factor
            }
            Travel::Bike => {
                if w.uses & use_bits::BIKE == 0 {
                    return None;
                }
                // Alpine routes are no place for a bike; demanding mountain trails are walked alongside it.
                match difficulty {
                    4.. => return None,
                    3 => 5.0,
                    _ => (if unpaved || w.class >= 8 { 12.0 } else { 18.0 }) * [1.0, 0.9, 0.6, 0.4][rough.min(3) as usize],
                }
            }
            Travel::Foot => {
                if w.uses & use_bits::FOOT == 0 {
                    return None;
                }
                // Walking pace by the trail's difficulty: a hiking trail at 4.8 km/h down to hands-on scrambling at 1.
                [4.8, 4.8, 4.0, 3.0, 2.0, 1.5, 1.0][difficulty.min(6) as usize]
            }
        };
        Some(kmh / 3.6)
    }

    /// What changing ways costs, seconds: a turn at an intersection.
    fn turn(self) -> f64 {
        match self {
            Travel::Car => 4.0,
            Travel::Bike => 2.0,
            Travel::Foot => 0.0,
        }
    }

    fn keeps_one_ways(self) -> bool {
        self != Travel::Foot
    }
}

/// One instruction: where a route turns onto a way.
#[derive(Clone, Debug)]
pub struct Step {
    pub name: String,
    pub turn: &'static str,
    pub at_m: f64,
}

#[derive(Clone, Debug)]
pub struct Route {
    pub travel: Travel,
    /// The way, as latitude and longitude.
    pub pts: Vec<(f64, f64)>,
    pub metres: f64,
    pub seconds: f64,
    pub climb_m: f64,
    pub steps: Vec<Step>,
    /// The ground along the way: the elevation at [`PROFILE_SAMPLES`] points evenly spaced from start to end, NaN where there is no terrain.
    pub profile: Vec<f32>,
    /// What draws the route again through finer cells as they arrive (Nick 2026-10-26: zooming in on a stretch fixes its line).
    pub(crate) again: Option<Again>,
}

/// A route's runs and ends, kept so it can be finished again once finer cells are on the device.
#[derive(Clone, Debug)]
pub(crate) struct Again {
    runs: Vec<Run>,
    from: (f64, f64),
    to: (f64, f64),
    a: T,
    b: T,
    seconds: f64,
    travel: Travel,
}

impl Route {
    /// How many of the route's runs are drawn coarser than the finest depth: finer cells arriving could refine them.
    pub fn coarse_runs(&self) -> usize {
        self.again.as_ref().map_or(0, |a| a.runs.iter().filter(|r| r.t.first().is_some_and(|t| t.depth < ROUTE_DEPTH)).count())
    }
}

/// The route drawn again through the finer cells the device holds now, or None when no run got finer.
pub fn refine_route(store: Arc<dyn CellStore>, dem: Arc<dyn CellStore>, route: &Route, cache: Option<Arc<RouteCache>>) -> Option<Route> {
    let again = route.again.as_ref()?;
    let mut fine = Cells { store: store.clone(), got: HashMap::new(), cache: cache.clone(), local: true };
    let mut runs = again.runs.clone();
    let mut finer = false;
    for run in &mut runs {
        let depth = run.t.first().map(|t| t.depth);
        refine(&mut fine, run);
        finer |= run.t.first().map(|t| t.depth) != depth;
    }
    if !finer {
        return None;
    }
    let mut cells = Cells { store, got: HashMap::new(), cache, local: true };
    finish_runs(&mut cells, &dem, again.travel, again.from, again.to, again.a, again.b, again.seconds, runs).ok()
}

/// How many points the route's elevation profile has.
pub const PROFILE_SAMPLES: usize = 512;

/// A trixel anywhere at the route depth: its diamond, its column and row of the diamond's grid, its half.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct T {
    d: u8,
    depth: u8,
    x: u32,
    y: u32,
    h: u8,
}


impl T {
    fn of(lat: f64, lon: f64) -> T {
        T::at(lat, lon, ROUTE_DEPTH)
    }

    /// The trixel holding a place at a depth.
    fn at(lat: f64, lon: f64, depth: u8) -> T {
        let c = Coord::from_lat_lon(lat, lon);
        let (iu, iv) = c.uv();
        let s = (1u64 << (30 - depth as u32 - mahere_tiles::TEX_BITS as u32)) as f64;
        let (x, y, h) = tri_at(iu as f64 / s, iv as f64 / s);
        T { d: c.diamond(), depth, x: x as u32, y: y as u32, h: h as u8 }
    }

    fn cell(self) -> (CellKey, usize) {
        let t = TEX as u32;
        (CellKey::from_grid(self.d, self.depth, (self.x / t) as u64, (self.y / t) as u64), tri_idx((self.x % t) as usize, (self.y % t) as usize, self.h as usize))
    }

    /// The trixel a depth up that holds this one.
    fn parent(self) -> T {
        let (x, y, h) = mahere_tiles::route::tri_parent(self.x as usize, self.y as usize, self.h as usize);
        T { d: self.d, depth: self.depth - 1, x: x as u32, y: y as u32, h: h as u8 }
    }

    /// The four trixels a depth down inside this one.
    fn children(self) -> [T; 4] {
        mahere_tiles::tri_children(self.x as usize, self.y as usize, self.h as usize).map(|(x, y, h)| T { d: self.d, depth: self.depth + 1, x: x as u32, y: y as u32, h: h as u8 })
    }

    /// The centroid, in the diamond's grid.
    fn centre(self) -> (f64, f64) {
        let off = if self.h == 0 { 1.0 / 3.0 } else { 2.0 / 3.0 };
        (self.x as f64 + off, self.y as f64 + off)
    }

    fn lat_lon(self) -> (f64, f64) {
        let (gx, gy) = self.centre();
        let side = side(self.depth) as f64;
        uv_to_lat_lon(self.d, gx / side, gy / side)
    }

    /// The trixels touching this one: both halves of the squares around it.
    fn around(self) -> impl Iterator<Item = T> {
        let me = self;
        (-1i64..=1).flat_map(move |dy| (-1i64..=1).flat_map(move |dx| (0..2u8).map(move |h| (dx, dy, h)))).filter_map(move |(dx, dy, h)| {
            let (x, y) = (me.x as i64 + dx, me.y as i64 + dy);
            let t = T { d: me.d, depth: me.depth, x: x as u32, y: y as u32, h };
            let s = side(me.depth) as i64;
            (x >= 0 && y >= 0 && x < s && y < s && t != me).then_some(t)
        })
    }
}

/// Metres between two points of a diamond's grid (its two axes 60° apart).
fn grid_m(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (du, dv) = (b.0 - a.0, b.1 - a.1);
    (du * du + dv * dv + du * dv).max(0.0).sqrt() * mahere_tiles::texel_m(ROUTE_DEPTH)
}

/// Cells' routing kept across searches (navigation asks again every few seconds): the decoded cells, their graphs built where the cell had none. Cleared whole past [`CACHE_CELLS`].
#[derive(Default)]
pub struct RouteCache(std::sync::Mutex<HashMap<CellKey, Option<Arc<RouteCell>>>>);

/// Cells a route cache holds before it is cleared: about 70 MB of planes at most.
const CACHE_CELLS: usize = 4000;

/// Cells' routing as the search reads it, each read once from the store, through the shared cache.
struct Cells {
    store: Arc<dyn CellStore>,
    got: HashMap<CellKey, Option<Arc<RouteCell>>>,
    cache: Option<Arc<RouteCache>>,
    /// Only what the device holds, never the network: the line's refinement reads the fine cells there are, as the map draws what it has.
    local: bool,
}

impl Cells {
    fn get(&mut self, key: CellKey) -> Option<Arc<RouteCell>> {
        if let Some(c) = self.got.get(&key) {
            return c.clone();
        }
        if let Some(shared) = self.cache.as_ref().and_then(|c| c.0.lock().ok().and_then(|m| m.get(&key).cloned())) {
            self.got.insert(key, shared.clone());
            return shared;
        }
        let fetched = if self.local { self.store.get_local(key) } else { self.store.get(key) };
        let c = match fetched {
            Fetch::Bytes(b) => mahere_tiles::decode_route(&b).map(|mut r| {
                // A cell baked before the graph: built here from its planes, the same graph the baker makes.
                if !r.has_graph() {
                    let t = std::time::Instant::now();
                    r.build_graph(key.depth);
                    if std::env::var_os("MAHERE_ROUTE_LOG").is_some() {
                        eprintln!("graph: {} ways, {} nodes, {} corridors built in {:.1} ms", r.ways.len(), r.nodes.len(), r.corridors.len(), t.elapsed().as_secs_f64() * 1e3);
                    }
                }
                Arc::new(r)
            }),
            _ => None,
        };
        self.got.insert(key, c.clone());
        if let Some(cache) = &self.cache {
            if let Ok(mut m) = cache.0.lock() {
                if m.len() >= CACHE_CELLS {
                    m.clear();
                }
                // Only what is there: an absent cell may arrive later.
                if c.is_some() {
                    m.insert(key, c.clone());
                }
            }
        }
        c
    }

    /// The ways on a trixel, by OSM id, with the cell's way record.
    fn ways(&mut self, t: T) -> Vec<(i64, mahere_tiles::route::Way, u16, Arc<RouteCell>)> {
        let (key, i) = t.cell();
        let Some(c) = self.get(key) else { return Vec::new() };
        c.ways_at(i).map(|w| (c.ways[w as usize - 1].osm, c.ways[w as usize - 1].clone(), w, c.clone())).collect()
    }
}

/// The ways a point could start or end on: the nearest trixels carrying a way this travel may use, found in rings out to [`SNAP_REACH`] (or [`SNAP_REACH_M`] at a coarser depth), each with the metres off the way.
fn snap(cells: &mut Cells, at: T, travel: Travel, point: (f64, f64)) -> Vec<(T, i64, f64)> {
    let mut found = Vec::new();
    let texel = mahere_tiles::texel_m(at.depth);
    let rings = if at.depth == ROUTE_DEPTH { SNAP_REACH } else { (SNAP_REACH_M / texel) as i64 };
    for r in 0..=rings {
        for dy in -r..=r {
            for dx in -r..=r {
                if dx.abs() != r && dy.abs() != r {
                    continue;
                }
                let (x, y) = (at.x as i64 + dx, at.y as i64 + dy);
                let s = side(at.depth) as i64;
                if x < 0 || y < 0 || x >= s || y >= s {
                    continue;
                }
                for h in 0..2u8 {
                    let t = T { d: at.d, depth: at.depth, x: x as u32, y: y as u32, h };
                    for (osm, way, _, _) in cells.ways(t) {
                        if travel.speed(&way).is_some() {
                            // From the point itself, not its trixel's centre: at a coarse depth a road a few hundred metres off can share the point's trixel.
                            found.push((t, osm, crow(point, t.lat_lon())));
                        }
                    }
                }
            }
        }
        // Rings a few trixels past the first find, so a closer way just beyond a corner is not missed.
        if let Some(&(_, _, best)) = found.iter().min_by(|a, b| a.2.total_cmp(&b.2)) {
            if r as f64 * texel > best + 10.0 {
                break;
            }
        }
    }
    found
}

/// A point's ways at the finest depth that has any near it, and at every coarser one that has (2026-10-27: a start on a 1 m area's edge snapped to a minor road the area's coarse cells cut off, a dead end, with I-90 in the planet's cells a few hundred metres further): each costs its walk to the way, so the search starts on whichever connects.
fn snap_down(cells: &mut Cells, lat: f64, lon: f64, travel: Travel) -> (T, Vec<(T, i64, f64)>) {
    let mut first: Option<T> = None;
    let mut all = Vec::new();
    for depth in (ROUTE_MIN..=ROUTE_DEPTH).rev() {
        let at = T::at(lat, lon, depth);
        let found = snap(cells, at, travel, (lat, lon));
        if !found.is_empty() {
            // A coarser depth's way is a fallback for when the finest one does not connect: its walk counts double and half a kilometre more, so a start on a side street stays on it.
            let fallback = first.is_some();
            first.get_or_insert(at);
            all.extend(found.into_iter().map(|(t, osm, m)| (t, osm, if fallback { m * 2.0 + 500.0 } else { m })));
        }
    }
    (first.unwrap_or_else(|| T::of(lat, lon)), all)
}

/// The best route from one point to another by stepping trixel to trixel: what [`find`] did before the graph (2026-10-25), kept to check the graph's routes against.
pub fn find_trixels(store: Arc<dyn CellStore>, dem: Arc<dyn CellStore>, from: (f64, f64), to: (f64, f64), travel: Travel) -> Result<Route, String> {
    let mut cells = Cells { store: store.clone(), got: HashMap::new(), cache: None, local: false };
    let (a, b) = (T::of(from.0, from.1), T::of(to.0, to.1));
    if a.d != b.d {
        return Err("the ends are in different diamonds of the map; routes cross no diamond yet".into());
    }
    let starts = snap(&mut cells, a, travel, from);
    let ends = snap(&mut cells, b, travel, to);
    if starts.is_empty() {
        return Err("no way to start on near here".into());
    }
    if ends.is_empty() {
        return Err("no way to arrive on near the target".into());
    }
    let goal: HashMap<T, f64> = ends.iter().map(|&(t, _, m)| (t, m)).collect();
    let goal_centre = b.centre();
    let top = travel.top_speed();
    // Getting to and from the way: walked, or driven slowly.
    let off_way = match travel {
        Travel::Car => 15.0 / 3.6,
        _ => 4.0 / 3.6,
    };
    let estimate = |t: T| grid_m(t.centre(), goal_centre) / top;

    // The states: a trixel on a way. Each keeps its best time and where it came from.
    let mut index: HashMap<(T, i64), usize> = HashMap::new();
    let mut states: Vec<(T, i64, f64, usize)> = Vec::new();
    let mut open: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    let key = |s: f64| (s * 1000.0) as u64;
    for &(t, osm, m) in &starts {
        let g = m / off_way;
        let i = states.len();
        states.push((t, osm, g, usize::MAX));
        index.insert((t, osm), i);
        open.push(Reverse((key(g + estimate(t)), i)));
    }
    let mut done: Option<(usize, f64)> = None;
    while let Some(Reverse((_, i))) = open.pop() {
        let (t, osm, g, _) = states[i];
        if done.is_some_and(|(_, best)| g >= best) {
            break;
        }
        if let Some(&m) = goal.get(&t) {
            let total = g + m / off_way;
            if done.is_none_or(|(_, best)| total < best) {
                done = Some((i, total));
            }
        }
        if states.len() > MAX_STATES {
            return Err(format!("gave up after {} states; too far for one search yet", states.len()));
        }
        let (tkey, ti) = t.cell();
        let Some(cell) = cells.get(tkey) else { continue };
        let Some(wnum) = cell.ways_at(ti).find(|&w| cell.ways[w as usize - 1].osm == osm) else { continue };
        let Some(speed) = travel.speed(&cell.ways[wnum as usize - 1]) else { continue };
        let mut push = |states: &mut Vec<(T, i64, f64, usize)>, open: &mut BinaryHeap<Reverse<(u64, usize)>>, nt: T, nosm: i64, ng: f64| {
            match index.get(&(nt, nosm)) {
                Some(&j) if states[j].2 <= ng => {}
                Some(&j) => {
                    states[j].2 = ng;
                    states[j].3 = i;
                    open.push(Reverse((key(ng + estimate(nt)), j)));
                }
                None => {
                    let j = states.len();
                    states.push((nt, nosm, ng, i));
                    index.insert((nt, nosm), j);
                    open.push(Reverse((key(ng + estimate(nt)), j)));
                }
            }
        };
        // Along the way.
        let dir_here = if travel.keeps_one_ways() { cell.direction(ti, wnum) } else { None };
        for nt in t.around() {
            let (nkey, ni) = nt.cell();
            let ncell = if nkey == tkey { cell.clone() } else {
                match cells.get(nkey) {
                    Some(c) => c,
                    None => continue,
                }
            };
            let Some(nw) = ncell.ways_at(ni).find(|&w| ncell.ways[w as usize - 1].osm == osm) else { continue };
            let (a, b) = (t.centre(), nt.centre());
            if travel.keeps_one_ways() {
                if let Some(dir) = dir_here.or_else(|| ncell.direction(ni, nw)) {
                    let (du, dv) = (b.0 - a.0, b.1 - a.1);
                    if du * dir.cos() as f64 + dv * dir.sin() as f64 <= -0.05 {
                        continue;
                    }
                }
            }
            let ng = g + grid_m(a, b) / speed;
            push(&mut states, &mut open, nt, osm, ng);
        }
        // Onto another way, where they meet.
        if cell.is_join(ti) {
            for w in cell.ways_at(ti) {
                let way = &cell.ways[w as usize - 1];
                if way.osm != osm && travel.speed(way).is_some() {
                    push(&mut states, &mut open, t, way.osm, g + travel.turn());
                }
            }
        }
    }
    if std::env::var_os("MAHERE_ROUTE_LOG").is_some() {
        eprintln!("route: {} states, {} cells read", states.len(), cells.got.len());
    }
    let Some((last, seconds)) = done else { return Err("no route between them on the ways baked so far".into()) };
    // The path back to its start.
    let mut chain = Vec::new();
    let mut i = last;
    while i != usize::MAX {
        chain.push((states[i].0, states[i].1));
        i = states[i].3;
    }
    chain.reverse();
    let chain = chain.iter().enumerate().map(|(k, &(t, osm))| (t, osm, if k == 0 { 0.0 } else { crow(chain[k - 1].0.lat_lon(), t.lat_lon()) })).collect();
    finish(&mut cells, &dem, travel, from, to, a, b, seconds, chain)
}

/// A node of the graph a search stands on: the cell, the node cluster's name and the way (by OSM id).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Node {
    key: CellKey,
    name: u32,
    osm: i64,
}

/// How a search reached a node: along a corridor of its cell, by turning onto another way at a junction, or by crossing the cell's edge.
#[derive(Clone, Copy, Debug)]
enum Via {
    Start,
    Corridor(u16),
    Turn,
    Hop,
    /// To the same place a depth up or down the pyramid: the trixel stood on there, and the corridor it lies in (0 at a node), whose run to the node reached is part of the way.
    /// `dead`: up from a node where the finer maze ended (an area's edge cuts the road): this node drives on at its own depth.
    Level { at: T, cor: u16, dead: bool },
}

impl T {
    /// A cell's trixel as a global one.
    fn from_cell(key: CellKey, i: usize) -> T {
        let (cu, cv) = key.grid();
        let (half, tx, ty) = (i & 1, (i >> 1) & (TEX - 1), i >> (1 + mahere_tiles::TEX_BITS));
        T { d: key.diamond(), depth: key.depth, x: (cu as u32) * TEX as u32 + tx as u32, y: (cv as u32) * TEX as u32 + ty as u32, h: half as u8 }
    }

    /// The trixels past the diamond's own grid around a trixel on its edge: a point a texel and a half outward, by latitude and longitude, and its surroundings in whatever diamond holds it.
    fn beyond(self) -> Vec<T> {
        let (gx, gy) = self.centre();
        let side = side(self.depth) as f64;
        let mut out = Vec::new();
        for (dx, dy) in [(-1.5, 0.0), (1.5, 0.0), (0.0, -1.5), (0.0, 1.5)] {
            let (px, py) = (gx + dx, gy + dy);
            if px >= 0.0 && py >= 0.0 && px < side && py < side {
                continue;
            }
            // Past the edge: the point in another diamond, found by its latitude and longitude.
            let (lat, lon) = uv_to_lat_lon(self.d, (px / side).clamp(-0.01, 1.01), (py / side).clamp(-0.01, 1.01));
            let t = T::at(lat, lon, self.depth);
            if t.d != self.d {
                out.push(t);
                out.extend(t.around());
            }
        }
        out
    }
}

/// Where a snapped trixel stands in the graph: on a node (that node, no distance), or in a corridor with the texel edges to each of its two nodes along the corridor's trixels.
enum Stand {
    At(Node),
    Between { cor: u16, a: Node, b: Node, to_a: f64, to_b: f64 },
}

/// The shortest run of trixels from `from` to any of `to`, through the trixels of corridor `cor` of way `w` and the clusters at its ends, in texel edges; None when there is none.
/// Any corridor trixel of the way may carry it, at four times the cost off its own: a hub (one run of trixels touching several nodes, a coarse cell's merged switchbacks) marks its trixels with its first corridor only, and the others share them (Nick 2026-10-26: the Snow Lake route jumped the switchbacks). Another node's trixels still block, so the run cannot stray past the two ends.
fn trace(cell: &RouteCell, w: u16, cor: u16, from: usize, to: &[usize], ends: [u32; 2]) -> Option<(f64, Vec<usize>)> {
    let allowed = |j: usize| cell.corridor_at(j, w) != 0 || cell.node_of(j, w).is_some_and(|n| n == ends[0] || n == ends[1]);
    let weight = |j: usize| if cell.corridor_at(j, w) == cor || cell.node_of(j, w).is_some() { 1.0 } else { 4.0 };
    let mut dist: HashMap<usize, (f64, usize)> = HashMap::new();
    dist.insert(from, (0.0, usize::MAX));
    let mut heap = BinaryHeap::new();
    heap.push(Reverse((0u64, from)));
    let mut hit = None;
    while let Some(Reverse((dk, k))) = heap.pop() {
        let d = dk as f64 / 256.0;
        if d > dist[&k].0 + 1e-9 {
            continue;
        }
        if to.contains(&k) {
            hit = Some(k);
            break;
        }
        for j in cell_around(k) {
            if !allowed(j) {
                continue;
            }
            let nd = d + grid_m(cell_centre(k), cell_centre(j)) / mahere_tiles::texel_m(ROUTE_DEPTH) * weight(j);
            if dist.get(&j).is_none_or(|o| nd < o.0) {
                dist.insert(j, (nd, k));
                heap.push(Reverse(((nd * 256.0) as u64, j)));
            }
        }
    }
    let end = hit?;
    let mut path = vec![end];
    let mut k = end;
    while let Some(&(_, p)) = dist.get(&k) {
        if p == usize::MAX {
            break;
        }
        path.push(p);
        k = p;
    }
    path.reverse();
    Some((dist[&end].0, path))
}

/// A cell trixel's neighbours and centroid, by memory index.
fn cell_around(i: usize) -> impl Iterator<Item = usize> {
    let (tx, ty) = ((i >> 1) & (TEX - 1), i >> (1 + mahere_tiles::TEX_BITS));
    (-1i64..=1).flat_map(move |dy| (-1i64..=1).flat_map(move |dx| (0..2usize).map(move |h| (dx, dy, h)))).filter_map(move |(dx, dy, h)| {
        let (x, y) = (tx as i64 + dx, ty as i64 + dy);
        if x < 0 || y < 0 || x >= TEX as i64 || y >= TEX as i64 {
            return None;
        }
        let j = tri_idx(x as usize, y as usize, h);
        (j != i).then_some(j)
    })
}

fn cell_centre(i: usize) -> (f64, f64) {
    let (half, tx, ty) = (i & 1, (i >> 1) & (TEX - 1), i >> (1 + mahere_tiles::TEX_BITS));
    mahere_tiles::tri_centroid(tx, ty, half)
}

/// Where a snapped trixel of a way stands.
fn stand(cells: &mut Cells, t: T, osm: i64) -> Option<(Stand, Arc<RouteCell>, u16)> {
    let (key, i) = t.cell();
    let cell = cells.get(key)?;
    let w = cell.ways_at(i).find(|&w| cell.ways[w as usize - 1].osm == osm)?;
    if let Some(name) = cell.node_of(i, w) {
        return Some((Stand::At(Node { key, name, osm }), cell, w));
    }
    let cor = cell.corridor_at(i, w);
    if cor == 0 {
        return None;
    }
    let c = cell.corridors[cor as usize - 1];
    let to = |name: u32| -> Option<f64> {
        let targets: Vec<usize> = cell.node_trixels(name, w).into_iter().map(|x| x as usize).collect();
        trace(&cell, w, cor, i, &targets, [c.a, c.b]).map(|(d, _)| d)
    };
    let (to_a, to_b) = (to(c.a)?, to(c.b)?);
    Some((Stand::Between { cor, a: Node { key, name: c.a, osm }, b: Node { key, name: c.b, osm }, to_a, to_b }, cell, w))
}

/// The best route from one point to another, or why there is none: a search over the cells' graphs, node to node along corridors, onto other ways at junctions, across cells (and diamonds) at the edges.
/// `dem` is where the terrain comes from for the climb (the base store itself for cells baked whole).
pub fn find(store: Arc<dyn CellStore>, dem: Arc<dyn CellStore>, from: (f64, f64), to: (f64, f64), travel: Travel) -> Result<Route, String> {
    find_cached(store, dem, from, to, travel, None)
}

/// [`find`] through a cache of cells kept across searches.
pub fn find_cached(store: Arc<dyn CellStore>, dem: Arc<dyn CellStore>, from: (f64, f64), to: (f64, f64), travel: Travel, cache: Option<Arc<RouteCache>>) -> Result<Route, String> {
    let mut cells = Cells { store: store.clone(), got: HashMap::new(), cache, local: false };
    let (a, starts) = snap_down(&mut cells, from.0, from.1, travel);
    let (b, ends) = snap_down(&mut cells, to.0, to.1, travel);
    if std::env::var_os("MAHERE_ROUTE_LOG").is_some() {
        let show = |v: &[(T, i64, f64)]| v.iter().take(6).map(|(t, osm, m)| format!("d{} way {osm} {m:.0} m", t.depth)).collect::<Vec<_>>().join(", ");
        eprintln!("snapped: start at depth {} [{}], end at depth {} [{}]", a.depth, show(&starts), b.depth, show(&ends));
    }
    if starts.is_empty() {
        return Err("no way to start on near here".into());
    }
    if ends.is_empty() {
        return Err("no way to arrive on near the target".into());
    }
    let top = travel.top_speed();
    // Getting to and from the way: walked, or driven slowly.
    let off_way = match travel {
        Travel::Car => 15.0 / 3.6,
        _ => 4.0 / 3.6,
    };
    let goal_ll = b.lat_lon();
    let start_ll = a.lat_lon();
    // Metres as the crow flies between a node and the goal, for the estimate (across diamonds too).
    let estimate = |ll: (f64, f64)| -> f64 { crow(ll, goal_ll) / top };
    // The states: a node, with the best time to it, where it came from and how.
    let mut index: HashMap<Node, usize> = HashMap::new();
    let mut states: Vec<(Node, f64, usize, Via)> = Vec::new();
    let mut open: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    let key_of = |s: f64| (s * 1000.0) as u64;
    let mut node_ll: HashMap<Node, (f64, f64)> = HashMap::new();
    let ll_of = |cells: &mut Cells, n: Node, node_ll: &mut HashMap<Node, (f64, f64)>| -> (f64, f64) {
        if let Some(&ll) = node_ll.get(&n) {
            return ll;
        }
        let ll = cells.get(n.key).and_then(|c| {
            let w = c.ways.iter().position(|w| w.osm == n.osm)? as u16 + 1;
            Some(T::from_cell(n.key, n.name as usize).lat_lon()).filter(|_| c.node_of(n.name as usize, w).is_some())
        }).unwrap_or(goal_ll);
        node_ll.insert(n, ll);
        ll
    };
    // The starts: the nodes at each end of the corridor a start stands in, the time along it included.
    let push = |states: &mut Vec<(Node, f64, usize, Via)>, open: &mut BinaryHeap<Reverse<(u64, usize)>>, index: &mut HashMap<Node, usize>, n: Node, g: f64, from: usize, via: Via, est: f64| {
        match index.get(&n) {
            Some(&j) if states[j].1 <= g => {}
            Some(&j) => {
                states[j].1 = g;
                states[j].2 = from;
                states[j].3 = via;
                open.push(Reverse((key_of(g + est), j)));
            }
            None => {
                let j = states.len();
                states.push((n, g, from, via));
                index.insert(n, j);
                open.push(Reverse((key_of(g + est), j)));
            }
        }
    };
    let mut start_stands: Vec<(T, i64, Stand)> = Vec::new();
    for &(t, osm, m) in &starts {
        let Some((st, cell, w)) = stand(&mut cells, t, osm) else { continue };
        let speed = travel.speed(&cell.ways[w as usize - 1]).unwrap_or(off_way);
        let texel = mahere_tiles::texel_m(t.depth);
        let g0 = m / off_way;
        match &st {
            Stand::At(n) => {
                let est = estimate(ll_of(&mut cells, *n, &mut node_ll));
                push(&mut states, &mut open, &mut index, *n, g0, usize::MAX, Via::Start, est);
            }
            Stand::Between { cor, a, b, to_a, to_b } => {
                let c = cell.corridors[*cor as usize - 1];
                let keep = travel.keeps_one_ways();
                if !keep || c.dir != 1 {
                    let est = estimate(ll_of(&mut cells, *a, &mut node_ll));
                    push(&mut states, &mut open, &mut index, *a, g0 + to_a * texel / speed, usize::MAX, Via::Start, est);
                }
                if !keep || c.dir != 2 {
                    let est = estimate(ll_of(&mut cells, *b, &mut node_ll));
                    push(&mut states, &mut open, &mut index, *b, g0 + to_b * texel / speed, usize::MAX, Via::Start, est);
                }
            }
        }
        start_stands.push((t, osm, st));
    }
    if states.is_empty() {
        return Err("no way to start on near here".into());
    }
    // The goals: each end's node, or the nodes of its corridor with the time from them to it.
    let mut goal: HashMap<Node, (f64, T, i64)> = HashMap::new();
    for &(t, osm, m) in &ends {
        let Some((st, cell, w)) = stand(&mut cells, t, osm) else { continue };
        let speed = travel.speed(&cell.ways[w as usize - 1]).unwrap_or(off_way);
        let texel = mahere_tiles::texel_m(t.depth);
        let extra = m / off_way;
        let mut add = |n: Node, g: f64| {
            if goal.get(&n).is_none_or(|o| g < o.0) {
                goal.insert(n, (g, t, osm));
            }
        };
        match st {
            Stand::At(n) => add(n, extra),
            Stand::Between { cor, a, b, to_a, to_b } => {
                let c = cell.corridors[cor as usize - 1];
                let keep = travel.keeps_one_ways();
                if !keep || c.dir != 2 {
                    add(a, extra + to_a * texel / speed);
                }
                if !keep || c.dir != 1 {
                    add(b, extra + to_b * texel / speed);
                }
            }
        }
    }
    if goal.is_empty() {
        return Err("no way to arrive on near the target".into());
    }
    let mut done: Option<(usize, f64)> = None;
    while let Some(Reverse((_, i))) = open.pop() {
        let (n, g, _, via_here) = states[i];
        if done.is_some_and(|(_, best)| g >= best) {
            break;
        }
        if let Some(&(extra, _, _)) = goal.get(&n) {
            let total = g + extra;
            if done.is_none_or(|(_, best)| total < best) {
                done = Some((i, total));
            }
        }
        if states.len() > MAX_STATES {
            return Err(format!("gave up after {} nodes; too far for one search yet", states.len()));
        }
        let Some(cell) = cells.get(n.key) else { continue };
        let Some(w) = cell.ways.iter().position(|x| x.osm == n.osm).map(|p| p as u16 + 1) else { continue };
        let Some(speed) = travel.speed(&cell.ways[w as usize - 1]) else { continue };
        let depth = n.key.depth;
        let texel = mahere_tiles::texel_m(depth);
        let here = ll_of(&mut cells, n, &mut node_ll);
        // A depth's maze is walked within its reach of an end and outside the finer depth's: the fine maze near the ends, the major roads' between, each depth taking over where the finer one's reach ends.
        let near = |ll: (f64, f64), r: f64| crow(ll, start_ll) < r || crow(ll, goal_ll) < r;
        let within = |r: f64| near(here, r);
        // Up from a dead end below (2026-10-27: I-90 ends at the Leavenworth area's edge in its fine cells; the planet's I-90 went back down into them instead of on, and Cle Elum to Ellensburg went by Chinook Pass): walked, whatever its depth.
        let forced = matches!(via_here, Via::Level { dead: true, .. });
        if std::env::var_os("MAHERE_ROUTE_TRACE").is_some() {
            eprintln!("expand\t{depth}\t{:.5}\t{:.5}\t{}\t{:.0}\t{:?}\t{forced}\t{}", here.0, here.1, n.osm, g, via_here, cell.ways[w as usize - 1].name);
        }
        let reached = within(reach(depth));
        let trixels = cell.node_trixels(n.name, w);
        // To the same place at another depth: a node there, or a point along a corridor there, whose two ends cost their way along it (the way is on every parent and child trixel of its own, the pyramid being a union).
        let level = |cells: &mut Cells, states: &mut Vec<(Node, f64, usize, Via)>, open: &mut BinaryHeap<Reverse<(u64, usize)>>, index: &mut HashMap<Node, usize>, node_ll: &mut HashMap<Node, (f64, f64)>, t: T, dead: bool| -> bool {
            let Some((st, c, cw)) = stand(cells, t, n.osm) else { return false };
            // Standing in a corridor there and going on to its end is a walk at that depth: not where a finer depth holds this spot and the spot is within its reach (Nick 2026-10-26: the Snow Lake walk was drawn at depth 8, its corridors a hair cheaper than the fine maze's for having fewer cells to cross).
            if matches!(st, Stand::Between { .. }) && t.depth < ROUTE_DEPTH && near(t.lat_lon(), reach(t.depth + 1)) {
                let finer = t.children().iter().any(|ct| {
                    let (ck, ci) = ct.cell();
                    cells.get(ck).is_some_and(|cc| cc.ways_at(ci).any(|x| cc.ways[x as usize - 1].osm == n.osm))
                });
                if finer {
                    return false;
                }
            }
            let sp = travel.speed(&c.ways[cw as usize - 1]).unwrap_or(speed);
            let tm = mahere_tiles::texel_m(t.depth);
            match st {
                Stand::At(m) => {
                    let est = estimate(ll_of(cells, m, node_ll));
                    push(states, open, index, m, g, i, Via::Level { at: t, cor: 0, dead }, est);
                }
                Stand::Between { cor, a: ma, b: mb, to_a, to_b } => {
                    let k = c.corridors[cor as usize - 1];
                    let keep = travel.keeps_one_ways();
                    if !keep || k.dir != 1 {
                        let est = estimate(ll_of(cells, ma, node_ll));
                        push(states, open, index, ma, g + to_a * tm / sp, i, Via::Level { at: t, cor, dead }, est);
                    }
                    if !keep || k.dir != 2 {
                        let est = estimate(ll_of(cells, mb, node_ll));
                        push(states, open, index, mb, g + to_b * tm / sp, i, Via::Level { at: t, cor, dead }, est);
                    }
                }
            }
            true
        };
        // Up a depth, where the way is kept there.
        let mut climbed = false;
        // A dead end here: the way has one corridor from this node and no other way joins it.
        let dead = cell.corridors.iter().filter(|c| c.way == w && (c.a == n.name || c.b == n.name)).count() <= 1 && !trixels.iter().any(|&t| cell.is_join(t as usize));
        if depth > ROUTE_MIN && mahere_tiles::route::class_floor(cell.ways[w as usize - 1].class) <= depth - 1 {
            for &t in &trixels {
                // The parent texel, else one beside it: a road drawn a texel wide at two depths does not always pass through the parent of each of its fine texels (2026-10-27: I-90 left the Leavenworth area's cells and never stepped up onto the planet's, so Seattle to Ellensburg went by way of Canada).
                let p = T::from_cell(n.key, t as usize).parent();
                if !level(&mut cells, &mut states, &mut open, &mut index, &mut node_ll, p, dead) {
                    for q in p.around() {
                        if level(&mut cells, &mut states, &mut open, &mut index, &mut node_ll, q, dead) {
                            climbed = true;
                            break;
                        }
                    }
                } else {
                    climbed = true;
                }
            }
        }
        // Down a depth, where the finer maze is this spot's (within its reach of an end): the fine maze for the first and last legs, and the depth the trip's middle belongs to.
        let mut descended = false;
        if depth < ROUTE_DEPTH && within(reach(depth + 1)) {
            for &t in &trixels {
                for ct in T::from_cell(n.key, t as usize).children() {
                    descended |= level(&mut cells, &mut states, &mut open, &mut index, &mut node_ll, ct, false);
                }
            }
        }
        // A depth is walked within its reach of an end and outside the finer depth's, or inside the finer one's where the finer maze has no way to come down to (2026-10-27: east of Cle Elum the Leavenworth area has no fine cells, so every coarser depth left I-90 to a finer one that was not there).
        let walk = forced || (reached && (depth == ROUTE_DEPTH || !within(reach(depth + 1)) || !descended));
        // A node that can neither climb (no coarser cell yet, a way not kept there, the coarsest depth) nor come down (no finer cell here) walks on at its own depth.
        // Either move is enough: the coarsest depth walked on wherever it could not climb (Nick 2026-10-26, the Snow Lake walk drawn in 100 m pieces), its corridors a hair cheaper than the fine maze's for having fewer cells to cross.
        if !walk && (climbed || descended) {
            continue;
        }
        // Along the corridors from this node.
        for (ci, c) in cell.corridors.iter().enumerate() {
            if c.way != w || (c.a != n.name && c.b != n.name) {
                continue;
            }
            let other = if c.a == n.name { c.b } else { c.a };
            if travel.keeps_one_ways() {
                let forward = c.a == n.name;
                if (c.dir == 1 && !forward) || (c.dir == 2 && forward) {
                    continue;
                }
            }
            let m = Node { key: n.key, name: other, osm: n.osm };
            let est = estimate(ll_of(&mut cells, m, &mut node_ll));
            push(&mut states, &mut open, &mut index, m, g + c.len as f64 / 256.0 * texel / speed, i, Via::Corridor(ci as u16 + 1), est);
        }
        for &t in &trixels {
            let ti = t as usize;
            // Onto another way, where they meet.
            if cell.is_join(ti) {
                for ow in cell.ways_at(ti) {
                    let way = &cell.ways[ow as usize - 1];
                    if way.osm == n.osm || travel.speed(way).is_none() {
                        continue;
                    }
                    let Some(oname) = cell.node_of(ti, ow) else { continue };
                    let m = Node { key: n.key, name: oname, osm: way.osm };
                    let est = estimate(ll_of(&mut cells, m, &mut node_ll));
                    push(&mut states, &mut open, &mut index, m, g + travel.turn(), i, Via::Turn, est);
                }
            }
            // Across the cell's edge: the same way in the cell beyond.
            let gt = T::from_cell(n.key, ti);
            let mut over: Vec<T> = gt.around().collect();
            over.extend(gt.beyond());
            for nt in over {
                let (nkey, ni) = nt.cell();
                if nkey == n.key {
                    continue;
                }
                let Some(ncell) = cells.get(nkey) else { continue };
                let Some(nw) = ncell.ways_at(ni).find(|&x| ncell.ways[x as usize - 1].osm == n.osm) else { continue };
                let Some(nname) = ncell.node_of(ni, nw) else { continue };
                let m = Node { key: nkey, name: nname, osm: n.osm };
                let est = estimate(ll_of(&mut cells, m, &mut node_ll));
                let step = if nt.d == gt.d { grid_m(gt.centre(), nt.centre()) / mahere_tiles::texel_m(ROUTE_DEPTH) * texel } else { texel };
                push(&mut states, &mut open, &mut index, m, g + step.max(texel) / speed, i, Via::Hop, est);
            }
        }
    }
    if std::env::var_os("MAHERE_ROUTE_LOG").is_some() {
        eprintln!("route: {} nodes, {} cells read", states.len(), cells.got.len());
    }
    let Some((last, seconds)) = done else { return Err("no route between them on the ways baked so far".into()) };
    // The nodes back to the start, then the trixels: along each corridor its traced run, at a turn or a hop the node itself.
    let mut nodes = Vec::new();
    let mut i = last;
    while i != usize::MAX {
        nodes.push(states[i]);
        i = states[i].2;
    }
    nodes.reverse();
    if std::env::var_os("MAHERE_ROUTE_LOG").is_some() {
        for (n, g, _, via) in &nodes {
            eprintln!("  depth {} cell {} node {} way {} at {:.0} s via {via:?}", n.key.depth, n.key.name(), n.name, n.osm, g);
        }
    }
    // The route as runs of trixels, each along one way: a whole corridor's run carries the corridor's own length (the road's, from the bake).
    let mut runs: Vec<Run> = Vec::new();
    let extend = |runs: &mut Vec<Run>, t: Vec<T>, osm: i64, arc: Option<f64>| runs.push(Run { t, osm, arc });
    // From the start trixel to the first node, along its corridor.
    let first = nodes[0].0;
    if let Some((t, osm, st)) = start_stands.iter().find(|(_, osm, st)| *osm == first.osm && match st { Stand::At(n) => *n == first, Stand::Between { a, b, .. } => *a == first || *b == first }) {
        let (key, ti) = t.cell();
        if let Some(cell) = cells.get(key) {
            let w = cell.ways.iter().position(|x| x.osm == *osm).map_or(0, |p| p as u16 + 1);
            match st {
                Stand::At(_) => extend(&mut runs, vec![*t], *osm, None),
                Stand::Between { cor, a, b, .. } => {
                    let targets: Vec<usize> = cell.node_trixels(first.name, w).into_iter().map(|x| x as usize).collect();
                    match trace(&cell, w, *cor, ti, &targets, [a.name, b.name]) {
                        Some((_, path)) => extend(&mut runs, path.into_iter().map(|j| T::from_cell(key, j)).collect(), *osm, None),
                        None => extend(&mut runs, vec![*t], *osm, None),
                    }
                }
            }
        }
    }
    for k in 1..nodes.len() {
        let (n, _, _, via) = nodes[k];
        let prev = nodes[k - 1].0;
        match via {
            Via::Corridor(cor) => {
                if let Some(cell) = cells.get(n.key) {
                    let w = cell.ways.iter().position(|x| x.osm == n.osm).map_or(0, |p| p as u16 + 1);
                    let c = cell.corridors[cor as usize - 1];
                    let from_t = cell.node_trixels(prev.name, w);
                    let targets: Vec<usize> = cell.node_trixels(n.name, w).into_iter().map(|x| x as usize).collect();
                    let traced = from_t.iter().find_map(|&f| trace(&cell, w, cor, f as usize, &targets, [c.a, c.b]));
                    if traced.is_none() && std::env::var_os("MAHERE_ROUTE_LOG").is_some() {
                        let marked = (0..mahere_tiles::TRI).filter(|&j| cell.corridor_at(j, w) == cor).count();
                        eprintln!("  untraced corridor {cor} of way {} in {} depth {}: nodes {} -> {} (a {} b {} len {} dir {}), from {:?} to {:?}, {marked} trixels marked", n.osm, n.key.name(), n.key.depth, prev.name, n.name, c.a, c.b, c.len, c.dir, from_t, targets);
                    }
                    let arc = c.len as f64 / 256.0 * mahere_tiles::texel_m(n.key.depth);
                    match traced {
                        Some((_, path)) => extend(&mut runs, path.into_iter().map(|j| T::from_cell(n.key, j)).collect(), n.osm, Some(arc)),
                        None => extend(&mut runs, vec![T::from_cell(n.key, n.name as usize)], n.osm, None),
                    }
                }
            }
            // A level move that stood in a corridor at the new depth walked that corridor to the node (Nick 2026-10-26: the line cut the switchbacks where the search climbed or came down mid-corridor): its run traced, as a corridor's is.
            Via::Level { at, cor, .. } if cor != 0 => {
                let (key, ti) = at.cell();
                let traced = cells.get(key).and_then(|cell| {
                    let w = cell.ways.iter().position(|x| x.osm == n.osm).map_or(0, |p| p as u16 + 1);
                    let c = cell.corridors[cor as usize - 1];
                    let targets: Vec<usize> = cell.node_trixels(n.name, w).into_iter().map(|x| x as usize).collect();
                    trace(&cell, w, cor, ti, &targets, [c.a, c.b]).map(|(_, path)| path)
                });
                match traced {
                    Some(path) => extend(&mut runs, path.into_iter().map(|j| T::from_cell(n.key, j)).collect(), n.osm, None),
                    None => extend(&mut runs, vec![T::from_cell(n.key, n.name as usize)], n.osm, None),
                }
            }
            Via::Turn | Via::Hop | Via::Start | Via::Level { .. } => extend(&mut runs, vec![T::from_cell(n.key, n.name as usize)], n.osm, None),
        }
    }
    // From the last node to the end trixel.
    let lastn = nodes[nodes.len() - 1].0;
    if let Some(&(_, t, osm)) = goal.get(&lastn) {
        let (key, ti) = t.cell();
        if let Some(cell) = cells.get(key) {
            let w = cell.ways.iter().position(|x| x.osm == osm).map_or(0, |p| p as u16 + 1);
            let cor = cell.corridor_at(ti, w);
            if cor != 0 {
                let c = cell.corridors[cor as usize - 1];
                let from_t = cell.node_trixels(lastn.name, w);
                if let Some((_, mut path)) = from_t.iter().find_map(|&f| trace(&cell, w, cor, f as usize, &[ti], [c.a, c.b])) {
                    path.remove(0);
                    extend(&mut runs, path.into_iter().map(|j| T::from_cell(key, j)).collect(), osm, None);
                }
            } else if runs.last().and_then(|r| r.t.last()).is_none_or(|l| *l != t) {
                extend(&mut runs, vec![t], osm, None);
            }
        }
    }
    // Every run drawn through the finest cells the device holds (Nick 2026-10-26: "no cutting switchbacks"): the search's coarse corridors are its hierarchy, the line is the road.
    let t_fine = std::time::Instant::now();
    let mut fine = Cells { store: store.clone(), got: HashMap::new(), cache: cells.cache.clone(), local: true };
    for run in &mut runs {
        refine(&mut fine, run);
    }
    if std::env::var_os("MAHERE_ROUTE_LOG").is_some() {
        eprintln!("refined {} runs through {} cells in {:.0} ms", runs.len(), fine.got.len(), t_fine.elapsed().as_secs_f64() * 1e3);
    }
    finish_runs(&mut cells, &dem, travel, from, to, a, b, seconds, runs)
}

/// A route from its runs: the chain of trixels with each step's metres (a corridor's length spread along its run, so the route measures the road and not the staircase of trixel centres, 3 percent long on a winding road), finished, and keeping its runs to be drawn again finer.
#[allow(clippy::too_many_arguments)]
fn finish_runs(cells: &mut Cells, dem: &Arc<dyn CellStore>, travel: Travel, from: (f64, f64), to: (f64, f64), a: T, b: T, seconds: f64, runs: Vec<Run>) -> Result<Route, String> {
    let mut chain: Vec<(T, i64, f64)> = Vec::new();
    for run in &runs {
        let start = chain.len();
        let mut prev = chain.last().map(|c| c.0.lat_lon());
        for &t in &run.t {
            let ll = t.lat_lon();
            chain.push((t, run.osm, prev.map_or(0.0, |p| crow(p, ll))));
            prev = Some(ll);
        }
        if let Some(arc) = run.arc {
            let sum: f64 = chain[start..].iter().map(|c| c.2).sum();
            if sum > 0.0 {
                for c in &mut chain[start..] {
                    c.2 *= arc / sum;
                }
            }
        }
    }
    // The same trixel twice in a row says nothing.
    chain.dedup_by(|x, y| x.0 == y.0 && x.1 == y.1);
    if std::env::var_os("MAHERE_ROUTE_LOG").is_some() {
        for k in 1..chain.len() {
            let gap = crow(chain[k - 1].0.lat_lon(), chain[k].0.lat_lon());
            if gap > 30.0 {
                eprintln!("  gap {gap:.0} m at {k}: {:?} way {} -> {:?} way {}", chain[k - 1].0, chain[k - 1].1, chain[k].0, chain[k].1);
            }
        }
    }
    let mut route = finish(cells, dem, travel, from, to, a, b, seconds, chain)?;
    route.again = Some(Again { runs, from, to, a, b, seconds, travel });
    Ok(route)
}

/// A run of a route: trixels along one way, at one depth, and the way's length over them when the bake knows it (a whole corridor).
#[derive(Clone, Debug)]
struct Run {
    t: Vec<T>,
    osm: i64,
    arc: Option<f64>,
}

/// A run drawn a depth finer at a time, as far as the device's cells go: at each depth the way's trixels under the run's (the pyramid is a union, so the road below lies under the band above), and the shortest path through them from under the run's first trixel to under its last. A depth whose cells are missing, or whose trixels do not join, leaves the run as it was.
fn refine(cells: &mut Cells, run: &mut Run) {
    while run.t.first().is_some_and(|t| t.depth < ROUTE_DEPTH) {
        // The candidates: the children carrying the way, sorted so a neighbour is found by binary search.
        let mut under: Vec<T> = Vec::new();
        for t in &run.t {
            for ct in t.children() {
                let (key, i) = ct.cell();
                let Some(cell) = cells.get(key) else { return };
                if cell.ways_at(i).any(|w| cell.ways[w as usize - 1].osm == run.osm) {
                    under.push(ct);
                }
            }
        }
        let rank = |t: &T| (t.x, t.y, t.h);
        under.sort_by_key(rank);
        under.dedup();
        let at = |t: &T| under.binary_search_by_key(&rank(t), rank).ok();
        let (first, last) = (run.t[0], run.t[run.t.len() - 1]);
        let is_child = |t: &T, of: T| t.depth == of.depth + 1 && mahere_tiles::route::tri_parent(t.x as usize, t.y as usize, t.h as usize) == (of.x as usize, of.y as usize, of.h as usize);
        let goals: Vec<usize> = under.iter().enumerate().filter(|(_, t)| is_child(t, last)).map(|(k, _)| k).collect();
        let mut dist = vec![f64::MAX; under.len()];
        let mut back = vec![usize::MAX; under.len()];
        let mut heap = BinaryHeap::new();
        for (k, t) in under.iter().enumerate() {
            if is_child(t, first) {
                dist[k] = 0.0;
                heap.push(Reverse((0u64, k)));
            }
        }
        let mut hit = None;
        while let Some(Reverse((dk, k))) = heap.pop() {
            let d = dk as f64 / 256.0;
            if d > dist[k] + 1e-9 {
                continue;
            }
            if goals.contains(&k) {
                hit = Some(k);
                break;
            }
            for nt in under[k].around() {
                let Some(j) = at(&nt) else { continue };
                let nd = d + grid_m(under[k].centre(), nt.centre());
                if nd < dist[j] {
                    dist[j] = nd;
                    back[j] = k;
                    heap.push(Reverse(((nd * 256.0) as u64, j)));
                }
            }
        }
        let Some(mut k) = hit else { return };
        let mut path = vec![under[k]];
        while back[k] != usize::MAX {
            k = back[k];
            path.push(under[k]);
        }
        path.reverse();
        run.t = path;
    }
}

/// A route from its chain of trixels and ways: the points, the length, the steps with their turns, and for a bike or on foot the time and climb from the ground.
#[allow(clippy::too_many_arguments)]
fn finish(cells: &mut Cells, dem: &Arc<dyn CellStore>, travel: Travel, from: (f64, f64), to: (f64, f64), a: T, b: T, seconds: f64, chain: Vec<(T, i64, f64)>) -> Result<Route, String> {
    if chain.is_empty() {
        return Err("no route between them on the ways baked so far".into());
    }
    let off_way = match travel {
        Travel::Car => 15.0 / 3.6,
        _ => 4.0 / 3.6,
    };
    let names: HashMap<i64, String> = chain.iter().filter_map(|s| cells.ways(s.0).into_iter().find(|w| w.0 == s.1).map(|w| (s.1, w.1.name))).collect();
    // A ferry rides at sea level: its stretch of the profile is the water, not the seabed under it (Nick 2026-10-26: Puget Sound's floor showed under the Southworth ferry).
    let ferries: Vec<i64> = chain.iter().map(|s| s.1).filter(|&osm| cells.ways(chain.iter().find(|s| s.1 == osm).unwrap().0).into_iter().any(|w| w.0 == osm && w.1.class == FERRY)).collect();
    let afloat: Vec<bool> = chain.iter().map(|s| ferries.contains(&s.1)).collect();
    let mut route = Route { travel, pts: Vec::with_capacity(chain.len()), metres: 0.0, seconds, climb_m: 0.0, steps: Vec::new(), profile: Vec::new(), again: None };
    route.pts.push(from);
    let mut last_way = None;
    for (n, s) in chain.iter().enumerate() {
        route.metres += s.2;
        if last_way != Some(s.1) {
            let name = names.get(&s.1).cloned().unwrap_or_default();
            let turn = if n == 0 { "Start" } else { turn_word(&chain, n) };
            if route.steps.last().is_none_or(|l| l.name != name) {
                route.steps.push(Step { name, turn, at_m: route.metres });
            }
            last_way = Some(s.1);
        }
        route.pts.push(s.0.lat_lon());
    }
    route.pts.push(to);
    // A short step says nothing a driver needs (a bridge's few metres, a ten-metre jog): folded into the one before it.
    let mut kept: Vec<Step> = Vec::new();
    for (k, step) in route.steps.iter().enumerate() {
        let len = route.steps.get(k + 1).map_or(f64::MAX, |n| n.at_m - step.at_m);
        let short = len < if step.name.is_empty() { 120.0 } else { 25.0 };
        if k > 0 && short {
            continue;
        }
        if kept.last().is_some_and(|l| l.name == step.name) {
            continue;
        }
        kept.push(step.clone());
    }
    route.steps = kept;
    let mut ground = Ground { store: dem.clone(), dems: HashMap::new() };
    // The profile: the ground at evenly spaced points along the chain.
    let mut along = 0.0;
    let mut k = 0;
    route.profile = (0..PROFILE_SAMPLES)
        .map(|i| {
            let want = route.metres * i as f64 / (PROFILE_SAMPLES - 1) as f64;
            while k + 1 < chain.len() && along + chain[k + 1].2 <= want {
                k += 1;
                along += chain[k].2;
            }
            if afloat[k] { 0.0 } else { ground.elev(chain[k].0).unwrap_or(f32::NAN) }
        })
        .collect();
    if travel != Travel::Car {
        let (time, climb) = ground_time(&mut ground, &chain.iter().map(|s| s.0).collect::<Vec<_>>(), &afloat, travel);
        route.climb_m = climb;
        if let Some(t) = time {
            route.seconds = t + crow(a.lat_lon(), chain[0].0.lat_lon()) / off_way + crow(chain.last().unwrap().0.lat_lon(), b.lat_lon()) / off_way;
        }
    }
    Ok(route)
}

/// Left, right or straight at step `n` of a chain: the turn between the heading over the thirty metres before and the thirty after.
fn turn_word(chain: &[(T, i64, f64)], n: usize) -> &'static str {
    let here = chain[n].0.lat_lon();
    let mut k = n;
    while k > 0 && crow(chain[k].0.lat_lon(), here) < 30.0 {
        k -= 1;
    }
    let back = chain[k].0.lat_lon();
    let mut k = n;
    while k + 1 < chain.len() && crow(chain[k].0.lat_lon(), here) < 30.0 {
        k += 1;
    }
    let ahead = chain[k].0.lat_lon();
    let k = here.0.to_radians().cos();
    let (ax, ay) = ((here.1 - back.1) * k, here.0 - back.0);
    let (bx, by) = ((ahead.1 - here.1) * k, ahead.0 - here.0);
    let angle = (ax * by - ay * bx).atan2(ax * bx + ay * by).to_degrees();
    if angle > 30.0 {
        "Turn left onto"
    } else if angle < -30.0 {
        "Turn right onto"
    } else {
        "Continue onto"
    }
}

/// The ground's elevation at a trixel, from the terrain layer, each cell read once.
struct Ground {
    store: Arc<dyn CellStore>,
    dems: HashMap<CellKey, Option<Arc<Vec<f32>>>>,
}

impl Ground {
    fn elev(&mut self, t: T) -> Option<f32> {
        let (key, i) = t.cell();
        let store = &self.store;
        let plane = self
            .dems
            .entry(key)
            .or_insert_with(|| match store.get(key) {
                Fetch::Bytes(b) => mahere_tiles::decode_cell(&b).ok().and_then(|p| p.dem).map(|d| Arc::new(d.elev)),
                _ => None,
            })
            .clone()?;
        plane.get(i).copied().filter(|e| e.is_finite())
    }
}

/// The time along a path on foot or a bike from the ground's own elevation, every few metres: Tobler's hiking function walking, and on a bike its pace plus a minute for every ten metres climbed. With the metres climbed. None when the elevation is not there.
fn ground_time(ground: &mut Ground, path: &[T], afloat: &[bool], travel: Travel) -> (Option<f64>, f64) {
    let mut elev = |k: usize| if afloat[k] { Some(0.0) } else { ground.elev(path[k]) };
    let (mut time, mut climb) = (0.0, 0.0);
    let step = 6usize;
    let mut prev: Option<(T, f32)> = None;
    for k in (0..path.len()).step_by(step).chain(std::iter::once(path.len().saturating_sub(1))) {
        let t = path[k];
        let Some(e) = elev(k) else { return (None, climb) };
        if let Some((pt, pe)) = prev {
            let len = crow(pt.lat_lon(), t.lat_lon());
            if len < 0.01 {
                continue;
            }
            let rise = (e - pe) as f64;
            climb += rise.max(0.0);
            time += match travel {
                Travel::Foot => len / (6.0 * (-3.5 * (rise / len + 0.05).abs()).exp() / 3.6),
                _ => len / (16.0 / 3.6) + rise.max(0.0) * 6.0,
            };
        }
        prev = Some((t, e));
    }
    (Some(time), climb)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ferries carry everyone at a ferry's pace; a trail's difficulty slows a walker and turns a bike away; a high-clearance road slows a car to a crawl.
    #[test]
    fn grades_and_ferries_set_the_pace() {
        let way = |class: u8, uses: u8, grade: u8| mahere_tiles::route::Way { class, uses, grade, ..Default::default() };
        let all = use_bits::FOOT | use_bits::BIKE | use_bits::MOTOR;
        for t in [Travel::Car, Travel::Bike, Travel::Foot] {
            assert_eq!(t.speed(&way(FERRY, all, 0)), Some(20.0 / 3.6), "{} by ferry", t.word());
        }
        let path = 9;
        assert_eq!(Travel::Foot.speed(&way(path, all, 1)), Some(4.8 / 3.6));
        assert_eq!(Travel::Foot.speed(&way(path, all, 6)), Some(1.0 / 3.6), "T6 is hands-on");
        assert_eq!(Travel::Bike.speed(&way(path, all, 4)), None, "no bikes on alpine routes");
        let track = 8;
        let smooth = Travel::Car.speed(&way(track, all, 0)).unwrap();
        let rough = Travel::Car.speed(&way(track, all, 3 << 4)).unwrap();
        assert!(rough < smooth * 0.2);
    }
    use mahere_tiles::osm::{Road, RoadClass, Travel as Tags};

    struct Baked(HashMap<CellKey, Vec<u8>>);
    impl CellStore for Baked {
        fn get(&self, key: CellKey) -> Fetch {
            self.0.get(&key).map_or(Fetch::Absent, |b| Fetch::Bytes(b.clone()))
        }
    }

    fn road(osm: i64, pts: Vec<(f32, f32)>, oneway: i8, layer: i8) -> Road {
        Road { class: RoadClass::Residential, pts, weight: 0.5, uses: use_bits::MOTOR | use_bits::FOOT | use_bits::BIKE, name: osm as u32, travel: Tags { way: osm, speed_kmh: 0, oneway, layer, ..Default::default() } }
    }

    fn baked(roads: &[Road], junctions: &[(f32, f32)]) -> Arc<dyn CellStore> {
        let names: Vec<mahere_tiles::Named> = ["", "East Road", "North Road", "The Bridge", "The Highway"].iter().map(|n| mahere_tiles::Named { kind: 6, rank: 0, name: n.to_string(), reference: String::new() }).collect();
        let cells = mahere_tiles::route::bake_routes(roads, &names, junctions, ROUTE_DEPTH);
        let cells = mahere_tiles::route::pyramid_routes(cells, ROUTE_DEPTH, ROUTE_MIN);
        let bytes = cells.into_iter().map(|(k, r)| (k, mahere_tiles::Cell { route: Some(r), ..Default::default() }.quantize().encode(&mahere_tiles::Loss { dem_m: 0.0, img: 0 }).unwrap())).collect();
        Arc::new(Baked(bytes))
    }

    #[test]
    fn a_route_turns_only_where_ways_meet() {
        // A road east, a road north off its middle at a junction, and a bridge north over the east road's far end with no junction.
        let (lat, lon) = (47.60f32, -122.33f32);
        let east = road(1, vec![(lat, lon - 0.004), (lat, lon), (lat, lon + 0.004)], 0, 0);
        let north = road(2, vec![(lat, lon), (lat + 0.004, lon)], 0, 0);
        let bridge = road(3, vec![(lat - 0.004, lon + 0.003), (lat + 0.004, lon + 0.003)], 0, 1);
        let store = baked(&[east, north, bridge], &[(lat, lon)]);
        let to_north = find(store.clone(), store.clone(), (lat as f64, lon as f64 - 0.0035), (lat as f64 + 0.0035, lon as f64), Travel::Car).unwrap();
        let said: Vec<String> = to_north.steps.iter().map(|s| format!("{} {}", s.turn, s.name)).collect();
        assert_eq!(said, ["Start East Road", "Turn left onto North Road"], "{:?}", to_north.steps);
        // The bridge's far end is reachable only by the bridge: from the east road there is no way up.
        let err = find(store.clone(), store, (lat as f64, lon as f64 - 0.0035), (lat as f64 + 0.0035, lon as f64 + 0.003), Travel::Car);
        assert!(err.is_err() || err.unwrap().metres > 2000.0, "a route climbed onto the bridge from the road beneath");
    }

    /// A long trip climbs the pyramid: from a street onto the highway, along it at a coarse depth, down to the street at the far end. Found quickly, the right length, the streets named at both ends.
    #[test]
    fn a_long_trip_climbs_the_pyramid() {
        let (lat, lon) = (47.60f32, -122.33f32);
        // 30 km of highway east, a side street at each end meeting it.
        let (w, e) = (lon - 0.2, lon + 0.2);
        let highway = Road { class: RoadClass::Primary, pts: (0..=40).map(|k| (lat, w + (e - w) * k as f32 / 40.0)).collect(), weight: 0.5, uses: use_bits::MOTOR | use_bits::FOOT | use_bits::BIKE, name: 4, travel: Tags { way: 4, speed_kmh: 90, oneway: 0, layer: 0, ..Default::default() } };
        let west = road(1, vec![(lat - 0.003, w + 0.01), (lat, w + 0.01)], 0, 0);
        let east = road(2, vec![(lat, e - 0.01), (lat + 0.003, e - 0.01)], 0, 0);
        let store = baked(&[highway, west, east], &[(lat, w + 0.01), (lat, e - 0.01)]);
        let t = std::time::Instant::now();
        let r = find(store.clone(), store, (lat as f64 - 0.0025, (w + 0.01) as f64), (lat as f64 + 0.0025, (e - 0.01) as f64), Travel::Car).unwrap();
        let took = t.elapsed();
        let named: Vec<&str> = r.steps.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(named, ["East Road", "The Highway", "North Road"], "{:?}", r.steps);
        // The middle is measured on the coarse grid, a staircase along the highway: within a tenth.
        assert!((r.metres - 29_000.0).abs() < 2900.0, "{} m", r.metres);
        // The cells decode slowly unoptimised (93 of them): the limit is the optimised build's.
        let limit = if cfg!(debug_assertions) { 8000 } else { 1500 };
        assert!(took.as_millis() < limit, "{took:?}");
    }

    /// A long trip's line follows the road at every depth it was found at (Nick 2026-10-26, the Snow Lake walk: the line cut the switchbacks where the search climbed or came down mid-corridor): a zigzag highway's every corner lies on the route drawn, and the length is the road's.
    #[test]
    fn a_long_trip_is_drawn_along_the_road() {
        let (lat, lon) = (47.60f32, -122.33f32);
        // 30 km east, swinging 400 m north and south every 1.5 km.
        let (w, e) = (lon - 0.2, lon + 0.2);
        let pts: Vec<(f32, f32)> = (0..=20).map(|k| (lat + if k % 2 == 0 { -0.0036 } else { 0.0036 }, w + (e - w) * k as f32 / 20.0)).collect();
        let road_m: f64 = pts.windows(2).map(|p| crow((p[0].0 as f64, p[0].1 as f64), (p[1].0 as f64, p[1].1 as f64))).sum();
        let highway = Road { class: RoadClass::Primary, pts: pts.clone(), weight: 0.5, uses: use_bits::MOTOR | use_bits::FOOT | use_bits::BIKE, name: 4, travel: Tags { way: 4, speed_kmh: 90, oneway: 0, layer: 0, ..Default::default() } };
        let (p0, pn) = (pts[0], pts[20]);
        // A side street south off its west end, one north off its east end.
        let west = road(1, vec![(p0.0 - 0.003, p0.1), p0], 0, 0);
        let east = road(2, vec![pn, (pn.0 + 0.003, pn.1)], 0, 0);
        let store = baked(&[highway, west, east], &[p0, pn]);
        let r = find(store.clone(), store, (p0.0 as f64 - 0.0025, p0.1 as f64), (pn.0 as f64 + 0.0025, pn.1 as f64), Travel::Car).unwrap();
        // Metres from a point to the nearest piece of the route drawn.
        let off = |q: (f64, f64)| -> f64 {
            let k = q.0.to_radians().cos();
            r.pts
                .windows(2)
                .map(|s| {
                    let (ax, ay) = ((s[0].1 - q.1) * k, s[0].0 - q.0);
                    let (bx, by) = ((s[1].1 - q.1) * k, s[1].0 - q.0);
                    let (dx, dy) = (bx - ax, by - ay);
                    let t = if dx * dx + dy * dy > 0.0 { (-(ax * dx + ay * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0) } else { 0.0 };
                    ((ax + t * dx).powi(2) + (ay + t * dy).powi(2)).sqrt() * 111_320.0
                })
                .fold(f64::MAX, f64::min)
        };
        let offs: Vec<f64> = pts[1..20].iter().map(|p| off((p.0 as f64, p.1 as f64))).collect();
        // Drawn through the finest cells (all of them are here): a corner is within a few texels of the line, not the coarse depth's forty metres.
        for (p, d) in pts[1..20].iter().zip(&offs) {
            assert!(*d < 12.0, "the corner at {p:?} is {d:.0} m off the line");
        }
        assert_eq!(r.profile.len(), PROFILE_SAMPLES);
        let side_m = 2.0 * 0.003 * 111_320.0;
        assert!((r.metres - road_m - side_m).abs() < 0.02 * road_m, "{} m drawn, {} m of road", r.metres, road_m + side_m);
    }

    #[test]
    fn a_car_keeps_to_a_one_way() {
        let (lat, lon) = (47.60f32, -122.33f32);
        let one = road(1, vec![(lat, lon - 0.003), (lat, lon + 0.003)], 1, 0);
        let store = baked(&[one], &[]);
        assert!(find(store.clone(), store.clone(), (lat as f64, lon as f64 - 0.002), (lat as f64, lon as f64 + 0.002), Travel::Car).is_ok());
        let wrong = find(store.clone(), store.clone(), (lat as f64, lon as f64 + 0.002), (lat as f64, lon as f64 - 0.002), Travel::Car);
        assert!(wrong.is_err(), "drove the wrong way down a one-way: {:?}", wrong.map(|r| (r.metres, r.pts.len(), r.steps)));
        assert!(find(store.clone(), store, (lat as f64, lon as f64 + 0.002), (lat as f64, lon as f64 - 0.002), Travel::Foot).is_ok(), "walking ignores one-ways");
    }
}
