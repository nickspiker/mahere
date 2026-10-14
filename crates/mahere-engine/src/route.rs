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

/// The most states a search looks at before giving up.
const MAX_STATES: usize = 6_000_000;

/// How far from a way a start or an end may be, trixels (about 340 m at depth 14).
const SNAP_REACH: i64 = 200;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Travel {
    Car,
    Bike,
    Foot,
}

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
        let kmh = match self {
            Travel::Car => {
                if w.uses & use_bits::MOTOR == 0 {
                    return None;
                }
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
                posted * if unpaved { 0.6 } else { 1.0 } * penalty
            }
            Travel::Bike => {
                if w.uses & use_bits::BIKE == 0 {
                    return None;
                }
                if unpaved || w.class >= 8 { 12.0 } else { 18.0 }
            }
            Travel::Foot => {
                if w.uses & use_bits::FOOT == 0 {
                    return None;
                }
                4.8
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
}

/// A trixel anywhere at the route depth: its diamond, its column and row of the diamond's grid, its half.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct T {
    d: u8,
    x: u32,
    y: u32,
    h: u8,
}

const SIDE: u32 = (1u32 << ROUTE_DEPTH) * TEX as u32;

impl T {
    fn of(lat: f64, lon: f64) -> T {
        let c = Coord::from_lat_lon(lat, lon);
        let (iu, iv) = c.uv();
        let s = (1u64 << (30 - ROUTE_DEPTH as u32 - mahere_tiles::TEX_BITS as u32)) as f64;
        let (x, y, h) = tri_at(iu as f64 / s, iv as f64 / s);
        T { d: c.diamond(), x: x as u32, y: y as u32, h: h as u8 }
    }

    fn cell(self) -> (CellKey, usize) {
        let t = TEX as u32;
        (CellKey::from_grid(self.d, ROUTE_DEPTH, (self.x / t) as u64, (self.y / t) as u64), tri_idx((self.x % t) as usize, (self.y % t) as usize, self.h as usize))
    }

    /// The centroid, in the diamond's grid.
    fn centre(self) -> (f64, f64) {
        let off = if self.h == 0 { 1.0 / 3.0 } else { 2.0 / 3.0 };
        (self.x as f64 + off, self.y as f64 + off)
    }

    fn lat_lon(self) -> (f64, f64) {
        let (gx, gy) = self.centre();
        uv_to_lat_lon(self.d, gx / SIDE as f64, gy / SIDE as f64)
    }

    /// The trixels touching this one: both halves of the squares around it.
    fn around(self) -> impl Iterator<Item = T> {
        let me = self;
        (-1i64..=1).flat_map(move |dy| (-1i64..=1).flat_map(move |dx| (0..2u8).map(move |h| (dx, dy, h)))).filter_map(move |(dx, dy, h)| {
            let (x, y) = (me.x as i64 + dx, me.y as i64 + dy);
            let t = T { d: me.d, x: x as u32, y: y as u32, h };
            (x >= 0 && y >= 0 && x < SIDE as i64 && y < SIDE as i64 && t != me).then_some(t)
        })
    }
}

/// Metres between two points of a diamond's grid (its two axes 60° apart).
fn grid_m(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (du, dv) = (b.0 - a.0, b.1 - a.1);
    (du * du + dv * dv + du * dv).max(0.0).sqrt() * mahere_tiles::texel_m(ROUTE_DEPTH)
}

/// Cells' routing as the search reads it, each read once from the store.
struct Cells {
    store: Arc<dyn CellStore>,
    got: HashMap<CellKey, Option<Arc<RouteCell>>>,
}

impl Cells {
    fn get(&mut self, key: CellKey) -> Option<Arc<RouteCell>> {
        if let Some(c) = self.got.get(&key) {
            return c.clone();
        }
        let c = match self.store.get(key) {
            Fetch::Bytes(b) => mahere_tiles::decode_route(&b).map(Arc::new),
            _ => None,
        };
        self.got.insert(key, c.clone());
        c
    }

    /// The ways on a trixel, by OSM id, with the cell's way record.
    fn ways(&mut self, t: T) -> Vec<(i64, mahere_tiles::route::Way, u16, Arc<RouteCell>)> {
        let (key, i) = t.cell();
        let Some(c) = self.get(key) else { return Vec::new() };
        c.ways_at(i).map(|w| (c.ways[w as usize - 1].osm, c.ways[w as usize - 1].clone(), w, c.clone())).collect()
    }
}

/// The ways a point could start or end on: the nearest trixels carrying a way this travel may use, found in rings out to [`SNAP_REACH`], each with the metres off the way.
fn snap(cells: &mut Cells, at: T, travel: Travel) -> Vec<(T, i64, f64)> {
    let mut found = Vec::new();
    for r in 0..=SNAP_REACH {
        for dy in -r..=r {
            for dx in -r..=r {
                if dx.abs() != r && dy.abs() != r {
                    continue;
                }
                let (x, y) = (at.x as i64 + dx, at.y as i64 + dy);
                if x < 0 || y < 0 || x >= SIDE as i64 || y >= SIDE as i64 {
                    continue;
                }
                for h in 0..2u8 {
                    let t = T { d: at.d, x: x as u32, y: y as u32, h };
                    for (osm, way, _, _) in cells.ways(t) {
                        if travel.speed(&way).is_some() {
                            found.push((t, osm, grid_m(at.centre(), t.centre())));
                        }
                    }
                }
            }
        }
        // Rings a few trixels past the first find, so a closer way just beyond a corner is not missed.
        if let Some(&(_, _, best)) = found.iter().min_by(|a, b| a.2.total_cmp(&b.2)) {
            if r as f64 * mahere_tiles::texel_m(ROUTE_DEPTH) > best + 10.0 {
                break;
            }
        }
    }
    found
}

/// The best route from one point to another, or why there is none.
pub fn find(store: Arc<dyn CellStore>, from: (f64, f64), to: (f64, f64), travel: Travel) -> Result<Route, String> {
    let mut cells = Cells { store: store.clone(), got: HashMap::new() };
    let (a, b) = (T::of(from.0, from.1), T::of(to.0, to.1));
    if a.d != b.d {
        return Err("the ends are in different diamonds of the map; routes cross no diamond yet".into());
    }
    let starts = snap(&mut cells, a, travel);
    let ends = snap(&mut cells, b, travel);
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
        chain.push(states[i]);
        i = states[i].3;
    }
    chain.reverse();
    let names: HashMap<i64, String> = chain.iter().filter_map(|s| cells.ways(s.0).into_iter().find(|w| w.0 == s.1).map(|w| (s.1, w.1.name))).collect();
    let mut route = Route { travel, pts: Vec::with_capacity(chain.len()), metres: 0.0, seconds, climb_m: 0.0, steps: Vec::new() };
    route.pts.push(from);
    let mut last_way = None;
    let mut last_pt: Option<(f64, f64)> = None;
    for (n, s) in chain.iter().enumerate() {
        if let Some(p) = last_pt {
            route.metres += grid_m(p, s.0.centre());
        }
        last_pt = Some(s.0.centre());
        if last_way != Some(s.1) {
            let name = names.get(&s.1).cloned().unwrap_or_default();
            // Which way it turns, from a few trixels before and after.
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
    if travel != Travel::Car {
        let (time, climb) = ground_time(&store, &chain.iter().map(|s| s.0).collect::<Vec<_>>(), travel);
        route.climb_m = climb;
        if let Some(t) = time {
            route.seconds = t + grid_m(a.centre(), chain[0].0.centre()) / off_way + grid_m(chain.last().unwrap().0.centre(), b.centre()) / off_way;
        }
    }
    Ok(route)
}

/// Left, right or straight at step `n` of a chain: the turn between the heading over the few trixels before and the few after.
fn turn_word(chain: &[(T, i64, f64, usize)], n: usize) -> &'static str {
    let back = chain[n.saturating_sub(8)].0.lat_lon();
    let here = chain[n].0.lat_lon();
    let ahead = chain[(n + 8).min(chain.len() - 1)].0.lat_lon();
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

/// The time along a path on foot or a bike from the ground's own elevation, every few metres: Tobler's hiking function walking, and on a bike its pace plus a minute for every ten metres climbed. With the metres climbed. None when the elevation is not there.
fn ground_time(store: &Arc<dyn CellStore>, path: &[T], travel: Travel) -> (Option<f64>, f64) {
    let mut dems: HashMap<CellKey, Option<Arc<Vec<f32>>>> = HashMap::new();
    let mut elev = |t: T| -> Option<f32> {
        let (key, i) = t.cell();
        let plane = dems
            .entry(key)
            .or_insert_with(|| match store.get(key) {
                Fetch::Bytes(b) => mahere_tiles::decode_cell(&b).ok().and_then(|p| p.dem).map(|d| Arc::new(d.elev)),
                _ => None,
            })
            .clone()?;
        plane.get(i).copied().filter(|e| e.is_finite())
    };
    let (mut time, mut climb) = (0.0, 0.0);
    let step = 6usize;
    let mut prev: Option<(T, f32)> = None;
    for k in (0..path.len()).step_by(step).chain(std::iter::once(path.len().saturating_sub(1))) {
        let t = path[k];
        let Some(e) = elev(t) else { return (None, climb) };
        if let Some((pt, pe)) = prev {
            let len = grid_m(pt.centre(), t.centre());
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
    use mahere_tiles::osm::{Road, RoadClass, Travel as Tags};

    struct Baked(HashMap<CellKey, Vec<u8>>);
    impl CellStore for Baked {
        fn get(&self, key: CellKey) -> Fetch {
            self.0.get(&key).map_or(Fetch::Absent, |b| Fetch::Bytes(b.clone()))
        }
    }

    fn road(osm: i64, pts: Vec<(f32, f32)>, oneway: i8, layer: i8) -> Road {
        Road { class: RoadClass::Residential, pts, weight: 0.5, uses: use_bits::MOTOR | use_bits::FOOT | use_bits::BIKE, name: osm as u32, travel: Tags { way: osm, speed_kmh: 0, oneway, layer } }
    }

    fn baked(roads: &[Road], junctions: &[(f32, f32)]) -> Arc<dyn CellStore> {
        let names: Vec<mahere_tiles::Named> = ["", "East Road", "North Road", "The Bridge"].iter().map(|n| mahere_tiles::Named { kind: 6, rank: 0, name: n.to_string(), reference: String::new() }).collect();
        let cells = mahere_tiles::route::bake_routes(roads, &names, junctions, ROUTE_DEPTH);
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
        let to_north = find(store.clone(), (lat as f64, lon as f64 - 0.0035), (lat as f64 + 0.0035, lon as f64), Travel::Car).unwrap();
        let said: Vec<String> = to_north.steps.iter().map(|s| format!("{} {}", s.turn, s.name)).collect();
        assert_eq!(said, ["Start East Road", "Turn left onto North Road"], "{:?}", to_north.steps);
        // The bridge's far end is reachable only by the bridge: from the east road there is no way up.
        let err = find(store, (lat as f64, lon as f64 - 0.0035), (lat as f64 + 0.0035, lon as f64 + 0.003), Travel::Car);
        assert!(err.is_err() || err.unwrap().metres > 2000.0, "a route climbed onto the bridge from the road beneath");
    }

    #[test]
    fn a_car_keeps_to_a_one_way() {
        let (lat, lon) = (47.60f32, -122.33f32);
        let one = road(1, vec![(lat, lon - 0.003), (lat, lon + 0.003)], 1, 0);
        let store = baked(&[one], &[]);
        assert!(find(store.clone(), (lat as f64, lon as f64 - 0.002), (lat as f64, lon as f64 + 0.002), Travel::Car).is_ok());
        assert!(find(store.clone(), (lat as f64, lon as f64 + 0.002), (lat as f64, lon as f64 - 0.002), Travel::Car).is_err(), "drove the wrong way down a one-way");
        assert!(find(store, (lat as f64, lon as f64 + 0.002), (lat as f64, lon as f64 - 0.002), Travel::Foot).is_ok(), "walking ignores one-ways");
    }
}
