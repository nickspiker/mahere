//! Routing in cells (Nick 2026-10-13: "flood fill our way to a fairly reasonable navigation platform", overpasses as "a special dual/triple case list"). The ways' centrelines are stamped a trixel wide at the base depth, apart from the drawn lines (whose widths overlap at every crossing). A cell holds a table of the ways in it (by OSM id, so a route knows a way across cells), the plane naming one way per trixel, the stack of further ways sharing a trixel without meeting there (a bridge over a road, a road over a tunnel), the trixels where ways truly meet (a node they share; a route changes ways only there), and each one-way trixel's direction of travel.
use std::collections::HashMap;

use mahere_coord::Coord;
use mahere_osm::Road;

use super::{disk_to_mem, mem_to_disk, raw_u16, texel_of, tri_at, tri_idx, CellKey, Named, TEX, TRI};

/// One way as a route sees it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Way {
    pub osm: i64,
    /// The line class id (1 = motorway .. 9 = path).
    pub class: u8,
    /// Who may use it ([`mahere_osm::use_bits`]).
    pub uses: u8,
    /// The posted limit, km/h; 0 untagged.
    pub speed_kmh: u8,
    pub oneway: i8,
    pub layer: i8,
    /// What turn-by-turn calls it.
    pub name: String,
}

/// A cell's routing: its ways and where each runs. Way numbers count from 1 (0 = no way).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouteCell {
    /// A way per trixel, memory order; empty when no way runs through the cell.
    pub way: Vec<u16>,
    pub ways: Vec<Way>,
    /// Further ways on a trixel the plane already gives another: (trixel, way).
    pub stack: Vec<(u32, u16)>,
    /// Trixels where ways meet.
    pub joins: Vec<u32>,
    /// A one-way trixel's direction of travel, per way: (trixel, way, angle in 256ths of a turn, in the cell's grid).
    pub dir: Vec<(u32, u16, u8)>,
}

impl RouteCell {
    /// The ways on a trixel: the plane's and the stack's.
    pub fn ways_at(&self, i: usize) -> impl Iterator<Item = u16> + '_ {
        let first = self.way.get(i).copied().filter(|&w| w != 0);
        let i = i as u32;
        let lo = self.stack.partition_point(|&(t, _)| t < i);
        first.into_iter().chain(self.stack[lo..].iter().take_while(move |&&(t, _)| t == i).map(|&(_, w)| w))
    }

    pub fn is_join(&self, i: usize) -> bool {
        self.joins.binary_search(&(i as u32)).is_ok()
    }

    /// A one-way trixel's direction of travel for a way, radians in the cell's grid (u along x, v along y).
    pub fn direction(&self, i: usize, way: u16) -> Option<f32> {
        let i = i as u32;
        let lo = self.dir.partition_point(|&(t, _, _)| t < i);
        self.dir[lo..].iter().take_while(|&&(t, _, _)| t == i).find(|&&(_, w, _)| w == way).map(|&(_, _, a)| a as f32 / 256.0 * std::f32::consts::TAU)
    }

    fn add(&mut self, i: usize, w: u16) {
        if self.way.is_empty() {
            self.way = vec![0; TRI];
        }
        if self.way[i] == 0 {
            self.way[i] = w;
        } else if self.way[i] != w && !self.stack.contains(&(i as u32, w)) {
            self.stack.push((i as u32, w));
        }
    }

    /// The cell's `route` section.
    pub fn section(&self) -> Option<vsf::VsfSection> {
        use vsf::{types::Tensor, VsfType};
        if self.ways.is_empty() {
            return None;
        }
        let n = self.ways.len();
        let mut s = vsf::VsfSection::new("route");
        let narrow = n < 256;
        let disk = mem_to_disk(&self.way);
        s.add_field("way", if narrow { VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], disk.iter().map(|&w| w as u8).collect())) } else { VsfType::t_u4(Tensor::new(vec![2, TEX * TEX], disk)) });
        s.add_field("osm", VsfType::t_u6(Tensor::new(vec![n], self.ways.iter().map(|w| w.osm as u64).collect())));
        s.add_field("class", VsfType::t_u3(Tensor::new(vec![n], self.ways.iter().map(|w| w.class).collect())));
        s.add_field("uses", VsfType::t_u3(Tensor::new(vec![n], self.ways.iter().map(|w| w.uses).collect())));
        s.add_field("speed", VsfType::t_u3(Tensor::new(vec![n], self.ways.iter().map(|w| w.speed_kmh).collect())));
        s.add_field("oneway", VsfType::t_u3(Tensor::new(vec![n], self.ways.iter().map(|w| (w.oneway + 1) as u8).collect())));
        s.add_field("layer", VsfType::t_u3(Tensor::new(vec![n], self.ways.iter().map(|w| (w.layer as i16 + 128) as u8).collect())));
        s.add_field("name", VsfType::x(self.ways.iter().map(|w| w.name.replace(['\n', '\r'], " ")).collect::<Vec<_>>().join("\n")));
        // The sparse lists, each in two or three parallel tensors; an empty list is left out.
        if !self.stack.is_empty() {
            s.add_field("stack", VsfType::t_u5(Tensor::new(vec![self.stack.len()], self.stack.iter().map(|s| s.0).collect())));
            s.add_field("stackway", VsfType::t_u4(Tensor::new(vec![self.stack.len()], self.stack.iter().map(|s| s.1).collect())));
        }
        if !self.joins.is_empty() {
            s.add_field("join", VsfType::t_u5(Tensor::new(vec![self.joins.len()], self.joins.clone())));
        }
        if !self.dir.is_empty() {
            s.add_field("dir", VsfType::t_u5(Tensor::new(vec![self.dir.len()], self.dir.iter().map(|d| d.0).collect())));
            s.add_field("dirway", VsfType::t_u4(Tensor::new(vec![self.dir.len()], self.dir.iter().map(|d| d.1).collect())));
            s.add_field("dirangle", VsfType::t_u3(Tensor::new(vec![self.dir.len()], self.dir.iter().map(|d| d.2).collect())));
        }
        Some(s)
    }

    /// A `route` section's fields back into a cell's routing.
    pub fn decode(fields: &HashMap<String, Vec<vsf::VsfType>>) -> Option<RouteCell> {
        use vsf::VsfType;
        let first = |name: &str| fields.get(name).and_then(|v| v.first());
        let u8s = |name: &str| -> Vec<u8> {
            match first(name) {
                Some(VsfType::t_u3(t)) => t.data.clone(),
                Some(VsfType::v_u3(t)) => t.data.clone(),
                _ => Vec::new(),
            }
        };
        let u16s = |name: &str| -> Vec<u16> { first(name).and_then(raw_u16).unwrap_or_default() };
        let u32s = |name: &str| -> Vec<u32> {
            match first(name) {
                Some(VsfType::t_u5(t)) => t.data.clone(),
                Some(VsfType::v_u5(t)) => t.data.clone(),
                _ => Vec::new(),
            }
        };
        let osm: Vec<i64> = match first("osm")? {
            VsfType::t_u6(t) => t.data.iter().map(|&v| v as i64).collect(),
            VsfType::v_u6(t) => t.data.iter().map(|&v| v as i64).collect(),
            _ => return None,
        };
        let n = osm.len();
        let (class, uses, speed, oneway, layer) = (u8s("class"), u8s("uses"), u8s("speed"), u8s("oneway"), u8s("layer"));
        let names: Vec<String> = match first("name") {
            Some(VsfType::x(s) | VsfType::a(s)) => s.split('\n').map(str::to_string).collect(),
            _ => vec![String::new(); n],
        };
        if [class.len(), uses.len(), speed.len(), oneway.len(), layer.len(), names.len()].iter().any(|&l| l != n) {
            return None;
        }
        let ways = (0..n).map(|i| Way { osm: osm[i], class: class[i], uses: uses[i], speed_kmh: speed[i], oneway: oneway[i] as i8 - 1, layer: (layer[i] as i16 - 128) as i8, name: names[i].clone() }).collect();
        let way = first("way").and_then(raw_u16).filter(|d| d.len() == TRI).map(|d| disk_to_mem(&d)).unwrap_or_default();
        let stack = u32s("stack").into_iter().zip(u16s("stackway")).collect();
        let dir = u32s("dir").into_iter().zip(u16s("dirway")).zip(u8s("dirangle")).map(|((t, w), a)| (t, w, a)).collect();
        Some(RouteCell { way, ways, stack, joins: u32s("join"), dir })
    }
}

/// Every routable way's centreline stamped a trixel wide at `depth`, with the stacks, the joins (the junctions' trixels) and the one-way directions. Only base-depth cells: routing reads the finest level.
pub fn bake_routes(roads: &[Road], names: &[Named], junctions: &[(f32, f32)], depth: u8) -> HashMap<CellKey, RouteCell> {
    let mut cells: HashMap<CellKey, RouteCell> = HashMap::new();
    // Each cell's way numbers by OSM id.
    let mut numbers: HashMap<CellKey, HashMap<i64, u16>> = HashMap::new();
    // Each cell's one-way trixels already given a direction.
    let mut directed: HashMap<CellKey, std::collections::HashSet<(u32, u16)>> = HashMap::new();
    let locate = |lat: f32, lon: f32| -> (u8, f64, f64) { texel_of(Coord::from_lat_lon(lat as f64, lon as f64), depth) };
    let at = |d: u8, gx: f64, gy: f64| -> (CellKey, usize) {
        let (tx, ty, half) = tri_at(gx, gy);
        (CellKey::from_grid(d, depth, (tx / TEX) as u64, (ty / TEX) as u64), tri_idx(tx % TEX, ty % TEX, half))
    };
    for road in roads.iter().filter(|r| r.class.routable()) {
        let way = Way {
            osm: road.travel.way,
            class: road.class_id(),
            uses: road.uses,
            speed_kmh: road.travel.speed_kmh,
            oneway: road.travel.oneway,
            layer: road.travel.layer,
            name: names.get(road.name as usize).map(|n| n.label()).unwrap_or_default(),
        };
        let mut prev: Option<(u8, f64, f64)> = None;
        for &(lat, lon) in &road.pts {
            let (d, gx, gy) = locate(lat, lon);
            if let Some((pd, px, py)) = prev.filter(|p| p.0 == d) {
                let (dx, dy) = (gx - px, gy - py);
                let len = dx.hypot(dy);
                // A quarter trixel a step, so successive trixels always touch.
                let steps = (len * 4.0).ceil().max(1.0) as usize;
                let travel = if road.travel.oneway < 0 { (-dy).atan2(-dx) } else { dy.atan2(dx) };
                let angle = (travel.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU * 256.0).round() as u32 as u8;
                for k in 0..=steps {
                    let t = k as f64 / steps as f64;
                    let (key, i) = at(pd, px + dx * t, py + dy * t);
                    let cell = cells.entry(key).or_default();
                    let nums = numbers.entry(key).or_default();
                    let w = *nums.entry(way.osm).or_insert_with(|| {
                        cell.ways.push(way.clone());
                        cell.ways.len() as u16
                    });
                    cell.add(i, w);
                    if road.travel.oneway != 0 && directed.entry(key).or_default().insert((i as u32, w)) {
                        cell.dir.push((i as u32, w, angle));
                    }
                }
            }
            prev = Some((d, gx, gy));
        }
    }
    for &(lat, lon) in junctions {
        let (d, gx, gy) = locate(lat, lon);
        let (key, i) = at(d, gx, gy);
        if let Some(cell) = cells.get_mut(&key) {
            cell.joins.push(i as u32);
        }
    }
    for cell in cells.values_mut() {
        cell.stack.sort_unstable();
        cell.stack.dedup();
        cell.joins.sort_unstable();
        cell.joins.dedup();
        cell.dir.sort_unstable();
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahere_osm::{RoadClass, Travel};

    fn road(class: RoadClass, osm: i64, pts: Vec<(f32, f32)>, oneway: i8, layer: i8) -> Road {
        Road { class, pts, weight: 0.5, uses: 0, name: 0, travel: Travel { way: osm, speed_kmh: 0, oneway, layer } }
    }

    #[test]
    fn an_overpass_stacks_and_a_junction_joins() {
        // A bridge west to east and a road south to north crossing under it, and a third road meeting the bridge at a node they share.
        let (lat, lon) = (47.60f32, -122.33f32);
        let bridge = road(RoadClass::Primary, 1, vec![(lat, lon - 0.002), (lat, lon + 0.002)], 1, 1);
        let under = road(RoadClass::Residential, 2, vec![(lat - 0.002, lon), (lat + 0.002, lon)], 0, 0);
        let side = road(RoadClass::Residential, 3, vec![(lat - 0.002, lon + 0.001), (lat, lon + 0.001)], 0, 0);
        let cells = bake_routes(&[bridge, under, side], &[Named::default()], &[(lat, lon + 0.001)], 14);
        let stacks: usize = cells.values().map(|c| c.stack.len()).sum();
        let joins: usize = cells.values().map(|c| c.joins.len()).sum();
        assert!(stacks >= 2, "the crossings stack: {stacks}");
        assert_eq!(joins, 1, "one junction, where the side road meets the bridge");
        let (key, c) = cells.iter().find(|(_, c)| !c.joins.is_empty()).unwrap();
        let j = c.joins[0] as usize;
        let at_join: Vec<i64> = c.ways_at(j).map(|w| c.ways[w as usize - 1].osm).collect();
        assert!(at_join.contains(&1) && at_join.contains(&3), "{key:?}: {at_join:?}");
        // The bridge is one-way east: its trixels carry the direction from its west end to its east end in the grid.
        let (_, wx, wy) = texel_of(Coord::from_lat_lon(lat as f64, (lon - 0.002) as f64), 14);
        let (_, ex, ey) = texel_of(Coord::from_lat_lon(lat as f64, (lon + 0.002) as f64), 14);
        let want = ((ey - wy).atan2(ex - wx).rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU * 256.0).round() as i32;
        let east = c.dir.iter().find(|d| c.ways[d.1 as usize - 1].osm == 1).unwrap();
        assert!((east.2 as i32 - want).rem_euclid(256).min((want - east.2 as i32).rem_euclid(256)) <= 1, "{} against {want}", east.2);
        // And the section round-trips.
        let s = c.section().unwrap();
        let fields: HashMap<String, Vec<vsf::VsfType>> = s.fields.into_iter().map(|f| (f.name, f.values)).collect();
        assert_eq!(RouteCell::decode(&fields).as_ref(), Some(c));
    }
}
