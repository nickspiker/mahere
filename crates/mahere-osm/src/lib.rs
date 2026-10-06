//! OSM ingest boundary. This is the one crate where protobuf exists: OSM's distribution format (.osm.pbf) is read here and nothing protobuf-shaped leaves. Every coordinate is round-tripped through [`mahere_coord::Coord`] on the way out, so downstream consumers see codec-quantized positions — the same positions cells will carry.
//!
//! Three parallel passes over the extract: relations (multipolygon areas), ways (lines, closed-way areas, relation members), then only the referenced nodes. Waterways get a weight from their upstream network length — the catchment proxy OSM can give — so a map can draw the Waikato and a headwater trickle differently without a styling table.

use std::collections::HashMap;

use mahere_coord::Coord;
use osmpbf::{Element, ElementReader};

/// Road / trail classification, ordered major → minor. Ordering is the default draw priority (minor classes draw first, major on top).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum RoadClass {
    Motorway,
    Trunk,
    Primary,
    Secondary,
    Tertiary,
    Residential,
    Service,
    Track,
    Path,
    /// Railways (active lines; abandoned grades read as Track via OSM tags).
    Rail,
    /// Transmission and distribution lines — classic navigation cues.
    Power,
    /// Rivers, streams, canals, irrigation ditches — drawn under everything.
    Waterway,
}

pub const CLASS_COUNT: usize = 12;

impl RoadClass {
    /// Map an OSM `highway=` tag value. `None` = a highway type mahere doesn't draw (construction, proposed, bus_stop, ...).
    fn from_tag(v: &str) -> Option<RoadClass> {
        use RoadClass::*;
        Some(match v {
            "motorway" | "motorway_link" => Motorway,
            "trunk" | "trunk_link" => Trunk,
            "primary" | "primary_link" => Primary,
            "secondary" | "secondary_link" => Secondary,
            "tertiary" | "tertiary_link" => Tertiary,
            "residential" | "unclassified" | "living_street" => Residential,
            "service" => Service,
            "track" => Track,
            "path" | "footway" | "cycleway" | "bridleway" | "steps" => Path,
            _ => return None,
        })
    }

    fn from_waterway(v: &str) -> Option<RoadClass> {
        match v {
            "river" | "stream" | "canal" | "ditch" | "drain" => Some(RoadClass::Waterway),
            _ => None,
        }
    }

    fn from_railway(v: &str) -> Option<RoadClass> {
        match v {
            "rail" | "light_rail" | "narrow_gauge" | "preserved" => Some(RoadClass::Rail),
            _ => None,
        }
    }

    fn from_power(v: &str) -> Option<RoadClass> {
        match v {
            "line" | "minor_line" => Some(RoadClass::Power),
            _ => None,
        }
    }

    /// Physical width on the ground, in meters, for stamping at base depth.
    /// A motorway really is 28 m of pavement; at 1 m data it looks like one.
    pub fn width_m(self) -> f32 {
        use RoadClass::*;
        match self {
            Motorway => 28.0,
            Trunk => 20.0,
            Primary => 14.0,
            Secondary => 11.0,
            Tertiary => 9.0,
            Residential => 7.0,
            Service => 4.0,
            Track => 3.0,
            Path => 1.5,
            Rail => 4.0,
            Power => 2.0,
            Waterway => 2.0,
        }
    }
}

/// Land cover and water areas: what's on the ground. Ordered so that a higher class wins where polygons overlap (water over everything, built-up over vegetation, vegetation over the generic classes).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum AreaClass {
    Grass,
    Farmland,
    Orchard,
    Scrub,
    Forest,
    Wetland,
    Sand,
    Rock,
    Glacier,
    Quarry,
    Industrial,
    Urban,
    Water,
}

pub const AREA_CLASS_COUNT: usize = 13;

impl AreaClass {
    fn from_tags<'a>(tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<AreaClass> {
        use AreaClass::*;
        let mut best: Option<AreaClass> = None;
        let mut area_no = false;
        for (k, v) in tags {
            let c = match (k, v) {
                ("area", "no") => {
                    area_no = true;
                    None
                }
                ("natural", "water") | ("landuse", "reservoir") | ("landuse", "basin") => Some(Water),
                ("natural", "wetland") | ("natural", "mud") => Some(Wetland),
                ("natural", "wood") | ("landuse", "forest") => Some(Forest),
                ("natural", "scrub") | ("natural", "heath") => Some(Scrub),
                ("natural", "grassland")
                | ("landuse", "grass")
                | ("landuse", "meadow")
                | ("landuse", "recreation_ground")
                | ("landuse", "cemetery")
                | ("leisure", "park")
                | ("leisure", "golf_course")
                | ("leisure", "pitch") => Some(Grass),
                ("landuse", "farmland") | ("landuse", "farmyard") => Some(Farmland),
                ("landuse", "orchard") | ("landuse", "vineyard") => Some(Orchard),
                ("landuse", "residential") | ("landuse", "commercial") | ("landuse", "retail") => Some(Urban),
                ("landuse", "industrial") | ("landuse", "railway") | ("landuse", "military") => Some(Industrial),
                ("natural", "sand") | ("natural", "beach") | ("natural", "shoal") => Some(Sand),
                ("natural", "glacier") => Some(Glacier),
                ("natural", "bare_rock") | ("natural", "scree") | ("natural", "shingle") => Some(Rock),
                ("landuse", "quarry") => Some(Quarry),
                _ => None,
            };
            if let Some(c) = c {
                best = Some(best.map_or(c, |b| b.max(c)));
            }
        }
        if area_no { None } else { best }
    }
}

/// One drawable way. Points are (lat, lon) degrees, already quantized by a round trip through the mahere coordinate codec. `weight` is 0..1:
/// for waterways, log-scaled upstream network length (a catchment proxy); for everything else it's unused (width comes from the class).
pub struct Road {
    pub class: RoadClass,
    pub pts: Vec<(f32, f32)>,
    pub weight: f32,
}

impl Road {
    /// Stamp width in meters: the class width, and for waterways 1.5 m at a headwater growing to ~20 m for a major river (big rivers also carry riverbank polygons in the water layer, which give the true width).
    pub fn width_m(&self) -> f32 {
        match self.class {
            RoadClass::Waterway => 1.5 + 18.0 * self.weight * self.weight * self.weight,
            c => c.width_m(),
        }
    }

    /// Peak coverage at the centreline: the NZ render's log-brightness, on the waterway's weight; everything else is opaque.
    pub fn cov_max(&self) -> u8 {
        match self.class {
            RoadClass::Waterway => (120.0 + 135.0 * self.weight) as u8,
            _ => 255,
        }
    }
}

/// One polygon feature: rings (outer and inner alike — even-odd fill sorts them out) of codec-quantized (lat, lon) points.
pub struct Area {
    pub class: AreaClass,
    pub rings: Vec<Vec<(f32, f32)>>,
}

pub struct Features {
    pub roads: Vec<Road>,
    pub areas: Vec<Area>,
}

enum WayRec {
    Line(RoadClass, Vec<i64>),
    Area(AreaClass, Vec<i64>),
    /// A multipolygon member, kept by id for ring assembly.
    Member(i64, Vec<i64>),
}

/// Load every drawable line and area from a .osm.pbf extract.
pub fn load_features(path: &str) -> Result<Features, osmpbf::Error> {
    // Pass 1: multipolygon relations with a drawable class.
    let relations: Vec<(AreaClass, Vec<i64>)> = ElementReader::from_path(path)?.par_map_reduce(
        |element| match element {
            Element::Relation(r) => {
                let is_mp = r.tags().any(|(k, v)| k == "type" && v == "multipolygon");
                match (is_mp, AreaClass::from_tags(r.tags())) {
                    (true, Some(class)) => {
                        let members: Vec<i64> = r
                            .members()
                            .filter(|m| m.member_type == osmpbf::RelMemberType::Way)
                            .map(|m| m.member_id)
                            .collect();
                        vec![(class, members)]
                    }
                    _ => Vec::new(),
                }
            }
            _ => Vec::new(),
        },
        Vec::new,
        |mut a, mut b| {
            a.append(&mut b);
            a
        },
    )?;
    let mut member_ids: Vec<i64> = relations.iter().flat_map(|(_, m)| m.iter().copied()).collect();
    member_ids.sort_unstable();
    member_ids.dedup();

    // Pass 2: ways — lines, closed-way areas, relation members.
    let ways: Vec<WayRec> = ElementReader::from_path(path)?.par_map_reduce(
        |element| match element {
            Element::Way(w) => {
                let mut out = Vec::new();
                let refs: Vec<i64> = w.refs().collect();
                let is_member = member_ids.binary_search(&w.id()).is_ok();
                if is_member {
                    out.push(WayRec::Member(w.id(), refs.clone()));
                }
                let line = w.tags().find_map(|(k, v)| match k {
                    "highway" => RoadClass::from_tag(v),
                    "waterway" => RoadClass::from_waterway(v),
                    "railway" => RoadClass::from_railway(v),
                    "power" => RoadClass::from_power(v),
                    _ => None,
                });
                if let Some(class) = line {
                    out.push(WayRec::Line(class, refs.clone()));
                }
                let closed = refs.len() >= 4 && refs.first() == refs.last();
                if closed && !is_member {
                    if let Some(class) = AreaClass::from_tags(w.tags()) {
                        out.push(WayRec::Area(class, refs));
                    }
                }
                out
            }
            _ => Vec::new(),
        },
        Vec::new,
        |mut a, mut b| {
            a.append(&mut b);
            a
        },
    )?;

    let mut needed: Vec<i64> = ways
        .iter()
        .flat_map(|r| match r {
            WayRec::Line(_, refs) | WayRec::Area(_, refs) | WayRec::Member(_, refs) => refs.iter().copied(),
        })
        .collect();
    needed.sort_unstable();
    needed.dedup();

    // Pass 3: just the referenced node locations.
    let mut located: Vec<(i64, f64, f64)> = ElementReader::from_path(path)?.par_map_reduce(
        |element| {
            let (id, lat, lon) = match element {
                Element::Node(n) => (n.id(), n.lat(), n.lon()),
                Element::DenseNode(n) => (n.id(), n.lat(), n.lon()),
                _ => return Vec::new(),
            };
            if needed.binary_search(&id).is_ok() {
                vec![(id, lat, lon)]
            } else {
                Vec::new()
            }
        },
        Vec::new,
        |mut a, mut b| {
            a.append(&mut b);
            a
        },
    )?;
    located.sort_unstable_by_key(|(id, _, _)| *id);
    let locate = |id: i64| -> Option<(f32, f32)> {
        let i = located.binary_search_by_key(&id, |(id, _, _)| *id).ok()?;
        let (_, lat, lon) = located[i];
        let (lat, lon) = Coord::from_lat_lon(lat, lon).to_lat_lon();
        Some((lat as f32, lon as f32))
    };
    let locate_all = |refs: &[i64]| -> Vec<(f32, f32)> { refs.iter().filter_map(|&id| locate(id)).collect() };

    // Lines, with the waterway network weighted.
    let mut lines: Vec<(RoadClass, Vec<i64>)> = Vec::new();
    let mut areas: Vec<Area> = Vec::new();
    let mut members: HashMap<i64, Vec<i64>> = HashMap::new();
    for rec in ways {
        match rec {
            WayRec::Line(class, refs) => lines.push((class, refs)),
            WayRec::Area(class, refs) => {
                let ring = locate_all(&refs);
                if ring.len() >= 4 {
                    areas.push(Area { class, rings: vec![ring] });
                }
            }
            WayRec::Member(id, refs) => {
                members.insert(id, refs);
            }
        }
    }
    let weights = waterway_weights(&lines, &locate);
    let mut roads = Vec::with_capacity(lines.len());
    for (i, (class, refs)) in lines.into_iter().enumerate() {
        let pts = locate_all(&refs);
        if pts.len() >= 2 {
            roads.push(Road { class, pts, weight: weights[i] });
        }
    }

    // Multipolygons: assemble member ways into closed rings.
    for (class, member_ids) in relations {
        let parts: Vec<&Vec<i64>> = member_ids.iter().filter_map(|id| members.get(id)).collect();
        let rings: Vec<Vec<(f32, f32)>> = assemble_rings(&parts)
            .into_iter()
            .map(|ring| locate_all(&ring))
            .filter(|r| r.len() >= 4)
            .collect();
        if !rings.is_empty() {
            areas.push(Area { class, rings });
        }
    }

    Ok(Features { roads, areas })
}

/// Join way fragments end-to-end by shared node ids into closed rings.
/// Fragments that never close are dropped (a broken relation draws nothing rather than a wrong fill).
fn assemble_rings(parts: &[&Vec<i64>]) -> Vec<Vec<i64>> {
    let mut used = vec![false; parts.len()];
    let mut by_end: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, p) in parts.iter().enumerate() {
        if let (Some(&a), Some(&b)) = (p.first(), p.last()) {
            by_end.entry(a).or_default().push(i);
            by_end.entry(b).or_default().push(i);
        }
    }
    let mut rings = Vec::new();
    for start in 0..parts.len() {
        if used[start] || parts[start].len() < 2 {
            continue;
        }
        used[start] = true;
        let mut ring: Vec<i64> = parts[start].clone();
        while ring.first() != ring.last() {
            let tail = *ring.last().unwrap();
            let next = by_end
                .get(&tail)
                .and_then(|c| c.iter().copied().find(|&j| !used[j]));
            let Some(j) = next else { break };
            used[j] = true;
            let p = parts[j];
            if p.first() == Some(&tail) {
                ring.extend_from_slice(&p[1..]);
            } else {
                ring.extend(p[..p.len() - 1].iter().rev());
            }
        }
        if ring.len() >= 4 && ring.first() == ring.last() {
            rings.push(ring);
        }
    }
    rings
}

/// Waterway weight 0..1 from upstream network length: a way's length plus everything that flows into it (OSM draws waterways in flow direction, so a tributary's last node lies on its receiver). Log-scaled: 0.1 km of headwater is 0, 1000 km of river is 1 — the NZ render's catchment rule with the data OSM actually has.
fn waterway_weights(lines: &[(RoadClass, Vec<i64>)], locate: &dyn Fn(i64) -> Option<(f32, f32)>) -> Vec<f32> {
    let n = lines.len();
    let mut weights = vec![0.0f32; n];
    let water: Vec<usize> = (0..n).filter(|&i| lines[i].0 == RoadClass::Waterway).collect();
    if water.is_empty() {
        return weights;
    }
    // node -> waterways passing through it
    let mut at_node: HashMap<i64, Vec<usize>> = HashMap::new();
    for &i in &water {
        for &id in &lines[i].1 {
            at_node.entry(id).or_default().push(i);
        }
    }
    // Own length in km.
    let own: HashMap<usize, f64> = water
        .iter()
        .map(|&i| {
            let pts: Vec<(f32, f32)> = lines[i].1.iter().filter_map(|&id| locate(id)).collect();
            let km: f64 = pts.windows(2).map(|w| haversine_km(w[0], w[1])).sum();
            (i, km)
        })
        .collect();
    // Tributaries: ways whose last node lies on this way.
    let mut incoming: HashMap<usize, Vec<usize>> = HashMap::new();
    for &i in &water {
        if let Some(&last) = lines[i].1.last() {
            if let Some(hosts) = at_node.get(&last) {
                for &h in hosts {
                    if h != i {
                        incoming.entry(h).or_default().push(i);
                    }
                }
            }
        }
    }
    // Memoized accumulation with a cycle guard.
    let mut total: HashMap<usize, f64> = HashMap::new();
    fn acc(
        i: usize,
        own: &HashMap<usize, f64>,
        incoming: &HashMap<usize, Vec<usize>>,
        total: &mut HashMap<usize, f64>,
        stack: &mut Vec<usize>,
    ) -> f64 {
        if let Some(&t) = total.get(&i) {
            return t;
        }
        if stack.contains(&i) {
            return 0.0;
        }
        stack.push(i);
        let mut t = own.get(&i).copied().unwrap_or(0.0);
        if let Some(ins) = incoming.get(&i) {
            for &u in ins {
                t += acc(u, own, incoming, total, stack);
            }
        }
        stack.pop();
        total.insert(i, t);
        t
    }
    for &i in &water {
        let km = acc(i, &own, &incoming, &mut total, &mut Vec::new());
        weights[i] = ((km.max(0.01).log10() + 1.0) / 4.0).clamp(0.0, 1.0) as f32;
    }
    weights
}

fn haversine_km(a: (f32, f32), b: (f32, f32)) -> f64 {
    let (la1, lo1) = (a.0 as f64).to_radians_pair(a.1 as f64);
    let (la2, lo2) = (b.0 as f64).to_radians_pair(b.1 as f64);
    let dlat = la2 - la1;
    let dlon = lo2 - lo1;
    let h = (dlat / 2.0).sin().powi(2) + la1.cos() * la2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * 6371.0 * h.sqrt().asin()
}

trait RadPair {
    fn to_radians_pair(self, other: f64) -> (f64, f64);
}
impl RadPair for f64 {
    fn to_radians_pair(self, other: f64) -> (f64, f64) {
        (self.to_radians(), other.to_radians())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rings_assemble_from_fragments_in_any_direction() {
        // Square 1-2-3-4-1 given as three fragments, one reversed.
        let a = vec![1, 2, 3];
        let b = vec![4, 3]; // reversed: 3 -> 4
        let c = vec![4, 1];
        let rings = assemble_rings(&[&a, &b, &c]);
        assert_eq!(rings, vec![vec![1, 2, 3, 4, 1]]);
    }

    #[test]
    fn tributaries_accumulate_into_the_receiver() {
        // Two 1 km streams feeding a 1 km river: river total 3 km.
        let pts: HashMap<i64, (f32, f32)> = [
            (1, (46.0, -121.0)),
            (2, (46.009, -121.0)),
            (3, (46.0, -121.013)),
            (4, (46.009, -121.013)),
            (5, (46.018, -121.0)),
        ]
        .into_iter()
        .collect();
        let locate = |id: i64| pts.get(&id).copied();
        let lines = vec![
            (RoadClass::Waterway, vec![2, 5]), // river: 2 -> 5
            (RoadClass::Waterway, vec![1, 2]), // stream into node 2
            (RoadClass::Waterway, vec![3, 4, 2]), // stream into node 2
            (RoadClass::Path, vec![1, 5]),
        ];
        let w = waterway_weights(&lines, &locate);
        assert!(w[0] > w[1] && w[0] > w[2], "receiver must outweigh tributaries: {w:?}");
        assert_eq!(w[3], 0.0);
    }
}
