//! OSM ingest boundary. This is the one crate where protobuf exists: OSM's distribution format (.osm.pbf) is read here and nothing protobuf-shaped leaves. Every coordinate is round-tripped through [`mahere_coord::Coord`] on the way out, so downstream consumers see codec-quantized positions — the same positions cells will carry.
//!
//! Three parallel passes over the extract: relations (multipolygon areas), ways (lines, closed-way areas, relation members), then only the referenced nodes. Waterways get a weight from their upstream network length — the catchment proxy OSM can give — so a map can draw the Waikato and a headwater trickle differently without a styling table.

pub mod hydro;
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
    /// Boundaries, stamped as lines from the relation rings: national parks, wilderness, national forests, other protected land, state and county lines.
    NationalPark,
    Wilderness,
    NationalForest,
    Protected,
    Admin,
}

pub const CLASS_COUNT: usize = 17;

/// Line `use` bits: what a way is for, from its tags. A theme draws a path differently when bikes or horses are allowed, hides motor roads on a ski map, and so on.
pub mod use_bits {
    pub const FOOT: u8 = 1 << 0;
    pub const BIKE: u8 = 1 << 1;
    pub const HORSE: u8 = 1 << 2;
    pub const SKI: u8 = 1 << 3;
    pub const MOTOR: u8 = 1 << 4;
    pub const UNPAVED: u8 = 1 << 5;
    pub const FREEWAY: u8 = 1 << 6;
    /// Seasonal, gated, private, or otherwise not generally open.
    pub const RESTRICTED: u8 = 1 << 7;
}

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
            NationalPark | Wilderness | NationalForest => 6.0,
            Protected => 5.0,
            Admin => 8.0,
        }
    }

    /// A boundary relation's class from its tags, if it is one worth a line.
    fn from_boundary<'a>(tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<RoadClass> {
        let (mut boundary, mut protect, mut operator, mut admin, mut reserve) = (None, None, String::new(), None, false);
        for (k, v) in tags {
            match k {
                "boundary" => boundary = Some(v.to_string()),
                "protect_class" => protect = Some(v.to_string()),
                "operator" => operator = v.to_lowercase(),
                "admin_level" => admin = v.parse::<u8>().ok(),
                "leisure" if v == "nature_reserve" => reserve = true,
                _ => {}
            }
        }
        match boundary.as_deref() {
            Some("national_park") => Some(RoadClass::NationalPark),
            Some("protected_area") => Some(match protect.as_deref() {
                Some("1") | Some("1a") | Some("1b") => RoadClass::Wilderness,
                Some("2") => RoadClass::NationalPark,
                Some("6") => RoadClass::NationalForest,
                _ if operator.contains("forest service") => RoadClass::NationalForest,
                _ => RoadClass::Protected,
            }),
            // Countries, states and counties.
            Some("administrative") => matches!(admin, Some(2) | Some(4) | Some(6)).then_some(RoadClass::Admin),
            _ => reserve.then_some(RoadClass::Protected),
        }
    }

    /// How a boundary weighs against the others of its kind: the line's magnitude.
    fn boundary_weight(self) -> f32 {
        match self {
            RoadClass::Admin => 1.0,
            RoadClass::NationalPark => 0.9,
            RoadClass::Wilderness => 0.8,
            RoadClass::NationalForest => 0.7,
            _ => 0.5,
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

/// One drawable way. Points are (lat, lon) degrees, already quantized by a round trip through the mahere coordinate codec. `weight` is 0..1: for waterways, log-scaled upstream network length (a catchment proxy) set after the network pass; for other classes a log-scaled magnitude from tags (power: voltage; roads: lanes until traffic counts exist; rail: usage). `uses` are the [`use_bits`].
#[derive(Clone)]
pub struct Road {
    pub class: RoadClass,
    pub pts: Vec<(f32, f32)>,
    pub weight: f32,
    pub uses: u8,
}

/// What a way is for and how big it is, from its tags, for the non-waterway classes (waterways get their magnitude from the network pass).
fn uses_and_magnitude<'a>(class: RoadClass, tags: impl Iterator<Item = (&'a str, &'a str)>) -> (u8, f32) {
    use RoadClass::*;
    use use_bits::*;
    let mut uses = match class {
        Motorway => MOTOR | FREEWAY,
        Trunk | Primary | Secondary | Tertiary | Residential | Service => MOTOR | BIKE | FOOT,
        Track => MOTOR | BIKE | FOOT | HORSE | UNPAVED,
        Path => FOOT | BIKE,
        Rail | Power | Waterway | NationalPark | Wilderness | NationalForest | Protected | Admin => 0,
    };
    let mut mag: Option<f32> = None;
    let mut lanes: Option<f32> = None;
    let no = |v: &str| matches!(v, "no" | "private" | "discouraged");
    let yes = |v: &str| matches!(v, "yes" | "designated" | "permissive" | "official");
    for (k, v) in tags {
        match k {
            "foot" if no(v) => uses &= !FOOT,
            "foot" if yes(v) => uses |= FOOT,
            "bicycle" if no(v) => uses &= !BIKE,
            "bicycle" if yes(v) => uses |= BIKE,
            "horse" if yes(v) => uses |= HORSE,
            "horse" if no(v) => uses &= !HORSE,
            "ski" if yes(v) => uses |= SKI,
            "piste:type" => uses |= SKI,
            "motor_vehicle" | "motorcar" if no(v) => uses &= !MOTOR,
            "highway" if v == "bridleway" => uses |= HORSE,
            "highway" if v == "cycleway" => uses |= BIKE,
            "highway" if v == "footway" || v == "steps" => uses &= !BIKE,
            "surface" => {
                if matches!(v, "gravel" | "dirt" | "ground" | "unpaved" | "compacted" | "fine_gravel" | "sand" | "grass" | "earth" | "mud" | "rock" | "pebblestone" | "wood") {
                    uses |= UNPAVED;
                } else if matches!(v, "asphalt" | "paved" | "concrete" | "paving_stones") {
                    uses &= !UNPAVED;
                }
            }
            "tracktype" if v != "grade1" => uses |= UNPAVED,
            "access" if no(v) => uses |= RESTRICTED,
            "seasonal" if v != "no" => uses |= RESTRICTED,
            "barrier" if v == "gate" => uses |= RESTRICTED,
            "lanes" => lanes = v.split(';').next().and_then(|x| x.parse().ok()),
            "voltage" if class == Power => {
                // Highest circuit, log over 400 V .. 765 kV.
                if let Some(volts) = v.split(';').filter_map(|x| x.trim().parse::<f32>().ok()).fold(None, |m: Option<f32>, x| Some(m.map_or(x, |m| m.max(x)))) {
                    mag = Some(((volts.max(400.0).log10() - 2.6) / (5.9 - 2.6)).clamp(0.0, 1.0));
                }
            }
            "usage" if class == Rail => {
                mag = Some(match v {
                    "main" => 1.0,
                    "branch" => 0.6,
                    "industrial" | "military" => 0.4,
                    _ => 0.3,
                });
            }
            _ => {}
        }
    }
    let mag = mag.unwrap_or_else(|| match class {
        Motorway => 1.0,
        Trunk => 0.85,
        Primary => 0.7,
        Secondary => 0.6,
        Tertiary => 0.5,
        Residential => 0.35,
        Service => 0.25,
        Track => 0.2,
        Path => 0.15,
        Rail => 0.5,
        Power => 0.3,
        NationalPark | Wilderness | NationalForest | Protected | Admin => class.boundary_weight(),
        Waterway => 0.0,
    });
    // Lanes refine a road's magnitude until traffic counts exist: log over 1..8 lanes, blended with the class default.
    let mag = match lanes {
        Some(l) if class != Waterway && class != Power && class != Rail => (mag * 0.6 + (l.clamp(1.0, 8.0).log2() / 3.0) * 0.4).clamp(0.0, 1.0),
        _ => mag,
    };
    (uses, mag)
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

    /// The texel class id (0 = empty): `RoadClass + 1`. Size and kind travel in the line layer's `mag` and `use` planes, not the id.
    pub fn class_id(&self) -> u8 {
        self.class as u8 + 1
    }

    /// Log-scaled magnitude as a texel byte (1..=255 so a stamped texel is never "no data").
    pub fn mag_byte(&self) -> u8 {
        1 + (self.weight.clamp(0.0, 1.0) * 254.0) as u8
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
    Line(RoadClass, Vec<i64>, u8, f32),
    Area(AreaClass, Vec<i64>),
    /// A multipolygon member, kept by id for ring assembly.
    Member(i64, Vec<i64>),
}

/// Load every drawable line and area from a .osm.pbf extract.
pub fn load_features(path: &str) -> Result<Features, osmpbf::Error> {
    // Pass 1: relations — multipolygons with a drawable area class, and boundaries, which become rings of line.
    let rels: Vec<(Option<AreaClass>, Option<RoadClass>, Vec<i64>)> = ElementReader::from_path(path)?.par_map_reduce(
        |element| match element {
            Element::Relation(r) => {
                let is_mp = r.tags().any(|(k, v)| k == "type" && v == "multipolygon");
                let area = if is_mp { AreaClass::from_tags(r.tags()) } else { None };
                let bound = RoadClass::from_boundary(r.tags());
                if area.is_none() && bound.is_none() {
                    return Vec::new();
                }
                let members: Vec<i64> = r
                    .members()
                    .filter(|m| m.member_type == osmpbf::RelMemberType::Way)
                    .map(|m| m.member_id)
                    .collect();
                vec![(area, bound, members)]
            }
            _ => Vec::new(),
        },
        Vec::new,
        |mut a, mut b| {
            a.append(&mut b);
            a
        },
    )?;
    let relations: Vec<(AreaClass, Vec<i64>)> = rels.iter().filter_map(|(a, _, m)| a.map(|a| (a, m.clone()))).collect();
    let bounds: Vec<(RoadClass, Vec<i64>)> = rels.iter().filter_map(|(_, b, m)| b.map(|b| (b, m.clone()))).collect();
    let mut member_ids: Vec<i64> = rels.iter().flat_map(|(_, _, m)| m.iter().copied()).collect();
    member_ids.sort_unstable();
    member_ids.dedup();

    // Pass 2: ways — lines, closed-way areas, relation members.
    let ways: Vec<WayRec> = ElementReader::from_path(path)?.par_map_reduce(
        |element| match element {
            Element::Way(w) => {
                let mut out = Vec::new();
                let refs: Vec<i64> = w.refs().collect();
                let is_member = member_ids.binary_search(&w.id()).is_ok();
                // A state's or county's seaward run (OSM tags it maritime) and any run along the coast draw nothing: the coast is the terrain's and a line through the sea around an island outlines it for no reader. The territorial limit itself (boundary=maritime, border_type=territorial, twelve nautical miles out) is a legal boundary and stays (Nick 2026-10-09).
                let seaward = w.tags().any(|(k, v)| (k == "maritime" && v == "yes") || (k == "natural" && v == "coastline"));
                if is_member && !seaward {
                    out.push(WayRec::Member(w.id(), refs.clone()));
                }
                // Boundaries from the ways themselves, since a region extract carries no boundary relations: a country's, state's or county's border on land (administrative, levels 2, 4 and 6, not maritime), and the territorial sea's limit twelve nautical miles out, which is the country's legal edge and stays though it is maritime; the contiguous zone and the EEZ beyond it are zones, not borders, and draw nothing (Nick 2026-10-09).
                let (mut admin_level, mut administrative, mut maritime, mut territorial) = (None, false, false, false);
                for (k, v) in w.tags() {
                    match k {
                        "boundary" if v == "administrative" => administrative = true,
                        "admin_level" => admin_level = v.parse::<u8>().ok(),
                        "maritime" if v == "yes" => maritime = true,
                        "border_type" if v == "territorial" => territorial = true,
                        _ => {}
                    }
                }
                let border = territorial || (administrative && matches!(admin_level, Some(2) | Some(4) | Some(6)) && !maritime);
                let line = w.tags().find_map(|(k, v)| match k {
                    "highway" => RoadClass::from_tag(v),
                    "waterway" => RoadClass::from_waterway(v),
                    "railway" => RoadClass::from_railway(v),
                    "power" => RoadClass::from_power(v),
                    _ => None,
                }).or(border.then_some(RoadClass::Admin));
                // Sidewalks, crossings and traffic islands are footways that only shadow a street: a trail map leaves them out, or every downtown street grows a trail-coloured fringe.
                let street_furniture = w.tags().any(|(k, v)| k == "footway" && matches!(v, "sidewalk" | "crossing" | "traffic_island" | "access_aisle"));
                if let Some(class) = line.filter(|_| !street_furniture) {
                    let (uses, mag) = uses_and_magnitude(class, w.tags());
                    out.push(WayRec::Line(class, refs.clone(), uses, mag));
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
            WayRec::Line(_, refs, _, _) | WayRec::Area(_, refs) | WayRec::Member(_, refs) => refs.iter().copied(),
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
    let mut line_attrs: Vec<(u8, f32)> = Vec::new();
    let mut areas: Vec<Area> = Vec::new();
    let mut members: HashMap<i64, Vec<i64>> = HashMap::new();
    for rec in ways {
        match rec {
            WayRec::Line(class, refs, uses, mag) => {
                lines.push((class, refs));
                line_attrs.push((uses, mag));
            }
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
            let (uses, mag) = line_attrs[i];
            let weight = if class == RoadClass::Waterway { weights[i] } else { mag };
            roads.push(Road { class, pts, weight, uses });
        }
    }

    // Boundaries: each closed ring of a relation is a line of its class.
    for (class, member_ids) in bounds {
        let parts: Vec<&Vec<i64>> = member_ids.iter().filter_map(|id| members.get(id)).collect();
        for ring in assemble_rings(&parts) {
            let pts = locate_all(&ring);
            if pts.len() >= 2 {
                roads.push(Road { class, pts, weight: class.boundary_weight(), uses: 0 });
            }
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
