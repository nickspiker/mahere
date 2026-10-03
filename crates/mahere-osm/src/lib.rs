//! OSM ingest boundary. This is the one crate where protobuf exists: OSM's
//! distribution format (.osm.pbf) is read here and nothing protobuf-shaped
//! leaves. Every coordinate is round-tripped through [`mahere_coord::Coord`]
//! on the way out, so downstream consumers see codec-quantized positions —
//! the same positions tiles will carry.

use mahere_coord::Coord;
use osmpbf::{Element, ElementReader};

/// Road / trail classification, ordered major → minor. Ordering is the
/// default draw priority (minor classes draw first, major on top).
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
}

pub const CLASS_COUNT: usize = 9;

impl RoadClass {
    /// Map an OSM `highway=` tag value. `None` = a highway type mahere
    /// doesn't draw (construction, proposed, bus_stop, ...).
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
}

/// One drawable way. Points are (lat, lon) degrees, already quantized by a
/// round trip through the mahere coordinate codec.
pub struct Road {
    pub class: RoadClass,
    pub pts: Vec<(f32, f32)>,
}

/// Load every drawable highway from a .osm.pbf extract.
///
/// Two parallel passes: ways first (classes + node refs), then only the
/// referenced nodes. Node lookup is sorted-slice binary search rather than a
/// hash map — the id sets run tens of millions for a state extract.
pub fn load_roads(path: &str) -> Result<Vec<Road>, osmpbf::Error> {
    // Pass 1: highway ways.
    let ways: Vec<(RoadClass, Vec<i64>)> = ElementReader::from_path(path)?.par_map_reduce(
        |element| match element {
            Element::Way(w) => {
                let class = w
                    .tags()
                    .find(|(k, _)| *k == "highway")
                    .and_then(|(_, v)| RoadClass::from_tag(v));
                match class {
                    Some(class) => vec![(class, w.refs().collect())],
                    None => Vec::new(),
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

    let mut needed: Vec<i64> = ways.iter().flat_map(|(_, refs)| refs.iter().copied()).collect();
    needed.sort_unstable();
    needed.dedup();

    // Pass 2: just the referenced node locations.
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

    // Assemble, round-tripping every point through the codec.
    let mut roads = Vec::with_capacity(ways.len());
    for (class, refs) in ways {
        let mut pts = Vec::with_capacity(refs.len());
        for id in refs {
            if let Ok(i) = located.binary_search_by_key(&id, |(id, _, _)| *id) {
                let (_, lat, lon) = located[i];
                let (lat, lon) = Coord::from_lat_lon(lat, lon).to_lat_lon();
                pts.push((lat as f32, lon as f32));
            }
        }
        if pts.len() >= 2 {
            roads.push(Road { class, pts });
        }
    }
    Ok(roads)
}
