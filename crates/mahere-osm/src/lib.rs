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
    /// Railways (active lines; abandoned grades read as Track via OSM tags).
    Rail,
    /// Transmission and distribution lines — classic navigation cues.
    Power,
    /// Rivers, streams, canals, irrigation ditches — drawn under everything.
    Waterway,
}

pub const CLASS_COUNT: usize = 12;

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
}

/// One drawable way. Points are (lat, lon) degrees, already quantized by a
/// round trip through the mahere coordinate codec.
pub struct Road {
    pub class: RoadClass,
    pub pts: Vec<(f32, f32)>,
}

/// Load every drawable highway AND waterway line from a .osm.pbf extract.
///
/// Two parallel passes: ways first (classes + node refs), then only the
/// referenced nodes. Node lookup is sorted-slice binary search rather than a
/// hash map — the id sets run tens of millions for a state extract.
pub fn load_features(path: &str) -> Result<Vec<Road>, osmpbf::Error> {
    // Pass 1: highway ways.
    let ways: Vec<(RoadClass, Vec<i64>)> = ElementReader::from_path(path)?.par_map_reduce(
        |element| match element {
            Element::Way(w) => {
                let class = w.tags().find_map(|(k, v)| match k {
                    "highway" => RoadClass::from_tag(v),
                    "waterway" => RoadClass::from_waterway(v),
                    "railway" => RoadClass::from_railway(v),
                    "power" => RoadClass::from_power(v),
                    _ => None,
                });
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

/// Back-compat alias.
pub fn load_roads(path: &str) -> Result<Vec<Road>, osmpbf::Error> {
    load_features(path)
}

// ==================== FEATPACK (VSF) ====================
// Compact on-device feature file: one VSF section "features" with three
// tensors — per-feature class (u8), per-feature point count (u32), and the
// flat (lat, lon) f64 point stream. Machine artifact, so VSF per house rule.

use vsf::types::Tensor;
use vsf::{VsfBuilder, VsfType};

/// Clip features to bboxes (runs of in-box points, one-point margin kept so
/// strokes exit the box cleanly) and write a featpack.
pub fn write_featpack(
    path: &str,
    feats: &[Road],
    bboxes: &[(f64, f64, f64, f64)], // (lat0, lon0, lat1, lon1)
) -> Result<(), String> {
    let inside = |lat: f32, lon: f32| {
        bboxes.iter().any(|&(a0, o0, a1, o1)| {
            (lat as f64) >= a0 && (lat as f64) <= a1 && (lon as f64) >= o0 && (lon as f64) <= o1
        })
    };
    let mut classes: Vec<u8> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let mut points: Vec<f64> = Vec::new();
    for r in feats {
        let mut run: Vec<(f32, f32)> = Vec::new();
        let flush = |run: &mut Vec<(f32, f32)>,
                     classes: &mut Vec<u8>,
                     counts: &mut Vec<u32>,
                     points: &mut Vec<f64>| {
            if run.len() >= 2 {
                classes.push(r.class as u8);
                counts.push(run.len() as u32);
                for &(la, lo) in run.iter() {
                    points.push(la as f64);
                    points.push(lo as f64);
                }
            }
            run.clear();
        };
        for (i, &(la, lo)) in r.pts.iter().enumerate() {
            if inside(la, lo) {
                if run.is_empty() && i > 0 {
                    run.push(r.pts[i - 1]); // entry margin
                }
                run.push((la, lo));
            } else if !run.is_empty() {
                run.push((la, lo)); // exit margin
                flush(&mut run, &mut classes, &mut counts, &mut points);
            }
        }
        flush(&mut run, &mut classes, &mut counts, &mut points);
    }
    let n = classes.len();
    let total = points.len();
    let bytes = VsfBuilder::new()
        .add_section(
            "features",
            vec![
                ("classes".to_string(), VsfType::t_u3(Tensor::new(vec![n], classes))),
                ("counts".to_string(), VsfType::t_u5(Tensor::new(vec![n], counts))),
                ("points".to_string(), VsfType::t_f6(Tensor::new(vec![total], points))),
            ],
        )
        .build()
        .map_err(|e| format!("featpack build: {e:?}"))?;
    std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))
}

/// Read a featpack back into drawable features.
pub fn read_featpack(path: &str) -> Result<Vec<Road>, String> {
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let (header, header_end) =
        vsf::VsfHeader::decode(&data).map_err(|e| format!("{path}: header: {e}"))?;
    let section = header
        .primary_section(&data, header_end)
        .map_err(|e| format!("{path}: {e}"))?;
    let mut classes: Option<Vec<u8>> = None;
    let mut counts: Option<Vec<u32>> = None;
    let mut points: Option<Vec<f64>> = None;
    if section.name == "features" {
        for f in section.fields {
            match (f.name.as_str(), f.values.into_iter().next()) {
                ("classes", Some(VsfType::t_u3(t))) => classes = Some(t.data),
                ("counts", Some(VsfType::t_u5(t))) => counts = Some(t.data),
                ("points", Some(VsfType::t_f6(t))) => points = Some(t.data),
                _ => {}
            }
        }
    }
    let (classes, counts, points) = match (classes, counts, points) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return Err(format!("{path}: missing featpack fields")),
    };
    const ALL: [RoadClass; CLASS_COUNT] = [
        RoadClass::Motorway,
        RoadClass::Trunk,
        RoadClass::Primary,
        RoadClass::Secondary,
        RoadClass::Tertiary,
        RoadClass::Residential,
        RoadClass::Service,
        RoadClass::Track,
        RoadClass::Path,
        RoadClass::Rail,
        RoadClass::Power,
        RoadClass::Waterway,
    ];
    let mut out = Vec::with_capacity(classes.len());
    let mut cursor = 0usize;
    for (ci, &count) in classes.iter().zip(counts.iter()) {
        let n = count as usize;
        let mut pts = Vec::with_capacity(n);
        for k in 0..n {
            let la = points[(cursor + k) * 2] as f32;
            let lo = points[(cursor + k) * 2 + 1] as f32;
            pts.push((la, lo));
        }
        cursor += n;
        out.push(Road { class: ALL[(*ci as usize).min(CLASS_COUNT - 1)], pts });
    }
    Ok(out)
}
