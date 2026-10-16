//! Land ownership (PAD-US 4.1, USGS) and motor vehicle use (MVUM, USDA Forest Service), read from the shapefiles `scripts/landuse-prep.py` writes: who manages the ground, where bikes are banned, and what may be driven on which Forest Service road or trail (Nick 2026-10-10).
//! This is the one table of classes: the prep step keeps the sources' own words, and they are mapped here.

use shapefile::dbase::FieldValue;

/// Ownership classes, a texel's class in the ownership plane (0 = none known, which is mostly private land). `WILD` is added on top where a wilderness or wilderness study area lies over it.
pub mod own {
    pub const FOREST_SERVICE: u8 = 1;
    pub const BLM: u8 = 2;
    pub const PARK_SERVICE: u8 = 3;
    pub const FISH_WILDLIFE_SERVICE: u8 = 4;
    pub const OTHER_FEDERAL: u8 = 5;
    pub const STATE_TRUST: u8 = 6;
    pub const STATE_PARK: u8 = 7;
    pub const STATE_WILDLIFE: u8 = 8;
    pub const TRIBAL: u8 = 9;
    pub const LOCAL: u8 = 10;
    pub const CONSERVATION: u8 = 11;
    pub const OTHER: u8 = 12;
    /// Added to a class where wilderness lies over it: no bikes, no motors.
    pub const WILD: u8 = 16;
    pub const COUNT: usize = 32;

    pub const NAMES: [&str; 13] = ["", "Forest Service", "Bureau of Land Management", "National Park Service", "Fish and Wildlife Service", "Other federal", "State trust land", "State park", "State wildlife area", "Tribal", "Local government", "Conservation land", "Other public"];
}

/// MVUM classes, a texel's class in the motor vehicle plane: what may be driven there, odd yearlong and even seasonal.
pub mod mvum {
    pub const ROAD_HIGHWAY: u8 = 1;
    pub const ROAD_ALL: u8 = 3;
    pub const TRAIL_ALL: u8 = 5;
    pub const TRAIL_50: u8 = 7;
    pub const TRAIL_MOTORCYCLE: u8 = 9;
    pub const TRAIL_OTHER: u8 = 11;
    pub const SPECIAL: u8 = 13;
    pub const COUNT: usize = 16;

    pub const NAMES: [&str; 15] = ["", "Road, highway-legal vehicles", "Road, highway-legal vehicles, seasonal", "Road, all vehicles", "Road, all vehicles, seasonal", "Trail, all vehicles", "Trail, all vehicles, seasonal", "Trail, vehicles 50\" or less", "Trail, vehicles 50\" or less, seasonal", "Trail, motorcycles", "Trail, motorcycles, seasonal", "Trail, other OHVs", "Trail, other OHVs, seasonal", "Special designation", "Special designation, seasonal"];

    /// A route's ground width as drawn, metres: wide, since the plane is drawn under the map's own lines as a coloured band around them.
    pub fn width_m(class: u8) -> f32 {
        if class <= 4 { 10.0 } else { 6.0 }
    }
}

/// An ownership class from PAD-US's manager name and type.
pub fn own_class(manager: &str, mtype: &str) -> u8 {
    use own::*;
    match manager {
        "Forest Service" => FOREST_SERVICE,
        "Bureau of Land Management" => BLM,
        "National Park Service" => PARK_SERVICE,
        "U.S. Fish and Wildlife Service" => FISH_WILDLIFE_SERVICE,
        "State Department of Natural Resources" | "State Land Board" | "Other or Unknown State Land" => STATE_TRUST,
        "State Park and Recreation" => STATE_PARK,
        "State Fish and Wildlife" => STATE_WILDLIFE,
        "Tribal" => TRIBAL,
        "Non-Governmental Organization" => CONSERVATION,
        _ => match mtype {
            "Federal" => OTHER_FEDERAL,
            "State" => STATE_TRUST,
            "American Indian Lands" | "Tribal" => TRIBAL,
            "Local Government" | "Regional Agency Special District" => LOCAL,
            "Non-Governmental Organization" | "Private" => CONSERVATION,
            _ => OTHER,
        },
    }
}

/// An MVUM class from the map's legend text, the route kind (road or trail) and its season.
pub fn mvum_class(symbol: &str, kind: &str, seasonal: &str) -> u8 {
    use mvum::*;
    let s = symbol.to_ascii_lowercase();
    let base = if s.contains("special designation") {
        SPECIAL
    } else if s.starts_with("roads open to highway legal") {
        ROAD_HIGHWAY
    } else if s.starts_with("roads open to all") {
        ROAD_ALL
    } else if s.starts_with("trails open to all") {
        TRAIL_ALL
    } else if s.contains("motorcycle") {
        TRAIL_MOTORCYCLE
    } else if s.contains("50\" or less") || s.contains("<50") {
        TRAIL_50
    } else if kind == "road" {
        ROAD_ALL
    } else {
        TRAIL_OTHER
    };
    base + (s.contains("seasonal") || seasonal.eq_ignore_ascii_case("seasonal")) as u8
}

/// A polygon with its class: rings of (lat, lon), outer and inner alike.
pub struct ClassedArea {
    pub class: u8,
    pub rings: Vec<Vec<(f64, f64)>>,
}

/// A line with its class: (lat, lon) points.
pub struct ClassedLine {
    pub class: u8,
    pub pts: Vec<(f64, f64)>,
}

fn text(rec: &shapefile::dbase::Record, name: &str) -> String {
    match rec.get(name) {
        Some(FieldValue::Character(Some(s))) => s.trim().to_string(),
        _ => String::new(),
    }
}

fn polygon_rings(shape: shapefile::Shape) -> Vec<Vec<(f64, f64)>> {
    let rings: Vec<Vec<(f64, f64)>> = match shape {
        shapefile::Shape::Polygon(p) => p.rings().iter().map(|r| r.points().iter().map(|q| (q.y, q.x)).collect()).collect(),
        shapefile::Shape::PolygonZ(p) => p.rings().iter().map(|r| r.points().iter().map(|q| (q.y, q.x)).collect()).collect(),
        shapefile::Shape::PolygonM(p) => p.rings().iter().map(|r| r.points().iter().map(|q| (q.y, q.x)).collect()).collect(),
        _ => Vec::new(),
    };
    rings.into_iter().filter(|r| r.len() >= 3).collect()
}

/// The ownership polygons, in painting order: reservations first (the ground inside is tribal unless a fee parcel says otherwise), then the fee parcels.
pub fn load_ownership(shp: &str) -> Result<Vec<ClassedArea>, String> {
    let mut reader = shapefile::Reader::from_path(shp).map_err(|e| format!("{shp}: {e}"))?;
    let mut tribal = Vec::new();
    let mut fee = Vec::new();
    for rec in reader.iter_shapes_and_records() {
        let (shape, record) = rec.map_err(|e| format!("{shp}: {e}"))?;
        let manager = text(&record, "MANAGER");
        let class = own_class(&manager, &text(&record, "MTYPE"));
        let area = ClassedArea { class, rings: polygon_rings(shape) };
        if area.rings.is_empty() {
            continue;
        }
        if manager == "Tribal" { tribal.push(area) } else { fee.push(area) }
    }
    tribal.extend(fee);
    Ok(tribal)
}

/// Wilderness and wilderness study areas, as rings.
pub fn load_wilderness(shp: &str) -> Result<Vec<Vec<Vec<(f64, f64)>>>, String> {
    let mut reader = shapefile::Reader::from_path(shp).map_err(|e| format!("{shp}: {e}"))?;
    let mut out = Vec::new();
    for rec in reader.iter_shapes_and_records() {
        let (shape, _) = rec.map_err(|e| format!("{shp}: {e}"))?;
        let rings = polygon_rings(shape);
        if !rings.is_empty() {
            out.push(rings);
        }
    }
    Ok(out)
}

/// MVUM routes, one line per part.
pub fn load_mvum(shp: &str) -> Result<Vec<ClassedLine>, String> {
    let mut reader = shapefile::Reader::from_path(shp).map_err(|e| format!("{shp}: {e}"))?;
    let mut out = Vec::new();
    for rec in reader.iter_shapes_and_records() {
        let (shape, record) = rec.map_err(|e| format!("{shp}: {e}"))?;
        let class = mvum_class(&text(&record, "SYMBOL"), &text(&record, "KIND"), &text(&record, "SEASONAL"));
        let parts: Vec<Vec<(f64, f64)>> = match shape {
            shapefile::Shape::Polyline(p) => p.parts().iter().map(|part| part.iter().map(|q| (q.y, q.x)).collect()).collect(),
            shapefile::Shape::PolylineZ(p) => p.parts().iter().map(|part| part.iter().map(|q| (q.y, q.x)).collect()).collect(),
            shapefile::Shape::PolylineM(p) => p.parts().iter().map(|part| part.iter().map(|q| (q.y, q.x)).collect()).collect(),
            _ => continue,
        };
        for pts in parts {
            if pts.len() >= 2 {
                out.push(ClassedLine { class, pts });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_from_the_sources_words() {
        assert_eq!(own_class("Forest Service", "Federal"), own::FOREST_SERVICE);
        assert_eq!(own_class("Bureau of Reclamation", "Federal"), own::OTHER_FEDERAL);
        assert_eq!(own_class("City Land", "Local Government"), own::LOCAL);
        assert_eq!(own_class("Tribal", "American Indian Lands"), own::TRIBAL);
        assert_eq!(mvum_class("Roads open to highway legal vehicles only, Seasonal", "road", "seasonal"), mvum::ROAD_HIGHWAY + 1);
        assert_eq!(mvum_class("Trails open to motorcycles, Yearlong", "trail", "yearlong"), mvum::TRAIL_MOTORCYCLE);
        assert_eq!(mvum_class("Trails open to vehicles 50\" or less in width, Seasonal", "trail", "seasonal"), mvum::TRAIL_50 + 1);
        assert_eq!(mvum_class("Special Designation, Yearlong", "trail", "yearlong"), mvum::SPECIAL);
        assert_eq!(mvum_class("Wheeled OHV <50\", Seasonal", "trail", "seasonal"), mvum::TRAIL_50 + 1);
        assert_eq!(own::NAMES.len(), own::OTHER as usize + 1);
        assert_eq!(mvum::NAMES.len(), mvum::SPECIAL as usize + 2);
    }
}

/// The sea (Nick 2026-10-16: Puget Sound drew as land in the 1 m areas, whose OSM extracts mark the sea only by its coastline): OSM's water polygons (osmdata.openstreetmap.de, water-polygons-split-4326, built from the same coastline as everything else), those touching a box (lat0, lon0, lat1, lon1), each as its rings.
pub fn load_sea(shp: &str, bbox: (f64, f64, f64, f64)) -> Result<Vec<Vec<Vec<(f64, f64)>>>, String> {
    let (lat0, lon0, lat1, lon1) = bbox;
    let mut reader = shapefile::ShapeReader::from_path(shp).map_err(|e| format!("{shp}: {e}"))?;
    let mut out = Vec::new();
    for shape in reader.iter_shapes() {
        let rings = polygon_rings(shape.map_err(|e| format!("{shp}: {e}"))?);
        let touches = rings.iter().flatten().fold((90.0f64, 180.0f64, -90.0f64, -180.0f64), |(a, b, c, d), &(lat, lon)| (a.min(lat), b.min(lon), c.max(lat), d.max(lon)));
        if touches.0 <= lat1 && touches.2 >= lat0 && touches.1 <= lon1 && touches.3 >= lon0 && !rings.is_empty() {
            out.push(rings);
        }
    }
    Ok(out)
}

/// A ring clipped to a box (lat0, lon0, lat1, lon1), Sutherland and Hodgman: edges along the box where the ring leaves it, which an even-odd fill inside the box reads the same as the whole ring.
pub fn clip_ring(ring: &[(f64, f64)], bbox: (f64, f64, f64, f64)) -> Vec<(f64, f64)> {
    let (lat0, lon0, lat1, lon1) = bbox;
    // Each side of the box: inside test and the crossing point along an edge.
    let sides: [(&dyn Fn((f64, f64)) -> bool, &dyn Fn((f64, f64), (f64, f64)) -> (f64, f64)); 4] = [
        (&|p| p.0 >= lat0, &|a, b| (lat0, a.1 + (lat0 - a.0) / (b.0 - a.0) * (b.1 - a.1))),
        (&|p| p.0 <= lat1, &|a, b| (lat1, a.1 + (lat1 - a.0) / (b.0 - a.0) * (b.1 - a.1))),
        (&|p| p.1 >= lon0, &|a, b| (a.0 + (lon0 - a.1) / (b.1 - a.1) * (b.0 - a.0), lon0)),
        (&|p| p.1 <= lon1, &|a, b| (a.0 + (lon1 - a.1) / (b.1 - a.1) * (b.0 - a.0), lon1)),
    ];
    let mut pts: Vec<(f64, f64)> = ring.to_vec();
    if pts.first() == pts.last() {
        pts.pop();
    }
    for (inside, cross) in sides {
        if pts.is_empty() {
            break;
        }
        let mut next = Vec::with_capacity(pts.len() + 4);
        for i in 0..pts.len() {
            let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
            match (inside(a), inside(b)) {
                (true, true) => next.push(b),
                (true, false) => next.push(cross(a, b)),
                (false, true) => {
                    next.push(cross(a, b));
                    next.push(b);
                }
                (false, false) => {}
            }
        }
        pts = next;
    }
    if pts.len() >= 3 {
        pts.push(pts[0]);
        pts
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod sea_tests {
    use super::*;

    /// A square straddling a box keeps the part inside, closed.
    #[test]
    fn a_ring_clips_to_its_box() {
        let square = vec![(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)];
        let clipped = clip_ring(&square, (0.0, 0.0, 2.0, 2.0));
        assert_eq!(clipped.first(), clipped.last());
        assert!(clipped.iter().all(|&(lat, lon)| (0.0..=1.0).contains(&lat) && (0.0..=1.0).contains(&lon)));
        assert_eq!(clipped.len(), 5, "the quarter inside is a square: {clipped:?}");
        assert!(clip_ring(&square, (5.0, 5.0, 6.0, 6.0)).is_empty());
    }
}
