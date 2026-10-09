//! HydroRIVERS (Lehner & Grill 2013, hydrosheds.org, CC BY 4.0): the global river network as reaches with a long-term average discharge, which is the magnitude OSM cannot give a river. At the global depth the water lines come from here alone.

use crate::{Road, RoadClass};
use shapefile::dbase::FieldValue;

/// A reach's weight 0..1 from its discharge in m³/s, log scaled: 0.01 is 0, 10,000 (the Mississippi) is 1, the Amazon clamps.
pub fn discharge_weight(cms: f64) -> f32 {
    (((cms.max(0.01)).log10() + 2.0) / 6.0).clamp(0.0, 1.0) as f32
}

/// Every reach as a waterway line, its weight from `DIS_AV_CMS`. Reaches under `min_cms` are left out (a headwater trickle at 100 m texels is noise).
pub fn load_hydrorivers(shp: &str, min_cms: f64) -> Result<Vec<Road>, String> {
    let mut reader = shapefile::Reader::from_path(shp).map_err(|e| format!("{shp}: {e}"))?;
    let mut out = Vec::new();
    for rec in reader.iter_shapes_and_records() {
        let (shape, record) = rec.map_err(|e| format!("{shp}: {e}"))?;
        let cms = match record.get("DIS_AV_CMS") {
            Some(FieldValue::Numeric(Some(v))) => *v,
            Some(FieldValue::Float(Some(v))) => *v as f64,
            _ => 0.0,
        };
        if cms < min_cms {
            continue;
        }
        let parts: Vec<Vec<(f32, f32)>> = match shape {
            shapefile::Shape::Polyline(p) => p.parts().iter().map(|part| part.iter().map(|q| (q.y as f32, q.x as f32)).collect()).collect(),
            shapefile::Shape::PolylineZ(p) => p.parts().iter().map(|part| part.iter().map(|q| (q.y as f32, q.x as f32)).collect()).collect(),
            shapefile::Shape::PolylineM(p) => p.parts().iter().map(|part| part.iter().map(|q| (q.y as f32, q.x as f32)).collect()).collect(),
            _ => continue,
        };
        let weight = discharge_weight(cms);
        for pts in parts {
            if pts.len() >= 2 {
                out.push(Road { class: RoadClass::Waterway, pts, weight, uses: 0 });
            }
        }
    }
    Ok(out)
}
