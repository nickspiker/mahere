//! Land ownership (PAD-US 4.1) and motor vehicle use (USFS MVUM) for one area, as plain shapefiles in latitude and longitude for the overlay baker (mahere-overlay): reprojected, clipped to the area's box, the source fields kept as text. The classes are decided in [`crate::landuse`], so there is one table. It replaced a Python script (Nick 2026-11-21: no Python in any repo), and reads the geodatabase itself ([`crate::filegdb`]) and inverts PAD-US's Albers projection here.
//!
//! The clip is Sutherland and Hodgman's ([`crate::landuse::clip_ring`]) where the Python used GEOS's validity repair and intersection: the even-odd fill the baker runs over each feature's rings reads both the same inside the box, which is padded past the area so no texel of it sits on a clipped edge.

use std::path::Path;

use shapefile::dbase::{FieldName, FieldValue, Record, TableWriterBuilder};
use shapefile::{Point, Polygon, PolygonRing, Polyline};

use crate::filegdb::{Table, Value, table_path};
use crate::landuse::clip_ring;

/// How far past the area's box the sources are clipped, in degrees, so polygons and lines reach its edge cells whole.
const PAD: f64 = 0.02;

/// The width of every text field written, in bytes: GDAL's default for a string field, which the Python's shapefiles carried, a longer value cut at a character boundary below it.
const TEXT_WIDTH: u8 = 80;

/// The prep: `bbox` is (lat0, lon0, lat1, lon1); `gdbs` the PAD-US state geodatabases (PADUS4_1_State<XX>.gdb); `mvum` the roads and trails shapefiles. Writes own.shp, wild.shp and mvum.shp into `out`.
pub fn prep(bbox: (f64, f64, f64, f64), out: &Path, gdbs: &[&Path], mvum: &[&Path]) -> Result<(), String> {
    let (lat0, lon0, lat1, lon1) = bbox;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let clip = (lat0 - PAD, lon0 - PAD, lat1 + PAD, lon1 + PAD);
    // The box in PAD-US's Albers, from its edges a hundredth of a degree at a time (an edge bows in the projection).
    let (mut ax0, mut ay0, mut ax1, mut ay1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    let corners = [(clip.0, clip.1), (clip.0, clip.3), (clip.2, clip.3), (clip.2, clip.1), (clip.0, clip.1)];
    for w in corners.windows(2) {
        let ((la0, lo0), (la1, lo1)) = (w[0], w[1]);
        let n = (((la1 - la0).abs() + (lo1 - lo0).abs()) / (1. / (1 << 7) as f64)).ceil() as usize;
        for k in 0..=n {
            let t = k as f64 / n as f64;
            let (x, y) = albers_forward(la0 + (la1 - la0) * t, lo0 + (lo1 - lo0) * t);
            (ax0, ay0, ax1, ay1) = (ax0.min(x), ay0.min(y), ax1.max(x), ay1.max(y));
        }
    }
    let alb = (ax0, ay0, ax1, ay1);

    let mut own: Vec<(Vec<Vec<(f64, f64)>>, [String; 4])> = Vec::new();
    let mut wild: Vec<(Vec<Vec<(f64, f64)>>, [String; 2])> = Vec::new();
    for gdb in gdbs {
        let st = gdb.file_name().and_then(|n| n.to_str()).ok_or_else(|| format!("{}: not a geodatabase name", gdb.display()))?.replace("PADUS4_1_State", "").replace(".gdb", "");
        // The fee layer: who manages the ground.
        each_area(gdb, &format!("PADUS4_1Fee_State_{st}"), &["d_Mang_Name", "d_Mang_Type", "d_Des_Tp", "d_Pub_Access"], alb, clip, |rings, f| {
            own.push((rings, [f[0].clone(), f[1].clone(), f[2].clone(), f[3].clone()]));
        })?;
        // Reservations are proclamation boundaries, not fee: the ground inside is tribal unless a fee parcel says otherwise (the baker paints them first).
        each_area(gdb, &format!("PADUS4_1Proclamation_State_{st}"), &["d_Mang_Type", "Unit_Nm"], alb, clip, |rings, f| {
            if f[0] == "American Indian Lands" {
                let des = if f[1].is_empty() { "Reservation".to_string() } else { f[1].clone() };
                own.push((rings, ["Tribal".into(), "American Indian Lands".into(), des, "Unknown".into()]));
            }
        })?;
        // The designation layer: wilderness and wilderness study areas, where bikes are banned.
        each_area(gdb, &format!("PADUS4_1Designation_State_{st}"), &["d_Des_Tp", "Unit_Nm"], alb, clip, |rings, f| {
            if f[0] == "Wilderness Area" || f[0] == "Wilderness Study Area" {
                wild.push((rings, [f[0].clone(), f[1].clone()]));
            }
        })?;
    }
    write_areas(&out.join("own.shp"), &["MANAGER", "MTYPE", "DES", "ACCESS"], &own)?;
    write_areas(&out.join("wild.shp"), &["DES", "NAME"], &wild)?;

    // MVUM: roads and trails, already in latitude and longitude (NAD83, within a metre of WGS 84).
    let mut routes: Vec<(Vec<Vec<(f64, f64)>>, [String; 4])> = Vec::new();
    for (path, kind) in mvum.iter().zip(["road", "trail"]) {
        let err = |e: shapefile::Error| format!("{}: {e}", path.display());
        let mut shapes = shapefile::ShapeReader::from_path(path).map_err(err)?;
        // The shapes first, the records of only those that reach the box after: the national files hold hundreds of thousands.
        let mut hits: Vec<(usize, Vec<Vec<(f64, f64)>>)> = Vec::new();
        for (i, shape) in shapes.iter_shapes().enumerate() {
            let parts: Vec<Vec<(f64, f64)>> = match shape.map_err(err)? {
                shapefile::Shape::Polyline(p) => p.parts().iter().map(|q| q.iter().map(|p| (p.y, p.x)).collect()).collect(),
                shapefile::Shape::PolylineM(p) => p.parts().iter().map(|q| q.iter().map(|p| (p.y, p.x)).collect()).collect(),
                shapefile::Shape::PolylineZ(p) => p.parts().iter().map(|q| q.iter().map(|p| (p.y, p.x)).collect()).collect(),
                _ => continue,
            };
            let clipped: Vec<Vec<(f64, f64)>> = parts.iter().flat_map(|p| clip_line(p, clip)).collect();
            if !clipped.is_empty() {
                hits.push((i, clipped));
            }
        }
        let mut table = shapefile::dbase::Reader::from_path(path.with_extension("dbf")).map_err(|e| format!("{}: {e}", path.display()))?;
        for (i, parts) in hits {
            table.seek(i).map_err(|e| format!("{} record {i}: {e}", path.display()))?;
            let rec = table.iter_records().next().ok_or_else(|| format!("{}: no record {i}", path.display()))?.map_err(|e| format!("{} record {i}: {e}", path.display()))?;
            let text = |name: &str| match rec.get(name) {
                Some(FieldValue::Character(Some(s))) => s.trim_end().to_string(),
                _ => String::new(),
            };
            routes.push((parts, [text("MVUM_SYMBO"), kind.to_string(), text("SEASONAL"), text("NAME")]));
        }
    }
    write_lines(&out.join("mvum.shp"), &["SYMBOL", "KIND", "SEASONAL", "NAME"], &routes)
}

/// Each feature of a PAD-US layer whose box meets `alb` (the Albers box), its rings in latitude and longitude clipped to `clip` (lat0, lon0, lat1, lon1), with the named fields as text (a null as empty); a feature with nothing left inside is skipped.
fn each_area(gdb: &Path, layer: &str, cols: &[&str], alb: (f64, f64, f64, f64), clip: (f64, f64, f64, f64), mut keep: impl FnMut(Vec<Vec<(f64, f64)>>, &[String])) -> Result<(), String> {
    let t = Table::open(&table_path(gdb, layer)?)?;
    let col = |n: &str| t.field(n).ok_or_else(|| format!("{layer}: no field {n}"));
    let cols: Vec<usize> = cols.iter().map(|n| col(n)).collect::<Result<_, _>>()?;
    let shape = t.fields.iter().position(|f| f.name.eq_ignore_ascii_case("shape")).ok_or_else(|| format!("{layer}: no shape field"))?;
    for (fid, row) in t.rows() {
        let row = row?;
        let Value::Shape(s) = row[shape] else { continue };
        let Some(b) = t.shape_box(s).map_err(|e| format!("{layer} row {fid}: {e}"))? else { continue };
        if b.0 > alb.2 || b.2 < alb.0 || b.1 > alb.3 || b.3 < alb.1 {
            continue;
        }
        let rings: Vec<Vec<(f64, f64)>> = t
            .shape_parts(s)
            .map_err(|e| format!("{layer} row {fid}: {e}"))?
            .iter()
            .map(|r| clip_ring(&r.iter().map(|&(x, y)| albers_inverse(x, y)).collect::<Vec<_>>(), clip))
            .filter(|r| r.len() >= 4 && ring_area(r) != 0.)
            .collect();
        if rings.is_empty() {
            continue;
        }
        let fields: Vec<String> = cols
            .iter()
            .map(|&c| match row[c] {
                Value::Text(s) => s.to_string(),
                _ => String::new(),
            })
            .collect();
        keep(rings, &fields);
    }
    Ok(())
}

/// Twice a ring's signed area in (lat, lon): positive when it runs clockwise on the map (east as x, north as y), as a shapefile's outer rings do.
fn ring_area(r: &[(f64, f64)]) -> f64 {
    r.windows(2).map(|w| w[0].0 * w[1].1 - w[1].0 * w[0].1).sum()
}

/// A line clipped to a box (lat0, lon0, lat1, lon1), each segment by Liang and Barsky: the runs inside, each its own part.
fn clip_line(pts: &[(f64, f64)], b: (f64, f64, f64, f64)) -> Vec<Vec<(f64, f64)>> {
    let mut out: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut run: Vec<(f64, f64)> = Vec::new();
    for w in pts.windows(2) {
        let (p, q) = (w[0], w[1]);
        let d = (q.0 - p.0, q.1 - p.1);
        let (mut t0, mut t1) = (0f64, 1f64);
        let mut inside = true;
        for (num, den) in [(p.0 - b.0, -d.0), (b.2 - p.0, d.0), (p.1 - b.1, -d.1), (b.3 - p.1, d.1)] {
            if den == 0. {
                if num < 0. {
                    inside = false;
                }
            } else {
                let t = num / den;
                if den < 0. {
                    t0 = t0.max(t);
                } else {
                    t1 = t1.min(t);
                }
            }
        }
        if !inside || t0 > t1 {
            if run.len() >= 2 {
                out.push(std::mem::take(&mut run));
            }
            run.clear();
            continue;
        }
        let a = (p.0 + d.0 * t0, p.1 + d.1 * t0);
        let z = (p.0 + d.0 * t1, p.1 + d.1 * t1);
        if run.last() != Some(&a) {
            if run.len() >= 2 {
                out.push(std::mem::take(&mut run));
            }
            run.clear();
            run.push(a);
        }
        run.push(z);
        // Leaving the box ends the run.
        if t1 < 1. {
            if run.len() >= 2 {
                out.push(std::mem::take(&mut run));
            }
            run.clear();
        }
    }
    if run.len() >= 2 {
        out.push(run);
    }
    out
}

/// A text value cut to the field's width at a character boundary.
fn fit(s: &str) -> String {
    let mut end = s.len().min(TEXT_WIDTH as usize);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

fn table(names: &[&str]) -> Result<TableWriterBuilder, String> {
    let mut t = TableWriterBuilder::new();
    for n in names {
        t = t.add_character_field(FieldName::try_from(*n).map_err(|e| format!("{n}: {e:?}"))?, TEXT_WIDTH);
    }
    Ok(t)
}

fn record<const N: usize>(names: &[&str], fields: &[String; N]) -> Record {
    let mut r = Record::default();
    for (n, v) in names.iter().zip(fields) {
        r.insert(n.to_string(), FieldValue::Character(Some(fit(v))));
    }
    r
}

/// The shapefile's side files: its text is UTF-8, its coordinates WGS 84 latitude and longitude.
fn side_files(shp: &Path) -> Result<(), String> {
    let wgs84 = "GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]]";
    std::fs::write(shp.with_extension("cpg"), "UTF-8").map_err(|e| format!("{}: {e}", shp.display()))?;
    std::fs::write(shp.with_extension("prj"), wgs84).map_err(|e| format!("{}: {e}", shp.display()))
}

fn write_areas<const N: usize>(shp: &Path, names: &[&str], areas: &[(Vec<Vec<(f64, f64)>>, [String; N])]) -> Result<(), String> {
    let err = |e: shapefile::Error| format!("{}: {e}", shp.display());
    let mut w = shapefile::Writer::from_path(shp, table(names)?).map_err(err)?;
    for (rings, fields) in areas {
        // Clockwise rings are outer, as the source drew them; the clip keeps their turn.
        let rings: Vec<PolygonRing<Point>> = rings
            .iter()
            .map(|r| {
                let pts: Vec<Point> = r.iter().map(|&(lat, lon)| Point::new(lon, lat)).collect();
                if ring_area(r) > 0. { PolygonRing::Outer(pts) } else { PolygonRing::Inner(pts) }
            })
            .collect();
        w.write_shape_and_record(&Polygon::with_rings(rings), &record(names, fields)).map_err(err)?;
    }
    drop(w);
    side_files(shp)?;
    println!("{}: {} features", shp.file_name().unwrap().to_string_lossy(), areas.len());
    Ok(())
}

fn write_lines<const N: usize>(shp: &Path, names: &[&str], lines: &[(Vec<Vec<(f64, f64)>>, [String; N])]) -> Result<(), String> {
    let err = |e: shapefile::Error| format!("{}: {e}", shp.display());
    let mut w = shapefile::Writer::from_path(shp, table(names)?).map_err(err)?;
    for (parts, fields) in lines {
        let parts: Vec<Vec<Point>> = parts.iter().map(|p| p.iter().map(|&(lat, lon)| Point::new(lon, lat)).collect()).collect();
        w.write_shape_and_record(&Polyline::with_parts(parts), &record(names, fields)).map_err(err)?;
    }
    drop(w);
    side_files(shp)?;
    println!("{}: {} features", shp.file_name().unwrap().to_string_lossy(), lines.len());
    Ok(())
}

// PAD-US's projection, USA Contiguous Albers Equal Area Conic, USGS version (ESRI:102039): on GRS 80, standard parallels 29.5° and 45.5°, origin 23° N 96° W, no false easting or northing. Its NAD83 is taken as WGS 84, as PROJ's NAD83 to WGS 84 does without a grid (a null shift).
const A: f64 = 6_378_137.;
const INV_F: f64 = 298.257_222_101;
const LAT1: f64 = 29.5;
const LAT2: f64 = 45.5;
const LAT0: f64 = 23.;
const LON0: f64 = -96.;

/// The ellipsoid's eccentricity e, and Snyder's constants for the cone: n, C and ρ0 (Map Projections: A Working Manual, 14-3 to 14-6).
fn cone() -> (f64, f64, f64, f64) {
    let f = 1. / INV_F;
    let e = (f * (2. - f)).sqrt();
    let m = |lat: f64| {
        let s = lat.to_radians().sin();
        lat.to_radians().cos() / (1. - e * e * s * s).sqrt()
    };
    let (m1, m2) = (m(LAT1), m(LAT2));
    let (q1, q2, q0) = (q(e, LAT1.to_radians()), q(e, LAT2.to_radians()), q(e, LAT0.to_radians()));
    let n = (m1 * m1 - m2 * m2) / (q2 - q1);
    let c = m1 * m1 + n * q1;
    let rho0 = A * (c - n * q0).sqrt() / n;
    (e, n, c, rho0)
}

/// Snyder's q (3-12): the authalic function of latitude φ (radians).
fn q(e: f64, phi: f64) -> f64 {
    let s = phi.sin();
    (1. - e * e) * (s / (1. - e * e * s * s) - (1. / (2. * e)) * ((1. - e * s) / (1. + e * s)).ln())
}

/// Latitude and longitude (degrees) to Albers metres (x, y).
pub fn albers_forward(lat: f64, lon: f64) -> (f64, f64) {
    let (e, n, c, rho0) = cone();
    let rho = A * (c - n * q(e, lat.to_radians())).sqrt() / n;
    let theta = n * (lon - LON0).to_radians();
    (rho * theta.sin(), rho0 - rho * theta.cos())
}

/// Albers metres (x, y) to (latitude, longitude) in degrees: the latitude by Snyder's iteration (3-16) until it moves less than 2⁻⁴⁰ radians (about 6 µm).
pub fn albers_inverse(x: f64, y: f64) -> (f64, f64) {
    let (e, n, c, rho0) = cone();
    let rho = (x * x + (rho0 - y) * (rho0 - y)).sqrt();
    let theta = x.atan2(rho0 - y);
    let qv = (c - rho * rho * n * n / (A * A)) / n;
    let mut phi = (qv / 2.).asin();
    for _ in 0..(1 << 4) {
        let s = phi.sin();
        let one = 1. - e * e * s * s;
        let d = one * one / (2. * phi.cos()) * (qv / (1. - e * e) - s / one + (1. / (2. * e)) * ((1. - e * s) / (1. + e * s)).ln());
        phi += d;
        if d.abs() < 1. / (1u64 << 40) as f64 {
            break;
        }
    }
    (phi.to_degrees(), LON0 + (theta / n).to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A point goes to Albers and back to where it was.
    #[test]
    fn albers_round_trips() {
        for &(lat, lon) in &[(47.5, -121.5), (23., -96.), (48.99, -124.7), (25.1, -80.2)] {
            let (x, y) = albers_forward(lat, lon);
            let (la, lo) = albers_inverse(x, y);
            assert!((la - lat).abs() < 1e-10 && (lo - lon).abs() < 1e-10, "{lat},{lon} came back {la},{lo}");
        }
        // The origin is at (0, 0).
        let (x, y) = albers_forward(23., -96.);
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6);
    }

    /// A segment crossing the box keeps the part inside; one outside is gone; one leaving and coming back is two parts.
    #[test]
    fn a_line_clips_to_its_box() {
        let b = (0., 0., 1., 1.);
        assert_eq!(clip_line(&[(0.5, -1.), (0.5, 2.)], b), vec![vec![(0.5, 0.), (0.5, 1.)]]);
        assert!(clip_line(&[(2., 2.), (3., 3.)], b).is_empty());
        let parts = clip_line(&[(0.2, 0.5), (0.2, 2.), (0.8, 2.), (0.8, 0.5)], b);
        assert_eq!(parts.len(), 2, "{parts:?}");
    }
}
