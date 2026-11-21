//! Esri File Geodatabase tables, read for PAD-US (Nick 2026-11-21: no Python in any repo, so the land-ownership prep reads the geodatabase itself). The layout is the one GDAL's OpenFileGDB driver reads, as Even Rouault documented it (github.com/rouault/dump_gdbtable/wiki/FGDB-Spec): only what PAD-US's ArcGIS 10 tables use: polygons and lines, their true curves (circular and elliptic arcs, Béziers) made into lines as GDAL makes them, and their text and number fields. A 64-bit table (ArcGIS Pro 3.2+), a raster field or a point geometry is refused by name rather than read wrong.
//!
//! Every read of the file's bytes goes through [`Cursor`], which checks the length first: the file is outside input, and a truncated or foreign table must end in an error naming it, not a read past the buffer.

use std::path::{Path, PathBuf};

/// The table named `name` in the geodatabase directory `gdb`: the catalog (a00000001.gdbtable) lists every table, the n-th row naming the file a<n as 8 hex digits>.gdbtable.
pub fn table_path(gdb: &Path, name: &str) -> Result<PathBuf, String> {
    let catalog = Table::open(&gdb.join("a00000001.gdbtable"))?;
    let col = catalog.field("Name").ok_or("the catalog has no Name field")?;
    for (fid, row) in catalog.rows() {
        if let Value::Text(n) = row?[col] {
            if n == name {
                return Ok(gdb.join(format!("a{fid:08x}.gdbtable")));
            }
        }
    }
    Err(format!("{}: no table {name}", gdb.display()))
}

/// A field of a table: its name, its type code and whether a row may leave it null.
pub struct Field {
    pub name: String,
    kind: u8,
    nullable: bool,
}

/// A value of a row: the shape is left as its encoded bytes, for [`Table::shape_box`] and [`Table::shape_parts`].
pub enum Value<'a> {
    Null,
    Int(i64),
    Real(f64),
    Text(&'a str),
    Shape(&'a [u8]),
    /// A date, a GUID, an XML document or a blob: kept but not decoded.
    Other,
}

/// The geometry field's grid: a stored coordinate is an integer, scaled and offset.
struct Grid {
    x0: f64,
    y0: f64,
    scale: f64,
}

/// One table, held whole in memory, with each row's offset from the .gdbtablx beside it.
pub struct Table {
    path: PathBuf,
    bytes: Vec<u8>,
    /// Each feature id's row offset in `bytes` (id 1 first); 0 where the row was deleted.
    offsets: Vec<u64>,
    pub fields: Vec<Field>,
    grid: Option<Grid>,
}

// Field type codes.
const INT16: u8 = 0;
const INT32: u8 = 1;
const FLOAT32: u8 = 2;
const FLOAT64: u8 = 3;
const STRING: u8 = 4;
const DATETIME: u8 = 5;
const OBJECTID: u8 = 6;
const GEOMETRY: u8 = 7;
const BINARY: u8 = 8;
const RASTER: u8 = 9;
const GUID: u8 = 10;
const GLOBALID: u8 = 11;
const XML: u8 = 12;
const INT64: u8 = 13;
const DATE: u8 = 14;
const TIME: u8 = 15;
const DATETIME_OFFSET: u8 = 16;

impl Table {
    pub fn open(path: &Path) -> Result<Table, String> {
        let err = |e: String| format!("{}: {e}", path.display());
        let bytes = std::fs::read(path).map_err(|e| err(e.to_string()))?;
        let tablx = std::fs::read(path.with_extension("gdbtablx")).map_err(|e| err(format!("its .gdbtablx: {e}")))?;
        let mut c = Cursor::new(&bytes);
        let magic = c.u32()?;
        if magic != 3 {
            return Err(err(format!("table version {magic}: only the ArcGIS 10 layout (3) is read")));
        }
        c.at = 32;
        let desc = c.u64()? as usize;
        let mut c = Cursor::new(&bytes);
        c.at = desc;
        let _desc_len = c.u32()?;
        let _version = c.u32()?;
        let _layer_flags = c.take(4)?;
        let n = c.u16()?;
        let mut fields = Vec::with_capacity(n as usize);
        let mut grid = None;
        for _ in 0..n {
            let name = c.utf16_name()?;
            let _alias = c.utf16_name()?;
            let kind = c.u8()?;
            let nullable = match kind {
                GEOMETRY => {
                    c.u8()?;
                    let flags = c.u8()?;
                    let wkt_len = c.u16()? as usize;
                    c.take(wkt_len)?;
                    let zm = c.u8()?;
                    let (has_m, has_z) = (zm & 2 != 0, zm & 4 != 0);
                    let (x0, y0, scale) = (c.f64()?, c.f64()?, c.f64()?);
                    // M and Z origin and scale, the tolerances, the extent: then the spatial index's grid sizes, a count after a zero byte.
                    c.take(8 * (2 * has_m as usize + 2 * has_z as usize + 1 + has_m as usize + has_z as usize + 4))?;
                    let pad = c.u8()?;
                    if pad != 0 {
                        return Err(err(format!("geometry field {name}: an unknown layout after the extent ({pad})")));
                    }
                    let grids = c.u32()? as usize;
                    c.take(8 * grids)?;
                    grid = Some(Grid { x0, y0, scale });
                    flags & 1 != 0
                }
                STRING => {
                    c.u32()?;
                    let flags = c.u8()?;
                    let default = c.varuint()? as usize;
                    c.take(default)?;
                    flags & 1 != 0
                }
                OBJECTID | BINARY | GUID | GLOBALID | XML => {
                    c.u8()?;
                    let flags = c.u8()?;
                    flags & 1 != 0
                }
                RASTER => return Err(err(format!("field {name}: a raster field"))),
                INT16 | INT32 | FLOAT32 | FLOAT64 | DATETIME | INT64 | DATE | TIME | DATETIME_OFFSET => {
                    c.u8()?;
                    let flags = c.u8()?;
                    let default = c.u8()? as usize;
                    c.take(default)?;
                    flags & 1 != 0
                }
                k => return Err(err(format!("field {name}: unknown type {k}"))),
            };
            fields.push(Field { name, kind, nullable });
        }
        let offsets = row_offsets(&tablx).map_err(|e| err(format!("its .gdbtablx: {e}")))?;
        Ok(Table { path: path.to_path_buf(), bytes, offsets, fields, grid })
    }

    /// The column of the field named `name`.
    pub fn field(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    /// Every row present, in feature id order, with its id: a value per field, the object id as [`Value::Int`].
    pub fn rows(&self) -> impl Iterator<Item = (u64, Result<Vec<Value<'_>>, String>)> + '_ {
        self.offsets.iter().enumerate().filter(|(_, o)| **o != 0).map(|(i, &o)| {
            let fid = i as u64 + 1;
            (fid, self.row(fid, o as usize).map_err(|e| format!("{} row {fid}: {e}", self.path.display())))
        })
    }

    fn row(&self, fid: u64, at: usize) -> Result<Vec<Value<'_>>, String> {
        let mut c = Cursor::new(&self.bytes);
        c.at = at;
        let len = c.u32()? as usize;
        let mut c = Cursor::new(c.take(len)?);
        let nullable = self.fields.iter().filter(|f| f.nullable).count();
        let nulls = c.take(nullable.div_ceil(8))?;
        let mut k = 0;
        let mut out = Vec::with_capacity(self.fields.len());
        for f in &self.fields {
            if f.kind == OBJECTID {
                out.push(Value::Int(fid as i64));
                continue;
            }
            if f.nullable {
                let null = nulls[k / 8] & (1 << (k % 8)) != 0;
                k += 1;
                if null {
                    out.push(Value::Null);
                    continue;
                }
            }
            out.push(match f.kind {
                INT16 => Value::Int(i16::from_le_bytes(c.array()?) as i64),
                INT32 => Value::Int(i32::from_le_bytes(c.array()?) as i64),
                INT64 => Value::Int(i64::from_le_bytes(c.array()?)),
                FLOAT32 => Value::Real(f32::from_le_bytes(c.array()?) as f64),
                FLOAT64 => Value::Real(c.f64()?),
                STRING => {
                    let n = c.varuint()? as usize;
                    Value::Text(std::str::from_utf8(c.take(n)?).map_err(|e| format!("field {}: {e}", f.name))?)
                }
                GEOMETRY => {
                    let n = c.varuint()? as usize;
                    Value::Shape(c.take(n)?)
                }
                BINARY | XML => {
                    let n = c.varuint()? as usize;
                    c.take(n)?;
                    Value::Other
                }
                DATETIME | DATE | TIME => {
                    c.take(8)?;
                    Value::Other
                }
                DATETIME_OFFSET => {
                    c.take(10)?;
                    Value::Other
                }
                GUID | GLOBALID => {
                    c.take(16)?;
                    Value::Other
                }
                k => return Err(format!("field {}: type {k}", f.name)),
            });
        }
        Ok(out)
    }

    /// A shape's box (xmin, ymin, xmax, ymax) from its header alone, None for an empty shape.
    pub fn shape_box(&self, shape: &[u8]) -> Result<Option<(f64, f64, f64, f64)>, String> {
        let g = self.grid.as_ref().ok_or("a table without a geometry field")?;
        let mut c = Cursor::new(shape);
        let kind = Kind::of(c.varuint()?)?;
        if c.varuint()? == 0 {
            return Ok(None);
        }
        c.varuint()?;
        if kind.curves {
            c.varuint()?;
        }
        let (x, y) = (c.varuint()?, c.varuint()?);
        let (dx, dy) = (c.varuint()?, c.varuint()?);
        Ok(Some((x as f64 / g.scale + g.x0, y as f64 / g.scale + g.y0, (x + dx) as f64 / g.scale + g.x0, (y + dy) as f64 / g.scale + g.y0)))
    }

    /// A polygon's rings or a line's parts, each as (x, y) points in the table's own coordinates.
    pub fn shape_parts(&self, shape: &[u8]) -> Result<Vec<Vec<(f64, f64)>>, String> {
        let g = self.grid.as_ref().ok_or("a table without a geometry field")?;
        let mut c = Cursor::new(shape);
        let kind = Kind::of(c.varuint()?)?;
        let points = c.varuint()? as usize;
        if points == 0 {
            return Ok(Vec::new());
        }
        let parts = c.varuint()? as usize;
        let curves = if kind.curves { c.varuint()? as usize } else { 0 };
        for _ in 0..4 {
            c.varuint()?;
        }
        let mut sizes = Vec::with_capacity(parts);
        for _ in 1..parts {
            sizes.push(c.varuint()? as usize);
        }
        let last = points.checked_sub(sizes.iter().sum()).ok_or("part sizes past the point count")?;
        sizes.push(last);
        // Each point a signed step from the one before, across every part.
        let (mut x, mut y) = (0i64, 0i64);
        let mut pts = Vec::with_capacity(points);
        for _ in 0..points {
            x += c.varint()?;
            y += c.varint()?;
            pts.push((x as f64 / g.scale + g.x0, y as f64 / g.scale + g.y0));
        }
        // The curves follow the Z and M steps: each says which segment it bends (by the index of its first point) and how.
        let mut bends: Vec<(usize, Vec<(f64, f64)>)> = Vec::with_capacity(curves);
        if curves > 0 {
            if kind.z {
                for _ in 0..points {
                    c.varint()?;
                }
            }
            if kind.m {
                // A lone 0x42 says every M is unset.
                if c.b.get(c.at) == Some(&0x42) {
                    c.u8()?;
                } else {
                    for _ in 0..points {
                        c.varint()?;
                    }
                }
            }
            for _ in 0..curves {
                let at = c.varuint()? as usize;
                let (p0, p1) = match (pts.get(at), pts.get(at + 1)) {
                    (Some(&p0), Some(&p1)) => (p0, p1),
                    _ => return Err(format!("a curve on segment {at} of {points} points")),
                };
                let between = match c.varuint()? {
                    ARC => {
                        let (u, v, bits) = (c.f64()?, c.f64()?, c.u32()?);
                        arc_points(p0, p1, (u, v), bits)
                    }
                    BEZIER => {
                        let (c0, c1) = ((c.f64()?, c.f64()?), (c.f64()?, c.f64()?));
                        bezier_points(p0, c0, c1, p1)
                    }
                    ELLIPSE => {
                        let (centre, rotation, major, ratio, bits) = ((c.f64()?, c.f64()?), c.f64()?, c.f64()?, c.f64()?, c.u32()?);
                        ellipse_points(p0, p1, centre, rotation, major, ratio, bits)
                    }
                    t => return Err(format!("curve segment type {t}: only circular and elliptic arcs and Béziers are read")),
                };
                bends.push((at, between));
            }
            bends.sort_by_key(|b| b.0);
        }
        let mut out = Vec::with_capacity(parts);
        let mut next = 0;
        let mut bend = bends.iter().peekable();
        for n in sizes {
            let mut part = Vec::with_capacity(n);
            for i in next..next + n {
                part.push(pts[i]);
                // A bent segment's points, between this point and the next, when the next is in the same part.
                while let Some((_, between)) = bend.next_if(|b| b.0 == i) {
                    if i + 1 < next + n {
                        part.extend_from_slice(between);
                    }
                }
            }
            next += n;
            out.push(part);
        }
        Ok(out)
    }
}

// Curve segment types, and the bits of circular and elliptic arcs (Esri's extended shape buffer).
const ARC: u64 = 1;
const BEZIER: u64 = 4;
const ELLIPSE: u64 = 5;
const ARC_EMPTY: u32 = 0x1;
const ARC_CCW: u32 = 0x8;
const ARC_LINE: u32 = 0x20;
const ARC_POINT: u32 = 0x40;
const ARC_INTERIOR: u32 = 0x80;
const ELLIPSE_EMPTY: u32 = 0x1;
const ELLIPSE_LINE: u32 = 0x40;
const ELLIPSE_POINT: u32 = 0x80;
const ELLIPSE_CENTRE_TO: u32 = 0x200;
const ELLIPSE_CENTRE_FROM: u32 = 0x400;
const ELLIPSE_MINOR: u32 = 0x1000;
const ELLIPSE_COMPLETE: u32 = 0x2000;

// The curves are made into lines the way GDAL's OpenFileGDB driver does it (ogrpgeogeometry.cpp and ogrgeometryfactory.cpp, GDAL 3.12), so the ground they cover is the ground the PAD-US prep has always drawn: its step, its rounding and its odd counts are GDAL's, kept as they are.

/// GDAL's default angular step for an arc (OGR_ARC_STEPSIZE), in degrees.
const ARC_STEP_DEG: f64 = 4.;

/// The points strictly between a circular arc's ends: (u, v) is a point on it when the interior-point bit is set, its centre otherwise, the direction then in the bits.
fn arc_points(p0: (f64, f64), p1: (f64, f64), uv: (f64, f64), bits: u32) -> Vec<(f64, f64)> {
    if bits & ARC_INTERIOR != 0 && bits & ARC_LINE == 0 {
        if p0 != p1 {
            return stroke(p0, uv, p1);
        }
        if p0 == uv {
            return Vec::new();
        }
        // A whole circle, p0 and the interior point at either end of a diameter: two half circles, through the points a quarter turn either side.
        let c = ((p0.0 + uv.0) / 2., (p0.1 + uv.1) / 2.);
        let q0 = (c.0 - (uv.1 - c.1), c.1 + (p0.0 - c.0));
        let q1 = (c.0 + (uv.1 - c.1), c.1 - (p0.0 - c.0));
        let mut out = stroke(p0, q0, uv);
        out.push(uv);
        out.extend(stroke(uv, q1, p1));
        return out;
    }
    if bits & (ARC_EMPTY | ARC_LINE | ARC_POINT) != 0 {
        return Vec::new();
    }
    // The centre and a direction: the arc's middle, found the way round the bits say, stands in for an interior point.
    let c = uv;
    let a0 = (p0.1 - c.1).atan2(p0.0 - c.0);
    let mut a1 = (p1.1 - c.1).atan2(p1.0 - c.0);
    let r = (p1.0 - c.0).hypot(p1.1 - c.1);
    let tau = std::f64::consts::TAU;
    if bits & ARC_CCW != 0 {
        if a0 >= a1 {
            a1 += tau;
        }
    } else if a0 <= a1 {
        a1 -= tau;
    }
    let mid = (a0 + a1) / 2.;
    stroke(p0, (c.0 + r * mid.cos(), c.1 + r * mid.sin()), p1)
}

/// The points strictly between p0 and p2 on the circle through p0, p1 and p2 (GDAL's curveToLineString, its default "stealth" stepping): the arc is always stroked from the same end, so a shared edge comes out the same from either side.
fn stroke(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> Vec<(f64, f64)> {
    if p0.0 < p2.0 || (p0.0 == p2.0 && p0.1 < p2.1) {
        let mut out = stroke(p2, p1, p0);
        out.reverse();
        return out;
    }
    let Some((c, r, a0, a1, a2)) = circle(p0, p1, p2) else {
        // Three points in a line: the middle one is kept.
        return vec![p1];
    };
    let sign = if a1 >= a0 { 1. } else { -1. };
    let step = ARC_STEP_DEG.to_radians();
    // GDAL's count: the nearest whole number of steps, then at least seven and odd.
    let mut n = ((a2 - a0).abs() / step + 0.5) as i64;
    n = if n < 7 { 7 } else { 7 + 2 * ((n - 7 + 1) / 2) };
    let step = sign * ((a2 - a0) / n as f64).abs();
    let mut out = Vec::with_capacity(n as usize);
    let mut a = a0 + step;
    while (a - a2) * sign < -1e-8 {
        out.push((c.0 + r * a.cos(), c.1 + r * a.sin()));
        a += step;
    }
    out
}

/// The circle through three points (GDAL's GetCurveParameters): its centre, radius, and the angles of the three points, unwound so they run one way. None when the points are in a line.
fn circle(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> Option<((f64, f64), f64, f64, f64, f64)> {
    let pi = std::f64::consts::PI;
    if p0 == p2 {
        if p0 == p1 {
            return None;
        }
        let c = ((p0.0 + p1.0) / 2., (p0.1 + p1.1) / 2.);
        let a0 = (p0.1 - c.1).atan2(p0.0 - c.0);
        return Some((c, (p0.0 - c.0).hypot(p0.1 - c.1), a0, a0 + pi, a0 + 2. * pi));
    }
    let (mut dx01, mut dy01, mut dx12, mut dy12) = (p1.0 - p0.0, p1.1 - p0.1, p2.0 - p1.0, p2.1 - p1.1);
    // Normalized by the largest step, so the differences stay small.
    let scale = dx01.abs().max(dy01.abs()).max(dx12.abs()).max(dy12.abs());
    let inv = 1. / scale;
    dx01 *= inv;
    dy01 *= inv;
    dx12 *= inv;
    dy12 *= inv;
    let det = dx01 * dy12 - dx12 * dy01;
    if det.abs() < 1e-8 || det.is_nan() {
        return None;
    }
    let c01 = dx01 * (p0.0 + p1.0) * inv + dy01 * (p0.1 + p1.1) * inv;
    let c12 = dx12 * (p1.0 + p2.0) * inv + dy12 * (p1.1 + p2.1) * inv;
    let c = (0.5 * scale * (c01 * dy12 - c12 * dy01) / det, 0.5 * scale * (-c01 * dx12 + c12 * dx01) / det);
    let a0 = ((p0.1 - c.1) * inv).atan2((p0.0 - c.0) * inv);
    let mut a1 = ((p1.1 - c.1) * inv).atan2((p1.0 - c.0) * inv);
    let mut a2 = ((p2.1 - c.1) * inv).atan2((p2.0 - c.0) * inv);
    if det < 0. {
        if a1 > a0 {
            a1 -= 2. * pi;
        }
        if a2 > a1 {
            a2 -= 2. * pi;
        }
    } else {
        if a1 < a0 {
            a1 += 2. * pi;
        }
        if a2 < a1 {
            a2 += 2. * pi;
        }
    }
    Some((c, (p0.0 - c.0).hypot(p0.1 - c.1), a0, a1, a2))
}

/// The points strictly between an elliptic arc's ends (GDAL's approximateArcAngles): the ellipse's centre, its rotation (radians, counter-clockwise), its semi-major axis and the minor's ratio to it; which way round comes from the minor and complete bits. An ellipse given by its ends' centres is drawn straight, as GDAL does.
fn ellipse_points(p0: (f64, f64), p1: (f64, f64), centre: (f64, f64), rotation: f64, major: f64, ratio: f64, bits: u32) -> Vec<(f64, f64)> {
    if bits & (ELLIPSE_CENTRE_TO | ELLIPSE_CENTRE_FROM | ELLIPSE_EMPTY | ELLIPSE_LINE | ELLIPSE_POINT) != 0 {
        return Vec::new();
    }
    let minor = major * ratio;
    // approximateArcAngles turns clockwise: the rotation and the angles change sign.
    let rot = -rotation.to_degrees();
    let (s, c) = rot.to_radians().sin_cos();
    let on = |p: (f64, f64)| -> f64 {
        if major == 0. || minor == 0. {
            return 0.;
        }
        let (dx, dy) = (p.0 - centre.0, p.1 - centre.1);
        let a = ((s * dx + c * dy) / minor).atan2((c * dx - s * dy) / major).to_degrees();
        if a < -180. { a + 360. } else { a }
    };
    let start = -on(p0);
    let mut end = -on(p1);
    if bits & ELLIPSE_COMPLETE != 0 {
        end = start + 360.;
    } else if bits & ELLIPSE_MINOR != 0 {
        if end > start + 180. {
            end -= 360.;
        } else if end < start - 180. {
            end += 360.;
        }
    } else if end > start && end < start + 180. {
        end -= 360.;
    } else if end < start && end > start - 180. {
        end += 360.;
    }
    let (start, end) = (-start, -end);
    // GDAL's count: at least the two ends, a vertex every step or less.
    let vertices = 2.max(((end - start).abs() / ARC_STEP_DEG).ceil() as i64 + 1);
    let slice = (end - start) / (vertices - 1) as f64;
    (1..vertices - 1)
        .map(|i| {
            let t = (start + i as f64 * slice).to_radians();
            let (x, y) = (t.cos() * major, t.sin() * minor);
            (centre.0 + x * c + y * s, centre.1 - x * s + y * c)
        })
        .collect()
}

/// The points strictly between a cubic Bézier's ends, at even steps of its parameter: as many as GDAL takes, from the turn between its end tangents and how long they are against the chord.
fn bezier_points(p0: (f64, f64), c0: (f64, f64), c1: (f64, f64), p1: (f64, f64)) -> Vec<(f64, f64)> {
    let pi = std::f64::consts::PI;
    let mut a0 = (c0.1 - p0.1).atan2(c0.0 - p0.0);
    let mut a1 = (p1.1 - c1.1).atan2(p1.0 - c1.0);
    if a0 + pi < a1 {
        a0 += 2. * pi;
    } else if a1 + pi < a0 {
        a1 += 2. * pi;
    }
    let sq = |a: (f64, f64), b: (f64, f64)| (b.0 - a.0) * (b.0 - a.0) + (b.1 - a.1) * (b.1 - a.1);
    let (t0, t1, chord) = (sq(p0, c0), sq(c1, p1), sq(p0, p1));
    // GDAL's own bounds on the count (at least two steps of turn, at most a thousand points), kept so the line matches.
    let n = if chord < 1e-9 { 1 } else { 1000f64.min(((a1 - a0).abs() / ARC_STEP_DEG.to_radians()).max(2.) * 1f64.max(5. * (t0 + t1) / chord)).ceil() as usize };
    (1..n)
        .map(|k| {
            let t = k as f64 / n as f64;
            let s = 1. - t;
            let (w0, w1, w2, w3) = (s * s * s, 3. * s * s * t, 3. * s * t * t, t * t * t);
            (w0 * p0.0 + w1 * c0.0 + w2 * c1.0 + w3 * p1.0, w0 * p0.1 + w1 * c0.1 + w2 * c1.1 + w3 * p1.1)
        })
        .collect()
}

/// A shape's type: only polylines and polygons, with whether Z, M and curves follow the points.
struct Kind {
    z: bool,
    m: bool,
    curves: bool,
}

impl Kind {
    fn of(code: u64) -> Result<Kind, String> {
        // The low byte is the shapefile type (or Esri's general polyline 50 and polygon 51); the high bits flag Z, M and curves.
        let t = code & 0xff;
        let z = code & 0x8000_0000 != 0 || matches!(t, 10 | 13 | 15 | 19);
        let m = code & 0x4000_0000 != 0 || matches!(t, 13 | 15 | 23 | 25);
        let curves = code & 0x2000_0000 != 0;
        match t {
            3 | 5 | 10 | 13 | 15 | 19 | 23 | 25 | 50 | 51 => Ok(Kind { z, m, curves }),
            t => Err(format!("shape type {t}: only lines and polygons are read")),
        }
    }
}

/// The .gdbtablx: each feature id's row offset, 0 where none. Blocks of 1024 ids that hold no row may be left out, a bitmap saying which are present.
fn row_offsets(x: &[u8]) -> Result<Vec<u64>, String> {
    let mut c = Cursor::new(x);
    let magic = c.u32()?;
    if magic != 3 {
        return Err(format!("index version {magic}"));
    }
    let blocks = c.u32()? as usize;
    let ids = c.u32()? as usize;
    let size = c.u32()? as usize;
    if !(4..=6).contains(&size) {
        return Err(format!("offset size {size}"));
    }
    let table = c.take(blocks * 1024 * size)?;
    let entry = |i: usize| {
        let mut b = [0u8; 8];
        b[..size].copy_from_slice(&table[i * size..(i + 1) * size]);
        u64::from_le_bytes(b)
    };
    // The bitmap trails the offsets, when there are blocks.
    let bitmap_words = if blocks > 0 { c.u32()? as usize } else { 0 };
    let mut out = vec![0u64; ids];
    if bitmap_words == 0 {
        // Every block present: an id's entry is its own place, in the table when the blocks cover every id.
        if ids > blocks * 1024 {
            return Err(format!("{ids} ids in {blocks} blocks"));
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o = entry(i);
        }
        return Ok(out);
    }
    c.take(12)?;
    // A bit per block of 1024 ids; the present blocks' entries follow one another in the table.
    let bitmap = c.take(bitmap_words * 4)?;
    if bitmap.len() * 8 < ids.div_ceil(1024) {
        return Err(format!("a bitmap of {} bits for {ids} ids", bitmap.len() * 8));
    }
    let mut rank = 0;
    for (b, chunk) in out.chunks_mut(1024).enumerate() {
        if bitmap[b / 8] & (1 << (b % 8)) != 0 {
            if rank == blocks {
                return Err(format!("the bitmap marks more than its {blocks} blocks"));
            }
            for (i, o) in chunk.iter_mut().enumerate() {
                *o = entry(rank * 1024 + i);
            }
            rank += 1;
        }
    }
    Ok(out)
}

/// A read position in a byte slice from the file, every read checked against its length.
struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Cursor<'a> {
        Cursor { b, at: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.at..self.at + n).ok_or_else(|| format!("truncated at byte {} (wanted {n} more)", self.at))?;
        self.at += n;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        Ok(self.take(N)?.try_into().unwrap())
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    /// A name: a count of UTF-16 units, then the units.
    fn utf16_name(&mut self) -> Result<String, String> {
        let n = self.u8()? as usize;
        let units: Vec<u16> = self.take(2 * n)?.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
        String::from_utf16(&units).map_err(|e| e.to_string())
    }

    /// An unsigned varint: seven bits a byte, low first, the top bit saying another follows.
    fn varuint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = self.u8()?;
            if shift > 63 {
                return Err("a varint longer than 64 bits".into());
            }
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
        }
    }

    /// A signed varint: the first byte holds the sign in bit 6 and six bits of the magnitude, the rest seven each.
    fn varint(&mut self) -> Result<i64, String> {
        let b = self.u8()?;
        let negative = b & 0x40 != 0;
        let mut v = (b & 0x3f) as u64;
        let mut more = b & 0x80 != 0;
        let mut shift = 6;
        while more {
            let b = self.u8()?;
            if shift > 63 {
                return Err("a varint longer than 64 bits".into());
            }
            v |= ((b & 0x7f) as u64) << shift;
            more = b & 0x80 != 0;
            shift += 7;
        }
        Ok(if negative { -(v as i64) } else { v as i64 })
    }
}
