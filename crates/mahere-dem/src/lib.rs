//! Raster elevation boundary. GeoTIFF dies here: USGS 3DEP tiles are decoded
//! once at load and everything downstream sees `elevation(lat, lon)` and
//! `elev_and_gradient(lat, lon)` queries in plain WGS84/NAD83 degrees
//! (the two datums differ by under a meter — ignored, like every consumer
//! of this data).
//!
//! Two grid kinds: geographic (the 1/3 and 1 arc-second products, degrees
//! per pixel) and projected UTM (the 1 m lidar products, meters per pixel).
//! A UTM tile is queried by projecting the lat/lon forward onto its zone;
//! the grid convergence (< 2° within a zone) is ignored for the gradient.

use std::fs::File;
use std::io::BufReader;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// How a tile's pixel grid maps to the world.
#[derive(Clone, Copy, Debug)]
enum Grid {
    /// Degrees per pixel (lon, lat).
    Geographic,
    /// UTM zone (northern hemisphere), meters per pixel.
    UtmNorth(u8),
}

/// One loaded DEM tile: an f32 elevation raster on a geographic or UTM grid.
pub struct DemTile {
    width: usize,
    height: usize,
    grid: Grid,
    /// Top-left corner of the top-left pixel in grid units (x, y).
    origin: (f64, f64),
    /// Grid units per pixel (x, y); y step is positive (rows go south).
    step: (f64, f64),
    data: Vec<f32>,
}

/// A set of tiles answering point queries; tiles are searched in order, so
/// overlapping collars resolve to the first tile loaded.
pub struct DemStore {
    tiles: Vec<DemTile>,
}

/// GeoKey IDs (OGC GeoTIFF).
const GT_MODEL_TYPE: u16 = 1024;
const PROJECTED_CS_TYPE: u16 = 3072;

impl DemTile {
    pub fn load(path: &str) -> Result<DemTile, String> {
        let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut dec = Decoder::new(BufReader::new(file))
            .map_err(|e| format!("{path}: {e}"))?
            .with_limits(tiff::decoder::Limits::unlimited());
        let (w, h) = dec.dimensions().map_err(|e| format!("{path}: {e}"))?;
        let scale = dec
            .get_tag_f64_vec(Tag::ModelPixelScaleTag)
            .map_err(|e| format!("{path}: ModelPixelScale: {e}"))?;
        let tie = dec
            .get_tag_f64_vec(Tag::ModelTiepointTag)
            .map_err(|e| format!("{path}: ModelTiepoint: {e}"))?;
        if scale.len() < 2 || tie.len() < 5 {
            return Err(format!("{path}: malformed georeferencing tags"));
        }
        let grid = match dec.get_tag_u32_vec(Tag::GeoKeyDirectoryTag) {
            Ok(keys) => grid_of(&keys).ok_or_else(|| format!("{path}: unsupported CRS in GeoKeyDirectory"))?,
            Err(e) => return Err(format!("{path}: GeoKeyDirectory: {e}")),
        };
        eprintln!("dem tile {path}: {w}x{h} {grid:?}");
        // Tiepoint maps raster (i, j) -> world (x, y); USGS ties pixel (0, 0)
        // to the top-left corner.
        let origin = (tie[3] - tie[0] * scale[0], tie[4] + tie[1] * scale[1]);
        let data = match dec.read_image().map_err(|e| format!("{path}: {e}"))? {
            DecodingResult::F32(v) => v,
            other => {
                return Err(format!("{path}: expected F32 samples, got {:?} variant", sample_kind(&other)))
            }
        };
        if data.len() != w as usize * h as usize {
            return Err(format!("{path}: pixel count mismatch"));
        }
        Ok(DemTile { width: w as usize, height: h as usize, grid, origin, step: (scale[0], scale[1]), data })
    }

    /// Fractional pixel coordinates of (lat, lon), if inside this tile
    /// (with half a pixel of slack for the bilinear footprint).
    fn pixel_at(&self, lat: f64, lon: f64) -> Option<(f64, f64)> {
        let (x, y) = match self.grid {
            Grid::Geographic => (lon, lat),
            Grid::UtmNorth(zone) => utm_forward(lat, lon, zone),
        };
        let px = (x - self.origin.0) / self.step.0 - 0.5;
        let py = (self.origin.1 - y) / self.step.1 - 0.5;
        if px < 0.0 || py < 0.0 || px > (self.width - 1) as f64 || py > (self.height - 1) as f64 {
            return None;
        }
        Some((px, py))
    }

    /// Meters per pixel along x and y at this latitude.
    fn m_per_px(&self, lat: f64) -> (f32, f32) {
        match self.grid {
            Grid::Geographic => (
                (self.step.0.to_radians() * EARTH_RADIUS_M * lat.to_radians().cos()) as f32,
                (self.step.1.to_radians() * EARTH_RADIUS_M) as f32,
            ),
            Grid::UtmNorth(_) => (self.step.0 as f32, self.step.1 as f32),
        }
    }

    fn grid(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    fn bilinear(&self, px: f64, py: f64) -> f32 {
        let x0 = px.floor() as usize;
        let y0 = py.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = (px - x0 as f64) as f32;
        let fy = (py - y0 as f64) as f32;
        let top = self.grid(x0, y0) * (1.0 - fx) + self.grid(x1, y0) * fx;
        let bot = self.grid(x0, y1) * (1.0 - fx) + self.grid(x1, y1) * fx;
        top * (1.0 - fy) + bot * fy
    }
}

/// Decode the GeoKeyDirectory (u16 quads: key, tag location, count, value)
/// into a grid kind. Geographic models and northern UTM zones (WGS84
/// 326xx, NAD83 269xx, NAD83(2011) 6329-6348) are supported.
fn grid_of(keys: &[u32]) -> Option<Grid> {
    let mut model = None;
    let mut pcs = None;
    for q in keys.chunks(4).skip(1) {
        if q.len() < 4 || q[1] != 0 {
            continue;
        }
        match q[0] as u16 {
            GT_MODEL_TYPE => model = Some(q[3]),
            PROJECTED_CS_TYPE => pcs = Some(q[3]),
            _ => {}
        }
    }
    match (model, pcs) {
        (Some(2), _) | (None, None) => Some(Grid::Geographic),
        (Some(1), Some(epsg)) => {
            let zone = match epsg {
                32601..=32660 => epsg - 32600,
                26901..=26923 => epsg - 26900,
                6329..=6348 => epsg - 6329 + 1,
                _ => return None,
            };
            Some(Grid::UtmNorth(zone as u8))
        }
        _ => None,
    }
}

/// Transverse Mercator forward projection (WGS84/GRS80 ellipsoid — identical
/// to sub-millimeter), UTM parameters: k0 = 0.9996, false easting 500 km.
pub fn utm_forward(lat: f64, lon: f64, zone: u8) -> (f64, f64) {
    const A: f64 = 6_378_137.0;
    const F: f64 = 1.0 / 298.257_223_563;
    const K0: f64 = 0.9996;
    let e2 = F * (2.0 - F);
    let ep2 = e2 / (1.0 - e2);
    let lon0 = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
    let phi = lat.to_radians();
    let dl = lon.to_radians() - lon0;
    let (sp, cp) = phi.sin_cos();
    let n = A / (1.0 - e2 * sp * sp).sqrt();
    let t = sp / cp;
    let t2 = t * t;
    let c = ep2 * cp * cp;
    let a = cp * dl;
    // Meridional arc length.
    let e4 = e2 * e2;
    let e6 = e4 * e2;
    let m = A
        * ((1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * phi
            - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * phi).sin()
            + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * phi).sin()
            - (35.0 * e6 / 3072.0) * (6.0 * phi).sin());
    let a2 = a * a;
    let a3 = a2 * a;
    let a4 = a3 * a;
    let a5 = a4 * a;
    let a6 = a5 * a;
    let x = K0 * n * (a + (1.0 - t2 + c) * a3 / 6.0 + (5.0 - 18.0 * t2 + t2 * t2 + 72.0 * c - 58.0 * ep2) * a5 / 120.0)
        + 500_000.0;
    let y = K0
        * (m + n * t * (a2 / 2.0 + (5.0 - t2 + 9.0 * c + 4.0 * c * c) * a4 / 24.0
            + (61.0 - 58.0 * t2 + t2 * t2 + 600.0 * c - 330.0 * ep2) * a6 / 720.0));
    (x, y)
}

impl DemStore {
    pub fn load(paths: &[String]) -> Result<DemStore, String> {
        let mut tiles = Vec::new();
        for p in paths {
            tiles.push(DemTile::load(p)?);
        }
        Ok(DemStore { tiles })
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Per-tile diagnostics: grid, extent, and value statistics.
    pub fn describe(&self) -> Vec<String> {
        self.tiles
            .iter()
            .map(|t| {
                let finite = t.data.iter().filter(|v| v.is_finite() && **v > -10_000.0).count();
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                for &v in &t.data {
                    if v.is_finite() && v > -10_000.0 {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
                let c = t.grid(t.width / 2, t.height / 2);
                format!(
                    "{}x{} {:?} origin ({:.3}, {:.3}) step ({}, {}) valid {}/{} range {lo}..{hi} centre {c}",
                    t.width, t.height, t.grid, t.origin.0, t.origin.1, t.step.0, t.step.1, finite, t.data.len()
                )
            })
            .collect()
    }

    /// Bilinear elevation in meters, or None outside coverage / on nodata.
    pub fn elevation(&self, lat: f64, lon: f64) -> Option<f32> {
        for t in &self.tiles {
            if let Some((px, py)) = t.pixel_at(lat, lon) {
                let e = t.bilinear(px, py);
                if e > -10_000.0 {
                    return Some(e);
                }
            }
        }
        None
    }

    /// Elevation plus gradient (east, north) in meters of rise per meter of
    /// ground, from central differences on the source grid.
    pub fn elev_and_gradient(&self, lat: f64, lon: f64) -> Option<(f32, (f32, f32))> {
        for t in &self.tiles {
            if let Some((px, py)) = t.pixel_at(lat, lon) {
                let e = t.bilinear(px, py);
                if e <= -10_000.0 {
                    continue;
                }
                let east = t.bilinear((px + 1.0).min((t.width - 1) as f64), py)
                    - t.bilinear((px - 1.0).max(0.0), py);
                let south = t.bilinear(px, (py + 1.0).min((t.height - 1) as f64))
                    - t.bilinear(px, (py - 1.0).max(0.0));
                let (mx, my) = t.m_per_px(lat);
                let ge = east / (2.0 * mx);
                let gn = -south / (2.0 * my);
                return Some((e, (ge, gn)));
            }
        }
        None
    }
}

fn sample_kind(r: &DecodingResult) -> &'static str {
    match r {
        DecodingResult::U8(_) => "U8",
        DecodingResult::U16(_) => "U16",
        DecodingResult::U32(_) => "U32",
        DecodingResult::U64(_) => "U64",
        DecodingResult::I8(_) => "I8",
        DecodingResult::I16(_) => "I16",
        DecodingResult::I32(_) => "I32",
        DecodingResult::I64(_) => "I64",
        DecodingResult::F32(_) => "F32",
        DecodingResult::F64(_) => "F64",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port Orchard, UTM 10N — reference values from PROJ (EPSG:26910).
    #[test]
    fn utm_forward_matches_proj() {
        let (x, y) = utm_forward(47.5404, -122.6363, 10);
        assert!((x - 527_370.459).abs() < 0.01, "easting {x}");
        assert!((y - 5_265_283.737).abs() < 0.01, "northing {y}");
        let (x, y) = utm_forward(46.2024, -121.4909, 10);
        assert!((x - 616_425.756).abs() < 0.01, "easting {x}");
        assert!((y - 5_117_642.733).abs() < 0.01, "northing {y}");
        // The central meridian maps to the false easting exactly.
        let (x0, _) = utm_forward(45.0, -123.0, 10);
        assert!((x0 - 500_000.0).abs() < 1e-3);
    }

    #[test]
    fn geokeys_decode_utm_and_geographic() {
        // Header quad then (key, loc, count, value) entries.
        let utm = [1u32, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 26910];
        assert!(matches!(grid_of(&utm), Some(Grid::UtmNorth(10))));
        let geo = [1u32, 1, 0, 1, 1024, 0, 1, 2];
        assert!(matches!(grid_of(&geo), Some(Grid::Geographic)));
    }
}
