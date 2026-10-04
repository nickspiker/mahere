//! Raster elevation boundary. GeoTIFF dies here: USGS 3DEP tiles are decoded
//! once at load and everything downstream sees `elevation(lat, lon)` and
//! `elev_and_gradient(lat, lon)` queries in plain WGS84/NAD83 degrees
//! (the two datums differ by under a meter — ignored, like every consumer
//! of this data).

use std::fs::File;
use std::io::BufReader;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// One loaded DEM tile: a lat/lon-gridded f32 elevation raster.
pub struct DemTile {
    width: usize,
    height: usize,
    /// Top-left corner of the top-left pixel (lon, lat) in degrees.
    origin: (f64, f64),
    /// Degrees per pixel (lon, lat); lat step is positive (rows go south).
    step: (f64, f64),
    data: Vec<f32>,
}

/// A set of tiles answering point queries; tiles are searched in order, so
/// overlapping collars resolve to the first tile loaded.
pub struct DemStore {
    tiles: Vec<DemTile>,
}

impl DemTile {
    pub fn load(path: &str) -> Result<DemTile, String> {
        let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut dec = Decoder::new(BufReader::new(file)).map_err(|e| format!("{path}: {e}"))?;
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
        // Tiepoint maps raster (i, j) -> world (x=lon, y=lat); USGS ties
        // pixel (0, 0) to the top-left corner.
        let origin = (tie[3] - tie[0] * scale[0], tie[4] + tie[1] * scale[1]);
        let data = match dec.read_image().map_err(|e| format!("{path}: {e}"))? {
            DecodingResult::F32(v) => v,
            other => {
                return Err(format!(
                    "{path}: expected F32 samples, got {:?} variant",
                    sample_kind(&other)
                ))
            }
        };
        if data.len() != w as usize * h as usize {
            return Err(format!("{path}: pixel count mismatch"));
        }
        Ok(DemTile {
            width: w as usize,
            height: h as usize,
            origin,
            step: (scale[0], scale[1]),
            data,
        })
    }

    /// Fractional pixel coordinates of (lat, lon), if inside this tile
    /// (with half a pixel of slack for the bilinear footprint).
    fn pixel_at(&self, lat: f64, lon: f64) -> Option<(f64, f64)> {
        let px = (lon - self.origin.0) / self.step.0 - 0.5;
        let py = (self.origin.1 - lat) / self.step.1 - 0.5;
        if px < 0.0 || py < 0.0 || px > (self.width - 1) as f64 || py > (self.height - 1) as f64 {
            return None;
        }
        Some((px, py))
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
                let m_per_px_lon =
                    (t.step.0.to_radians() * EARTH_RADIUS_M * lat.to_radians().cos()) as f32;
                let m_per_px_lat = (t.step.1.to_radians() * EARTH_RADIUS_M) as f32;
                let ge = east / (2.0 * m_per_px_lon);
                let gn = -south / (2.0 * m_per_px_lat);
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
