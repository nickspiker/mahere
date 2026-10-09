//! Raster elevation boundary. GeoTIFF dies here: USGS 3DEP tiles are decoded once at load and everything downstream sees `elevation(lat, lon)` and `elev_and_gradient(lat, lon)` queries in plain WGS84/NAD83 degrees (the two datums differ by under a meter — ignored, like every consumer of this data).
//!
//! Two grid kinds: geographic (the 1/3 and 1 arc-second products, degrees per pixel) and projected UTM (the 1 m lidar products, meters per pixel).
//! A UTM tile is queried by projecting the lat/lon forward onto its zone; the grid convergence (< 2° within a zone) is ignored for the gradient.

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

/// A set of tiles answering point queries; tiles are searched in order, so overlapping collars resolve to the first tile loaded.
pub struct DemStore {
    tiles: Vec<DemTile>,
    /// Every one-degree square the source publishes a tile for, when the source is a global set (Copernicus publishes a tile wherever there is land): a point in a square outside it is open sea.
    land_squares: Option<std::collections::HashSet<(i32, i32)>>,
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
        // Tiepoint maps raster (i, j) -> world (x, y); USGS ties pixel (0, 0) to the top-left corner.
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

    /// Fractional pixel coordinates of (lat, lon), if inside this tile (with half a pixel of slack for the bilinear footprint).
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

/// Decode the GeoKeyDirectory (u16 quads: key, tag location, count, value) into a grid kind. Geographic models and northern UTM zones (WGS84 326xx, NAD83 269xx, NAD83(2011) 6329-6348) are supported.
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

/// Transverse Mercator forward projection (WGS84/GRS80 ellipsoid — identical to sub-millimeter), UTM parameters: k0 = 0.9996, false easting 500 km.
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
    /// Tiles decode in parallel, kept in the order given: a sample takes the first tile with data, so earlier paths win.
    pub fn load(paths: &[String]) -> Result<DemStore, String> {
        use rayon::prelude::*;
        let tiles: Result<Vec<DemTile>, String> = paths.par_iter().map(|p| DemTile::load(p)).collect();
        Ok(DemStore { tiles: tiles?, land_squares: None })
    }

    /// Declare the source global: the one-degree squares (floor of latitude, floor of longitude) it has tiles for. Points outside every one of them read as sea level.
    pub fn set_land_squares(&mut self, squares: std::collections::HashSet<(i32, i32)>) {
        self.land_squares = Some(squares);
    }

    /// Elevation, or 0 in a square the global source has no tile for (open sea).
    pub fn elevation_or_sea(&self, p: (f64, f64)) -> Option<f32> {
        if let Some(e) = self.elevation(p.0, p.1) {
            return Some(e);
        }
        let land = self.land_squares.as_ref()?;
        let lon = (p.1 + 180.0).rem_euclid(360.0) - 180.0;
        let sq = (p.0.floor() as i32, lon.floor() as i32);
        (!land.contains(&sq)).then_some(0.0)
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

    /// Elevation plus gradient (east, north) in meters of rise per meter of ground, from central differences on the source grid.
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
mod img_tests {
    #[test]
    fn a_flat_reflector_is_neutral() {
        let m = super::sentinel_to_vsf();
        let v = super::mul3(m, [0.3, 0.3, 0.3]);
        for c in v {
            assert!((c - 0.3).abs() < 1e-3, "{v:?}");
        }
        let w = super::srgb_to_vsf([1.0, 1.0, 1.0]);
        for c in w {
            assert!((c - 1.0).abs() < 2e-3, "{w:?}");
        }
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

// ==================== IMAGERY: ORTHOIMAGES TO SCENE LIGHT ====================

/// A multi-band orthoimage tile, geographic or UTM grid like a dem tile, interleaved samples kept as they came: 8-bit display-referred (NAIP: red, green, blue, near-infrared at 60 cm) or 16-bit reflectance (Sentinel-2: B04, B03, B02, B08, or one near-infrared band alone). [`ImgStore::sample`] turns either into scene light in VSF RGB.
pub struct ImgTile {
    width: usize,
    height: usize,
    bands: usize,
    grid: Grid,
    origin: (f64, f64),
    step: (f64, f64),
    data: Samples,
}

enum Samples {
    /// Display-referred bytes, taken as sRGB.
    Display(Vec<u8>),
    /// Reflectance, 10000 per unit plus `offset`.
    Reflectance { data: Vec<u16>, offset: u16 },
}

/// Scene light 1 (paper white) for a reflectance source: 30% for the visible bands, 50% for the near-infrared, where leaves are bright. Reflectance is against a flat white diffuser, so sunlit slopes and snow pass 1; the stored tone has headroom for five times paper white.
pub const VISIBLE_WHITE: f32 = 0.30;
pub const NIR_WHITE: f32 = 0.50;

/// Sentinel-2's red, green and blue bands (665, 560, 490 nm) to VSF RGB, row-major. Three narrow bands sample a reflectance spectrum that is smooth, so the spectrum is taken as the straight-line interpolation between the band centres, held flat beyond the end bands, and integrated against the Stockman & Sharpe 2000 10° cone fundamentals under Illuminant E, then into VSF RGB. A flat reflector is Illuminant E exactly, so equal reflectance comes out neutral with no fitted weights. (Taking the bands as monochromatic primaries instead fails: a line at 490 nm is cyan, more M than S, and white then needs a negative 560 nm band.)
pub fn sentinel_to_vsf() -> &'static [[f32; 3]; 3] {
    static M: std::sync::OnceLock<[[f32; 3]; 3]> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        let cones = &vsf::colour::LMS_2000_10DEG_1NM;
        let start = cones.start_nm as usize;
        let n = cones.data.len() / 3;
        // The interpolation's weight on each band at a wavelength: hats between the centres, flat past the ends. Bands in order blue, green, red.
        let nodes = [490.0f32, 560.0, 665.0];
        let weight = |nm: f32| -> [f32; 3] {
            if nm <= nodes[0] {
                [1.0, 0.0, 0.0]
            } else if nm >= nodes[2] {
                [0.0, 0.0, 1.0]
            } else if nm <= nodes[1] {
                let t = (nm - nodes[0]) / (nodes[1] - nodes[0]);
                [1.0 - t, t, 0.0]
            } else {
                let t = (nm - nodes[1]) / (nodes[2] - nodes[1]);
                [0.0, 1.0 - t, t]
            }
        };
        // c[cone][band]: each band's basis spectrum seen by each cone (the cones sum to 1 over the table, so Illuminant E is (1, 1, 1)).
        let mut c = [[0f32; 3]; 3];
        for k in 0..n {
            let w = weight((start + k) as f32);
            for cone in 0..3 {
                for band in 0..3 {
                    c[cone][band] += cones.data[k * 3 + cone] * w[band];
                }
            }
        }
        // vsf's matrices are column-major.
        let l = &vsf::colour::LMS2VSF_RGB;
        let to_vsf = [[l[0], l[3], l[6]], [l[1], l[4], l[7]], [l[2], l[5], l[8]]];
        // Columns reordered to the samples' order: red (665), green (560), blue (490).
        let m: [[f32; 3]; 3] = std::array::from_fn(|r| std::array::from_fn(|band| (0..3).map(|j| to_vsf[r][j] * c[j][band]).sum()));
        std::array::from_fn(|r| [m[r][2], m[r][1], m[r][0]])
    })
}

/// sRGB bytes to linear light, the sRGB transfer exactly (it is what a display-referred source was made for).
fn srgb_table() -> &'static [f32; 256] {
    static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        std::array::from_fn(|b| {
            let c = b as f32 / 255.0;
            if c <= 0.040_45 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
        })
    })
}

fn mul3(m: &[[f32; 3]; 3], x: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| m[r][0] * x[0] + m[r][1] * x[1] + m[r][2] * x[2])
}

pub struct ImgStore {
    tiles: Vec<ImgTile>,
    /// Geographic tiles by the one-degree squares they overlap, so a store of dozens of tiles samples without scanning them all; tiles in other grids are always scanned.
    by_square: std::collections::HashMap<(i32, i32), Vec<usize>>,
    others: Vec<usize>,
}

impl ImgTile {
    pub fn load(path: &str) -> Result<ImgTile, String> {
        let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut dec = Decoder::new(BufReader::new(file)).map_err(|e| format!("{path}: {e}"))?.with_limits(tiff::decoder::Limits::unlimited());
        let (w, h) = dec.dimensions().map_err(|e| format!("{path}: {e}"))?;
        let scale = dec.get_tag_f64_vec(Tag::ModelPixelScaleTag).map_err(|e| format!("{path}: ModelPixelScale: {e}"))?;
        let tie = dec.get_tag_f64_vec(Tag::ModelTiepointTag).map_err(|e| format!("{path}: ModelTiepoint: {e}"))?;
        let grid = match dec.get_tag_u32_vec(Tag::GeoKeyDirectoryTag) {
            Ok(keys) => grid_of(&keys).ok_or_else(|| format!("{path}: unsupported CRS"))?,
            Err(e) => return Err(format!("{path}: GeoKeyDirectory: {e}")),
        };
        let origin = (tie[3] - tie[0] * scale[0], tie[4] + tie[1] * scale[1]);
        let bits = dec.get_tag_u32_vec(Tag::BitsPerSample).ok().and_then(|b| b.first().copied());
        // A palette image (ESA WorldCover's class map: one byte a pixel, the colours in a table) the decoder refuses as well; its bytes are the classes, read by hand.
        let palette = dec.get_tag_u32(Tag::PhotometricInterpretation).ok() == Some(3);
        if bits == Some(8) && palette {
            let data = read_tiled_u8(path, &mut dec, w as usize, h as usize)?;
            eprintln!("img tile {path}: {w}x{h} x1 (classes) {grid:?}");
            return Ok(ImgTile { width: w as usize, height: h as usize, bands: 1, grid, origin, step: (scale[0], scale[1]), data: Samples::Display(data) });
        }
        if bits == Some(15) {
            // Sentinel-2 L2A as served by the Planetary Computer: 15-bit samples, which the decoder refuses; read the tiles by hand.
            let data = read_tiled_u15(path, &mut dec, w as usize, h as usize)?;
            eprintln!("img tile {path}: {w}x{h} x1 (15-bit) {grid:?}");
            return Ok(ImgTile { width: w as usize, height: h as usize, bands: 1, grid, origin, step: (scale[0], scale[1]), data: Samples::Reflectance { data, offset: 1000 } });
        }
        // GDAL metadata names the reflectance offset when the producer removed it (the ESA WorldCover composite: offset 0); a bare Sentinel-2 L2A band from processing baseline 4 on carries the 1000 offset in its samples.
        let meta = dec.get_tag_ascii_string(Tag::Unknown(42112)).unwrap_or_default();
        let offset: u16 = if meta.contains("role=\"offset\">0<") { 0 } else { 1000 };
        let (data, n) = match dec.read_image().map_err(|e| format!("{path}: {e}"))? {
            DecodingResult::U8(v) => {
                let n = v.len();
                (Samples::Display(v), n)
            }
            DecodingResult::U16(v) => {
                let n = v.len();
                (Samples::Reflectance { data: v, offset }, n)
            }
            other => return Err(format!("{path}: expected U8 or U16 samples, got {}", sample_kind(&other))),
        };
        let px = w as usize * h as usize;
        if n % px != 0 {
            return Err(format!("{path}: sample count {n} is not a multiple of {px} pixels"));
        }
        let bands = n / px;
        eprintln!("img tile {path}: {w}x{h} x{bands} {grid:?}");
        Ok(ImgTile { width: w as usize, height: h as usize, bands, grid, origin, step: (scale[0], scale[1]), data })
    }

    fn pixel_at(&self, lat: f64, lon: f64) -> Option<(usize, usize)> {
        let (x, y) = match self.grid {
            Grid::Geographic => (lon, lat),
            Grid::UtmNorth(zone) => utm_forward(lat, lon, zone),
        };
        let px = (x - self.origin.0) / self.step.0;
        let py = (self.origin.1 - y) / self.step.1;
        if px < 0.0 || py < 0.0 || px >= self.width as f64 || py >= self.height as f64 {
            return None;
        }
        Some((px as usize, py as usize))
    }

    /// The scene light of the pixel whose first sample is at `i`, or None when every band is 0.
    fn light(&self, i: usize) -> Option<[f32; 4]> {
        let n = self.bands.min(4);
        match &self.data {
            Samples::Display(d) => {
                let px = &d[i..i + n];
                if px.iter().all(|&v| v == 0) {
                    return None;
                }
                let t = srgb_table();
                if n == 1 {
                    return Some([t[px[0] as usize], 0.0, 0.0, 0.0]);
                }
                let rgb = [t[px[0] as usize], t[px[1] as usize], t[px.get(2).copied().unwrap_or(0) as usize]];
                let v = srgb_to_vsf(rgb);
                Some([v[0], v[1], v[2], px.get(3).map_or(0.0, |&b| t[b as usize])])
            }
            Samples::Reflectance { data, offset } => {
                let px = &data[i..i + n];
                if px.iter().all(|&v| v == 0) {
                    return None;
                }
                let r = |d: u16| d.saturating_sub(*offset) as f32 / 10000.0;
                if n == 1 {
                    return Some([r(px[0]) / NIR_WHITE, 0.0, 0.0, 0.0]);
                }
                let v = mul3(sentinel_to_vsf(), [r(px[0]), r(px[1]), r(px.get(2).copied().unwrap_or(0))]);
                Some([v[0] / VISIBLE_WHITE, v[1] / VISIBLE_WHITE, v[2] / VISIBLE_WHITE, px.get(3).map_or(0.0, |&d| r(d) / NIR_WHITE)])
            }
        }
    }
}

/// Linear sRGB to VSF RGB (vsf's matrix, D65 adapted to E; column-major there).
fn srgb_to_vsf(x: [f32; 3]) -> [f32; 3] {
    let m = &vsf::colour::SRGB2VSF_RGB;
    [m[0] * x[0] + m[3] * x[1] + m[6] * x[2], m[1] * x[0] + m[4] * x[1] + m[7] * x[2], m[2] * x[0] + m[5] * x[1] + m[8] * x[2]]
}

impl ImgStore {
    /// Tiles decode in parallel, kept in the order given: a sample takes the first tile with data, so earlier paths win.
    pub fn load(paths: &[String]) -> Result<ImgStore, String> {
        use rayon::prelude::*;
        let tiles: Vec<ImgTile> = paths.par_iter().map(|p| ImgTile::load(p)).collect::<Result<_, _>>()?;
        let mut by_square: std::collections::HashMap<(i32, i32), Vec<usize>> = std::collections::HashMap::new();
        let mut others = Vec::new();
        for (i, t) in tiles.iter().enumerate() {
            match t.grid {
                Grid::Geographic => {
                    let (lon0, lat1) = t.origin;
                    let (lon1, lat0) = (lon0 + t.width as f64 * t.step.0, lat1 - t.height as f64 * t.step.1);
                    for la in lat0.floor() as i32..lat1.ceil() as i32 {
                        for lo in lon0.floor() as i32..lon1.ceil() as i32 {
                            by_square.entry((la, lo)).or_default().push(i);
                        }
                    }
                }
                _ => others.push(i),
            }
        }
        Ok(ImgStore { tiles, by_square, others })
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// The raw byte of a one-band tile at (lat, lon), longitude wrapped: a class raster's class (0 = no data). None outside coverage.
    pub fn class_at(&self, lat: f64, lon: f64) -> Option<u8> {
        let lon = (lon + 180.0).rem_euclid(360.0) - 180.0;
        let near = self.by_square.get(&(lat.floor() as i32, lon.floor() as i32)).map_or(&[][..], |v| &v[..]);
        for &ti in near.iter().chain(&self.others) {
            let t = &self.tiles[ti];
            if t.bands != 1 {
                continue;
            }
            if let Some((px, py)) = t.pixel_at(lat, lon) {
                if let Samples::Display(d) = &t.data {
                    let v = d[py * t.width + px];
                    if v != 0 {
                        return Some(v);
                    }
                }
            }
        }
        None
    }

    /// [`ImgStore::sample`] at a (lat, lon) pair, longitude wrapped into -180..180 as the source tiles are.
    pub fn sample_lat_lon(&self, p: (f64, f64)) -> Option<[f32; 4]> {
        self.sample(p.0, (p.1 + 180.0).rem_euclid(360.0) - 180.0)
    }

    /// Nearest-pixel scene light at (lat, lon): VSF RGB red, green, blue (paper white 1), then near-infrared (paper white 1 at [`NIR_WHITE`]); a single-band tile is near-infrared alone and comes back in the first slot. None outside coverage or where every band is 0 (the no-data collar).: None outside coverage or where every band is 0 (NAIP's no-data collar).
    pub fn sample(&self, lat: f64, lon: f64) -> Option<[f32; 4]> {
        let near = self.by_square.get(&(lat.floor() as i32, lon.floor() as i32)).map_or(&[][..], |v| &v[..]);
        for &ti in near.iter().chain(&self.others) {
            let t = &self.tiles[ti];
            if let Some((px, py)) = t.pixel_at(lat, lon) {
                let i = (py * t.width + px) * t.bands;
                if let Some(x) = t.light(i) {
                    return Some(x);
                }
            }
        }
        None
    }
}

// ==================== LIDAR INTENSITY: LAZ -> 1 m GRID ====================

/// The 1064 nm return intensity of a lidar point cloud, binned to a 1 m UTM grid (mean of first returns per cell) and stretched to 1..=255 between the 1st and 99th percentile; 0 = no returns. An active-illumination near-infrared image: no sun, no shadows.
pub struct IntensityStore {
    zone: u8,
    x0: f64,
    y0: f64,
    width: usize,
    height: usize,
    data: Vec<u8>,
    /// Canopy height per cell, metres (first-return top minus ground), 0 = none or no ground return.
    canopy: Vec<u8>,
}

impl IntensityStore {
    /// Build from LAZ tiles (all in one northern UTM `zone`, metres). Tiles decode in parallel; the grid covers their union.
    pub fn from_laz(paths: &[String], zone: u8) -> Result<IntensityStore, String> {
        use rayon::prelude::*;
        // Pass 1: bounds.
        let bounds: Vec<las::Bounds> = paths
            .par_iter()
            .map(|p| las::Reader::from_path(p).map(|r| r.header().bounds()).map_err(|e| format!("{p}: {e}")))
            .collect::<Result<_, _>>()?;
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for b in &bounds {
            x0 = x0.min(b.min.x);
            y0 = y0.min(b.min.y);
            x1 = x1.max(b.max.x);
            y1 = y1.max(b.max.y);
        }
        let (x0, y0) = (x0.floor(), y0.floor());
        let width = (x1.ceil() - x0) as usize + 1;
        let height = (y1.ceil() - y0) as usize + 1;
        eprintln!("intensity grid {width}x{height} m from {} tiles", paths.len());
        // Pass 2: per-tile accumulation into the shared grid (tiles don't overlap, so disjoint writes; merged by addition).
        // Per tile: intensity sum/count of first returns, the highest first return (canopy top) and the lowest ground-classified return per cell — on the TILE's own sub-grid (its bounds, not the whole box), copied into the shared grids afterwards. Tiles don't overlap.
        struct TileGrid {
            ox: usize,
            oy: usize,
            w: usize,
            h: usize,
            sum: Vec<u32>,
            cnt: Vec<u16>,
            top: Vec<f32>,
            ground: Vec<f32>,
        }
        let y_top = y1.ceil();
        let partial: Vec<TileGrid> = paths
            .par_iter()
            .zip(bounds.par_iter())
            .filter_map(|(p, b)| {
                let ox = (b.min.x.floor() - x0).max(0.0) as usize;
                let oy = (y_top - b.max.y.ceil()).max(0.0) as usize;
                let w = ((b.max.x.ceil() - b.min.x.floor()) as usize + 1).min(width - ox);
                let h = ((b.max.y.ceil() - b.min.y.floor()) as usize + 1).min(height - oy);
                let mut g = TileGrid { ox, oy, w, h, sum: vec![0; w * h], cnt: vec![0; w * h], top: vec![f32::MIN; w * h], ground: vec![f32::MAX; w * h] };
                let data = las::Reader::from_path(p).and_then(|mut r| r.read_all()).ok()?;
                // Column accessors skip the full-record decode.
                for (((((x, y), z), inten), rn), cls) in
                    data.x().zip(data.y()).zip(data.z()).zip(data.intensity()).zip(data.return_number()).zip(data.classification())
                {
                    let gx = (x - x0) as isize - ox as isize;
                    let gy = (y_top - y) as isize - oy as isize;
                    if gx < 0 || gy < 0 || gx as usize >= w || gy as usize >= h {
                        continue;
                    }
                    let i = gy as usize * w + gx as usize;
                    if cls == 2 {
                        g.ground[i] = g.ground[i].min(z as f32);
                    }
                    if rn != 1 {
                        continue;
                    }
                    g.sum[i] = g.sum[i].saturating_add(inten as u32);
                    g.cnt[i] = g.cnt[i].saturating_add(1);
                    g.top[i] = g.top[i].max(z as f32);
                }
                Some(g)
            })
            .collect();
        let mut sum = vec![0u32; width * height];
        let mut cnt = vec![0u16; width * height];
        let mut top = vec![f32::MIN; width * height];
        let mut ground = vec![f32::MAX; width * height];
        for g in partial {
            for y in 0..g.h {
                let src = y * g.w;
                let dst = (g.oy + y) * width + g.ox;
                for x in 0..g.w {
                    sum[dst + x] = sum[dst + x].saturating_add(g.sum[src + x]);
                    cnt[dst + x] = cnt[dst + x].saturating_add(g.cnt[src + x]);
                    top[dst + x] = top[dst + x].max(g.top[src + x]);
                    ground[dst + x] = ground[dst + x].min(g.ground[src + x]);
                }
            }
        }
        // Canopy: top minus the lowest ground return within a 3x3 m neighbourhood (ground returns are sparse under dense canopy), metres, clamped to 255.
        let canopy: Vec<u8> = (0..sum.len())
            .map(|i| {
                if top[i] == f32::MIN {
                    return 0;
                }
                let (cx, cy) = ((i % width) as isize, (i / width) as isize);
                let mut g = f32::MAX;
                for dy in -1..=1isize {
                    for dx in -1..=1isize {
                        let (x, y) = (cx + dx, cy + dy);
                        if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
                            g = g.min(ground[y as usize * width + x as usize]);
                        }
                    }
                }
                if g == f32::MAX {
                    return 0;
                }
                (top[i] - g).clamp(0.0, 255.0) as u8
            })
            .collect();
        // Percentile stretch over cells with returns.
        let mut means: Vec<u32> = (0..sum.len()).filter(|&i| cnt[i] > 0).map(|i| sum[i] / cnt[i] as u32).collect();
        if means.is_empty() {
            return Err("no returns".into());
        }
        means.sort_unstable();
        let lo = means[means.len() / 100] as f32;
        let hi = means[means.len() * 99 / 100].max(means[means.len() / 100] + 1) as f32;
        let data: Vec<u8> = (0..sum.len())
            .map(|i| {
                if cnt[i] == 0 {
                    0
                } else {
                    let m = (sum[i] / cnt[i] as u32) as f32;
                    (1.0 + 254.0 * ((m - lo) / (hi - lo)).clamp(0.0, 1.0)) as u8
                }
            })
            .collect();
        eprintln!("intensity stretch {lo}..{hi}, {} of {} cells lit", means.len(), data.len());
        Ok(IntensityStore { zone, x0, y0: y1.ceil(), width, height, data, canopy })
    }

    /// Canopy height in metres at (lat, lon); None outside the grid or without returns.
    pub fn canopy(&self, lat: f64, lon: f64) -> Option<u8> {
        let (x, y) = utm_forward(lat, lon, self.zone);
        let gx = x - self.x0;
        let gy = self.y0 - y;
        if gx < 0.0 || gy < 0.0 || gx >= self.width as f64 || gy >= self.height as f64 {
            return None;
        }
        Some(self.canopy[gy as usize * self.width + gx as usize])
    }

    /// Nearest 1 m cell at (lat, lon); None outside the grid or with no returns.
    pub fn sample(&self, lat: f64, lon: f64) -> Option<u8> {
        let (x, y) = utm_forward(lat, lon, self.zone);
        let gx = x - self.x0;
        let gy = self.y0 - y;
        if gx < 0.0 || gy < 0.0 || gx >= self.width as f64 || gy >= self.height as f64 {
            return None;
        }
        let v = self.data[gy as usize * self.width + gx as usize];
        (v != 0).then_some(v)
    }
}

/// A tiled, deflate-compressed, single-band TIFF of bytes (no predictor), read tile by tile as they are.
fn read_tiled_u8<R: std::io::Read + std::io::Seek>(path: &str, dec: &mut Decoder<R>, w: usize, h: usize) -> Result<Vec<u8>, String> {
    use std::io::{Read, Seek, SeekFrom};
    let get = |dec: &mut Decoder<R>, t: Tag| dec.get_tag_u32_vec(t).ok().and_then(|v| v.first().copied());
    let (tw, th) = (get(dec, Tag::TileWidth).ok_or("no TileWidth")? as usize, get(dec, Tag::TileLength).ok_or("no TileLength")? as usize);
    if get(dec, Tag::Compression) != Some(8) || get(dec, Tag::Predictor).unwrap_or(1) != 1 {
        return Err(format!("{path}: byte reader handles deflate without a predictor only"));
    }
    let offsets = dec.get_tag_u64_vec(Tag::TileOffsets).map_err(|e| format!("{path}: {e}"))?;
    let counts = dec.get_tag_u64_vec(Tag::TileByteCounts).map_err(|e| format!("{path}: {e}"))?;
    let across = w.div_ceil(tw);
    let mut file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = vec![0u8; w * h];
    let mut raw = Vec::new();
    for (t, (&off, &len)) in offsets.iter().zip(&counts).enumerate() {
        raw.resize(len as usize, 0);
        file.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
        file.read_exact(&mut raw).map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity(tw * th);
        flate2::read::ZlibDecoder::new(&raw[..]).read_to_end(&mut bytes).map_err(|e| format!("{path}: tile {t}: {e}"))?;
        let (x0, y0) = ((t % across) * tw, (t / across) * th);
        for ry in 0..th {
            let y = y0 + ry;
            if y >= h {
                break;
            }
            let row = &bytes[ry * tw..((ry + 1) * tw).min(bytes.len())];
            let n = row.len().min(w.saturating_sub(x0));
            out[y * w + x0..y * w + x0 + n].copy_from_slice(&row[..n]);
        }
    }
    Ok(out)
}

/// A tiled, deflate-compressed, single-band TIFF of 15-bit samples (no predictor), read tile by tile into 16-bit samples as they are.
fn read_tiled_u15<R: std::io::Read + std::io::Seek>(path: &str, dec: &mut Decoder<R>, w: usize, h: usize) -> Result<Vec<u16>, String> {
    use std::io::{Read, Seek, SeekFrom};
    let get = |dec: &mut Decoder<R>, t: Tag| dec.get_tag_u32_vec(t).ok().and_then(|v| v.first().copied());
    let (tw, th) = (get(dec, Tag::TileWidth).ok_or("no TileWidth")? as usize, get(dec, Tag::TileLength).ok_or("no TileLength")? as usize);
    if get(dec, Tag::Compression) != Some(8) || get(dec, Tag::Predictor).unwrap_or(1) != 1 {
        return Err(format!("{path}: 15-bit reader handles deflate without a predictor only"));
    }
    let offsets = dec.get_tag_u64_vec(Tag::TileOffsets).map_err(|e| format!("{path}: {e}"))?;
    let counts = dec.get_tag_u64_vec(Tag::TileByteCounts).map_err(|e| format!("{path}: {e}"))?;
    let across = w.div_ceil(tw);
    let mut file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = vec![0u16; w * h];
    let mut raw = Vec::new();
    for (t, (&off, &len)) in offsets.iter().zip(&counts).enumerate() {
        raw.resize(len as usize, 0);
        file.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
        file.read_exact(&mut raw).map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity(tw * th * 15 / 8 + 8);
        flate2::read::ZlibDecoder::new(&raw[..]).read_to_end(&mut bytes).map_err(|e| format!("{path}: tile {t}: {e}"))?;
        let (x0, y0) = ((t % across) * tw, (t / across) * th);
        let row_bytes = (tw * 15).div_ceil(8);
        for ry in 0..th {
            let y = y0 + ry;
            if y >= h {
                break;
            }
            let row = &bytes[ry * row_bytes..((ry + 1) * row_bytes).min(bytes.len())];
            let (mut acc, mut nbits, mut pos) = (0u32, 0u32, 0usize);
            for rx in 0..tw {
                while nbits < 15 {
                    acc = (acc << 8) | row.get(pos).copied().unwrap_or(0) as u32;
                    pos += 1;
                    nbits += 8;
                }
                nbits -= 15;
                let d = ((acc >> nbits) & 0x7FFF) as u16;
                let x = x0 + rx;
                if x < w {
                    out[y * w + x] = d;
                }
            }
        }
    }
    Ok(out)
}
