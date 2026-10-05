//! The cell pipeline: bake source data into dymaxion-cell rasters — the
//! #pagetable renderer's entire diet. Two layers today:
//!
//! - **dem**: per-cell elevation + gradient planes (f32), sampled from the
//!   source DEM at bake time (so cell-edge gradients are exact — the baker
//!   holds the whole source; no aprons needed).
//! - **line**: the linework pyramid. Every line feature stamped 1 texel wide
//!   at the base depth, then parents built by averaging 4 children per
//!   texel up the Morton tree — coverage IS box filtering, so opacity at
//!   every zoom is computed, not styled. No vectors exist at render time.
//!
//! Texels are (class, coverage) so styling stays a draw-time LUT. Cells are
//! 256x256, written as VSF at `{layer}/{depth:02}/{prefix:016x}.vsf` —
//! a directory layout that is byte-for-byte the future R2 bucket layout.

use std::collections::HashMap;
use std::path::Path;

use mahere_coord::{Coord, morton_compact, morton_spread, uv_to_lat_lon};
use mahere_dem::DemStore;
use mahere_osm::Road;
use rayon::prelude::*;
use vsf::types::Tensor;
use vsf::{VsfBuilder, VsfType};

/// Cells are TEX x TEX texels; TEX_BITS of Morton depth below the cell.
pub const TEX: usize = 256;
pub const TEX_BITS: u8 = 8;

/// A cell address: dymaxion Morton prefix (diamond in the top 4 bits of the
/// full-resolution coordinate, right-aligned here) at `depth`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CellKey {
    pub depth: u8,
    pub prefix: u64,
}

impl CellKey {
    /// The cell containing `c` at `depth`: diamond + 2*depth Morton bits,
    /// right-aligned (so prefixes sort and shift like integers).
    pub fn containing(c: Coord, depth: u8) -> CellKey {
        CellKey { depth, prefix: c.raw() >> (60 - 2 * depth as u32) }
    }

    pub fn parent(self) -> CellKey {
        CellKey { depth: self.depth - 1, prefix: self.prefix >> 2 }
    }

    /// Ancestor at a shallower `depth` (`depth <= self.depth`).
    pub fn ancestor(self, depth: u8) -> CellKey {
        CellKey { depth, prefix: self.prefix >> (2 * (self.depth - depth) as u32) }
    }

    /// Child 0-3 (Morton bit pair: u high, v low).
    pub fn child(self, q: u64) -> CellKey {
        CellKey { depth: self.depth + 1, prefix: (self.prefix << 2) | q }
    }

    pub fn diamond(self) -> u8 {
        (self.prefix >> (2 * self.depth)) as u8
    }

    /// Cell-grid coordinates (cu, cv) within the diamond, each `depth` bits.
    pub fn grid(self) -> (u64, u64) {
        let m = self.prefix & ((1u64 << (2 * self.depth)) - 1);
        // morton helpers operate on the 60-bit layout: left-align first.
        let aligned = m << (60 - 2 * self.depth as u32);
        (
            morton_compact(aligned >> 1) >> (30 - self.depth as u32),
            morton_compact(aligned) >> (30 - self.depth as u32),
        )
    }

    pub fn path(self, layer: &str) -> String {
        format!("{layer}/{:02}/{:016x}.vsf.zst", self.depth, self.prefix)
    }

    /// Diamond-UV rectangle covered by this cell.
    pub fn uv_rect(self) -> (f64, f64, f64) {
        let (cu, cv) = self.grid();
        let size = 1.0 / (1u64 << self.depth) as f64;
        (cu as f64 * size, cv as f64 * size, size)
    }
}

/// Global texel coordinates of a Coord at a given base depth: the texel grid
/// is the cell grid times TEX.
fn texel_of(c: Coord, depth: u8) -> (u8, u64, f64, f64) {
    let (iu, iv) = c.uv();
    let shift = 30 - depth as u32 - TEX_BITS as u32;
    let scale = 1.0 / (1u64 << shift) as f64;
    (c.diamond(), c.raw() >> (60 - 2 * depth as u32), iu as f64 * scale, iv as f64 * scale)
}

// ==================== LINE LAYER ====================

pub struct LineCell {
    pub class: Vec<u8>, // 0 = empty; else RoadClass as u8 + 1
    pub cov: Vec<u8>,
}

impl LineCell {
    fn new() -> LineCell {
        LineCell { class: vec![0; TEX * TEX], cov: vec![0; TEX * TEX] }
    }
}

/// Stamp every feature 1 texel wide into base-depth cells, then build the
/// pyramid up to `min_depth`. Returns cells keyed by (depth, prefix).
pub fn bake_lines(
    feats: &[Road],
    base_depth: u8,
    min_depth: u8,
) -> HashMap<CellKey, LineCell> {
    assert!(base_depth as u32 + TEX_BITS as u32 <= 30);
    let mut cells: HashMap<CellKey, LineCell> = HashMap::new();
    let mut cross_diamond = 0usize;
    for road in feats {
        let class = road.class as u8 + 1;
        let mut prev: Option<(u8, f64, f64)> = None;
        for &(lat, lon) in &road.pts {
            let c = Coord::from_lat_lon(lat as f64, lon as f64);
            let (d, _, gu, gv) = texel_of(c, base_depth);
            if let Some((pd, pu, pv)) = prev {
                if pd == d {
                    stamp_segment(&mut cells, base_depth, d, pu, pv, gu, gv, class);
                } else {
                    cross_diamond += 1;
                }
            }
            prev = Some((d, gu, gv));
        }
    }
    if cross_diamond > 0 {
        eprintln!("  (skipped {cross_diamond} cross-diamond segments)");
    }
    // Pyramid: average coverage, keep the most major class present.
    let mut depth = base_depth;
    while depth > min_depth {
        let child_keys: Vec<CellKey> =
            cells.keys().filter(|k| k.depth == depth).copied().collect();
        for key in child_keys {
            let parent = key.parent();
            let q = key.prefix & 3;
            let (ox, oy) = (((q >> 1) & 1) as usize * (TEX / 2), (q & 1) as usize * (TEX / 2));
            // Read child quad sums first (borrow discipline), then write.
            let mut patch = vec![(0u8, 0u16); (TEX / 2) * (TEX / 2)];
            {
                let child = &cells[&key];
                for ty in 0..TEX / 2 {
                    for tx in 0..TEX / 2 {
                        let mut covsum = 0u16;
                        let mut best = 0u8;
                        for (sx, sy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            let i = (ty * 2 + sy) * TEX + tx * 2 + sx;
                            covsum += child.cov[i] as u16;
                            let cl = child.class[i];
                            if cl != 0 && (best == 0 || cl < best) {
                                best = cl;
                            }
                        }
                        patch[ty * (TEX / 2) + tx] = (best, covsum / 4);
                    }
                }
            }
            let p = cells.entry(parent).or_insert_with(LineCell::new);
            for ty in 0..TEX / 2 {
                for tx in 0..TEX / 2 {
                    let (best, cov) = patch[ty * (TEX / 2) + tx];
                    let i = (oy + ty) * TEX + ox + tx;
                    p.cov[i] = cov as u8;
                    if best != 0 && (p.class[i] == 0 || best < p.class[i]) {
                        p.class[i] = best;
                    }
                }
            }
        }
        depth -= 1;
    }
    cells
}

#[allow(clippy::too_many_arguments)]
fn stamp_segment(
    cells: &mut HashMap<CellKey, LineCell>,
    depth: u8,
    diamond: u8,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    class: u8,
) {
    let steps = ((x1 - x0).abs().max((y1 - y0).abs()) * 2.0).ceil().max(1.0) as usize;
    let grid = 1u64 << depth;
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let gx = x0 + (x1 - x0) * t;
        let gy = y0 + (y1 - y0) * t;
        let (cu, tu) = ((gx as u64) / TEX as u64, (gx as u64) % TEX as u64);
        let (cv, tv) = ((gy as u64) / TEX as u64, (gy as u64) % TEX as u64);
        if cu >= grid || cv >= grid {
            continue;
        }
        let m = (morton_spread(cu << (30 - depth as u32)) << 1)
            | morton_spread(cv << (30 - depth as u32));
        let prefix = ((diamond as u64) << (2 * depth)) | (m >> (60 - 2 * depth as u32));
        let cell = cells
            .entry(CellKey { depth, prefix })
            .or_insert_with(LineCell::new);
        let i = tv as usize * TEX + tu as usize;
        cell.cov[i] = 255;
        if cell.class[i] == 0 || class < cell.class[i] {
            cell.class[i] = class;
        }
    }
}

// ==================== DEM LAYER ====================

pub struct DemCell {
    pub elev: Vec<f32>,
    pub ge: Vec<f32>,
    pub gn: Vec<f32>,
}

/// Bake dem cells covering `keys` from the source store.
pub fn bake_dem(dem: &DemStore, keys: &[CellKey]) -> Vec<(CellKey, DemCell)> {
    keys.par_iter()
        .map(|&key| {
            let (u0, v0, size) = key.uv_rect();
            let d = key.diamond();
            let step = size / TEX as f64;
            let mut cell = DemCell {
                elev: vec![f32::NAN; TEX * TEX],
                ge: vec![0.0; TEX * TEX],
                gn: vec![0.0; TEX * TEX],
            };
            for ty in 0..TEX {
                for tx in 0..TEX {
                    let u = u0 + (tx as f64 + 0.5) * step;
                    let v = v0 + (ty as f64 + 0.5) * step;
                    let (lat, lon) = uv_to_lat_lon(d, u, v);
                    if let Some((e, (ge, gn))) = dem.elev_and_gradient(lat, lon) {
                        let i = ty * TEX + tx;
                        cell.elev[i] = e;
                        cell.ge[i] = ge;
                        cell.gn[i] = gn;
                    }
                }
            }
            (key, cell)
        })
        .collect()
}

/// Every depth from `base.depth - 1` down to `min_depth`, each cell the
/// box-filter of its four children (mean of the children's elevations and
/// gradients, no-data ignored). Building the pyramid from the base set is
/// what guarantees every ancestor of a baked cell exists.
pub fn dem_pyramid(base: &[(CellKey, DemCell)], min_depth: u8) -> Vec<(CellKey, DemCell)> {
    let mut out: Vec<(CellKey, DemCell)> = Vec::new();
    // Start index of the most recent level inside `out` (None = base).
    let mut last: Option<usize> = None;
    loop {
        let cur: &[(CellKey, DemCell)] = match last {
            None => base,
            Some(i) => &out[i..],
        };
        let Some(&(k0, _)) = cur.first() else { break };
        if k0.depth <= min_depth {
            break;
        }
        let mut parents: Vec<CellKey> = cur.iter().map(|(k, _)| k.parent()).collect();
        parents.sort_by_key(|k| k.prefix);
        parents.dedup();
        let by_key: std::collections::HashMap<(u8, u64), &DemCell> =
            cur.iter().map(|(k, c)| ((k.depth, k.prefix), c)).collect();
        let next: Vec<(CellKey, DemCell)> = parents
            .par_iter()
            .map(|&p| {
                let mut cell = DemCell {
                    elev: vec![f32::NAN; TEX * TEX],
                    ge: vec![0.0; TEX * TEX],
                    gn: vec![0.0; TEX * TEX],
                };
                for q in 0..4u64 {
                    let ck = p.child(q);
                    let Some(child) = by_key.get(&(ck.depth, ck.prefix)) else { continue };
                    let (ox, oy) = (((q >> 1) as usize) * (TEX / 2), ((q & 1) as usize) * (TEX / 2));
                    for ty in 0..TEX / 2 {
                        for tx in 0..TEX / 2 {
                            let (mut e, mut ge, mut gn, mut n) = (0.0f32, 0.0f32, 0.0f32, 0u32);
                            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                                let i = (2 * ty + dy) * TEX + 2 * tx + dx;
                                if !child.elev[i].is_nan() {
                                    e += child.elev[i];
                                    ge += child.ge[i];
                                    gn += child.gn[i];
                                    n += 1;
                                }
                            }
                            if n > 0 {
                                let i = (oy + ty) * TEX + ox + tx;
                                cell.elev[i] = e / n as f32;
                                cell.ge[i] = ge / n as f32;
                                cell.gn[i] = gn / n as f32;
                            }
                        }
                    }
                }
                (p, cell)
            })
            .collect();
        last = Some(out.len());
        out.extend(next);
    }
    out
}

/// All cells at `depth` that intersect a lat/lon bbox. Scanned at half-cell
/// resolution, never coarser than an eighth of the box — a slanted rhombus
/// cell can cut through a box without containing any corner of it.
pub fn cells_covering(lat0: f64, lon0: f64, lat1: f64, lon1: f64, depth: u8) -> Vec<CellKey> {
    let mut keys = std::collections::HashSet::new();
    // Half a cell edge in degrees of latitude, as the scan step.
    let step = ((7054_000.0 / (1u64 << depth) as f64) / 111_320.0 / 2.0)
        .min((lat1 - lat0) / 8.0)
        .min((lon1 - lon0) / 8.0)
        .max(1e-4);
    let mut lat = lat0;
    while lat <= lat1 + step {
        let mut lon = lon0;
        while lon <= lon1 + step {
            keys.insert(CellKey::containing(
                Coord::from_lat_lon(lat.min(lat1), lon.min(lon1)),
                depth,
            ));
            lon += step;
        }
        lat += step;
    }
    let mut v: Vec<CellKey> = keys.into_iter().collect();
    v.sort_by_key(|k| k.prefix);
    v
}

// ==================== VSF I/O ====================

pub fn write_line_cell(out: &Path, key: CellKey, cell: &LineCell) -> Result<(), String> {
    let bytes = VsfBuilder::new()
        .add_section(
            "linecell",
            vec![
                ("depth".to_string(), VsfType::u(key.depth as usize, false)),
                ("prefix".to_string(), VsfType::u(key.prefix as usize, false)),
                ("class".to_string(), VsfType::t_u3(Tensor::new(vec![TEX, TEX], cell.class.clone()))),
                ("cov".to_string(), VsfType::t_u3(Tensor::new(vec![TEX, TEX], cell.cov.clone()))),
            ],
        )
        .build()
        .map_err(|e| format!("linecell build: {e:?}"))?;
    write_file(out, &key.path("line"), &bytes)
}

pub fn write_dem_cell(out: &Path, key: CellKey, cell: &DemCell) -> Result<(), String> {
    let bytes = VsfBuilder::new()
        .add_section(
            "demcell",
            vec![
                ("depth".to_string(), VsfType::u(key.depth as usize, false)),
                ("prefix".to_string(), VsfType::u(key.prefix as usize, false)),
                ("elev".to_string(), VsfType::t_f5(Tensor::new(vec![TEX, TEX], cell.elev.clone()))),
                ("ge".to_string(), VsfType::t_f5(Tensor::new(vec![TEX, TEX], cell.ge.clone()))),
                ("gn".to_string(), VsfType::t_f5(Tensor::new(vec![TEX, TEX], cell.gn.clone()))),
            ],
        )
        .build()
        .map_err(|e| format!("demcell build: {e:?}"))?;
    write_file(out, &key.path("dem"), &bytes)
}

fn write_file(out: &Path, rel: &str, bytes: &[u8]) -> Result<(), String> {
    let path = out.join(rel);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    // Whole-file zstd: VSF internals untouched, the R2 layout keeps the
    // .vsf.zst names, and near-empty line cells shrink ~100x.
    let z = zstd::encode_all(bytes, 3).map_err(|e| e.to_string())?;
    std::fs::write(&path, z).map_err(|e| format!("{}: {e}", path.display()))
}

/// Read any cell file's fields (width-agnostic, per VSF doctrine).
pub fn read_cell_fields(path: &Path) -> Result<HashMap<String, VsfType>, String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode_cell_fields(&data).map_err(|e| format!("{}: {e}", path.display()))
}

/// Decode a cell from raw file bytes (zstd or plain VSF) — the loader
/// thread's entry point; no filesystem coupling.
pub fn decode_cell_fields(data: &[u8]) -> Result<HashMap<String, VsfType>, String> {
    let plain: Vec<u8>;
    let data = if data.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
        plain = zstd::decode_all(data).map_err(|e| e.to_string())?;
        &plain[..]
    } else {
        data
    };
    let (header, end) = vsf::VsfHeader::decode(data).map_err(|e| e.to_string())?;
    let section = header.primary_section(data, end).map_err(|e| e.to_string())?;
    Ok(section
        .fields
        .into_iter()
        .filter_map(|f| f.values.into_iter().next().map(|v| (f.name, v)))
        .collect())
}

pub fn tensor_u8(v: &VsfType) -> Option<Vec<u8>> {
    match v {
        VsfType::t_u3(t) => Some(t.data.clone()),
        VsfType::v_u3(t) => Some(t.data.clone()),
        _ => None,
    }
}

pub fn tensor_f32(v: &VsfType) -> Option<Vec<f32>> {
    match v {
        VsfType::t_f5(t) => Some(t.data.clone()),
        VsfType::v_f5(t) => Some(t.data.clone()),
        VsfType::t_f6(t) => Some(t.data.iter().map(|&x| x as f32).collect()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_key_round_trips() {
        let c = Coord::from_lat_lon(46.2024, -121.4909);
        for depth in [6u8, 10, 13] {
            let key = CellKey::containing(c, depth);
            assert_eq!(key.diamond(), c.diamond());
            let (u0, v0, size) = key.uv_rect();
            let (iu, iv) = c.uv();
            let (u, v) = (iu as f64 / 2f64.powi(30), iv as f64 / 2f64.powi(30));
            assert!(u >= u0 && u < u0 + size, "u {u} not in [{u0}, {})", u0 + size);
            assert!(v >= v0 && v < v0 + size);
            assert_eq!(key.parent().child((key.prefix & 3) as u64), key);
        }
    }

    #[test]
    fn stamped_line_survives_pyramid() {
        // One diagonal trail across a cell at depth 12 -> coverage must
        // appear at 12 and fade (not vanish) at 10.
        let road = Road {
            class: mahere_osm::RoadClass::Path,
            pts: (0..200)
                .map(|i| {
                    let t = i as f32 / 199.0;
                    (46.15 + t * 0.02, -121.52 + t * 0.03)
                })
                .collect(),
        };
        let cells = bake_lines(&[road], 12, 10);
        let base_cov: u64 = cells
            .iter()
            .filter(|(k, _)| k.depth == 12)
            .map(|(_, c)| c.cov.iter().map(|&x| x as u64).sum::<u64>())
            .sum();
        let top_cov: u64 = cells
            .iter()
            .filter(|(k, _)| k.depth == 10)
            .map(|(_, c)| c.cov.iter().map(|&x| x as u64).sum::<u64>())
            .sum();
        assert!(base_cov > 0, "base stamping produced nothing");
        assert!(top_cov > 0, "pyramid lost the line");
        // Box filtering conserves total coverage (up to edge rounding).
        let ratio = base_cov as f64 / top_cov as f64;
        assert!(
            (0.7..=1.5).contains(&(ratio / 16.0)),
            "coverage not conserved through 2 levels: base {base_cov} top {top_cov}"
        );
    }
}
