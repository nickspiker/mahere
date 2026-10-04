//! The organic-pixel-grid terrain engine: a reservoir of exact point samples
//! of the DEM, filled by stratified quadtree refinement and splatted through
//! the camera each frame.
//!
//! Samples store truths — position, elevation, gradient — never styled
//! color. Hillshade is computed at splat time from the gradient, so the sun
//! can move (and the palette change) by re-splatting the reservoir without a
//! single re-evaluation. Camera motion reprojects existing samples; only
//! unseen screen cells are evaluated, coarsest level first, each level
//! quartering its cells and filling the missing three, then extra per-pixel
//! passes accumulate anti-aliasing.

use mahere_dem::DemStore;
use rayon::prelude::*;

use crate::Camera;

/// Coarsest refinement level: cells of 2^COARSEST pixels.
const COARSEST: u8 = 5;
/// Extra per-pixel jittered passes after level 0 completes (AA).
const AA_PASSES: u8 = 2;
/// Evaluation budget per tick, keeps frames interactive while converging.
const MAX_EVAL_PER_TICK: usize = 120_000;
/// Reservoir cap; eviction keeps the on-screen neighborhood.
const MAX_SAMPLES: usize = 5_000_000;

pub struct Sample {
    lat: f64,
    lon: f64,
    /// NaN = evaluated but no data (outside DEM coverage).
    elev: f32,
    ge: f32,
    gn: f32,
    level: u8,
}

pub struct Terrain {
    dem: DemStore,
    samples: Vec<Sample>,
    level: u8,
    aa_pass: u8,
    aa_cursor: usize,
    converged: bool,
    generation: u64,
    jitter_gen: u64,
    // Reused splat buffers for the fine-level accumulation.
    acc_rgb: Vec<[u32; 3]>,
    acc_n: Vec<u16>,
}

impl Terrain {
    pub fn new(dem: DemStore) -> Terrain {
        Terrain {
            dem,
            samples: Vec::new(),
            level: COARSEST,
            aa_pass: 0,
            aa_cursor: 0,
            converged: false,
            generation: 0,
            jitter_gen: 0,
            acc_rgb: Vec::new(),
            acc_n: Vec::new(),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Camera moved: restart refinement against the new view. Existing
    /// samples stay valid (they're exact); only coverage must be re-derived.
    pub fn note_camera(&mut self, w: usize, h: usize, cam: &Camera) {
        self.level = COARSEST;
        self.aa_pass = 0;
        self.aa_cursor = 0;
        self.converged = false;
        self.jitter_gen = self.jitter_gen.wrapping_add(1);
        if self.samples.len() > MAX_SAMPLES {
            // Keep what the camera can see (with margin); exact samples
            // off-screen are cheap to regenerate relative to their memory.
            let (mw, mh) = (w as f64 * 0.3, h as f64 * 0.3);
            self.samples.retain(|s| {
                let (x, y) = cam.geo_to_screen(s.lat, s.lon, w, h);
                x > -mw && x < w as f64 + mw && y > -mh && y < h as f64 + mh
            });
            self.samples.truncate(MAX_SAMPLES);
        }
    }

    /// Evaluate one budgeted batch toward convergence. Returns true if any
    /// samples were added (the host should redraw).
    pub fn tick(&mut self, w: usize, h: usize, cam: &Camera) -> bool {
        if self.converged || w == 0 || h == 0 {
            return false;
        }
        // Collect this batch's evaluation points (screen px centers+jitter).
        let mut points: Vec<(f64, f64, u8)> = Vec::new(); // (px, py, level)
        if self.level > 0 || self.aa_pass == 0 && self.level == 0 {
            let cs = 1usize << self.level;
            let (cw, ch) = (w.div_ceil(cs), h.div_ceil(cs));
            let mut occupied = vec![false; cw * ch];
            for s in &self.samples {
                let (x, y) = cam.geo_to_screen(s.lat, s.lon, w, h);
                if x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h {
                    occupied[(y as usize / cs) * cw + (x as usize / cs)] = true;
                }
            }
            for cy in 0..ch {
                for cx in 0..cw {
                    if occupied[cy * cw + cx] {
                        continue;
                    }
                    let (jx, jy) = jitter(cx as u64, cy as u64, self.level, self.jitter_gen);
                    let px = (cx * cs) as f64 + jx * cs as f64;
                    let py = (cy * cs) as f64 + jy * cs as f64;
                    if px < w as f64 && py < h as f64 {
                        points.push((px, py, self.level));
                    }
                    if points.len() >= MAX_EVAL_PER_TICK {
                        break;
                    }
                }
                if points.len() >= MAX_EVAL_PER_TICK {
                    break;
                }
            }
            if points.is_empty() {
                // Level complete: descend, or enter AA.
                if self.level > 0 {
                    self.level -= 1;
                } else {
                    self.aa_pass = 1;
                    self.aa_cursor = 0;
                }
                return self.tick(w, h, cam);
            }
        } else {
            // AA passes: one extra jittered sample per pixel per pass.
            if self.aa_pass > AA_PASSES {
                self.converged = true;
                return false;
            }
            let total = w * h;
            let end = (self.aa_cursor + MAX_EVAL_PER_TICK).min(total);
            for i in self.aa_cursor..end {
                let (pxi, pyi) = (i % w, i / w);
                let (jx, jy) =
                    jitter(pxi as u64, pyi as u64, COARSEST + self.aa_pass, self.jitter_gen);
                points.push((pxi as f64 + jx, pyi as f64 + jy, 0));
            }
            self.aa_cursor = end;
            if end == total {
                self.aa_pass += 1;
                self.aa_cursor = 0;
            }
            if points.is_empty() {
                self.converged = true;
                return false;
            }
        }

        // Evaluate the batch in parallel: exact world point -> DEM truth.
        let dem = &self.dem;
        let new: Vec<Sample> = points
            .par_iter()
            .map(|&(px, py, level)| {
                let (lat, lon) = cam.screen_to_geo(px, py, w, h);
                match dem.elev_and_gradient(lat, lon) {
                    Some((elev, (ge, gn))) => Sample { lat, lon, elev, ge, gn, level },
                    None => Sample { lat, lon, elev: f32::NAN, ge: 0.0, gn: 0.0, level },
                }
            })
            .collect();
        self.samples.extend(new);
        self.generation += 1;
        true
    }

    /// Draw the reservoir: coarse levels as rects (painter, coarse first, so
    /// refinement shows no holes), finest accumulated and averaged for AA.
    /// Sun parameters apply HERE — relighting never re-evaluates.
    pub fn splat(
        &mut self,
        canvas: &mut [u32],
        w: usize,
        h: usize,
        cam: &Camera,
        sun_az_deg: f32,
        sun_alt_deg: f32,
    ) {
        let (saz, caz) = sun_az_deg.to_radians().sin_cos();
        let (salt, calt) = sun_alt_deg.to_radians().sin_cos();
        let sun = [saz * calt, caz * calt, salt]; // (east, north, up)

        self.acc_rgb.clear();
        self.acc_rgb.resize(w * h, [0; 3]);
        self.acc_n.clear();
        self.acc_n.resize(w * h, 0);

        for level in (1..=COARSEST).rev() {
            for s in self.samples.iter().filter(|s| s.level == level) {
                let (x, y) = cam.geo_to_screen(s.lat, s.lon, w, h);
                let cs = 1isize << s.level;
                let x0 = (x as isize - cs / 2).clamp(0, w as isize);
                let y0 = (y as isize - cs / 2).clamp(0, h as isize);
                let x1 = (x as isize + cs / 2 + 1).clamp(0, w as isize);
                let y1 = (y as isize + cs / 2 + 1).clamp(0, h as isize);
                let rgb = shade(s, sun);
                for py in y0..y1 {
                    let row = &mut canvas[py as usize * w..py as usize * w + w];
                    for px in x0..x1 {
                        row[px as usize] = rgb;
                    }
                }
            }
        }
        for s in self.samples.iter().filter(|s| s.level == 0) {
            let (x, y) = cam.geo_to_screen(s.lat, s.lon, w, h);
            if x < 0.0 || y < 0.0 || x >= w as f64 || y >= h as f64 {
                continue;
            }
            let i = y as usize * w + x as usize;
            let rgb = shade(s, sun);
            let a = &mut self.acc_rgb[i];
            a[0] += (rgb >> 16) & 255;
            a[1] += (rgb >> 8) & 255;
            a[2] += rgb & 255;
            self.acc_n[i] += 1;
        }
        for (i, &n) in self.acc_n.iter().enumerate() {
            if n > 0 {
                let a = self.acc_rgb[i];
                let nn = n as u32;
                canvas[i] = ((a[0] / nn) << 16) | ((a[1] / nn) << 8) | (a[2] / nn);
            }
        }
    }
}

/// Style lives here, applied per splat: hypsometric tint shaded by Lambert
/// hillshade from the sample's stored gradient.
fn shade(s: &Sample, sun: [f32; 3]) -> u32 {
    if s.elev.is_nan() {
        return 0x12141A; // no data: background
    }
    if s.elev < 0.5 {
        // Sea level: Puget Sound and the strait read as water.
        return 0x1A3A52;
    }
    // Hypsometric ramp stops (elevation m, rgb).
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.0, [72.0, 96.0, 60.0]),
        (500.0, [110.0, 112.0, 70.0]),
        (1200.0, [138.0, 116.0, 84.0]),
        (2200.0, [160.0, 152.0, 146.0]),
        (3000.0, [238.0, 240.0, 245.0]),
    ];
    let mut tint = STOPS[STOPS.len() - 1].1;
    for i in 0..STOPS.len() - 1 {
        let (e0, c0) = STOPS[i];
        let (e1, c1) = STOPS[i + 1];
        if s.elev < e1 {
            let t = ((s.elev - e0) / (e1 - e0)).clamp(0.0, 1.0);
            tint = [
                c0[0] + (c1[0] - c0[0]) * t,
                c0[1] + (c1[1] - c0[1]) * t,
                c0[2] + (c1[2] - c0[2]) * t,
            ];
            break;
        }
    }
    // Lambert against the surface normal from the stored gradient.
    let inv = 1.0 / (1.0 + s.ge * s.ge + s.gn * s.gn).sqrt();
    let n = [-s.ge * inv, -s.gn * inv, inv];
    let diffuse = (n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2]).max(0.0);
    let b = 0.30 + 0.70 * diffuse;
    let r = (tint[0] * b) as u32;
    let g = (tint[1] * b) as u32;
    let bl = (tint[2] * b) as u32;
    (r << 16) | (g << 8) | bl
}

/// Deterministic per-cell jitter in [0, 1)^2 (splitmix-style hash).
fn jitter(x: u64, y: u64, level: u8, seed: u64) -> (f64, f64) {
    let mut v = x
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(y.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((level as u64) << 32)
        .wrapping_add(seed.wrapping_mul(0x94D0_49BB_1331_11EB));
    v ^= v >> 30;
    v = v.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v ^= v >> 27;
    v = v.wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^= v >> 31;
    let a = (v & 0xFFFF_FFFF) as f64 / 4_294_967_296.0;
    let b = (v >> 32) as f64 / 4_294_967_296.0;
    (a, b)
}
