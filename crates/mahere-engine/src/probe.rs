//! The real light: the front camera painting a sphere around the phone. The camera looks out of the screen; the orientation sensor says where the screen points; so every camera frame paints its patch of a world-fixed sphere, and turning the phone paints the rest. The sphere is the icosahedron at two subdivisions, 320 triangles of about 16°, which is finer than the nine harmonics the renderer lights by can tell apart, so the sun lands where it is. Unpainted triangles stay dark until the camera gets there: honest at every moment (Nick 2026-10-08).
//!
//! Units are integer and absolute: raw sensor counts above black, shifted up by the frame's stop (the exposure ladder is powers of two of one base), so frames at any exposure paint the same sphere. The level is set by the sphere itself, the brightest a normal can be lit, so walking indoors and painting over the sun brings the level down with it.
//!
//! Colour: the sensor's raw samples through Android's `SENSOR_COLOR_TRANSFORM` (XYZ to camera, a 1931 characterisation under the reference illuminant) inverted, then XYZ to VSF RGB with no adaptation: a blue sky lights the shadows blue, a tungsten lamp lights the map orange. The light's colour is the point.

use crate::sh::{Sh9, basis};
use mahere_coord::{Coord, uv_to_lat_lon};

/// Bayer colour filter arrangement, as Android numbers them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cfa {
    Rggb,
    Grbg,
    Gbrg,
    Bggr,
}

impl Cfa {
    pub fn from_android(n: i32) -> Option<Cfa> {
        match n {
            0 => Some(Cfa::Rggb),
            1 => Some(Cfa::Grbg),
            2 => Some(Cfa::Gbrg),
            3 => Some(Cfa::Bggr),
            _ => None,
        }
    }
}

/// A raw sensor frame: 16-bit little-endian samples, or Android's packed 10-bit (four samples' high bytes then a byte of their low two bits, which reads in five eighths of the bytes), `row_stride` bytes per row, the sensor's `white` level and its `black` pedestal at each of the four Bayer positions (row-major within the quad: (0,0), (1,0), (0,1), (1,1)). The pedestals differ a little per channel, and a frame near black is all pedestal, so each is taken from its own.
pub struct Raw<'a> {
    pub data: &'a [u8],
    pub w: usize,
    pub h: usize,
    pub row_stride: usize,
    pub packed10: bool,
    pub cfa: Cfa,
    pub black: [u16; 4],
    pub white: u16,
}

/// What the exposure loop steers on: the fraction of raw samples at white, and the level (as a fraction of white) under which 99.9% of them lie. A lamp or the sun in the frame clips at any exposure a phone has, so the loop asks for almost nothing clipped rather than nothing, and for the bulk of the frame well up the range.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub clipped: f32,
    pub p999: f32,
}

/// A binned frame: camera RGB per bin as raw counts above black (mean over the bin), how many of each bin's raw samples were at white, and the frame's exposure statistics.
pub struct Binned {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<[u32; 3]>,
    pub clipped: Vec<u16>,
    pub stats: Stats,
}

impl Binned {
    /// Fill the bins this frame clipped from a shorter exposure of the same scene, both brought to absolute units (each shifted up by its own stop): a lamp that is white at the long exposure gets its real brightness from the short one. A clipped bin's mean is a floor; the short frame's is the truth up to its own clip.
    pub fn fill_clipped(&mut self, short: &Binned, stop: u32, short_stop: u32) {
        if (short.w, short.h) != (self.w, self.h) {
            return;
        }
        for i in 0..self.rgb.len() {
            if self.clipped[i] > 0 {
                let s = short.rgb[i];
                self.rgb[i] = [s[0] << short_stop >> stop, s[1] << short_stop >> stop, s[2] << short_stop >> stop];
            }
        }
    }

    /// To absolute units: shifted up by the frame's stop.
    pub fn shift(&mut self, stop: u32) {
        for p in self.rgb.iter_mut() {
            *p = [p[0] << stop, p[1] << stop, p[2] << stop];
        }
    }
}

/// Bin a raw frame to about `cols` columns (each 2×2 Bayer quad is one sample, then an even box over those), counting the samples at white in each bin, with the frame's exposure statistics. Integers all the way through the samples, in two passes a row: the sums and the clip count over every sample (no scatter, so it vectorises), then the histogram over every sixteenth, which is plenty for a percentile.
pub fn bin(raw: &Raw, cols: usize) -> Binned {
    let (qw, qh) = (raw.w / 2, raw.h / 2);
    // An even number of quads a bin, so a bin is whole five-byte groups of a packed row.
    let f = ((qw / cols.max(1)) & !1).max(2);
    let (w, h) = (qw / f, qh / f);
    // Where red and blue sit in the quad, as positions 0..4 row-major; green is the other two.
    let (rp, bp) = match raw.cfa {
        Cfa::Rggb => (0usize, 3usize),
        Cfa::Grbg => (1, 2),
        Cfa::Gbrg => (2, 1),
        Cfa::Bggr => (3, 0),
    };
    let clip_at = raw.white.saturating_sub(2);
    // The histogram bins the sensor's own bits: a shift takes a sample to 0..255.
    let shift = (16 - raw.white.leading_zeros()).saturating_sub(8);
    use rayon::prelude::*;
    let rows: Vec<(Vec<[u32; 4]>, Vec<u16>, [u32; 256])> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut sums = vec![[0u32; 4]; w];
            let mut clipped = vec![0u16; w];
            let mut hist = [0u32; 256];
            for j in 0..f {
                let qy = (y * f + j) * 2;
                if raw.packed10 {
                    // Whole five-byte groups, a bin a run of f/2 of them: the high bytes shifted up, the low bits picked out of the fifth, even columns to one sum and odd to the other.
                    let row_bytes = raw.w / 4 * 5;
                    let row0 = &raw.data[qy * raw.row_stride..][..row_bytes];
                    let row1 = &raw.data[(qy + 1) * raw.row_stride..][..row_bytes];
                    let gb = f / 2 * 5;
                    for x in 0..w {
                        let (mut s, mut clip) = ([0u32; 4], 0u32);
                        for (k, row) in [row0, row1].into_iter().enumerate() {
                            for c in row[x * gb..x * gb + gb].chunks_exact(5) {
                                let lo = c[4];
                                let v0 = ((c[0] as u16) << 2) | (lo & 3) as u16;
                                let v1 = ((c[1] as u16) << 2) | ((lo >> 2) & 3) as u16;
                                let v2 = ((c[2] as u16) << 2) | ((lo >> 4) & 3) as u16;
                                let v3 = ((c[3] as u16) << 2) | ((lo >> 6) & 3) as u16;
                                s[2 * k] += v0 as u32 + v2 as u32;
                                s[2 * k + 1] += v1 as u32 + v3 as u32;
                                clip += (v0 >= clip_at) as u32 + (v1 >= clip_at) as u32 + (v2 >= clip_at) as u32 + (v3 >= clip_at) as u32;
                            }
                        }
                        for k in 0..4 {
                            sums[x][k] += s[k];
                        }
                        clipped[x] = clipped[x].saturating_add(clip.min(u16::MAX as u32) as u16);
                    }
                    for c in row0.chunks_exact(20) {
                        hist[((((c[0] as u16) << 2) | (c[4] & 3) as u16) >> shift) as usize] += 1;
                    }
                    continue;
                }
                let row0 = &raw.data[qy * raw.row_stride..][..raw.w * 2];
                let row1 = &raw.data[(qy + 1) * raw.row_stride..][..raw.w * 2];
                for x in 0..w {
                    let x0 = x * f * 4;
                    let (mut s, mut clip) = ([0u32; 4], 0u32);
                    for c in row0[x0..x0 + f * 4].chunks_exact(4) {
                        let (a, b) = (u16::from_le_bytes([c[0], c[1]]), u16::from_le_bytes([c[2], c[3]]));
                        s[0] += a as u32;
                        s[1] += b as u32;
                        clip += (a >= clip_at) as u32 + (b >= clip_at) as u32;
                    }
                    for c in row1[x0..x0 + f * 4].chunks_exact(4) {
                        let (a, b) = (u16::from_le_bytes([c[0], c[1]]), u16::from_le_bytes([c[2], c[3]]));
                        s[2] += a as u32;
                        s[3] += b as u32;
                        clip += (a >= clip_at) as u32 + (b >= clip_at) as u32;
                    }
                    for k in 0..4 {
                        sums[x][k] += s[k];
                    }
                    clipped[x] = clipped[x].saturating_add(clip.min(u16::MAX as u32) as u16);
                }
                for c in row0.chunks_exact(32) {
                    hist[((u16::from_le_bytes([c[0], c[1]]) >> shift) as usize).min(255)] += 1;
                }
            }
            (sums, clipped, hist)
        })
        .collect();
    let count = (f * f) as u32;
    let mut out = Vec::with_capacity(w * h);
    let mut clipped = Vec::with_capacity(w * h);
    let mut hist = [0u32; 256];
    for (sums, c, hh) in rows {
        for s in sums {
            // Each position's black off its own sum, then the mean over the bin; the greens together are two positions.
            let lin = |k: usize| s[k].saturating_sub(count * raw.black[k] as u32);
            let g = (lin(0) + lin(1) + lin(2) + lin(3) - lin(rp) - lin(bp)) / 2;
            out.push([lin(rp) / count, g / count, lin(bp) / count]);
        }
        clipped.extend(c);
        for k in 0..256 {
            hist[k] += hh[k];
        }
    }
    let total: u32 = hist.iter().sum();
    let clipped_frac = clipped.iter().map(|&c| c as u64).sum::<u64>() as f32 / (raw.w * raw.h).max(1) as f32;
    let mut seen = 0u64;
    let mut p999 = 1.0;
    for (i, &c) in hist.iter().enumerate() {
        seen += c as u64;
        if seen * 1000 >= total as u64 * 999 {
            p999 = i as f32 / 255.0;
            break;
        }
    }
    Binned { w, h, rgb: out, clipped, stats: Stats { clipped: clipped_frac, p999 } }
}

/// Camera RGB to VSF RGB in fixed point: `xyz_to_cam` is Android's row-major XYZ→camera matrix, inverted; XYZ then goes to VSF RGB colorimetrically (Illuminant E to white, no adaptation). The product is one matrix, held to twelve fractional bits; negative light clamps to zero.
pub fn to_vsf(rgb: &mut [[u32; 3]], xyz_to_cam: &[f32; 9]) {
    let Some(c2x) = invert3(xyz_to_cam) else { return };
    let x2v = &vsf::colour::XYZ2VSF_RGB;
    // Row-major product of XYZ→VSF (column-major in vsf) and camera→XYZ (row-major).
    let mut m = [0i64; 9];
    for r in 0..3 {
        for c in 0..3 {
            let v: f32 = (0..3).map(|k| x2v[r + 3 * k] * c2x[3 * k + c]).sum();
            m[3 * r + c] = (v * 4096.0).round() as i64;
        }
    }
    for p in rgb.iter_mut() {
        let x = [p[0] as i64, p[1] as i64, p[2] as i64];
        let mut o = [0u32; 3];
        for r in 0..3 {
            o[r] = ((m[3 * r] * x[0] + m[3 * r + 1] * x[1] + m[3 * r + 2] * x[2]) >> 12).max(0) as u32;
        }
        *p = o;
    }
}

fn invert3(m: &[f32; 9]) -> Option<[f32; 9]> {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-12 {
        return None;
    }
    let k = 1.0 / det;
    Some([(e * i - f * h) * k, (c * h - b * i) * k, (b * f - c * e) * k, (f * g - d * i) * k, (a * i - c * g) * k, (c * d - a * f) * k, (d * h - e * g) * k, (b * g - a * h) * k, (a * e - b * d) * k])
}

/// Turn a sensor-oriented binned frame upright: `orientation` is Android's `SENSOR_ORIENTATION`, the clockwise degrees the sensor image must turn to stand upright in portrait. Returns the upright frame and its size.
pub fn upright(w: usize, h: usize, rgb: &[[u32; 3]], orientation: i32) -> (usize, usize, Vec<[u32; 3]>) {
    let rot = |x: usize, y: usize| -> (usize, usize) {
        match orientation.rem_euclid(360) {
            90 => (h - 1 - y, x),
            180 => (w - 1 - x, h - 1 - y),
            270 => (y, w - 1 - x),
            _ => (x, y),
        }
    };
    let swap = matches!(orientation.rem_euclid(360), 90 | 270);
    let (ow, oh) = if swap { (h, w) } else { (w, h) };
    let mut out = vec![[0u32; 3]; ow * oh];
    for y in 0..h {
        for x in 0..w {
            let (ox, oy) = rot(x, y);
            out[oy * ow + ox] = rgb[y * w + x];
        }
    }
    (ow, oh, out)
}

/// The lens as the direction of every upright bin in the device frame (x right, y up the screen, z out of it), for a frame `w × h` with half-angle tangents `tan_w` across and `tan_h` down. The front camera faces the user, so what it sees on its image-right lies to the device's left: image x maps to −x.
pub struct Lens {
    pub w: usize,
    pub h: usize,
    pub tan_w: f32,
    pub tan_h: f32,
    pub dirs: Vec<[f32; 3]>,
}

impl Lens {
    pub fn new(w: usize, h: usize, tan_w: f32, tan_h: f32) -> Lens {
        let mut dirs = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let sx = ((x as f32 + 0.5) / w as f32 * 2.0 - 1.0) * tan_w;
                let sy = (1.0 - (y as f32 + 0.5) / h as f32 * 2.0) * tan_h;
                let inv = 1.0 / (1.0 + sx * sx + sy * sy).sqrt();
                dirs.push([-sx * inv, sy * inv, inv]);
            }
        }
        Lens { w, h, tan_w, tan_h, dirs }
    }

    pub fn fits(&self, w: usize, h: usize, tan_w: f32, tan_h: f32) -> bool {
        (self.w, self.h) == (w, h) && (self.tan_w - tan_w).abs() < 1e-4 && (self.tan_h - tan_h).abs() < 1e-4
    }
}

/// The icosahedron at two subdivisions: dymaxion depth 2, 160 diamond cells of two triangles each.
pub const TRIS: usize = 320;

/// The sphere of light around the phone, world-fixed (east, north, up): the radiance painted on each triangle in absolute integer units, and which have been painted.
pub struct Sphere {
    pub tris: Vec<[u32; 3]>,
    pub seen: Vec<bool>,
    /// Unit direction of each triangle's centre, in the world frame.
    pub centres: Vec<[f32; 3]>,
    /// The level the last light was held to: the brightest irradiance any normal sees, in the sphere's units.
    pub level: f32,
}

/// The triangle a world direction falls in: the depth-2 cell's diamond and grid, and which half by the carry of the fractional parts of u and v.
pub fn tri_of(d: [f32; 3]) -> usize {
    let c = Coord::from_xyz([d[0] as f64, d[1] as f64, d[2] as f64]);
    let (iu, iv) = c.uv();
    let (cu, cv) = ((iu >> 28) as usize, (iv >> 28) as usize);
    let m = (1u64 << 28) - 1;
    let half = (((iu & m) + (iv & m)) >> 28) as usize;
    ((c.diamond() as usize * 16 + cu * 4 + cv) << 1) | half
}

impl Default for Sphere {
    fn default() -> Self {
        Self::new()
    }
}

impl Sphere {
    pub fn new() -> Sphere {
        let mut centres = Vec::with_capacity(TRIS);
        for diamond in 0..10u8 {
            for cu in 0..4 {
                for cv in 0..4 {
                    for half in 0..2 {
                        // The centroid of a lower triangle is a third of the way in from its square's (0,0) corner; an upper's two thirds.
                        let t = if half == 0 { 1.0 / 3.0 } else { 2.0 / 3.0 };
                        let (lat, lon) = uv_to_lat_lon(diamond, (cu as f64 + t) / 4.0, (cv as f64 + t) / 4.0);
                        let (phi, lam) = (lat.to_radians(), lon.to_radians());
                        centres.push([(phi.cos() * lam.cos()) as f32, (phi.cos() * lam.sin()) as f32, phi.sin() as f32]);
                    }
                }
            }
        }
        Sphere { tris: vec![[0; 3]; TRIS], seen: vec![false; TRIS], centres, level: 0.0 }
    }

    /// Paint a frame: every bin's direction taken through the device's rotation `rot` (row-major, world = rot · device) into the world, the bins landing in a triangle averaged and written over it.
    pub fn paint(&mut self, rgb: &[[u32; 3]], lens: &Lens, rot: &[f32; 9]) {
        let mut sums = vec![[0u64; 3]; TRIS];
        let mut counts = vec![0u32; TRIS];
        for (p, d) in rgb.iter().zip(&lens.dirs) {
            let w = [rot[0] * d[0] + rot[1] * d[1] + rot[2] * d[2], rot[3] * d[0] + rot[4] * d[1] + rot[5] * d[2], rot[6] * d[0] + rot[7] * d[1] + rot[8] * d[2]];
            let t = tri_of(w);
            for k in 0..3 {
                sums[t][k] += p[k] as u64;
            }
            counts[t] += 1;
        }
        for t in 0..TRIS {
            if counts[t] > 0 {
                self.tris[t] = [(sums[t][0] / counts[t] as u64) as u32, (sums[t][1] / counts[t] as u64) as u32, (sums[t][2] / counts[t] as u64) as u32];
                self.seen[t] = true;
            }
        }
    }

    /// The painted sphere as the light, projected in the device frame for `rot` and held to its own level: the brightest a normal can be lit by it is one. None until something is painted.
    pub fn light(&mut self, rot: &[f32; 9]) -> Option<Sh9> {
        let omega = 4.0 * std::f32::consts::PI / TRIS as f32;
        let mut world = Sh9::ZERO;
        let mut device = Sh9::ZERO;
        let mut any = false;
        for t in 0..TRIS {
            if !self.seen[t] {
                continue;
            }
            any = true;
            let c = self.centres[t];
            // Device = rotᵀ · world.
            let dd = [rot[0] * c[0] + rot[3] * c[1] + rot[6] * c[2], rot[1] * c[0] + rot[4] * c[1] + rot[7] * c[2], rot[2] * c[0] + rot[5] * c[1] + rot[8] * c[2]];
            let (bw, bd) = (basis(c), basis(dd));
            for ch in 0..3 {
                let v = self.tris[t][ch] as f32 * omega;
                for k in 0..9 {
                    world.l[ch][k] += v * bw[k];
                    device.l[ch][k] += v * bd[k];
                }
            }
        }
        if !any {
            return None;
        }
        // The level: the brightest irradiance any normal sees, sampled at the triangle centres.
        let mut peak = 0f32;
        for c in &self.centres {
            let e = world.irradiance(*c);
            peak = peak.max(0.3 * e[0] + 0.6 * e[1] + 0.1 * e[2]);
        }
        if peak <= 0.0 {
            return None;
        }
        self.level = peak;
        Some(device.scaled(1.0 / peak))
    }

    /// The radiance painted in a world direction as light on the map's scale (a sphere of this radiance everywhere would light every normal to one), or None where nothing is painted yet.
    pub fn radiance(&self, world: [f32; 3]) -> Option<[f32; 3]> {
        if self.level <= 0.0 {
            return None;
        }
        let t = tri_of(world);
        if !self.seen[t] {
            return None;
        }
        let k = std::f32::consts::PI / self.level;
        let r = self.tris[t];
        Some([r[0] as f32 * k, r[1] as f32 * k, r[2] as f32 * k])
    }
}

impl Sh9 {
    pub fn scaled(&self, k: f32) -> Sh9 {
        let mut out = *self;
        for ch in 0..3 {
            for v in out.l[ch].iter_mut() {
                *v *= k;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    #[test]
    fn every_direction_has_a_triangle_and_the_centres_find_their_own() {
        let s = Sphere::new();
        for (t, c) in s.centres.iter().enumerate() {
            assert_eq!(tri_of(*c), t, "centre {t}");
        }
        let mut hit = vec![false; TRIS];
        for i in 0..2000 {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / 2000.0;
            let r = (1.0 - z * z).sqrt();
            let a = i as f32 * 2.399_963;
            hit[tri_of([r * a.cos(), r * a.sin(), z])] = true;
        }
        assert!(hit.iter().all(|&h| h), "{} triangles never hit", hit.iter().filter(|&&h| !h).count());
    }

    #[test]
    fn a_uniform_frame_paints_a_patch_that_lights_its_own_direction_most() {
        let lens = Lens::new(32, 24, 0.55, 0.77);
        let frame = vec![[1000u32; 3]; 32 * 24];
        let mut s = Sphere::new();
        s.paint(&frame, &lens, &IDENTITY);
        let seen = s.seen.iter().filter(|&&v| v).count();
        // A 57° × 75° field covers around a tenth of the sphere.
        assert!((15..=60).contains(&seen), "{seen} triangles painted");
        let sh = s.light(&IDENTITY).unwrap();
        // On the map's scale a sphere of radiance one lights every normal to one, so a patch a tenth of the sphere that lights its own direction to one is brighter than one.
        let r = s.radiance([0.0, 0.0, 1.0]).unwrap();
        assert!(r[1] > 1.0 && r[1] < 4.0, "{r:?}");
        assert!(s.radiance([0.0, 0.0, -1.0]).is_none());
        let out = sh.irradiance([0.0, 0.0, 1.0]);
        let back = sh.irradiance([0.0, 0.0, -1.0]);
        assert!(out[1] > 0.9 && out[1] <= 1.01, "{out:?}");
        assert!(back[1] < 0.2, "{back:?}");
    }

    #[test]
    fn light_on_the_image_left_comes_from_the_device_right() {
        let lens = Lens::new(32, 24, 0.55, 0.77);
        let mut frame = vec![[0u32; 3]; 32 * 24];
        for y in 8..16 {
            for x in 0..8 {
                frame[y * 32 + x] = [1000; 3];
            }
        }
        let mut s = Sphere::new();
        s.paint(&frame, &lens, &IDENTITY);
        let sh = s.light(&IDENTITY).unwrap();
        let right = sh.irradiance([0.7071, 0.0, 0.7071]);
        let left = sh.irradiance([-0.7071, 0.0, 0.7071]);
        assert!(right[1] > left[1], "right {right:?} left {left:?}");
    }

    #[test]
    fn the_sphere_is_world_fixed() {
        // A point painted with the phone turned a quarter turn about its own normal lands in a different world triangle than unturned, unless it is on the axis.
        let lens = Lens::new(32, 24, 0.55, 0.77);
        let mut frame = vec![[0u32; 3]; 32 * 24];
        frame[12 * 32 + 4] = [1000; 3];
        let mut a = Sphere::new();
        a.paint(&frame, &lens, &IDENTITY);
        let quarter = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut b = Sphere::new();
        b.paint(&frame, &lens, &quarter);
        assert_ne!(a.seen, b.seen);
        // And the light it gives, seen from the turned phone, is the same light turned back, to within the triangles' 16°.
        let la = a.light(&IDENTITY).unwrap();
        let lb = b.light(&quarter).unwrap();
        let _ = (&a, &b);
        for n in [[0.7071, 0.0, 0.7071], [0.0, 0.7071, 0.7071]] {
            let (ea, eb) = (la.irradiance(n), lb.irradiance(n));
            assert!((ea[1] - eb[1]).abs() < 0.05, "{n:?}: {ea:?} vs {eb:?}");
        }
    }

    #[test]
    fn bayer_bins_to_the_quad_mean_and_counts_the_clipped() {
        let (w, h) = (8usize, 4usize);
        let mut data = vec![0u8; w * h * 2];
        for y in 0..h {
            for x in 0..w {
                // RGGB: red 300, greens 200, blue 100 above a black of 64; one blue sample at white.
                let v: u16 = match (x % 2, y % 2) {
                    (0, 0) => 364,
                    (1, 1) if (x, y) == (7, 3) => 1023,
                    (1, 1) => 164,
                    _ => 264,
                };
                data[(y * w + x) * 2..(y * w + x) * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        let raw = Raw { data: &data, w, h, row_stride: w * 2, packed10: false, cfa: Cfa::Rggb, black: [64; 4], white: 1023 };
        let b = bin(&raw, 2);
        assert_eq!((b.w, b.h), (2, 1));
        assert_eq!(b.rgb[0], [300, 200, 100]);
        assert!((b.stats.clipped - 1.0 / 32.0).abs() < 1e-6, "{:?}", b.stats);
        assert_eq!(b.clipped, vec![0, 1]);
        // A short bracket three stops down fills that bin, in absolute units.
        let mut long = bin(&raw, 2);
        let mut short = bin(&raw, 2);
        short.rgb[1] = [10, 20, 30];
        long.fill_clipped(&short, 0, 3);
        assert_eq!(long.rgb[1], [80, 160, 240]);
        assert_eq!(long.rgb[0], short.rgb[0]);
    }

    #[test]
    fn packed_ten_bit_reads_as_sixteen() {
        let (w, h) = (8usize, 2usize);
        let vals: Vec<u16> = (0..w * h).map(|i| (i as u16 * 97 + 5) & 1023).collect();
        let mut wide = vec![0u8; w * h * 2];
        let mut packed10 = vec![0u8; w / 4 * 5 * h];
        for y in 0..h {
            for x in 0..w {
                let v = vals[y * w + x];
                wide[(y * w + x) * 2..(y * w + x) * 2 + 2].copy_from_slice(&v.to_le_bytes());
                let g = y * (w / 4 * 5) + (x / 4) * 5;
                packed10[g + x % 4] = (v >> 2) as u8;
                packed10[g + 4] |= ((v & 3) as u8) << (2 * (x % 4));
            }
        }
        let a = bin(&Raw { data: &wide, w, h, row_stride: w * 2, packed10: false, cfa: Cfa::Rggb, black: [0; 4], white: 1023 }, 2);
        let b = bin(&Raw { data: &packed10, w, h, row_stride: w / 4 * 5, packed10: true, cfa: Cfa::Rggb, black: [0; 4], white: 1023 }, 2);
        assert_eq!(a.rgb, b.rgb);
        assert_eq!(a.clipped, b.clipped);
    }

    #[test]
    fn a_flat_reflector_stays_neutral_through_the_fixed_point_matrix() {
        // Camera = XYZ (identity transform): grey in XYZ is Illuminant E, which VSF RGB calls white.
        let mut rgb = vec![[1000u32; 3]];
        to_vsf(&mut rgb, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        for c in rgb[0] {
            assert!((c as i32 - 1000).abs() <= 3, "{:?}", rgb[0]);
        }
    }

    #[test]
    fn upright_turns_a_landscape_sensor_to_portrait() {
        let rgb: Vec<[u32; 3]> = (0..6).map(|i| [i; 3]).collect();
        let (w, h, out) = upright(3, 2, &rgb, 90);
        assert_eq!((w, h), (2, 3));
        // The sensor's top-left lands top-right after a clockwise quarter turn.
        assert_eq!(out[1][0], 0);
    }
}
