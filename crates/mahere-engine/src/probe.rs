//! The real light: a front-camera frame as the lighting environment. The camera looks out of the screen, so its pixels are radiance samples in the device frame, the frame the light already lives in; each one, weighted by its solid angle, projects straight into the nine harmonics the renderer evaluates. The frame is in absolute units (raw fraction of white per second of exposure), so frames at different exposures agree; the level is anchored by the map core to the brightest light seen (Nick 2026-10-08: absolute, adapting only to a brighter light, so turning away from the light does not brighten the map back up).
//!
//! Colour: the sensor's raw samples through Android's `SENSOR_COLOR_TRANSFORM` (XYZ to camera, a 1931 characterisation under the reference illuminant) inverted, then XYZ to VSF RGB with no adaptation: a blue sky lights the shadows blue, a tungsten lamp lights the map orange. The light's colour is the point.

use crate::sh::{Sh9, basis};

/// A frame of linear VSF RGB radiance, upright (rows run down the screen, columns across it, as the user sees the world behind the phone), and the lens: the tangents of its half-angles across (`tan_w`) and down (`tan_h`).
pub struct Probe {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<[f32; 3]>,
    pub tan_w: f32,
    pub tan_h: f32,
}

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

/// A raw sensor frame: 16-bit little-endian samples, `row_stride` bytes per row, the sensor's `white` level and its `black` pedestal at each of the four Bayer positions (row-major within the quad: (0,0), (1,0), (0,1), (1,1)). The pedestals differ a little per channel, and a frame near black is all pedestal, so each is taken from its own.
pub struct Raw<'a> {
    pub data: &'a [u8],
    pub w: usize,
    pub h: usize,
    pub row_stride: usize,
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

/// A binned frame: linear camera RGB per bin, how many of each bin's raw samples were at white, and the frame's exposure statistics.
pub struct Binned {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<[f32; 3]>,
    pub clipped: Vec<u16>,
    pub stats: Stats,
}

impl Binned {
    /// Fill the bins this frame clipped from a shorter exposure of the same scene, scaled by the exposure ratio: a lamp that is white at the long exposure gets its real brightness from the short one. A clipped bin's mean is a floor; the short frame's is the truth up to its own clip.
    pub fn fill_clipped(&mut self, short: &Binned, ratio: f32) {
        if (short.w, short.h) != (self.w, self.h) {
            return;
        }
        for i in 0..self.rgb.len() {
            if self.clipped[i] > 0 {
                let s = short.rgb[i];
                self.rgb[i] = [s[0] * ratio, s[1] * ratio, s[2] * ratio];
            }
        }
    }
}

/// Bin a raw frame to about `cols` columns of linear camera RGB (each 2×2 Bayer quad is one sample, then an integer box over those), counting the samples at white in each bin, with the frame's exposure statistics.
pub fn bin(raw: &Raw, cols: usize) -> Binned {
    let (qw, qh) = (raw.w / 2, raw.h / 2);
    let f = (qw / cols.max(1)).max(1);
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
    // Integers all the way through the samples, in two passes a row: the sums and the clip count over every sample (no scatter, so it vectorises), then the histogram over every sixteenth, which is plenty for a percentile.
    use rayon::prelude::*;
    let rows: Vec<(Vec<[u32; 4]>, Vec<u16>, [u32; 256])> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut sums = vec![[0u32; 4]; w];
            let mut clipped = vec![0u16; w];
            let mut hist = [0u32; 256];
            for j in 0..f {
                let qy = (y * f + j) * 2;
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
    let black_mean = raw.black.iter().map(|&b| b as u32).sum::<u32>() / 4;
    let scale = 1.0 / ((raw.white as u32).saturating_sub(black_mean).max(1) as f32 * count as f32);
    let mut out = Vec::with_capacity(w * h);
    let mut clipped = Vec::with_capacity(w * h);
    let mut hist = [0u32; 256];
    for (sums, c, hh) in rows {
        for s in sums {
            let lin = |k: usize| s[k].saturating_sub(count * raw.black[k] as u32) as f32 * scale;
            let g = (lin(0) + lin(1) + lin(2) + lin(3) - lin(rp) - lin(bp)) * 0.5;
            out.push([lin(rp), g, lin(bp)]);
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

/// Camera RGB to VSF RGB: `xyz_to_cam` is Android's row-major XYZ→camera matrix, inverted here; XYZ then goes to VSF RGB colorimetrically (Illuminant E to white, no adaptation). Negative light clamps to zero.
pub fn to_vsf(rgb: &mut [[f32; 3]], xyz_to_cam: &[f32; 9]) {
    let Some(cam_to_xyz) = invert3(xyz_to_cam) else { return };
    let m = &vsf::colour::XYZ2VSF_RGB;
    for p in rgb.iter_mut() {
        let x = [cam_to_xyz[0] * p[0] + cam_to_xyz[1] * p[1] + cam_to_xyz[2] * p[2], cam_to_xyz[3] * p[0] + cam_to_xyz[4] * p[1] + cam_to_xyz[5] * p[2], cam_to_xyz[6] * p[0] + cam_to_xyz[7] * p[1] + cam_to_xyz[8] * p[2]];
        let v = vsf::colour::convert::apply_matrix_3x3_f32(m, &x);
        *p = [v[0].max(0.0), v[1].max(0.0), v[2].max(0.0)];
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

/// Turn a sensor-oriented frame upright: `orientation` is Android's `SENSOR_ORIENTATION`, the clockwise degrees the sensor image must turn to stand upright in portrait. The lens tangents turn with it.
pub fn upright(w: usize, h: usize, rgb: &[[f32; 3]], tan_w: f32, tan_h: f32, orientation: i32) -> Probe {
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
    let mut out = vec![[0f32; 3]; ow * oh];
    for y in 0..h {
        for x in 0..w {
            let (ox, oy) = rot(x, y);
            out[oy * ow + ox] = rgb[y * w + x];
        }
    }
    Probe { w: ow, h: oh, rgb: out, tan_w: if swap { tan_h } else { tan_w }, tan_h: if swap { tan_w } else { tan_h } }
}

impl Sh9 {
    /// The environment a probe frame saw, in the device frame (x right, y up the screen, z out of it), in the frame's own units. The largest circle that fits the frame is taken as the whole sphere: its centre is straight out of the screen, its rim is straight into the back, and every direction between is stretched in angle to match (Nick 2026-10-08: no corners, no interpolation, pretend the camera is a full sphere). The corners outside the circle are dropped. The front camera faces the user, so what it sees on its image-right lies to the device's left: image x maps to −x.
    pub fn from_probe(p: &Probe) -> Sh9 {
        let mut sh = Sh9::ZERO;
        let (dw, dh) = (2.0 * p.tan_w / p.w as f32, 2.0 * p.tan_h / p.h as f32);
        let r_max = p.tan_w.min(p.tan_h);
        let theta_max = r_max.atan();
        let stretch = std::f32::consts::PI / theta_max;
        for y in 0..p.h {
            for x in 0..p.w {
                let sx = ((x as f32 + 0.5) / p.w as f32 * 2.0 - 1.0) * p.tan_w;
                let sy = (1.0 - (y as f32 + 0.5) / p.h as f32 * 2.0) * p.tan_h;
                let r = (sx * sx + sy * sy).sqrt();
                if r > r_max {
                    continue;
                }
                // The pixel's own angle from the axis, stretched so the rim of the circle reaches the back of the sphere; its azimuth kept (mirrored in x).
                let theta = r.atan();
                let t = theta * stretch;
                let (st, ct) = t.sin_cos();
                let (ux, uy) = if r > 1e-6 { (-sx / r, sy / r) } else { (0.0, 0.0) };
                let d = [ux * st, uy * st, ct];
                // The solid angle the pixel covers after the stretch: sin t · dt · dφ, with dt = stretch · dθ, dθ = dr / (1 + r²), and the pixel's area dw·dh = r · dr · dφ.
                let omega = if r > 1e-6 { st * stretch / (1.0 + r * r) * (dw * dh / r) } else { stretch * stretch * dw * dh };
                let b = basis(d);
                let c = p.rgb[y * p.w + x];
                for ch in 0..3 {
                    let v = c[ch] * omega;
                    for k in 0..9 {
                        sh.l[ch][k] += v * b[k];
                    }
                }
            }
        }
        sh
    }

    /// The luminance of the irradiance on the screen's own normal: what the probe says falls on the phone.
    pub fn screen_luminance(&self) -> f32 {
        let e = self.irradiance([0.0, 0.0, 1.0]);
        0.3 * e[0] + 0.6 * e[1] + 0.1 * e[2]
    }

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

    #[test]
    fn a_flat_frame_is_a_uniform_sphere() {
        let p = Probe { w: 64, h: 48, rgb: vec![[1.0; 3]; 64 * 48], tan_w: 1.0, tan_h: 0.75 };
        let sh = Sh9::from_probe(&p);
        // A unit sphere of radiance delivers π to every normal.
        let front = sh.irradiance([0.0, 0.0, 1.0]);
        assert!((front[1] - std::f32::consts::PI).abs() < 0.05, "{front:?}");
        for n in [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]] {
            let e = sh.irradiance(n);
            assert!((e[1] - std::f32::consts::PI).abs() < 0.15, "{n:?} {e:?}");
        }
        assert!((sh.scaled(1.0 / sh.screen_luminance()).screen_luminance() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn the_corners_are_dropped() {
        // Light only in the corners of a wide frame: nothing reaches the sphere.
        let (w, h) = (64usize, 32usize);
        let mut rgb = vec![[0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                if x < 8 || x >= w - 8 {
                    rgb[y * w + x] = [1.0; 3];
                }
            }
        }
        let sh = Sh9::from_probe(&Probe { w, h, rgb, tan_w: 2.0, tan_h: 1.0 });
        assert!(sh.l.iter().all(|c| c.iter().all(|&v| v == 0.0)), "{:?}", sh.l[1]);
    }

    #[test]
    fn light_on_the_image_left_comes_from_the_device_right() {
        let (w, h) = (16usize, 12usize);
        let mut rgb = vec![[0f32; 3]; w * h];
        // Lit a little left of centre: after the stretch that is the device's right, in front of its horizon.
        for y in 4..8 {
            for x in 4..7 {
                rgb[y * w + x] = [1.0; 3];
            }
        }
        let sh = Sh9::from_probe(&Probe { w, h, rgb, tan_w: 1.2, tan_h: 0.9 });
        let right = sh.irradiance([0.7071, 0.0, 0.7071]);
        let left = sh.irradiance([-0.7071, 0.0, 0.7071]);
        assert!(right[1] > left[1], "right {right:?} left {left:?}");
    }

    #[test]
    fn bayer_bins_to_the_quad_mean_and_reports_the_peak() {
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
        let raw = Raw { data: &data, w, h, row_stride: w * 2, cfa: Cfa::Rggb, black: [64; 4], white: 1023 };
        let b = bin(&raw, 4);
        assert_eq!((b.w, b.h), (4, 2));
        let s = 1.0 / 959.0;
        assert!((b.rgb[0][0] - 300.0 * s).abs() < 1e-4 && (b.rgb[0][1] - 200.0 * s).abs() < 1e-4 && (b.rgb[0][2] - 100.0 * s).abs() < 1e-4, "{:?}", b.rgb[0]);
        // One sample of 32 at white: 3% clipped, in the last bin. The percentile is from a sixteenth of the samples, so a frame this small says little about it.
        assert!((b.stats.clipped - 1.0 / 32.0).abs() < 1e-6, "{:?}", b.stats);
        assert!((0.0..=1.0).contains(&b.stats.p999));
        assert_eq!(b.clipped, vec![0, 0, 0, 0, 0, 0, 0, 1]);
        // A short bracket at a quarter of the exposure fills that bin, scaled back up.
        let mut long = bin(&raw, 4);
        let mut short = bin(&raw, 4);
        short.rgb[7] = [0.1, 0.2, 0.3];
        long.fill_clipped(&short, 4.0);
        assert_eq!(long.rgb[7], [0.4, 0.8, 1.2]);
        assert_eq!(long.rgb[0], short.rgb[0]);
    }

    #[test]
    fn upright_turns_a_landscape_sensor_to_portrait() {
        let rgb: Vec<[f32; 3]> = (0..6).map(|i| [i as f32; 3]).collect();
        let p = upright(3, 2, &rgb, 1.5, 1.0, 90);
        assert_eq!((p.w, p.h), (2, 3));
        assert_eq!((p.tan_w, p.tan_h), (1.0, 1.5));
        // The sensor's top-left lands top-right after a clockwise quarter turn.
        assert_eq!(p.rgb[1][0], 0.0);
    }
}
