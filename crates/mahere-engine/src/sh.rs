//! Image-based lighting for the terrain: second-order spherical harmonics (Ramamoorthi & Hanrahan 2001). Nine coefficients per colour channel capture the irradiance an environment delivers to every normal direction to within a couple of percent for a diffuse surface, and evaluating it per pixel is a quadratic form in the normal â no table, no seams. The environment lives in the DEVICE frame (the sun stays top-left of the phone however the map turns), so normals are rotated by the bearing before lookup. Today the environment is a sun plus a sky dome; a front-camera capture is the same 27 numbers from a different source.

/// Per-channel quadratic form of the irradiance, ready for the pixel loop.
#[derive(Clone, Copy)]
pub struct Quad {
    pub k: [[f32; 10]; 3],
}

impl Quad {
    /// One directional light and nothing else, exactly: irradiance `rgb Â· (n Â· dir)` per channel, which the pixel loop clamps at zero â so a face turned away from the light, or a light under the landscape, is black. `dir` is in the device frame; `bearing_sc` conjugates it onto world-frame normals as `Sh9::quadratic` does (b′ = Rᵀ b).
    pub fn directional(dir: [f32; 3], rgb: [f32; 3], bearing_sc: (f32, f32)) -> Quad {
        let (sb, cb) = bearing_sc;
        let b = [cb * dir[0] + sb * dir[1], -sb * dir[0] + cb * dir[1], dir[2]];
        let mut q = Quad { k: [[0.0; 10]; 3] };
        for c in 0..3 {
            q.k[c][6] = rgb[c] * b[0];
            q.k[c][7] = rgb[c] * b[1];
            q.k[c][8] = rgb[c] * b[2];
        }
        q
    }

    /// Irradiance per channel for a unit normal in the world frame.
    #[inline(always)]
    pub fn eval(&self, x: f32, y: f32, z: f32) -> [f32; 3] {
        let m = [x * x, y * y, z * z, x * y, x * z, y * z, x, y, z, 1.0];
        let mut out = [0f32; 3];
        for c in 0..3 {
            let k = &self.k[c];
            let mut e = 0.0;
            for i in 0..10 {
                e += k[i] * m[i];
            }
            out[c] = e;
        }
        out
    }
}

#[derive(Clone, Copy)]
pub struct Sh9 {
    /// Radiance coefficients per channel, real SH order: 00, 1-1, 10, 11, 2-2, 2-1, 20, 21, 22.
    pub l: [[f32; 9]; 3],
}

/// Real SH basis up to l=2 at a unit direction.
#[inline(always)]
pub(crate) fn basis(d: [f32; 3]) -> [f32; 9] {
    let (x, y, z) = (d[0], d[1], d[2]);
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

impl Sh9 {
    pub const ZERO: Sh9 = Sh9 { l: [[0.0; 9]; 3] };

    /// A directional light of normal-incidence irradiance `rgb` from unit direction `dir`.
    pub fn add_sun(&mut self, dir: [f32; 3], rgb: [f32; 3]) {
        let b = basis(dir);
        for c in 0..3 {
            for k in 0..9 {
                self.l[c][k] += rgb[c] * b[k];
            }
        }
    }

    /// A uniform sky dome of radiance `rgb` over the hemisphere above `up`, projected numerically.
    pub fn add_sky(&mut self, up: [f32; 3], rgb: [f32; 3]) {
        let n = 24usize;
        let mut acc = [0f32; 9];
        let mut count = 0usize;
        for i in 0..n {
            for j in 0..(2 * n) {
                // Stratified over the sphere; keep the half above `up`.
                let ct = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
                let st = (1.0 - ct * ct).max(0.0).sqrt();
                let ph = std::f32::consts::TAU * (j as f32 + 0.5) / (2 * n) as f32;
                let d = [st * ph.cos(), st * ph.sin(), ct];
                if d[0] * up[0] + d[1] * up[1] + d[2] * up[2] <= 0.0 {
                    continue;
                }
                let b = basis(d);
                for k in 0..9 {
                    acc[k] += b[k];
                }
                count += 1;
            }
        }
        // Each sample carries the solid angle 2Ï / count of the hemisphere.
        let w = 2.0 * std::f32::consts::PI / count.max(1) as f32;
        for c in 0..3 {
            for k in 0..9 {
                self.l[c][k] += rgb[c] * acc[k] * w;
            }
        }
    }

    /// Irradiance at a unit normal, per channel (the cosine-lobe convolution folded into the constants).
    #[inline(always)]
    pub fn irradiance(&self, n: [f32; 3]) -> [f32; 3] {
        const C1: f32 = 0.429043;
        const C2: f32 = 0.511664;
        const C3: f32 = 0.743125;
        const C4: f32 = 0.886227;
        const C5: f32 = 0.247708;
        let (x, y, z) = (n[0], n[1], n[2]);
        let mut out = [0f32; 3];
        for c in 0..3 {
            let l = &self.l[c];
            out[c] = C1 * l[8] * (x * x - y * y)
                + C3 * l[6] * z * z
                + C4 * l[0]
                - C5 * l[6]
                + 2.0 * C1 * (l[4] * x * y + l[7] * x * z + l[5] * y * z)
                + 2.0 * C2 * (l[3] * x + l[1] * y + l[2] * z);
        }
        out
    }

    /// The irradiance as a quadratic form per channel â E(n) = náµ A n + bÂ·n + c â which is what the pixel loop evaluates: ten coefficients per channel, no basis functions. `bearing_sc` conjugates the device-frame environment onto world-frame normals so the loop never rotates.
    pub fn quadratic(&self, bearing_sc: (f32, f32)) -> Quad {
        const C1: f32 = 0.429043;
        const C2: f32 = 0.511664;
        const C3: f32 = 0.743125;
        const C4: f32 = 0.886227;
        const C5: f32 = 0.247708;
        let (sb, cb) = bearing_sc;
        let mut q = Quad { k: [[0.0; 10]; 3] };
        for c in 0..3 {
            let l = &self.l[c];
            // In the device frame: A (symmetric), b, const.
            let a = [[C1 * l[8], C1 * l[4], C1 * l[7]], [C1 * l[4], -C1 * l[8], C1 * l[5]], [C1 * l[7], C1 * l[5], C3 * l[6]]];
            let bv = [2.0 * C2 * l[3], 2.0 * C2 * l[1], 2.0 * C2 * l[2]];
            let k0 = C4 * l[0] - C5 * l[6];
            // n_dev = R n_world with R rotating (x, y) by the bearing: x' = x cb - y sb, y' = x sb + y cb. Conjugate: A' = Ráµ A R, b' = Ráµ b.
            let r = [[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]];
            let mut ar = [[0.0f32; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    ar[i][j] = (0..3).map(|m| a[i][m] * r[m][j]).sum();
                }
            }
            let mut ap = [[0.0f32; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    ap[i][j] = (0..3).map(|m| r[m][i] * ar[m][j]).sum();
                }
            }
            let bp = [
                r[0][0] * bv[0] + r[1][0] * bv[1] + r[2][0] * bv[2],
                r[0][1] * bv[0] + r[1][1] * bv[1] + r[2][1] * bv[2],
                r[0][2] * bv[0] + r[1][2] * bv[1] + r[2][2] * bv[2],
            ];
            // xÂ², yÂ², zÂ², xy, xz, yz (cross terms doubled), x, y, z, 1.
            q.k[c] = [ap[0][0], ap[1][1], ap[2][2], 2.0 * ap[0][1], 2.0 * ap[0][2], 2.0 * ap[1][2], bp[0], bp[1], bp[2], k0];
        }
        q
    }

    /// The default environment: a warm sun at (azimuth clockwise from screen-up, altitude) in the device frame plus a cool sky dome, scaled so flat ground under the sun's zenith reads about as bright as today's hillshade (ambient â 0.3, sun â 0.7).
    pub fn sun_and_sky(az_deg: f32, alt_deg: f32) -> Sh9 {
        Sh9::sun_and_sky_coloured(az_deg, alt_deg, [0.74, 0.70, 0.62], [0.85, 0.95, 1.15])
    }

    /// The same with a theme's sun and sky colours: the cartographic light, a sun the user places above a sky dome.
    pub fn sun_and_sky_coloured(az_deg: f32, alt_deg: f32, sun: [f32; 3], sky: [f32; 3]) -> Sh9 {
        let (az, alt) = (az_deg.to_radians(), alt_deg.to_radians());
        // Device frame: x right, y up (screen), z out of the screen; "up" for the sky is +z.
        let mut sh = Sh9::ZERO;
        sh.add_sun([az.sin() * alt.cos(), az.cos() * alt.cos(), alt.sin()], sun);
        // A sky of radiance S gives ÏS onto an upward normal: ÏS â 0.3 for a unit tint.
        let s = 0.3 / std::f32::consts::PI;
        sh.add_sky([0.0, 0.0, 1.0], [s * sky[0], s * sky[1], s * sky[2]]);
        sh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_ground_under_sun_and_sky_matches_the_old_shade() {
        let sh = Sh9::sun_and_sky(315.0, 40.0);
        let e = sh.irradiance([0.0, 0.0, 1.0]);
        // Old: 0.3 + 0.7 * sin(40Â°) = 0.75; SH's cosine lobe is a soft approximation, so allow a margin.
        let lum = 0.3 * e[0] + 0.6 * e[1] + 0.1 * e[2];
        assert!((0.6..0.9).contains(&lum), "luminance {lum}");
        // A slope facing away from the sun is darker than one facing it.
        let toward = sh.irradiance([-0.5, 0.5, 0.7071]);
        let away = sh.irradiance([0.5, -0.5, 0.7071]);
        assert!(toward[1] > away[1]);
        // Nothing is ever negative in a way that matters.
        assert!(away[1] > -0.05);
        // The quadratic form (bearing 0) reproduces the direct evaluation.
        let q = sh.quadratic((0.0, 1.0));
        let n = [-0.3, 0.4, 0.866];
        let (d, p) = (sh.irradiance(n), q.eval(n[0], n[1], n[2]));
        for c in 0..3 {
            assert!((d[c] - p[c]).abs() < 1e-4, "{d:?} vs {p:?}");
        }
        // With a bearing, the quadratic on a world normal equals the direct evaluation on the rotated normal.
        let (sb, cb) = (0.6f32, 0.8f32);
        let q = sh.quadratic((sb, cb));
        let rot = [n[0] * cb - n[1] * sb, n[0] * sb + n[1] * cb, n[2]];
        let (d, p) = (sh.irradiance(rot), q.eval(n[0], n[1], n[2]));
        for c in 0..3 {
            assert!((d[c] - p[c]).abs() < 1e-4, "{d:?} vs {p:?}");
        }
    }
}

/// Where the sun is: azimuth clockwise from true north and altitude above the horizon, degrees, for a place and a Unix time. NOAA's solar position approximation, good to a few tenths of a degree â the real sun the terrain can be lit by when the phone knows the time and where it is.
pub fn sun_position(lat_deg: f64, lon_deg: f64, unix_secs: f64) -> (f64, f64) {
    // Julian centuries since J2000.
    let jd = unix_secs / 86400.0 + 2440587.5;
    let t = (jd - 2451545.0) / 36525.0;
    let l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)).rem_euclid(360.0);
    let m = (357.52911 + t * (35999.05029 - 0.0001537 * t)).to_radians();
    let c = (1.914602 - t * (0.004817 + 0.000014 * t)) * m.sin() + (0.019993 - 0.000101 * t) * (2.0 * m).sin() + 0.000289 * (3.0 * m).sin();
    let true_long = l0 + c;
    let omega = (125.04 - 1934.136 * t).to_radians();
    let lambda = (true_long - 0.00569 - 0.00478 * omega.sin()).to_radians();
    let eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0;
    let eps = (eps0 + 0.00256 * omega.cos()).to_radians();
    let decl = (eps.sin() * lambda.sin()).asin();
    // Equation of time, minutes.
    let y = (eps / 2.0).tan().powi(2);
    let e = (0.016708634 - t * (0.000042037 + 0.0000001267 * t)) as f64;
    let l0r = l0.to_radians();
    let eot = 4.0 * (y * (2.0 * l0r).sin() - 2.0 * e * m.sin() + 4.0 * e * y * m.sin() * (2.0 * l0r).cos() - 0.5 * y * y * (4.0 * l0r).sin() - 1.25 * e * e * (2.0 * m).sin()).to_degrees();
    // True solar time and hour angle.
    let minutes = (unix_secs.rem_euclid(86400.0)) / 60.0;
    let tst = (minutes + eot + 4.0 * lon_deg).rem_euclid(1440.0);
    let ha = (tst / 4.0 - 180.0).to_radians();
    let lat = lat_deg.to_radians();
    let cos_zen = lat.sin() * decl.sin() + lat.cos() * decl.cos() * ha.cos();
    let zen = cos_zen.clamp(-1.0, 1.0).acos();
    let alt = 90.0 - zen.to_degrees();
    let az = {
        let d = lat.cos() * zen.sin();
        if d.abs() < 1e-9 {
            180.0
        } else {
            let cos_az = ((lat.sin() * zen.cos() - decl.sin()) / d).clamp(-1.0, 1.0);
            let a = cos_az.acos().to_degrees();
            if ha > 0.0 { (a + 180.0).rem_euclid(360.0) } else { (540.0 - a).rem_euclid(360.0) }
        }
    };
    (az, alt)
}

#[cfg(test)]
mod sun_tests {
    #[test]
    fn noon_sun_at_st_helens_in_october_is_south_and_low() {
        // 2026-10-06 20:00 UTC = 13:00 PDT at Mount St Helens: the sun is a little west of south, about 38Â° up.
        let (az, alt) = super::sun_position(46.2, -122.19, 1_791_316_800.0);
        assert!((170.0..200.0).contains(&az), "azimuth {az}");
        assert!((30.0..45.0).contains(&alt), "altitude {alt}");
        // Midnight: below the horizon.
        let (_, night) = super::sun_position(46.2, -122.19, 1_791_316_800.0 - 12.0 * 3600.0);
        assert!(night < 0.0, "altitude {night}");
    }
}

#[cfg(test)]
mod directional_tests {
    use super::*;

    #[test]
    fn the_bare_sun_is_the_cosine_and_nothing_else() {
        let q = Quad::directional([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], (0.0, 1.0));
        // Overhead: flat ground full, a 60° slope half, the underside negative (the pixel loop clamps it to black).
        assert!((q.eval(0.0, 0.0, 1.0)[0] - 1.0).abs() < 1e-6);
        let (s, c) = 60f32.to_radians().sin_cos();
        assert!((q.eval(s, 0.0, c)[0] - 0.5).abs() < 1e-5);
        // Under the landscape: flat ground gets nothing.
        let night = Quad::directional([0.0, 0.6, -0.8], [1.0, 1.0, 1.0], (0.0, 1.0));
        assert!(night.eval(0.0, 0.0, 1.0)[0] < 0.0);
        // With a bearing, the light on a world normal equals the light on that normal turned into the device frame.
        let (sb, cb) = (0.6f32, 0.8f32);
        let dir = [0.3, 0.5, 0.8124];
        let q = Quad::directional(dir, [1.0, 1.0, 1.0], (sb, cb));
        let n = [0.2, -0.4, 0.894];
        let dev = [n[0] * cb - n[1] * sb, n[0] * sb + n[1] * cb, n[2]];
        let want = dir[0] * dev[0] + dir[1] * dev[1] + dir[2] * dev[2];
        assert!((q.eval(n[0], n[1], n[2])[0] - want).abs() < 1e-5);
    }
}
