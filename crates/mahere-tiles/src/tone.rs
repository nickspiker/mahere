//! Gamma 2 and the stored highlight rolloff, shared by the baker and the renderers.
//!
//! Every byte of colour is VSF RGB at gamma 2, quantised ×256 by truncation: a byte `b` is the light `(b/256)²`, a light `x` is the byte `⌊√x·256⌋`. No rounding anywhere.
//!
//! Imagery has more range than a byte holds at paper white (sunlit snow, glint), so it is stored rolled: scene light `x` (paper white 1) becomes `curve(x/n)` at headroom `n`, then gamma 2. The agreed tone for every imagery cell is [`IMG_TAG`], Opsin's cubic rail at five times paper white; each cell carries its tag, so a reader knows exactly what went in and unrolls it before anything is mixed.

/// The light of a gamma-2 byte.
#[inline]
pub fn dec(b: u8) -> f32 {
    let x = b as f32 / 256.0;
    x * x
}

/// The gamma-2 byte of a light, truncated (`as` saturates at 255).
#[inline]
pub fn enc(x: f32) -> u8 {
    (x.max(0.0).sqrt() * 256.0) as u8
}

/// Opsin's highlight rail, `(3x − x³)/2`: slope 3/2 at black, flat at white. Clamped to [0, 1] first, because past 1 the cubic folds back.
#[inline]
pub fn rail(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    (3.0 * x - x * x * x) * 0.5
}

/// A highlight rolloff curve, stored in the data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tone {
    /// Opsin's rail: a fixed clip at the headroom, exactly invertible, `u = 2·sin(asin(y)/3)`.
    Cubic,
    /// `tanh u`: no clip, but at 8 bits the top byte holds everything past about 2.8, so only that much unrolls distinctly.
    Tanh,
}

impl Tone {
    pub fn roll(self, u: f32) -> f32 {
        match self {
            Tone::Cubic => rail(u),
            Tone::Tanh => u.max(0.0).tanh(),
        }
    }

    pub fn unroll(self, y: f32) -> f32 {
        let y = y.clamp(0.0, 1.0);
        match self {
            Tone::Cubic => 2.0 * (y.asin() / 3.0).sin(),
            Tone::Tanh => y.min(0.999_999).atanh(),
        }
    }
}

/// A tone as one byte: the curve in the high nibble (0 none, 1 cubic, 2 tanh), the headroom in the low (1..=15 times paper white). 0 is plain gamma 2, no curve: the light is the scene.
pub fn tag(curve: Tone, headroom: u8) -> u8 {
    let c = match curve {
        Tone::Cubic => 1,
        Tone::Tanh => 2,
    };
    (c << 4) | headroom.clamp(1, 15)
}

/// The agreed imagery tone: the cubic rail at five times paper white. Paper white is reflectance 0.3 for a reflectance source, byte 255 for a display-referred one; reflectance against a flat white diffuser passes 1 on sunlit slopes and snow (the 2021 Sentinel-2 composite's Rainier square: 0.08% above 0.9, the brightest 1.65), and five times 0.3 holds up to 1.5.
pub const IMG_TAG: u8 = (1 << 4) | 5;

fn parts(t: u8) -> (Option<Tone>, f32) {
    let curve = match t >> 4 {
        1 => Some(Tone::Cubic),
        2 => Some(Tone::Tanh),
        _ => None,
    };
    (curve, (t & 15).max(1) as f32)
}

/// Scene light from a stored byte under tone `t`, read at the middle of the byte's bucket: truncation put the value somewhere in `[b, b+1)/256`, and the middle is what averaging and re-rolling (the pyramid, every level) need to land back in the same bucket instead of drifting a code darker each time.
pub fn unroll(t: u8, b: u8) -> f32 {
    let y = (b as f32 + 0.5) / 256.0;
    let y = y * y;
    match parts(t) {
        (Some(c), n) => c.unroll(y) * n,
        (None, _) => y,
    }
}

/// The stored byte of scene light `x` under tone `t`. Never 0 for a lit sample (0 is no data).
pub fn roll(t: u8, x: f32) -> u8 {
    let y = match parts(t) {
        (Some(c), n) => c.roll(x / n),
        (None, _) => x,
    };
    enc(y).max(1)
}

/// The scene light of every byte under tone `t`, built once per tag.
pub fn table(t: u8) -> &'static [f32; 256] {
    static TABLES: [std::sync::OnceLock<[f32; 256]>; 256] = [const { std::sync::OnceLock::new() }; 256];
    TABLES[t as usize].get_or_init(|| std::array::from_fn(|b| unroll(t, b as u8)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_round_trip_by_truncation() {
        for b in 0..=255u8 {
            assert_eq!(enc(dec(b)), b, "byte {b}");
        }
    }

    #[test]
    fn the_curves_unroll() {
        for c in [Tone::Cubic, Tone::Tanh] {
            for i in 0..100 {
                let u = i as f32 / 100.0;
                assert!((c.unroll(c.roll(u)) - u).abs() < 1e-3, "{c:?} at {u}");
            }
        }
    }

    #[test]
    fn the_imagery_tone_round_trips_its_bytes() {
        for b in 1..=255u8 {
            assert_eq!(roll(IMG_TAG, unroll(IMG_TAG, b)), b, "byte {b}");
        }
        assert_eq!(roll(IMG_TAG, 5.0), 255);
        assert_eq!(roll(IMG_TAG, 50.0), 255);
    }
}
