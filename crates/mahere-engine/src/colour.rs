//! The colour path. Every colour the map holds is VSF RGB (703/523/462 nm, Illuminant E) at gamma 2, quantised ×256 by truncation: a byte `b` is the light `(b/256)²`, a light `x` is the byte `⌊√x·256⌋`. No rounding anywhere: it is slower and biases brightness.
//!
//! Light is mixed and shaded linear. There is one display encode, at the very end: the display exposure, VSF RGB to the display's primaries, the clamp, optionally the highlight rail, the square root.
//!
//! The rail is Opsin's HDR rolloff, `(3x − x³)/2`: slope 3/2 at black, flat at white. With the exposure at 2/3 the shadows and mid-tones pass at unity, and everything from there to 1.5 rolls smoothly into white instead of clipping. Off, the same exposure is shown straight: the shadows 1.5× darker, nothing clipping before 1.5.
//!
//! Data with more range than a byte holds (imagery: snow, glint, concrete) is stored rolled through a tagged tone ([`mahere_tiles::tone`]) and unrolled to scene light (a 256-entry table) before anything is mixed.

/// The display exposure before the rail: 2/3 cancels the rail's slope at black, so dark and mid colours land where they were authored.
pub const EXPOSURE: f32 = 2.0 / 3.0;

pub use mahere_tiles::tone::{dec, enc, rail};

/// The light of a gamma-2 colour.
#[inline]
pub fn lin(c: [u8; 3]) -> [f32; 3] {
    [dec(c[0]), dec(c[1]), dec(c[2])]
}

/// The scene light of every stored imagery byte under the agreed tone; the loader brings any other tone to it.
pub fn img_table() -> &'static [f32; 256] {
    mahere_tiles::tone::table(mahere_tiles::tone::IMG_TAG)
}

/// VSF RGB to the display, Photon's rule: macOS surfaces are tagged VSF RGB and take it as is; Android and Linux surfaces are tagged (or assumed) BT.2020, so the primaries convert. Column-major, as vsf writes its matrices.
pub fn display_matrix() -> [f32; 9] {
    if cfg!(target_os = "macos") { [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] } else { vsf::colour::VSF_RGB2REC2020 }
}

/// The encode boundary: linear VSF RGB to the display's gamma-2 bytes.
#[derive(Clone, Copy, Debug)]
pub struct Display {
    m: [f32; 9],
}

impl Default for Display {
    fn default() -> Self {
        Display { m: display_matrix() }
    }
}

impl Display {
    /// The matrix as three rows, for a shader.
    pub fn rows(&self) -> [[f32; 4]; 3] {
        let m = &self.m;
        [[m[0], m[3], m[6], 0.0], [m[1], m[4], m[7], 0.0], [m[2], m[5], m[8], 0.0]]
    }

    /// Rail on: the highlights roll into white. Rail off: the same exposure straight, shadows 1.5× darker (0.58 stop), clipping only past 1.5.
    pub fn encode(&self, x: [f32; 3], rolled: bool) -> [u8; 3] {
        let m = &self.m;
        let d = [m[0] * x[0] + m[3] * x[1] + m[6] * x[2], m[1] * x[0] + m[4] * x[1] + m[7] * x[2], m[2] * x[0] + m[5] * x[1] + m[8] * x[2]];
        let t = |v: f32| if rolled { rail(v * EXPOSURE) } else { v * EXPOSURE };
        [enc(t(d[0])), enc(t(d[1])), enc(t(d[2]))]
    }

    /// An authored colour as the map shows it unlit: for legend swatches and anything else drawn beside the map.
    pub fn swatch(&self, c: [u8; 3]) -> [u8; 3] {
        self.encode(lin(c), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rail_is_unity_in_the_dark_and_flat_at_white() {
        let x = 0.01;
        assert!((rail(x * EXPOSURE) - x).abs() < 1e-4);
        assert_eq!(rail(1.0), 1.0);
        assert_eq!(rail(7.0), 1.0);
    }
}
