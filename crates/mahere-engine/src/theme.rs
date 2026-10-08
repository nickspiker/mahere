//! Themes: every colour the renderers use, as tables the client owns. A theme is the hypsometric ramp, the land and line tables, the water and contour inks with the contours' opacity, the flat ground and background, the sun and sky of the lighting environment, and the layers it turns on by default. Nothing in a cell knows about any of it.
//!
//! The three built-ins share one hierarchy so any of them reads at a glance: the ground is quiet and the shading soft (mostly sky, a little sun), so relief shows without crushing steep faces to black; roads are darker or more saturated than the ground they cross, heavier and warmer as they get bigger; trails are the most saturated thing on the map; water is unmistakably blue and a lake covers the river running into it; contours and boundaries sit under everything, faint enough to read through.

use crate::raster::{LayerMask, Style};

/// Bumped whenever the built-ins change, so a vault holding an older copy of one is refreshed.
pub const BUILTIN_REVISION: u32 = 2;

#[derive(Clone, Debug)]
pub struct Theme {
    pub name: String,
    /// Which built-in revision this is; a stored built-in older than [`BUILTIN_REVISION`] is replaced.
    pub revision: u32,
    /// Hypsometric stops: metres and the colour there, interpolated between; the sea below half a metre takes `sea`.
    pub hypso: [(f32, [u8; 3]); 5],
    pub sea: [u8; 3],
    /// Ground with the elevation tint off.
    pub flat: [u8; 3],
    /// Where there is no terrain at all.
    pub bg: [u8; 3],
    /// Ground with the terrain layer off.
    pub no_dem: [u8; 3],
    pub land: [[u8; 3]; 14],
    pub line: [[u8; 3]; 18],
    pub water: [u8; 3],
    pub contour: [u8; 3],
    pub contour_index: [u8; 3],
    /// Opacity of a full contour line, ordinary and index.
    pub contour_alpha: [f32; 2],
    /// Normal-incidence irradiance of the sun and the radiance tint of the sky dome; a sky of 1 lights flat ground to 0.3.
    pub sun: [f32; 3],
    pub sky: [f32; 3],
    pub layers: LayerMask,
}

/// Land cover classes in table order (1-based ids; 0 is empty).
pub const LAND_NAMES: [&str; 14] = ["", "Grass", "Farmland", "Orchard", "Scrub", "Forest", "Wetland", "Sand", "Rock", "Glacier", "Quarry", "Industrial", "Urban", "Water"];

/// Line classes in table order (1-based ids; 0 is empty).
pub const CLASS_NAMES: [&str; 18] = ["", "Motorway", "Trunk", "Primary", "Secondary", "Tertiary", "Residential", "Service", "Track", "Path", "Rail", "Power", "Waterway", "National park", "Wilderness", "National forest", "Protected", "County and state"];

const ALL: LayerMask = LayerMask { dem: true, land: true, water: true, line: true, debug: false, imagery: false, contours: true, slope: false, infrared: false, hypso: true, boundaries: true };

/// The field map: a pale, warm ground that greens in the valleys and lightens toward the peaks, soft relief, roads in dark greys rising to amber and red-orange, trails in crimson.
pub fn trail() -> Theme {
    Theme {
        name: "Trail".to_string(),
        revision: BUILTIN_REVISION,
        hypso: [(0.0, [196, 210, 176]), (600.0, [208, 212, 180]), (1200.0, [220, 214, 192]), (1900.0, [228, 224, 214]), (2600.0, [242, 242, 244])],
        sea: [150, 192, 228],
        flat: [232, 232, 228],
        bg: [44, 48, 56],
        no_dem: [206, 206, 200],
        land: [[0, 0, 0], [190, 214, 160], [222, 214, 172], [198, 214, 160], [200, 206, 170], [154, 190, 142], [172, 206, 196], [232, 222, 184], [206, 202, 196], [234, 242, 250], [202, 194, 186], [210, 204, 214], [222, 216, 210], [150, 192, 228]],
        line: [[0, 0, 0], [214, 92, 42], [222, 120, 46], [222, 152, 52], [196, 156, 66], [140, 130, 110], [118, 118, 124], [146, 146, 152], [140, 96, 58], [206, 34, 78], [76, 76, 86], [146, 136, 168], [62, 130, 206], [64, 150, 80], [104, 160, 76], [118, 140, 66], [76, 150, 118], [150, 98, 172]],
        water: [150, 192, 228],
        contour: [150, 122, 92],
        contour_index: [124, 94, 64],
        contour_alpha: [0.32, 0.55],
        sun: [0.52, 0.50, 0.46],
        sky: [1.75, 1.80, 1.90],
        layers: ALL,
    }
}

/// A printed quad: white ground with no elevation tint, woodland in pale green, brown contours with heavier index lines, red highways, black roads, magenta trails, blue water.
pub fn topo() -> Theme {
    Theme {
        name: "Topo".to_string(),
        revision: BUILTIN_REVISION,
        hypso: [(0.0, [236, 240, 230]), (600.0, [240, 240, 232]), (1200.0, [242, 240, 234]), (1900.0, [244, 242, 240]), (2600.0, [250, 250, 252])],
        sea: [160, 204, 240],
        flat: [246, 246, 242],
        bg: [226, 226, 222],
        no_dem: [236, 236, 232],
        land: [[0, 0, 0], [222, 236, 206], [240, 236, 214], [224, 236, 206], [226, 232, 210], [204, 226, 192], [208, 230, 226], [244, 236, 214], [228, 226, 222], [246, 250, 254], [226, 220, 214], [232, 226, 234], [236, 230, 228], [160, 204, 240]],
        line: [[0, 0, 0], [196, 36, 36], [200, 46, 44], [204, 60, 50], [70, 66, 66], [80, 78, 78], [58, 58, 62], [102, 102, 106], [120, 92, 62], [186, 24, 128], [44, 44, 52], [140, 120, 170], [40, 120, 204], [64, 150, 84], [110, 160, 90], [120, 140, 70], [80, 150, 126], [142, 92, 168]],
        water: [160, 204, 240],
        contour: [178, 130, 84],
        contour_index: [146, 98, 56],
        contour_alpha: [0.42, 0.72],
        sun: [0.40, 0.40, 0.40],
        sky: [2.05, 2.05, 2.10],
        layers: LayerMask { hypso: false, ..ALL },
    }
}

/// For a dark room or a five o'clock start: slate ground with its relief still readable, land cover as quiet tints, light roads, amber highways, a bright coral for trails, water a clear deep blue rather than a hole.
pub fn night() -> Theme {
    Theme {
        name: "Night".to_string(),
        revision: BUILTIN_REVISION,
        hypso: [(0.0, [62, 72, 74]), (600.0, [68, 74, 76]), (1200.0, [74, 78, 80]), (1900.0, [84, 88, 92]), (2600.0, [108, 112, 120])],
        sea: [30, 62, 100],
        flat: [82, 86, 92],
        bg: [14, 16, 20],
        no_dem: [60, 64, 70],
        land: [[0, 0, 0], [70, 92, 66], [88, 86, 64], [72, 92, 64], [76, 84, 66], [54, 80, 60], [58, 84, 82], [108, 100, 80], [88, 88, 90], [124, 132, 142], [82, 78, 74], [80, 78, 90], [92, 88, 90], [30, 62, 100]],
        line: [[0, 0, 0], [250, 168, 72], [246, 180, 84], [240, 196, 108], [222, 206, 150], [206, 202, 188], [182, 182, 188], [148, 148, 156], [196, 154, 112], [255, 112, 142], [156, 156, 166], [150, 140, 182], [86, 156, 236], [100, 180, 110], [130, 190, 108], [150, 170, 96], [100, 180, 150], [190, 140, 214]],
        water: [36, 78, 122],
        contour: [140, 140, 128],
        contour_index: [172, 168, 150],
        contour_alpha: [0.28, 0.50],
        sun: [0.50, 0.50, 0.56],
        sky: [1.60, 1.65, 1.80],
        layers: ALL,
    }
}

/// The built-in themes, as authored: VSF RGB, gamma 2. A host converts them to its display before use.
pub fn builtin() -> Vec<Theme> {
    vec![trail(), topo(), night()]
}

impl Theme {
    /// The hypsometric table indexed by elevation quantum >> 4 (4 m buckets); the sea below half a metre, the background in the last row for no data.
    pub fn hypso_lut(&self) -> Box<[[u8; 3]; 4096]> {
        let mut lut = Box::new([[0u8; 3]; 4096]);
        for (i, out) in lut.iter_mut().enumerate() {
            let elev = (i as f32 * 16.0) / 4.0 - 500.0;
            // Below the first stop (land under sea level, the Dead Sea, a polder) takes the first stop; the sea itself is told apart at draw time, where elevation is exactly zero and the ground dead flat.
            let _ = self.sea;
            let last = self.hypso[self.hypso.len() - 1].1;
            let mut tint = [last[0] as f32, last[1] as f32, last[2] as f32];
            for w in self.hypso.windows(2) {
                let (e0, c0) = w[0];
                let (e1, c1) = w[1];
                if elev < e1 {
                    let t = ((elev - e0) / (e1 - e0)).clamp(0.0, 1.0);
                    tint = [c0[0] as f32 + (c1[0] as f32 - c0[0] as f32) * t, c0[1] as f32 + (c1[1] as f32 - c0[1] as f32) * t, c0[2] as f32 + (c1[2] as f32 - c0[2] as f32) * t];
                    break;
                }
            }
            *out = [tint[0] as u8, tint[1] as u8, tint[2] as u8];
        }
        lut[4095] = self.bg;
        lut
    }

    pub fn style(&self) -> Style {
        Style { land: self.land, line: self.line, water: self.water, contour: self.contour, contour_index: self.contour_index, contour_alpha: self.contour_alpha, sea: self.sea, flat: self.flat, bg: self.bg, no_dem: self.no_dem }
    }
}
