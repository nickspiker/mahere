//! Themes: every colour the renderers use, as tables the client owns. A theme is the hypsometric ramp, the land and line tables, the water and contour inks, the flat ground and background, the sun and sky colours of the lighting environment, and the layers it turns on by default. Nothing in a cell knows about any of it. Three organic ones to start — Trail, Topo, Night — and the shape is what a user-authored VSF theme would fill.

use crate::raster::{LayerMask, Style};

#[derive(Clone, Debug)]
pub struct Theme {
    pub name: String,
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
    /// Normal-incidence irradiance of the sun and the radiance tint of the sky dome.
    pub sun: [f32; 3],
    pub sky: [f32; 3],
    pub layers: LayerMask,
}

/// Land cover classes in table order (1-based ids; 0 is empty).
pub const LAND_NAMES: [&str; 14] = ["", "Grass", "Farmland", "Orchard", "Scrub", "Forest", "Wetland", "Sand", "Rock", "Glacier", "Quarry", "Industrial", "Urban", "Water"];

/// Line classes in table order (1-based ids; 0 is empty).
pub const CLASS_NAMES: [&str; 18] = ["", "Motorway", "Trunk", "Primary", "Secondary", "Tertiary", "Residential", "Service", "Track", "Path", "Rail", "Power", "Waterway", "National park", "Wilderness", "National forest", "Protected", "County and state"];

pub fn trail() -> Theme {
    Theme {
    name: "Trail".to_string(),
    hypso: [(0.0, [72, 96, 60]), (500.0, [110, 112, 70]), (1200.0, [138, 116, 84]), (2200.0, [160, 152, 146]), (3000.0, [238, 240, 245])],
    sea: [26, 58, 82],
    flat: [232, 232, 230],
    bg: [18, 20, 26],
    no_dem: [96, 100, 96],
    land: [[0, 0, 0], [122, 162, 90], [178, 170, 108], [138, 160, 88], [118, 138, 88], [66, 108, 66], [92, 140, 128], [214, 200, 160], [150, 146, 140], [226, 232, 240], [130, 120, 110], [140, 132, 142], [152, 140, 138], [26, 58, 82]],
    line: [[0, 0, 0], [245, 150, 60], [238, 175, 62], [240, 208, 84], [212, 212, 168], [182, 192, 182], [142, 147, 158], [112, 117, 128], [152, 120, 88], [80, 230, 120], [125, 122, 128], [148, 136, 160], [120, 190, 255], [96, 200, 96], [150, 210, 120], [140, 160, 80], [110, 190, 150], [190, 150, 210]],
    water: [26, 58, 82],
    contour: [92, 62, 34],
    contour_index: [64, 40, 18],
    sun: [0.74, 0.70, 0.62],
    sky: [0.85, 0.95, 1.15],
    layers: LayerMask { dem: true, land: true, water: true, line: true, debug: false, imagery: false, contours: true, slope: false, infrared: false, hypso: true, boundaries: true },
    }
}

/// A printed quad: near-white ground, brown contours with heavy index lines, dark roads, blue water, the land cover pale enough to read through.
pub fn topo() -> Theme {
    Theme {
    name: "Topo".to_string(),
    hypso: [(0.0, [226, 232, 220]), (500.0, [232, 230, 218]), (1200.0, [236, 230, 216]), (2200.0, [240, 238, 232]), (3000.0, [250, 250, 252])],
    sea: [170, 205, 235],
    flat: [238, 238, 234],
    bg: [226, 226, 222],
    no_dem: [206, 206, 202],
    land: [[0, 0, 0], [208, 226, 190], [226, 220, 184], [210, 222, 186], [204, 214, 184], [178, 204, 170], [190, 214, 206], [236, 228, 200], [214, 212, 208], [240, 244, 250], [214, 206, 198], [218, 212, 222], [224, 216, 214], [170, 205, 235]],
    line: [[0, 0, 0], [60, 60, 64], [70, 70, 74], [84, 84, 88], [100, 100, 104], [118, 118, 122], [136, 136, 140], [156, 156, 160], [150, 110, 70], [168, 80, 40], [96, 96, 100], [150, 130, 170], [40, 90, 160], [90, 160, 90], [120, 170, 110], [130, 140, 70], [100, 160, 130], [140, 90, 170]],
    water: [150, 190, 225],
    contour: [120, 80, 40],
    contour_index: [80, 50, 20],
    sun: [0.70, 0.70, 0.68],
    sky: [1.0, 1.0, 1.05],
    layers: LayerMask { dem: true, land: true, water: true, line: true, debug: false, imagery: false, contours: true, slope: false, infrared: false, hypso: false, boundaries: true },
    }
}

/// A trailhead at five in the morning: dark ground, a dim warm sun, amber lines, water near black.
pub fn night() -> Theme {
    Theme {
    name: "Night".to_string(),
    hypso: [(0.0, [18, 28, 20]), (500.0, [28, 30, 22]), (1200.0, [38, 32, 26]), (2200.0, [44, 42, 40]), (3000.0, [70, 72, 78])],
    sea: [10, 22, 40],
    flat: [28, 30, 34],
    bg: [6, 8, 12],
    no_dem: [30, 32, 36],
    land: [[0, 0, 0], [30, 44, 26], [42, 40, 26], [34, 42, 24], [30, 36, 24], [20, 32, 22], [24, 36, 34], [50, 46, 36], [40, 40, 40], [58, 62, 70], [34, 32, 30], [36, 34, 38], [40, 36, 36], [10, 22, 40]],
    line: [[0, 0, 0], [230, 150, 50], [220, 160, 60], [210, 170, 70], [190, 160, 90], [160, 140, 100], [130, 120, 100], [110, 100, 90], [150, 110, 70], [240, 200, 90], [110, 100, 110], [120, 100, 140], [60, 110, 170], [80, 140, 80], [100, 150, 90], [110, 120, 60], [80, 130, 100], [140, 100, 160]],
    water: [10, 22, 40],
    contour: [90, 74, 44],
    contour_index: [120, 98, 60],
    sun: [0.40, 0.33, 0.24],
    sky: [0.35, 0.45, 0.70],
    layers: LayerMask { dem: true, land: true, water: true, line: true, debug: false, imagery: false, contours: true, slope: false, infrared: false, hypso: true, boundaries: true },
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
            if elev < 0.5 {
                *out = self.sea;
                continue;
            }
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
        Style { land: self.land, line: self.line, water: self.water, contour: self.contour, contour_index: self.contour_index, flat: self.flat, bg: self.bg, no_dem: self.no_dem }
    }
}
