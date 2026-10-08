//! Themes: every colour the renderers use, as tables the client owns. A theme is the hypsometric ramp, the land and line tables, the water and contour inks with the contours' opacity, the flat ground and background, the sun and sky of the lighting environment, and the layers it turns on by default. Nothing in a cell knows about any of it.
//!
//! The three built-ins share one hierarchy so any of them reads at a glance: the ground is quiet and the shading soft (mostly sky, a little sun), so relief shows without crushing steep faces to black; roads are darker or more saturated than the ground they cross, heavier and warmer as they get bigger; trails are the most saturated thing on the map; water is unmistakably blue and a lake covers the river running into it; contours and boundaries sit under everything, faint enough to read through.

use crate::raster::{LayerMask, Style};

/// Bumped whenever the built-ins change, so a vault holding an older copy of one is refreshed.
pub const BUILTIN_REVISION: u32 = 3;

#[derive(Clone, Debug)]
pub struct Theme {
    pub name: String,
    /// Which built-in revision this is; a stored built-in older than [`BUILTIN_REVISION`] is replaced, unless the user has edited it.
    pub revision: u32,
    /// The user has changed it from what shipped: an update never overwrites it, and a reset takes the shipped one back.
    pub edited: bool,
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

// The light is linear: flat ground under a 40° sun sees about `sun · 0.64 + sky · 0.3`, a slope turned from the sun about `sky · 0.25`, a slope facing it up to `sun + sky · 0.3`, which the display's rail rolls into white. A sun near 1 over a sky near 1.2 keeps flat ground at its authored colour with the shade side a third of it.

/// The field map: a pale, warm ground that greens in the valleys and lightens toward the peaks, honest relief under a warm sun and a cool sky, roads in greys rising to amber and red-orange, trails in crimson.
pub fn trail() -> Theme {
    Theme {
        name: "Trail".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [180, 206, 154]), (600.0, [198, 210, 162]), (1200.0, [218, 212, 178]), (1900.0, [230, 222, 204]), (2600.0, [244, 244, 246])],
        sea: [150, 192, 228],
        flat: [234, 230, 222],
        bg: [44, 48, 56],
        no_dem: [206, 206, 200],
        land: [[0, 0, 0], [190, 214, 160], [222, 214, 172], [198, 214, 160], [200, 206, 170], [150, 188, 138], [172, 206, 196], [232, 222, 184], [206, 202, 196], [234, 242, 250], [202, 194, 186], [210, 204, 214], [222, 216, 210], [150, 192, 228]],
        line: [[0, 0, 0], [208, 80, 34], [218, 110, 40], [218, 144, 46], [190, 148, 58], [126, 116, 98], [104, 104, 112], [130, 130, 138], [124, 84, 48], [200, 24, 70], [64, 64, 74], [140, 128, 164], [52, 120, 200], [56, 140, 72], [96, 152, 68], [110, 132, 58], [68, 142, 110], [144, 90, 166]],
        water: [146, 190, 228],
        contour: [140, 110, 80],
        contour_index: [112, 82, 54],
        contour_alpha: [0.40, 0.66],
        sun: [1.02, 0.95, 0.82],
        sky: [1.08, 1.12, 1.22],
        layers: ALL,
    }
}

/// A printed quad: white ground with no elevation tint, woodland in pale green, brown contours with heavier index lines, red highways, black roads, magenta trails, blue water. The relief is soft, as a print's shaded relief is.
pub fn topo() -> Theme {
    Theme {
        name: "Topo".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [236, 240, 230]), (600.0, [240, 240, 232]), (1200.0, [242, 240, 234]), (1900.0, [244, 242, 240]), (2600.0, [250, 250, 252])],
        sea: [160, 204, 240],
        flat: [246, 246, 242],
        bg: [226, 226, 222],
        no_dem: [236, 236, 232],
        land: [[0, 0, 0], [222, 236, 206], [240, 236, 214], [224, 236, 206], [226, 232, 210], [200, 224, 188], [208, 230, 226], [244, 236, 214], [228, 226, 222], [246, 250, 254], [226, 220, 214], [232, 226, 234], [236, 230, 228], [160, 204, 240]],
        line: [[0, 0, 0], [190, 30, 30], [196, 40, 38], [200, 54, 44], [60, 56, 56], [70, 68, 68], [48, 48, 52], [92, 92, 96], [110, 82, 54], [180, 16, 120], [36, 36, 44], [132, 112, 164], [34, 112, 200], [56, 142, 76], [100, 152, 80], [112, 132, 62], [72, 142, 118], [134, 84, 160]],
        water: [156, 202, 240],
        contour: [170, 120, 74],
        contour_index: [138, 88, 46],
        contour_alpha: [0.50, 0.80],
        sun: [0.72, 0.72, 0.72],
        sky: [1.55, 1.55, 1.60],
        layers: LayerMask { hypso: false, ..ALL },
    }
}

/// For a dark room or a five o'clock start: slate ground with its relief readable, land cover as quiet tints, light roads, amber highways, a bright coral for trails, water a clear deep blue rather than a hole.
pub fn night() -> Theme {
    Theme {
        name: "Night".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [80, 92, 94]), (600.0, [88, 96, 98]), (1200.0, [96, 100, 104]), (1900.0, [108, 112, 118]), (2600.0, [132, 136, 146])],
        sea: [34, 66, 104],
        flat: [86, 90, 96],
        bg: [14, 16, 20],
        no_dem: [62, 66, 72],
        land: [[0, 0, 0], [72, 94, 68], [90, 88, 66], [74, 94, 66], [78, 86, 68], [56, 82, 62], [60, 86, 84], [110, 102, 82], [90, 90, 92], [126, 134, 144], [84, 80, 76], [82, 80, 92], [94, 90, 92], [34, 66, 104]],
        line: [[0, 0, 0], [252, 172, 76], [248, 184, 88], [242, 200, 112], [226, 210, 156], [212, 208, 196], [190, 190, 196], [156, 156, 164], [204, 160, 116], [255, 116, 146], [164, 164, 174], [156, 146, 188], [92, 162, 240], [104, 184, 114], [134, 194, 112], [154, 174, 100], [104, 184, 154], [194, 144, 218]],
        water: [40, 82, 128],
        contour: [146, 146, 134],
        contour_index: [178, 174, 156],
        contour_alpha: [0.22, 0.44],
        sun: [1.15, 1.15, 1.26],
        sky: [0.80, 0.86, 1.02],
        layers: ALL,
    }
}

/// Snow and rock: a cool ground that runs from sage valleys to white summits, blue-grey contours as a Swiss sheet draws them on ice and scree, shadows tinted by a blue sky, charcoal roads, deep crimson trails.
pub fn alpine() -> Theme {
    Theme {
        name: "Alpine".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [178, 198, 186]), (800.0, [198, 212, 206]), (1500.0, [216, 226, 230]), (2200.0, [234, 240, 246]), (3000.0, [252, 253, 255])],
        sea: [126, 174, 222],
        flat: [236, 240, 244],
        bg: [40, 46, 58],
        no_dem: [214, 220, 226],
        land: [[0, 0, 0], [190, 212, 186], [214, 216, 190], [192, 212, 180], [196, 206, 190], [134, 172, 148], [170, 204, 206], [226, 220, 196], [198, 202, 208], [242, 248, 255], [198, 196, 194], [204, 204, 214], [216, 214, 218], [126, 174, 222]],
        line: [[0, 0, 0], [150, 46, 40], [164, 60, 46], [120, 64, 56], [72, 72, 82], [82, 82, 92], [92, 92, 102], [120, 120, 130], [104, 76, 54], [178, 18, 66], [48, 48, 60], [130, 124, 160], [36, 106, 188], [58, 132, 84], [92, 146, 82], [104, 126, 70], [66, 136, 120], [128, 88, 160]],
        water: [130, 180, 226],
        contour: [104, 124, 156],
        contour_index: [74, 94, 128],
        contour_alpha: [0.40, 0.64],
        sun: [1.05, 1.00, 0.94],
        sky: [1.00, 1.14, 1.40],
        layers: ALL,
    }
}

/// Canyon country: sandstone and ochre ground that pales on the high mesas, warm sun over a blue sky so the shade sides cool, red and rust roads, trails in a deep teal that stands off the rock.
pub fn desert() -> Theme {
    Theme {
        name: "Desert".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [214, 198, 162]), (600.0, [222, 196, 152]), (1200.0, [226, 190, 142]), (1900.0, [222, 190, 154]), (2600.0, [238, 228, 216])],
        sea: [104, 164, 206],
        flat: [234, 222, 198],
        bg: [52, 40, 32],
        no_dem: [220, 206, 182],
        land: [[0, 0, 0], [206, 206, 150], [222, 206, 150], [196, 204, 140], [212, 196, 150], [156, 174, 118], [176, 200, 170], [240, 222, 176], [214, 188, 160], [240, 244, 248], [206, 186, 166], [210, 196, 190], [222, 206, 190], [104, 164, 206]],
        line: [[0, 0, 0], [150, 38, 28], [172, 58, 32], [186, 88, 38], [160, 104, 54], [116, 92, 74], [94, 82, 74], [126, 114, 104], [108, 72, 42], [16, 112, 148], [62, 56, 54], [128, 118, 148], [38, 116, 186], [70, 132, 70], [104, 146, 72], [120, 128, 62], [76, 138, 112], [140, 88, 152]],
        water: [108, 168, 210],
        contour: [150, 102, 66],
        contour_index: [118, 74, 42],
        contour_alpha: [0.40, 0.66],
        sun: [1.06, 0.96, 0.82],
        sky: [1.00, 1.08, 1.28],
        layers: ALL,
    }
}

/// Black on paper: a monochrome sheet for printing or for when colour is noise. Grey relief, land cover as faint grey tints, every road in greys by weight, trails in solid black, water a cool grey.
pub fn ink() -> Theme {
    Theme {
        name: "Ink".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [240, 240, 238]), (600.0, [242, 242, 240]), (1200.0, [244, 244, 242]), (1900.0, [246, 246, 244]), (2600.0, [250, 250, 250])],
        sea: [200, 202, 208],
        flat: [244, 244, 242],
        bg: [30, 30, 30],
        no_dem: [236, 236, 234],
        land: [[0, 0, 0], [234, 234, 232], [238, 238, 236], [232, 232, 230], [232, 232, 230], [218, 218, 216], [226, 228, 230], [242, 242, 238], [228, 228, 228], [250, 250, 252], [226, 226, 224], [228, 228, 228], [232, 232, 230], [200, 202, 208]],
        line: [[0, 0, 0], [24, 24, 24], [32, 32, 32], [44, 44, 44], [58, 58, 58], [72, 72, 72], [88, 88, 88], [112, 112, 112], [92, 92, 92], [0, 0, 0], [40, 40, 40], [140, 140, 140], [104, 106, 114], [120, 120, 120], [120, 120, 120], [120, 120, 120], [120, 120, 120], [120, 120, 120]],
        water: [204, 206, 212],
        contour: [128, 128, 128],
        contour_index: [84, 84, 84],
        contour_alpha: [0.42, 0.72],
        sun: [0.86, 0.86, 0.86],
        sky: [1.36, 1.36, 1.36],
        layers: LayerMask { hypso: false, ..ALL },
    }
}

/// One editable value of a theme: a colour (VSF RGB bytes, as the file holds them), a stop's height, a contour opacity, or the sun's or sky's light per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Hypso(u8),
    Sea,
    Flat,
    Bg,
    NoDem,
    Water,
    Contour,
    ContourIndex,
    Land(u8),
    Line(u8),
    ContourAlpha(u8),
    Sun,
    Sky,
}

impl Field {
    /// Every field, in the order the editor lists them.
    pub fn all() -> Vec<Field> {
        let mut v: Vec<Field> = (0..5).map(Field::Hypso).collect();
        v.extend([Field::Sea, Field::Flat, Field::Bg, Field::NoDem]);
        v.extend((1..14).map(Field::Land));
        v.extend((1..18).map(Field::Line));
        v.extend([Field::Water, Field::Contour, Field::ContourIndex, Field::ContourAlpha(0), Field::ContourAlpha(1), Field::Sun, Field::Sky]);
        v
    }

    pub fn name(self) -> String {
        match self {
            Field::Hypso(i) => format!("Terrain stop {}", i + 1),
            Field::Sea => "Sea".into(),
            Field::Flat => "Flat ground".into(),
            Field::Bg => "No data".into(),
            Field::NoDem => "Terrain off".into(),
            Field::Water => "Water".into(),
            Field::Contour => "Contour".into(),
            Field::ContourIndex => "Index contour".into(),
            Field::Land(i) => LAND_NAMES[i as usize].to_string(),
            Field::Line(i) => CLASS_NAMES[i as usize].to_string(),
            Field::ContourAlpha(0) => "Contour opacity".into(),
            Field::ContourAlpha(_) => "Index opacity".into(),
            Field::Sun => "Sun".into(),
            Field::Sky => "Sky".into(),
        }
    }

    /// Whether the field is a colour (three VSF RGB bytes) as opposed to a light (three channels, 0 to 3) or a single number.
    pub fn is_colour(self) -> bool {
        !matches!(self, Field::ContourAlpha(_) | Field::Sun | Field::Sky)
    }
}

impl Theme {
    /// A field's values: a colour's bytes (and a stop's metres in the fourth slot), a light's channels, or an opacity in the first slot.
    pub fn get(&self, f: Field) -> [f32; 4] {
        let c = |c: [u8; 3]| [c[0] as f32, c[1] as f32, c[2] as f32, 0.0];
        match f {
            Field::Hypso(i) => {
                let (m, col) = self.hypso[i as usize];
                [col[0] as f32, col[1] as f32, col[2] as f32, m]
            }
            Field::Sea => c(self.sea),
            Field::Flat => c(self.flat),
            Field::Bg => c(self.bg),
            Field::NoDem => c(self.no_dem),
            Field::Water => c(self.water),
            Field::Contour => c(self.contour),
            Field::ContourIndex => c(self.contour_index),
            Field::Land(i) => c(self.land[i as usize]),
            Field::Line(i) => c(self.line[i as usize]),
            Field::ContourAlpha(i) => [self.contour_alpha[i as usize], 0.0, 0.0, 0.0],
            Field::Sun => [self.sun[0], self.sun[1], self.sun[2], 0.0],
            Field::Sky => [self.sky[0], self.sky[1], self.sky[2], 0.0],
        }
    }

    pub fn set(&mut self, f: Field, v: [f32; 4]) {
        let c = [v[0].clamp(0.0, 255.0) as u8, v[1].clamp(0.0, 255.0) as u8, v[2].clamp(0.0, 255.0) as u8];
        match f {
            Field::Hypso(i) => self.hypso[i as usize] = (v[3].max(0.0), c),
            Field::Sea => self.sea = c,
            Field::Flat => self.flat = c,
            Field::Bg => self.bg = c,
            Field::NoDem => self.no_dem = c,
            Field::Water => self.water = c,
            Field::Contour => self.contour = c,
            Field::ContourIndex => self.contour_index = c,
            Field::Land(i) => self.land[i as usize] = c,
            Field::Line(i) => self.line[i as usize] = c,
            Field::ContourAlpha(i) => self.contour_alpha[i as usize] = v[0].clamp(0.0, 1.0),
            Field::Sun => self.sun = [v[0].max(0.0), v[1].max(0.0), v[2].max(0.0)],
            Field::Sky => self.sky = [v[0].max(0.0), v[1].max(0.0), v[2].max(0.0)],
        }
    }

    /// One of the shipped themes, by name: edited in place, reset to what shipped, never deleted.
    pub fn is_builtin(&self) -> bool {
        BUILTIN_NAMES.contains(&self.name.as_str())
    }

    /// The shipped theme of this name, if it is one.
    pub fn shipped(&self) -> Option<Theme> {
        builtin().into_iter().find(|b| b.name == self.name)
    }
}

pub const BUILTIN_NAMES: [&str; 6] = ["Trail", "Topo", "Night", "Alpine", "Desert", "Ink"];

/// The built-in themes, as authored: VSF RGB, gamma 2. The renderers convert to the display at their one encode.
pub fn builtin() -> Vec<Theme> {
    vec![trail(), topo(), night(), alpine(), desert(), ink()]
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
