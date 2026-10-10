//! Themes: every colour the renderers use, as tables the client owns. A theme is the hypsometric ramp, the land and line tables, the water and contour inks with the contours' opacity, the flat ground and background, the sun and sky of the lighting environment, and the layers it turns on by default. Nothing in a cell knows about any of it.
//!
//! The three built-ins share one hierarchy so any of them reads at a glance: the ground is quiet and the shading soft (mostly sky, a little sun), so relief shows without crushing steep faces to black; roads are darker or more saturated than the ground they cross, heavier and warmer as they get bigger; trails are the most saturated thing on the map; water is unmistakably blue and a lake covers the river running into it; contours and boundaries sit under everything, faint enough to read through.

use crate::raster::{LayerMask, Style};

/// Bumped whenever the built-ins change, so a vault holding an older copy of one is refreshed.
pub const BUILTIN_REVISION: u32 = 6;

#[derive(Clone, Debug)]
pub struct Theme {
    pub name: String,
    /// Which built-in revision this is; a stored built-in older than [`BUILTIN_REVISION`] is replaced, unless the user has edited it.
    pub revision: u32,
    /// The user has changed it from what shipped: an update never overwrites it, and a reset takes the shipped one back.
    pub edited: bool,
    /// Hypsometric stops: metres and the colour there, interpolated between; the sea below half a metre takes `sea`.
    pub hypso: [(f32, [u8; 3]); 5],
    /// Every colour below carries its opacity as a fourth byte (Nick 2026-10-10): a line can be half seen, a land fill a wash, a contour faint; the ground colours' alpha means nothing and stays 255.
    pub sea: [u8; 4],
    /// Ground with the elevation tint off.
    pub flat: [u8; 4],
    /// Where there is no terrain at all.
    pub bg: [u8; 4],
    /// Ground with the terrain layer off.
    pub no_dem: [u8; 4],
    pub land: [[u8; 4]; 14],
    pub line: [[u8; 4]; 18],
    pub water: [u8; 4],
    pub contour: [u8; 4],
    pub contour_index: [u8; 4],
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
        sea: [150, 192, 228, 255],
        flat: [234, 230, 222, 255],
        bg: [44, 48, 56, 255],
        no_dem: [206, 206, 200, 255],
        land: [[0, 0, 0, 0], [190, 214, 160, 217], [222, 214, 172, 217], [198, 214, 160, 217], [200, 206, 170, 217], [150, 188, 138, 217], [172, 206, 196, 217], [232, 222, 184, 217], [206, 202, 196, 217], [234, 242, 250, 217], [202, 194, 186, 217], [210, 204, 214, 217], [222, 216, 210, 217], [150, 192, 228, 217]],
        line: [[0, 0, 0, 0], [208, 80, 34, 255], [218, 110, 40, 255], [218, 144, 46, 255], [190, 148, 58, 255], [126, 116, 98, 255], [104, 104, 112, 255], [130, 130, 138, 255], [124, 84, 48, 255], [200, 24, 70, 255], [64, 64, 74, 255], [140, 128, 164, 255], [52, 120, 200, 255], [56, 140, 72, 255], [96, 152, 68, 255], [110, 132, 58, 255], [68, 142, 110, 255], [144, 90, 166, 255]],
        water: [146, 190, 228, 255],
        contour: [140, 110, 80, 102],
        contour_index: [112, 82, 54, 168],
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
        sea: [160, 204, 240, 255],
        flat: [246, 246, 242, 255],
        bg: [226, 226, 222, 255],
        no_dem: [236, 236, 232, 255],
        land: [[0, 0, 0, 0], [222, 236, 206, 217], [240, 236, 214, 217], [224, 236, 206, 217], [226, 232, 210, 217], [200, 224, 188, 217], [208, 230, 226, 217], [244, 236, 214, 217], [228, 226, 222, 217], [246, 250, 254, 217], [226, 220, 214, 217], [232, 226, 234, 217], [236, 230, 228, 217], [160, 204, 240, 217]],
        line: [[0, 0, 0, 0], [190, 30, 30, 255], [196, 40, 38, 255], [200, 54, 44, 255], [60, 56, 56, 255], [70, 68, 68, 255], [48, 48, 52, 255], [92, 92, 96, 255], [110, 82, 54, 255], [180, 16, 120, 255], [36, 36, 44, 255], [132, 112, 164, 255], [34, 112, 200, 255], [56, 142, 76, 255], [100, 152, 80, 255], [112, 132, 62, 255], [72, 142, 118, 255], [134, 84, 160, 255]],
        water: [156, 202, 240, 255],
        contour: [170, 120, 74, 128],
        contour_index: [138, 88, 46, 204],
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
        sea: [34, 66, 104, 255],
        flat: [86, 90, 96, 255],
        bg: [14, 16, 20, 255],
        no_dem: [62, 66, 72, 255],
        land: [[0, 0, 0, 0], [72, 94, 68, 217], [90, 88, 66, 217], [74, 94, 66, 217], [78, 86, 68, 217], [56, 82, 62, 217], [60, 86, 84, 217], [110, 102, 82, 217], [90, 90, 92, 217], [126, 134, 144, 217], [84, 80, 76, 217], [82, 80, 92, 217], [94, 90, 92, 217], [34, 66, 104, 217]],
        line: [[0, 0, 0, 0], [252, 172, 76, 255], [248, 184, 88, 255], [242, 200, 112, 255], [226, 210, 156, 255], [212, 208, 196, 255], [190, 190, 196, 255], [156, 156, 164, 255], [204, 160, 116, 255], [255, 116, 146, 255], [164, 164, 174, 255], [156, 146, 188, 255], [92, 162, 240, 255], [104, 184, 114, 255], [134, 194, 112, 255], [154, 174, 100, 255], [104, 184, 154, 255], [194, 144, 218, 255]],
        water: [40, 82, 128, 255],
        contour: [146, 146, 134, 56],
        contour_index: [178, 174, 156, 112],
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
        sea: [126, 174, 222, 255],
        flat: [236, 240, 244, 255],
        bg: [40, 46, 58, 255],
        no_dem: [214, 220, 226, 255],
        land: [[0, 0, 0, 0], [190, 212, 186, 217], [214, 216, 190, 217], [192, 212, 180, 217], [196, 206, 190, 217], [134, 172, 148, 217], [170, 204, 206, 217], [226, 220, 196, 217], [198, 202, 208, 217], [242, 248, 255, 217], [198, 196, 194, 217], [204, 204, 214, 217], [216, 214, 218, 217], [126, 174, 222, 217]],
        line: [[0, 0, 0, 0], [150, 46, 40, 255], [164, 60, 46, 255], [120, 64, 56, 255], [72, 72, 82, 255], [82, 82, 92, 255], [92, 92, 102, 255], [120, 120, 130, 255], [104, 76, 54, 255], [178, 18, 66, 255], [48, 48, 60, 255], [130, 124, 160, 255], [36, 106, 188, 255], [58, 132, 84, 255], [92, 146, 82, 255], [104, 126, 70, 255], [66, 136, 120, 255], [128, 88, 160, 255]],
        water: [130, 180, 226, 255],
        contour: [104, 124, 156, 102],
        contour_index: [74, 94, 128, 163],
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
        sea: [104, 164, 206, 255],
        flat: [234, 222, 198, 255],
        bg: [52, 40, 32, 255],
        no_dem: [220, 206, 182, 255],
        land: [[0, 0, 0, 0], [206, 206, 150, 217], [222, 206, 150, 217], [196, 204, 140, 217], [212, 196, 150, 217], [156, 174, 118, 217], [176, 200, 170, 217], [240, 222, 176, 217], [214, 188, 160, 217], [240, 244, 248, 217], [206, 186, 166, 217], [210, 196, 190, 217], [222, 206, 190, 217], [104, 164, 206, 217]],
        line: [[0, 0, 0, 0], [150, 38, 28, 255], [172, 58, 32, 255], [186, 88, 38, 255], [160, 104, 54, 255], [116, 92, 74, 255], [94, 82, 74, 255], [126, 114, 104, 255], [108, 72, 42, 255], [16, 112, 148, 255], [62, 56, 54, 255], [128, 118, 148, 255], [38, 116, 186, 255], [70, 132, 70, 255], [104, 146, 72, 255], [120, 128, 62, 255], [76, 138, 112, 255], [140, 88, 152, 255]],
        water: [108, 168, 210, 255],
        contour: [150, 102, 66, 102],
        contour_index: [118, 74, 42, 168],
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
        sea: [200, 202, 208, 255],
        flat: [244, 244, 242, 255],
        bg: [30, 30, 30, 255],
        no_dem: [236, 236, 234, 255],
        land: [[0, 0, 0, 0], [234, 234, 232, 217], [238, 238, 236, 217], [232, 232, 230, 217], [232, 232, 230, 217], [218, 218, 216, 217], [226, 228, 230, 217], [242, 242, 238, 217], [228, 228, 228, 217], [250, 250, 252, 217], [226, 226, 224, 217], [228, 228, 228, 217], [232, 232, 230, 217], [200, 202, 208, 217]],
        line: [[0, 0, 0, 0], [24, 24, 24, 255], [32, 32, 32, 255], [44, 44, 44, 255], [58, 58, 58, 255], [72, 72, 72, 255], [88, 88, 88, 255], [112, 112, 112, 255], [92, 92, 92, 255], [0, 0, 0, 0], [40, 40, 40, 255], [140, 140, 140, 255], [104, 106, 114, 255], [120, 120, 120, 255], [120, 120, 120, 255], [120, 120, 120, 255], [120, 120, 120, 255], [120, 120, 120, 255]],
        water: [204, 206, 212, 255],
        contour: [128, 128, 128, 107],
        contour_index: [84, 84, 84, 184],
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
        v.extend([Field::Water, Field::Contour, Field::ContourIndex, Field::Sun, Field::Sky]);
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
            Field::Sun => "Sun".into(),
            Field::Sky => "Sky".into(),
        }
    }

    /// Whether the field is a colour (three VSF RGB bytes) as opposed to a light (three channels, 0 to 3) or a single number.
    pub fn is_colour(self) -> bool {
        !matches!(self, Field::Sun | Field::Sky)
    }
}

impl Theme {
    /// A field's values: a colour's bytes (and a stop's metres in the fourth slot), a light's channels, or an opacity in the first slot.
    pub fn get(&self, f: Field) -> [f32; 4] {
        let c = |c: [u8; 4]| [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32];
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
            Field::Sun => [self.sun[0], self.sun[1], self.sun[2], 0.0],
            Field::Sky => [self.sky[0], self.sky[1], self.sky[2], 0.0],
        }
    }

    pub fn set(&mut self, f: Field, v: [f32; 4]) {
        let c3 = [v[0].clamp(0.0, 255.0) as u8, v[1].clamp(0.0, 255.0) as u8, v[2].clamp(0.0, 255.0) as u8];
        let c = [c3[0], c3[1], c3[2], v[3].clamp(0.0, 255.0) as u8];
        match f {
            Field::Hypso(i) => self.hypso[i as usize] = (v[3].max(0.0), c3),
            Field::Sea => self.sea = c,
            Field::Flat => self.flat = c,
            Field::Bg => self.bg = c,
            Field::NoDem => self.no_dem = c,
            Field::Water => self.water = c,
            Field::Contour => self.contour = c,
            Field::ContourIndex => self.contour_index = c,
            Field::Land(i) => self.land[i as usize] = c,
            Field::Line(i) => self.line[i as usize] = c,
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

/// A cyanotype: Prussian-blue paper that deepens with height, white lines, pale cyan contours, water a paler wash. A survey blueprint, cool light from straight above.
pub fn blueprint() -> Theme {
    Theme {
        name: "Blueprint".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [34, 70, 128]), (600.0, [30, 62, 116]), (1200.0, [26, 54, 104]), (1900.0, [22, 46, 92]), (2600.0, [18, 38, 78])],
        sea: [22, 46, 92, 255],
        flat: [30, 62, 116, 255],
        bg: [10, 20, 40, 255],
        no_dem: [28, 56, 104, 255],
        land: [[0, 0, 0, 0], [40, 82, 136, 217], [44, 78, 132, 217], [40, 84, 138, 217], [38, 76, 128, 217], [30, 74, 124, 217], [36, 80, 140, 217], [50, 84, 134, 217], [44, 70, 118, 217], [70, 104, 150, 217], [42, 68, 112, 217], [46, 72, 120, 217], [50, 76, 124, 217], [22, 46, 92, 217]],
        line: [[0, 0, 0, 0], [255, 255, 255, 255], [246, 248, 255, 255], [232, 238, 250, 255], [214, 224, 242, 255], [196, 210, 232, 255], [178, 194, 220, 255], [150, 168, 200, 255], [204, 196, 160, 255], [255, 226, 120, 255], [200, 206, 220, 255], [160, 176, 204, 255], [140, 200, 240, 255], [170, 230, 210, 255], [180, 232, 190, 255], [170, 224, 170, 255], [160, 220, 220, 255], [230, 190, 240, 255]],
        water: [90, 150, 210, 255],
        contour: [140, 190, 235, 115],
        contour_index: [190, 225, 250, 191],
        sun: [0.90, 0.95, 1.05],
        sky: [1.10, 1.14, 1.24],
        layers: ALL,
    }
}

/// An atlas plate of the 1890s: cream paper that browns toward the hills, woodland in dusky green, oxblood roads, umber rail, water in a faded blue, contours in sepia. A warm lamp over the page.
pub fn atlas() -> Theme {
    Theme {
        name: "Atlas".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [236, 226, 196]), (600.0, [230, 214, 178]), (1200.0, [220, 198, 158]), (1900.0, [206, 180, 140]), (2600.0, [190, 162, 126])],
        sea: [176, 200, 206, 255],
        flat: [238, 230, 206, 255],
        bg: [60, 48, 36, 255],
        no_dem: [226, 216, 190, 255],
        land: [[0, 0, 0, 0], [210, 214, 164, 217], [228, 216, 170, 217], [206, 212, 160, 217], [212, 206, 160, 217], [164, 180, 130, 217], [188, 206, 184, 217], [236, 224, 180, 217], [206, 196, 176, 217], [238, 236, 222, 217], [204, 190, 166, 217], [210, 198, 186, 217], [218, 204, 186, 217], [176, 200, 206, 217]],
        line: [[0, 0, 0, 0], [122, 28, 30, 255], [136, 40, 34, 255], [150, 54, 40, 255], [112, 70, 46, 255], [98, 76, 56, 255], [86, 74, 62, 255], [120, 108, 92, 255], [110, 82, 50, 255], [160, 36, 60, 255], [70, 52, 40, 255], [130, 112, 92, 255], [80, 120, 150, 255], [96, 130, 90, 255], [110, 140, 90, 255], [120, 124, 70, 255], [100, 130, 110, 255], [140, 100, 130, 255]],
        water: [150, 184, 194, 255],
        contour: [150, 112, 70, 115],
        contour_index: [118, 82, 46, 184],
        sun: [1.04, 0.98, 0.86],
        sky: [1.06, 1.08, 1.12],
        layers: ALL,
    }
}

/// Volcanic: black ground that glows ember-orange toward the summits, water as cooled black glass with a blue edge, trails in bright yellow, roads in hot greys. A low red sun under a dark sky.
pub fn ember() -> Theme {
    Theme {
        name: "Ember".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [30, 24, 26]), (500.0, [64, 34, 30]), (1000.0, [124, 52, 30]), (1600.0, [196, 104, 40]), (2400.0, [244, 190, 90])],
        sea: [14, 18, 30, 255],
        flat: [48, 40, 40, 255],
        bg: [8, 6, 8, 255],
        no_dem: [40, 34, 34, 255],
        land: [[0, 0, 0, 0], [60, 56, 40, 217], [70, 58, 40, 217], [62, 60, 42, 217], [66, 56, 42, 217], [44, 50, 38, 217], [50, 58, 56, 217], [84, 70, 48, 217], [62, 58, 56, 217], [120, 118, 124, 217], [74, 64, 58, 217], [70, 62, 66, 217], [78, 70, 70, 217], [14, 18, 30, 217]],
        line: [[0, 0, 0, 0], [240, 110, 60, 255], [236, 130, 70, 255], [230, 150, 84, 255], [212, 170, 110, 255], [196, 186, 170, 255], [176, 172, 168, 255], [140, 136, 134, 255], [190, 140, 90, 255], [255, 230, 70, 255], [150, 150, 156, 255], [140, 130, 170, 255], [70, 130, 200, 255], [110, 170, 100, 255], [130, 180, 100, 255], [150, 160, 90, 255], [100, 170, 150, 255], [190, 130, 210, 255]],
        water: [30, 44, 70, 255],
        contour: [150, 110, 90, 76],
        contour_index: [200, 150, 110, 140],
        sun: [1.20, 0.80, 0.55],
        sky: [0.85, 0.84, 0.95],
        layers: ALL,
    }
}

/// Monsoon country: emerald lowlands rising through saturated greens to ochre ridges, teal water, hot-pink trails, roads in deep plum. Warm hazy light.
pub fn monsoon() -> Theme {
    Theme {
        name: "Monsoon".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [70, 160, 100]), (500.0, [116, 180, 96]), (1000.0, [184, 184, 100]), (1600.0, [222, 170, 104]), (2400.0, [240, 224, 190])],
        sea: [44, 150, 170, 255],
        flat: [160, 196, 140, 255],
        bg: [20, 40, 36, 255],
        no_dem: [140, 180, 130, 255],
        land: [[0, 0, 0, 0], [120, 196, 110, 217], [196, 204, 120, 217], [130, 200, 116, 217], [140, 190, 120, 217], [60, 150, 90, 217], [80, 180, 170, 217], [226, 212, 150, 217], [170, 176, 160, 217], [230, 240, 240, 217], [176, 160, 130, 217], [168, 160, 176, 217], [190, 184, 176, 217], [44, 150, 170, 217]],
        line: [[0, 0, 0, 0], [96, 30, 80, 255], [110, 40, 92, 255], [124, 52, 104, 255], [100, 60, 96, 255], [86, 66, 90, 255], [72, 62, 80, 255], [110, 100, 116, 255], [120, 80, 60, 255], [255, 60, 150, 255], [50, 40, 60, 255], [120, 110, 160, 255], [20, 120, 150, 255], [40, 110, 70, 255], [70, 130, 70, 255], [90, 120, 50, 255], [50, 130, 110, 255], [150, 90, 170, 255]],
        water: [50, 160, 180, 255],
        contour: [60, 90, 70, 117],
        contour_index: [30, 60, 44, 184],
        sun: [1.06, 1.00, 0.84],
        sky: [1.00, 1.08, 1.14],
        layers: ALL,
    }
}

/// An Admiralty chart: buff land kept flat and quiet, the water a pale white-blue that holds the eye, soundings-blue contours, black roads, magenta for every boundary as a chart draws its limits. Even light, no drama.
pub fn chart() -> Theme {
    Theme {
        name: "Chart".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [240, 232, 192]), (600.0, [240, 232, 192]), (1200.0, [240, 232, 192]), (1900.0, [240, 232, 192]), (2600.0, [240, 232, 192])],
        sea: [206, 230, 244, 255],
        flat: [240, 232, 192, 255],
        bg: [214, 222, 230, 255],
        no_dem: [228, 220, 186, 255],
        land: [[0, 0, 0, 0], [224, 224, 176, 217], [232, 224, 182, 217], [222, 224, 174, 217], [226, 220, 180, 217], [204, 212, 164, 217], [206, 224, 212, 217], [238, 230, 190, 217], [222, 216, 200, 217], [240, 244, 246, 217], [220, 210, 194, 217], [224, 214, 206, 217], [228, 220, 204, 217], [206, 230, 244, 217]],
        line: [[0, 0, 0, 0], [40, 40, 40, 255], [52, 52, 52, 255], [66, 66, 66, 255], [84, 84, 84, 255], [100, 100, 100, 255], [120, 120, 120, 255], [150, 150, 150, 255], [130, 100, 70, 255], [200, 30, 110, 255], [60, 60, 60, 255], [150, 140, 170, 255], [60, 130, 200, 255], [210, 60, 170, 255], [210, 60, 170, 255], [210, 60, 170, 255], [210, 60, 170, 255], [210, 60, 170, 255]],
        water: [190, 222, 240, 255],
        contour: [90, 140, 190, 102],
        contour_index: [50, 100, 160, 178],
        sun: [0.32, 0.32, 0.33],
        sky: [2.20, 2.22, 2.26],
        layers: LayerMask { hypso: false, ..ALL },
    }
}

/// Neon: near-black violet ground, ridges that light up magenta with height, cyan water, trails in electric yellow, roads in cool neon, contours as a faint blue grid. A magenta sun over a violet sky.
pub fn neon() -> Theme {
    Theme {
        name: "Neon".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [48, 30, 76]), (600.0, [74, 40, 108]), (1200.0, [118, 50, 144]), (1900.0, [180, 64, 170]), (2600.0, [244, 110, 210])],
        sea: [12, 44, 72, 255],
        flat: [60, 40, 90, 255],
        bg: [8, 6, 18, 255],
        no_dem: [52, 36, 80, 255],
        land: [[0, 0, 0, 0], [52, 90, 90, 217], [84, 64, 104, 217], [56, 92, 94, 217], [70, 72, 104, 217], [36, 82, 90, 217], [38, 90, 116, 217], [104, 84, 104, 217], [78, 70, 100, 217], [140, 140, 190, 217], [88, 64, 92, 217], [90, 60, 116, 217], [98, 70, 122, 217], [12, 44, 72, 217]],
        line: [[0, 0, 0, 0], [255, 90, 220, 255], [240, 110, 230, 255], [220, 130, 240, 255], [180, 140, 240, 255], [150, 150, 230, 255], [130, 140, 210, 255], [110, 110, 170, 255], [200, 120, 160, 255], [240, 255, 60, 255], [120, 120, 180, 255], [150, 110, 220, 255], [40, 220, 255, 255], [60, 240, 180, 255], [80, 240, 140, 255], [120, 230, 120, 255], [60, 230, 220, 255], [220, 120, 255, 255]],
        water: [20, 160, 220, 255],
        contour: [70, 100, 200, 102],
        contour_index: [110, 150, 255, 168],
        sun: [1.10, 0.70, 1.10],
        sky: [0.80, 0.80, 1.10],
        layers: ALL,
    }
}

/// First light: rose-gold valleys rising through peach to lavender summits, water a dusty teal, roads in plum, trails in deep rose. A long low sun from the side and a cool sky.
pub fn dawn() -> Theme {
    Theme {
        name: "Dawn".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [232, 196, 172]), (600.0, [238, 204, 176]), (1200.0, [240, 212, 190]), (1900.0, [228, 206, 214]), (2600.0, [214, 200, 236])],
        sea: [150, 190, 200, 255],
        flat: [238, 212, 196, 255],
        bg: [50, 40, 60, 255],
        no_dem: [226, 204, 190, 255],
        land: [[0, 0, 0, 0], [214, 206, 160, 217], [236, 210, 168, 217], [212, 208, 164, 217], [224, 200, 170, 217], [186, 180, 150, 217], [196, 206, 200, 217], [244, 220, 184, 217], [218, 200, 196, 217], [244, 240, 250, 217], [214, 196, 186, 217], [220, 200, 210, 217], [230, 210, 206, 217], [150, 190, 200, 217]],
        line: [[0, 0, 0, 0], [120, 40, 90, 255], [134, 52, 100, 255], [150, 66, 110, 255], [120, 78, 110, 255], [108, 88, 112, 255], [96, 84, 104, 255], [130, 116, 130, 255], [130, 84, 70, 255], [190, 30, 90, 255], [70, 56, 76, 255], [140, 120, 170, 255], [60, 130, 170, 255], [90, 140, 100, 255], [110, 150, 100, 255], [120, 130, 80, 255], [80, 140, 130, 255], [160, 100, 170, 255]],
        water: [140, 186, 198, 255],
        contour: [170, 120, 120, 97],
        contour_index: [140, 86, 96, 163],
        sun: [1.10, 0.92, 0.78],
        sky: [0.96, 1.04, 1.26],
        layers: ALL,
    }
}


// The seven after Nick's ratings (2026-10-09): three natural landscapes dark enough to read like imagery, a candy one, and three darker paper maps whose road and trail colours come in families.

/// Natural ground as a satellite sees it: dark forest, straw grass, tan farmland, grey-brown rock, snow; lowlands olive darkening through earth to a pale summit. Lines light against the dark ground, highways in amber, trails in hot orange. A high sun and a modest sky, so relief reads as it does from orbit.
pub fn terra() -> Theme {
    Theme {
        name: "Terra".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [92, 108, 60]), (600.0, [118, 112, 68]), (1200.0, [138, 120, 88]), (1900.0, [152, 142, 126]), (2600.0, [234, 236, 240])],
        sea: [34, 62, 92, 255],
        flat: [120, 116, 92, 255],
        bg: [14, 16, 20, 255],
        no_dem: [104, 104, 88, 255],
        land: [[0, 0, 0, 0], [122, 132, 72, 217], [154, 142, 88, 217], [92, 114, 60, 217], [110, 112, 68, 217], [56, 84, 46, 217], [70, 96, 80, 217], [192, 178, 140, 217], [128, 118, 104, 217], [230, 234, 240, 217], [150, 136, 120, 217], [130, 126, 124, 217], [140, 134, 130, 217], [34, 62, 92, 217]],
        line: [[0, 0, 0, 0], [244, 190, 88, 255], [236, 172, 80, 255], [224, 160, 82, 255], [206, 194, 152, 255], [192, 186, 160, 255], [172, 172, 166, 255], [150, 150, 146, 255], [204, 150, 90, 255], [255, 122, 56, 255], [92, 92, 96, 255], [170, 150, 190, 255], [92, 160, 220, 255], [120, 200, 120, 255], [140, 210, 110, 255], [160, 190, 90, 255], [110, 200, 170, 255], [200, 150, 220, 255]],
        water: [40, 76, 110, 255],
        contour: [220, 200, 160, 71],
        contour_index: [240, 226, 190, 128],
        sun: [1.16, 1.08, 0.96],
        sky: [0.84, 0.90, 1.04],
        layers: ALL,
    }
}

/// The far north from above: moss and lichen lowlands, grey scree, long snowfields, a steel sea; forest a dark blue-green, wetland rust. Lines in cool whites and sky blue, trails in coral. A low cold sun under a bright sky.
pub fn tundra() -> Theme {
    Theme {
        name: "Tundra".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [96, 112, 84]), (400.0, [118, 122, 96]), (800.0, [128, 126, 112]), (1300.0, [150, 152, 150]), (1800.0, [226, 232, 238])],
        sea: [52, 76, 96, 255],
        flat: [120, 124, 108, 255],
        bg: [16, 18, 22, 255],
        no_dem: [110, 114, 104, 255],
        land: [[0, 0, 0, 0], [120, 134, 90, 217], [140, 138, 100, 217], [96, 118, 84, 217], [112, 118, 92, 217], [52, 78, 66, 217], [128, 94, 64, 217], [172, 166, 142, 217], [134, 134, 130, 217], [228, 234, 240, 217], [150, 146, 136, 217], [128, 130, 132, 217], [140, 140, 142, 217], [52, 76, 96, 217]],
        line: [[0, 0, 0, 0], [236, 222, 180, 255], [232, 214, 170, 255], [226, 206, 160, 255], [206, 206, 196, 255], [190, 194, 190, 255], [172, 176, 176, 255], [150, 154, 156, 255], [196, 170, 120, 255], [255, 128, 100, 255], [84, 88, 96, 255], [160, 150, 190, 255], [120, 190, 236, 255], [120, 200, 150, 255], [140, 210, 130, 255], [160, 200, 100, 255], [110, 200, 190, 255], [190, 150, 220, 255]],
        water: [64, 96, 122, 255],
        contour: [210, 214, 206, 66],
        contour_index: [236, 238, 232, 122],
        sun: [1.00, 0.98, 0.96],
        sky: [1.00, 1.06, 1.18],
        layers: ALL,
    }
}

/// Dry country from above: red earth, ochre grass, dark gallery forest along the water, salt-pale sand, basalt rock; lowlands rust rising through ochre to a bleached summit. Lines in cream and chalk, highways in pale gold, trails in bright cyan so they stand off the red. A hot white sun.
pub fn savanna() -> Theme {
    Theme {
        name: "Savanna".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [134, 84, 54]), (400.0, [150, 108, 62]), (900.0, [164, 130, 76]), (1500.0, [176, 150, 104]), (2200.0, [222, 212, 190])],
        sea: [40, 72, 96, 255],
        flat: [146, 104, 66, 255],
        bg: [18, 12, 10, 255],
        no_dem: [130, 98, 70, 255],
        land: [[0, 0, 0, 0], [160, 136, 76, 217], [172, 150, 92, 217], [96, 108, 54, 217], [128, 112, 62, 217], [62, 82, 42, 217], [84, 100, 70, 217], [212, 194, 150, 217], [112, 92, 80, 217], [236, 236, 236, 217], [160, 130, 108, 217], [136, 120, 112, 217], [150, 130, 118, 217], [40, 72, 96, 217]],
        line: [[0, 0, 0, 0], [250, 214, 110, 255], [244, 206, 110, 255], [236, 198, 112, 255], [230, 220, 190, 255], [216, 208, 186, 255], [196, 190, 176, 255], [172, 166, 156, 255], [220, 180, 120, 255], [60, 230, 240, 255], [70, 60, 56, 255], [180, 150, 190, 255], [100, 180, 230, 255], [150, 210, 110, 255], [170, 220, 100, 255], [190, 200, 80, 255], [120, 210, 170, 255], [210, 150, 220, 255]],
        water: [52, 90, 118, 255],
        contour: [240, 214, 170, 71],
        contour_index: [250, 234, 200, 128],
        sun: [1.20, 1.14, 1.02],
        sky: [0.86, 0.90, 1.02],
        layers: ALL,
    }
}

/// Candy: lowlands in strawberry milk rising through mint and lemon to lavender and sugar-white summits, a bubblegum sea, forest in spearmint, farmland in banana. Roads in grape, cherry and cotton candy, trails in hot pink, contours in lilac. A bright sweet sun.
pub fn candy() -> Theme {
    Theme {
        name: "Candy".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [250, 206, 220]), (500.0, [206, 244, 220]), (1000.0, [252, 246, 190]), (1700.0, [222, 206, 250]), (2400.0, [255, 252, 255])],
        sea: [180, 226, 250, 255],
        flat: [250, 220, 232, 255],
        bg: [70, 40, 80, 255],
        no_dem: [244, 222, 236, 255],
        land: [[0, 0, 0, 0], [204, 246, 196, 217], [252, 238, 170, 217], [190, 244, 210, 217], [222, 240, 190, 217], [150, 230, 190, 217], [190, 236, 240, 217], [255, 236, 200, 217], [230, 214, 230, 217], [255, 255, 255, 217], [240, 210, 220, 217], [230, 206, 240, 217], [240, 212, 240, 217], [180, 226, 250, 217]],
        line: [[0, 0, 0, 0], [160, 60, 200, 255], [190, 70, 200, 255], [220, 80, 190, 255], [240, 100, 150, 255], [250, 130, 170, 255], [250, 160, 200, 255], [240, 180, 210, 255], [230, 150, 90, 255], [255, 40, 140, 255], [120, 90, 160, 255], [200, 160, 240, 255], [80, 170, 250, 255], [90, 210, 160, 255], [120, 220, 140, 255], [170, 220, 110, 255], [90, 210, 210, 255], [220, 140, 240, 255]],
        water: [150, 212, 250, 255],
        contour: [200, 160, 230, 102],
        contour_index: [170, 120, 220, 168],
        sun: [1.04, 1.00, 1.00],
        sky: [1.10, 1.08, 1.16],
        layers: ALL,
    }
}

/// A survey sheet on kraft paper: a darker tan base than Atlas, relief in warm browns, forest in dull olive, water a slate blue. Lines by family: the highways a brick family from dark to light, the local roads a brown family, the trails a red family, rail and power in near-black, the boundaries a green family. Brown contours.
pub fn survey() -> Theme {
    Theme {
        name: "Survey".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [200, 182, 142]), (600.0, [194, 170, 128]), (1200.0, [184, 156, 114]), (1900.0, [170, 140, 102]), (2600.0, [156, 126, 92])],
        sea: [142, 164, 170, 255],
        flat: [202, 186, 148, 255],
        bg: [40, 32, 24, 255],
        no_dem: [190, 176, 142, 255],
        land: [[0, 0, 0, 0], [178, 182, 128, 217], [196, 184, 132, 217], [170, 178, 122, 217], [180, 172, 124, 217], [138, 152, 104, 217], [160, 176, 154, 217], [206, 192, 146, 217], [174, 164, 142, 217], [224, 222, 206, 217], [172, 158, 132, 217], [180, 168, 150, 217], [186, 172, 150, 217], [142, 164, 170, 217]],
        line: [[0, 0, 0, 0], [104, 24, 22, 255], [126, 34, 28, 255], [148, 48, 36, 255], [96, 60, 36, 255], [112, 76, 48, 255], [124, 92, 62, 255], [138, 110, 82, 255], [110, 70, 40, 255], [168, 30, 44, 255], [36, 30, 26, 255], [60, 52, 48, 255], [60, 100, 130, 255], [58, 104, 62, 255], [74, 118, 66, 255], [92, 112, 54, 255], [66, 112, 96, 255], [84, 92, 60, 255]],
        water: [120, 148, 160, 255],
        contour: [120, 86, 50, 122],
        contour_index: [90, 60, 30, 189],
        sun: [1.02, 0.96, 0.86],
        sky: [1.04, 1.06, 1.08],
        layers: ALL,
    }
}

/// A woodcut print: grey-cream stock a shade darker than a page, relief in cool greys, forest in a soft grey-green, water a pale slate. Lines in one family of blacks: highways solid black, roads greys lightening as they shrink, trails in sepia red, boundaries in grey-green. Grey contours.
pub fn woodcut() -> Theme {
    Theme {
        name: "Woodcut".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [204, 198, 184]), (600.0, [196, 190, 176]), (1200.0, [186, 180, 168]), (1900.0, [174, 168, 158]), (2600.0, [160, 156, 150])],
        sea: [160, 170, 174, 255],
        flat: [206, 200, 186, 255],
        bg: [30, 30, 30, 255],
        no_dem: [196, 192, 182, 255],
        land: [[0, 0, 0, 0], [190, 192, 170, 217], [200, 194, 176, 217], [182, 188, 168, 217], [188, 186, 170, 217], [156, 166, 148, 217], [176, 186, 178, 217], [206, 200, 180, 217], [180, 178, 170, 217], [226, 226, 222, 217], [180, 176, 166, 217], [184, 182, 176, 217], [190, 188, 182, 217], [160, 170, 174, 217]],
        line: [[0, 0, 0, 0], [20, 18, 16, 255], [34, 32, 30, 255], [50, 48, 44, 255], [72, 70, 66, 255], [92, 90, 86, 255], [112, 110, 106, 255], [132, 130, 126, 255], [96, 76, 60, 255], [150, 50, 36, 255], [24, 22, 20, 255], [70, 66, 70, 255], [90, 104, 112, 255], [96, 112, 92, 255], [108, 122, 94, 255], [116, 118, 84, 255], [100, 118, 108, 255], [110, 100, 110, 255]],
        water: [140, 152, 158, 255],
        contour: [110, 106, 100, 107],
        contour_index: [70, 66, 62, 178],
        sun: [1.00, 1.00, 1.00],
        sky: [1.04, 1.04, 1.06],
        layers: ALL,
    }
}

/// A gazetteer on sage paper: a grey-green base darker than a page, relief in muted greens to a dun summit, forest a deeper green, water a dusty blue. Lines by family: the highways an indigo family, the local roads a slate family, the trails a burgundy family, boundaries in olive. Green-grey contours.
pub fn gazetteer() -> Theme {
    Theme {
        name: "Gazetteer".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [184, 192, 166]), (600.0, [178, 184, 156]), (1200.0, [176, 176, 146]), (1900.0, [172, 166, 138]), (2600.0, [166, 156, 132])],
        sea: [146, 166, 176, 255],
        flat: [186, 194, 168, 255],
        bg: [28, 34, 30, 255],
        no_dem: [178, 186, 164, 255],
        land: [[0, 0, 0, 0], [172, 190, 150, 217], [190, 190, 150, 217], [160, 184, 146, 217], [170, 180, 146, 217], [128, 158, 118, 217], [152, 180, 166, 217], [200, 194, 160, 217], [170, 170, 156, 217], [222, 226, 220, 217], [168, 162, 144, 217], [172, 172, 164, 217], [178, 178, 168, 217], [146, 166, 176, 217]],
        line: [[0, 0, 0, 0], [44, 40, 110, 255], [58, 54, 128, 255], [76, 72, 146, 255], [78, 86, 104, 255], [94, 102, 118, 255], [110, 118, 132, 255], [128, 134, 146, 255], [104, 86, 64, 255], [138, 32, 64, 255], [40, 42, 48, 255], [84, 78, 96, 255], [70, 112, 140, 255], [96, 112, 60, 255], [110, 124, 66, 255], [122, 128, 56, 255], [92, 120, 92, 255], [112, 104, 70, 255]],
        water: [122, 150, 164, 255],
        contour: [104, 112, 88, 117],
        contour_index: [70, 78, 56, 184],
        sun: [1.00, 0.98, 0.92],
        sky: [1.04, 1.06, 1.10],
        layers: ALL,
    }
}

pub const BUILTIN_NAMES: [&str; 20] = ["Trail", "Topo", "Night", "Alpine", "Desert", "Ink", "Blueprint", "Atlas", "Ember", "Monsoon", "Chart", "Neon", "Dawn", "Terra", "Tundra", "Savanna", "Candy", "Survey", "Woodcut", "Gazetteer"];

/// The built-in themes, as authored: VSF RGB, gamma 2. The renderers convert to the display at their one encode.
pub fn builtin() -> Vec<Theme> {
    vec![trail(), topo(), night(), alpine(), desert(), ink(), blueprint(), atlas(), ember(), monsoon(), chart(), neon(), dawn(), terra(), tundra(), savanna(), candy(), survey(), woodcut(), gazetteer()]
}

impl Theme {
    /// The hypsometric table indexed by elevation quantum >> 4 (8 m buckets); the sea below half a metre, the background in the last row for no data.
    pub fn hypso_lut(&self) -> Box<[[u8; 3]; 4096]> {
        let mut lut = Box::new([[0u8; 3]; 4096]);
        for (i, out) in lut.iter_mut().enumerate() {
            let elev = i as f32 * 16.0 * mahere_tiles::ELEV_STEP - mahere_tiles::ELEV_OFFSET;
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
        lut[4095] = [self.bg[0], self.bg[1], self.bg[2]];
        lut
    }

    pub fn style(&self) -> Style {
        Style { land: self.land, line: self.line, water: self.water, contour: self.contour, contour_index: self.contour_index, sea: self.sea, flat: self.flat, bg: self.bg, no_dem: self.no_dem }
    }
}
