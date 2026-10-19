//! Themes: every colour the renderers use, as tables the client owns. A theme is the hypsometric ramp, the land and line tables, the water and contour inks with the contours' opacity, the flat ground and background, the sun and sky of the lighting environment, and the layers it turns on by default. Nothing in a cell knows about any of it.
//!
//! The three built-ins share one hierarchy so any of them reads at a glance: the ground is quiet and the shading soft (mostly sky, a little sun), so relief shows without crushing steep faces to black; roads are darker or more saturated than the ground they cross, heavier and warmer as they get bigger; trails are the most saturated thing on the map; water is unmistakably blue and a lake covers the river running into it; contours and boundaries sit under everything, faint enough to read through.

use crate::raster::{LayerMask, Style};

/// Bumped whenever the built-ins change, so a vault holding an older copy of one is refreshed.
pub const BUILTIN_REVISION: u32 = 11;

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
    /// Land ownership, a colour per manager (`mahere_tiles::landuse::own`): its alpha is the edge's, the fill inside is a fifth of it, so the land reads as a wash and its boundary as a line.
    pub own: [[u8; 4]; 13],
    /// Wilderness and wilderness study areas over the ownership, in the same way.
    pub wild: [u8; 4],
    /// Motor vehicle use routes, a colour per class (`mahere_tiles::landuse::mvum`), drawn as a band under the map's own lines.
    pub mvum: [[u8; 4]; 15],
    /// The theme's own light, a probe in the screen's frame (x right, y up, z out), kept absolute as captured or authored: held to its level when used, so flat ground lights to one, and turned with the map's own sun. None: the sun and sky below.
    pub probe: Option<crate::sh::Sh9>,
    /// Normal-incidence irradiance of the sun and the radiance tint of the sky dome; a sky of 1 lights flat ground to 0.3.
    pub sun: [f32; 3],
    pub sky: [f32; 3],
    pub layers: LayerMask,
    /// The labels' colours, in [`LABEL_NAMES`] order, VSF RGB at gamma 2 with opacity, as every theme colour (Nick 2026-10-13: labels and their look in the theme).
    pub label: [[u8; 4]; 11],
    /// How labels are lettered: size (times the base), glow (the halo's width, a fraction of the letter), smoothing (the curve fit's radius, in letters), spacing (room kept around each label, a fraction of the letter).
    pub label_look: [f32; 4],
    /// How many labels (Nick 2026-10-14: sparse and big): density (how far down the order of kinds labels go, 0 towns only to 1 every kind) and repeat (how far apart a road's or river's name repeats, a fraction of the screen's long side).
    pub label_amount: [f32; 2],
    /// The lettering's typeface, one of [`LABEL_FONTS`] (Nick 2026-10-14: theme the map's type too, Oxanium for Neon and the like).
    pub label_font: String,
}

/// What each label colour is for, by index.
pub const LABEL_NAMES: [&str; 11] = ["Places", "Summits", "Water names", "Roads", "Trails", "Boundaries", "Area names", "Small marks", "Halo", "Route", "Route casing"];

/// Labels on a light ground: dark letters on a paper halo.
pub const LABELS_LIGHT: [[u8; 4]; 11] = [[30, 30, 34, 255], [90, 54, 26, 255], [26, 82, 134, 255], [44, 44, 48, 255], [116, 42, 24, 255], [42, 92, 44, 255], [46, 94, 48, 255], [70, 70, 76, 255], [242, 240, 232, 210], [30, 110, 240, 255], [255, 255, 255, 255]];

/// Labels on a dark ground: light letters on a dark halo.
pub const LABELS_DARK: [[u8; 4]; 11] = [[238, 238, 234, 255], [232, 200, 160, 255], [142, 196, 240, 255], [230, 230, 230, 255], [240, 160, 128, 255], [158, 214, 154, 255], [168, 216, 160, 255], [208, 208, 216, 255], [16, 20, 24, 200], [74, 154, 255, 255], [10, 14, 20, 255]];

/// The look labels start from: seven tenths of the panel's font (the panel's own size read big on the map, 2026-10-14), a glow of a ninth of a letter, a curve fit half a letter wide, an eighth of a letter of room.
pub const LABEL_LOOK: [f32; 4] = [0.7, 0.11, 0.5, 0.12];

/// The amount labels start from: every kind, a road's name every quarter of the screen.
pub const LABEL_AMOUNT: [f32; 2] = [1.0, 0.25];

/// What each step of the density slider letters, by the order labels are placed in: towns first, every kind last.
pub const LABEL_TIERS: [&str; 10] = ["towns", "summits", "lakes", "highways", "rivers", "villages", "roads, trails and parks", "streets and rail", "named areas", "every mark"];

/// The typefaces a theme can letter its map in: the panel's Open Sans, and static cuts of open fonts in five weights each (crates/mahere-panel/assets/fonts).
pub const LABEL_FONTS: [&str; 7] = ["Open Sans", "Oxanium", "EB Garamond", "Roboto Condensed", "Fredoka", "Cinzel", "Mahere Mono"];

/// The deepest tier a density lets through.
pub fn label_tier_limit(density: f32) -> u8 {
    (density.clamp(0.0, 1.0) * (LABEL_TIERS.len() - 1) as f32).round() as u8
}

/// Land cover classes in table order (1-based ids; 0 is empty).
pub const LAND_NAMES: [&str; 14] = ["", "Grass", "Farmland", "Orchard", "Scrub", "Forest", "Wetland", "Sand", "Rock", "Glacier", "Quarry", "Industrial", "Urban", "Water"];

/// Line classes in table order (1-based ids; 0 is empty).
pub const CLASS_NAMES: [&str; 18] = ["", "Motorway", "Trunk", "Primary", "Secondary", "Tertiary", "Residential", "Service", "Track", "Path", "Rail", "Power", "Waterway", "National park", "Wilderness", "National forest", "Protected", "County and state"];

/// A line class's word for a reader (the tap card, turn-by-turn): the editor's eighteen, then the classes added after them, whose colours each theme derives ([`Theme::lines`]).
pub fn class_word(class: u8) -> &'static str {
    match class {
        18 => "Ferry",
        19 => "Lift",
        20 => "Ridge",
        21 => "Cliff",
        c => CLASS_NAMES.get(c as usize).copied().filter(|s| !s.is_empty()).unwrap_or("Line"),
    }
}

const ALL: LayerMask = LayerMask { dem: true, land: true, water: true, line: true, debug: false, imagery: false, contours: true, slope: false, infrared: false, hypso: true, boundaries: true, own: true, mvum: true, raw: false, labels: true, geoid: false };

/// The ownership colours every built-in starts from (Nick 2026-10-10): the agencies' customary map colours, Forest Service green, BLM yellow, Park Service purple, state blue, tribal brown; the alpha is the boundary's, the wash a fifth of it.
pub const OWN: [[u8; 4]; 13] = [[0, 0, 0, 0], [70, 150, 70, 200], [236, 204, 70, 200], [150, 110, 200, 200], [70, 160, 180, 200], [160, 140, 120, 200], [80, 130, 226, 200], [90, 186, 150, 200], [70, 166, 206, 200], [200, 132, 74, 200], [214, 120, 156, 200], [150, 174, 90, 200], [170, 170, 170, 200]];

/// Wilderness: a deep green edge and wash over the manager's, where bikes and motors are banned.
pub const WILD: [u8; 4] = [30, 96, 40, 230];

/// Motor vehicle use: highway-legal roads blue, roads open to all vehicles green, trails for all vehicles darker green, 50-inch trails orange, motorcycle trails red, other OHV purple, special designations grey; seasonal ones at half the opacity.
pub const MVUM: [[u8; 4]; 15] = [[0, 0, 0, 0], [70, 120, 230, 220], [70, 120, 230, 120], [60, 190, 80, 220], [60, 190, 80, 120], [40, 150, 60, 220], [40, 150, 60, 120], [240, 150, 40, 220], [240, 150, 40, 120], [220, 60, 60, 220], [220, 60, 60, 120], [170, 80, 200, 220], [170, 80, 200, 120], [130, 130, 130, 220], [130, 130, 130, 120]];

// The light is linear: flat ground under a 40° sun sees about `sun · 0.64 + sky · 0.3`, a slope turned from the sun about `sky · 0.25`, a slope facing it up to `sun + sky · 0.3`, which the display's rail rolls into white. A sun near 1 over a sky near 1.2 keeps flat ground at its authored colour with the shade side a third of it.

/// The field map: a pale, warm ground that greens in the valleys and lightens toward the peaks, honest relief under a warm sun and a cool sky, roads in greys rising to amber and red-orange, trails in crimson.
pub fn trail() -> Theme {
    Theme {
        name: "Trail".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [113, 135, 86]), (600.0, [135, 144, 94]), (1200.0, [155, 144, 106]), (1900.0, [167, 158, 140]), (2600.0, [200, 200, 196])],
        sea: [92, 132, 170, 255],
        flat: [171, 166, 153, 255],
        bg: [44, 48, 56, 255],
        no_dem: [153, 153, 148, 255],
        land: [[0, 0, 0, 0], [144, 162, 121, 217], [167, 162, 130, 217], [149, 162, 121, 217], [151, 156, 129, 217], [113, 142, 104, 217], [130, 156, 148, 217], [176, 167, 140, 217], [156, 153, 148, 217], [177, 183, 189, 217], [153, 147, 140, 217], [158, 154, 162, 217], [167, 163, 158, 217], [113, 145, 173, 217]],
        line: [[0, 0, 0, 0], [208, 80, 34, 255], [218, 110, 40, 255], [218, 144, 46, 255], [190, 148, 58, 255], [126, 116, 98, 255], [104, 104, 112, 255], [130, 130, 138, 255], [124, 84, 48, 255], [200, 24, 70, 255], [64, 64, 74, 255], [140, 128, 164, 255], [52, 120, 200, 255], [56, 140, 72, 255], [96, 152, 68, 255], [110, 132, 58, 255], [68, 142, 110, 255], [144, 90, 166, 255]],
        water: [90, 136, 176, 255],
        contour: [140, 110, 80, 102],
        contour_index: [112, 82, 54, 168],
        sun: [1.02, 0.95, 0.82],
        sky: [1.08, 1.12, 1.22],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 40.0, [1.000, 0.880, 0.700], [0.080, 0.110, 0.170], [0.150, 0.150, 0.150], [0.060, 0.050, 0.035])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// A printed quad: white ground with no elevation tint, woodland in pale green, brown contours with heavier index lines, red highways, black roads, magenta trails, blue water. The relief is soft, as a print's shaded relief is.
pub fn topo() -> Theme {
    Theme {
        name: "Topo".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [185, 188, 174]), (600.0, [188, 188, 178]), (1200.0, [190, 187, 180]), (1900.0, [192, 190, 187]), (2600.0, [201, 201, 202])],
        sea: [128, 170, 206, 255],
        flat: [194, 192, 183, 255],
        bg: [226, 226, 222, 255],
        no_dem: [181, 181, 176, 255],
        land: [[0, 0, 0, 0], [172, 183, 159, 217], [186, 183, 165, 217], [173, 183, 159, 217], [175, 180, 163, 217], [155, 173, 145, 217], [161, 178, 175, 217], [189, 183, 165, 217], [177, 175, 172, 217], [190, 194, 197, 217], [175, 171, 165, 217], [180, 175, 181, 217], [183, 178, 177, 217], [124, 158, 186, 217]],
        line: [[0, 0, 0, 0], [190, 30, 30, 255], [196, 40, 38, 255], [200, 54, 44, 255], [60, 56, 56, 255], [70, 68, 68, 255], [48, 48, 52, 255], [92, 92, 96, 255], [110, 82, 54, 255], [180, 16, 120, 255], [36, 36, 44, 255], [132, 112, 164, 255], [34, 112, 200, 255], [56, 142, 76, 255], [100, 152, 80, 255], [112, 132, 62, 255], [72, 142, 118, 255], [134, 84, 160, 255]],
        water: [124, 168, 206, 255],
        contour: [170, 120, 74, 128],
        contour_index: [138, 88, 46, 204],
        sun: [0.72, 0.72, 0.72],
        sky: [1.55, 1.55, 1.60],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 50.0, [0.700, 0.700, 0.680], [0.130, 0.140, 0.150], [0.130, 0.130, 0.130], [0.050, 0.050, 0.050])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: LayerMask { hypso: false, ..ALL },
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// For a dark room or a five o'clock start: slate ground with its relief readable, land cover as quiet tints, light roads, amber highways, a bright coral for trails, water a clear deep blue rather than a hole.
pub fn night() -> Theme {
    Theme {
        name: "Night".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [64, 78, 74]), (600.0, [72, 82, 78]), (1200.0, [80, 86, 86]), (1900.0, [92, 96, 100]), (2600.0, [118, 122, 132])],
        sea: [26, 52, 84, 255],
        flat: [70, 76, 78, 255],
        bg: [14, 16, 20, 255],
        no_dem: [60, 62, 66, 255],
        land: [[0, 0, 0, 0], [65, 85, 61, 217], [81, 79, 59, 217], [67, 85, 59, 217], [70, 77, 61, 217], [50, 74, 56, 217], [54, 77, 76, 217], [99, 92, 74, 217], [81, 81, 83, 217], [113, 121, 130, 217], [76, 72, 68, 217], [74, 72, 83, 217], [85, 81, 83, 217], [31, 59, 94, 217]],
        line: [[0, 0, 0, 0], [252, 172, 76, 255], [248, 184, 88, 255], [242, 200, 112, 255], [226, 210, 156, 255], [212, 208, 196, 255], [190, 190, 196, 255], [156, 156, 164, 255], [204, 160, 116, 255], [255, 116, 146, 255], [164, 164, 174, 255], [156, 146, 188, 255], [92, 162, 240, 255], [104, 184, 114, 255], [134, 194, 112, 255], [154, 174, 100, 255], [104, 184, 154, 255], [194, 144, 218, 255]],
        water: [34, 70, 110, 255],
        contour: [146, 146, 134, 56],
        contour_index: [178, 174, 156, 112],
        sun: [1.15, 1.15, 1.26],
        sky: [0.80, 0.86, 1.02],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 35.0, [0.600, 0.700, 1.000], [0.050, 0.070, 0.120], [0.060, 0.070, 0.090], [0.020, 0.020, 0.025])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// Snow and rock: a cool ground that runs from sage valleys to white summits, blue-grey contours as a Swiss sheet draws them on ice and scree, shadows tinted by a blue sky, charcoal roads, deep crimson trails.
pub fn alpine() -> Theme {
    Theme {
        name: "Alpine".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [101, 124, 101]), (800.0, [126, 140, 124]), (1500.0, [148, 153, 149]), (2200.0, [176, 180, 184]), (3000.0, [212, 214, 216])],
        sea: [80, 122, 170, 255],
        flat: [176, 180, 184, 255],
        bg: [40, 46, 58, 255],
        no_dem: [158, 162, 166, 255],
        land: [[0, 0, 0, 0], [144, 160, 140, 217], [162, 163, 144, 217], [145, 160, 136, 217], [148, 156, 144, 217], [102, 130, 112, 217], [129, 154, 156, 217], [171, 166, 148, 217], [149, 153, 158, 217], [183, 187, 193, 217], [149, 148, 147, 217], [154, 154, 162, 217], [163, 162, 165, 217], [95, 131, 167, 217]],
        line: [[0, 0, 0, 0], [150, 46, 40, 255], [164, 60, 46, 255], [120, 64, 56, 255], [72, 72, 82, 255], [82, 82, 92, 255], [92, 92, 102, 255], [120, 120, 130, 255], [104, 76, 54, 255], [178, 18, 66, 255], [48, 48, 60, 255], [130, 124, 160, 255], [36, 106, 188, 255], [58, 132, 84, 255], [92, 146, 82, 255], [104, 126, 70, 255], [66, 136, 120, 255], [128, 88, 160, 255]],
        water: [84, 128, 176, 255],
        contour: [104, 124, 156, 102],
        contour_index: [74, 94, 128, 163],
        sun: [1.05, 1.00, 0.94],
        sky: [1.00, 1.14, 1.40],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 45.0, [1.000, 0.970, 0.920], [0.080, 0.100, 0.150], [0.160, 0.165, 0.175], [0.080, 0.078, 0.074])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// Canyon country: sandstone and ochre ground that pales on the high mesas, warm sun over a blue sky so the shade sides cool, red and rust roads, trails in a deep teal that stands off the rock.
pub fn desert() -> Theme {
    Theme {
        name: "Desert".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [162, 138, 101]), (600.0, [169, 134, 92]), (1200.0, [166, 123, 81]), (1900.0, [156, 121, 92]), (2600.0, [197, 184, 169])],
        sea: [74, 126, 166, 255],
        flat: [180, 162, 132, 255],
        bg: [52, 40, 32, 255],
        no_dem: [166, 156, 138, 255],
        land: [[0, 0, 0, 0], [161, 161, 118, 217], [174, 161, 118, 217], [154, 159, 109, 217], [166, 154, 118, 217], [122, 136, 92, 217], [138, 156, 132, 217], [188, 174, 138, 217], [167, 147, 125, 217], [188, 190, 194, 217], [161, 145, 130, 217], [164, 154, 149, 217], [174, 161, 149, 217], [81, 128, 161, 217]],
        line: [[0, 0, 0, 0], [150, 38, 28, 255], [172, 58, 32, 255], [186, 88, 38, 255], [160, 104, 54, 255], [116, 92, 74, 255], [94, 82, 74, 255], [126, 114, 104, 255], [108, 72, 42, 255], [16, 112, 148, 255], [62, 56, 54, 255], [128, 118, 148, 255], [38, 116, 186, 255], [70, 132, 70, 255], [104, 146, 72, 255], [120, 128, 62, 255], [76, 138, 112, 255], [140, 88, 152, 255]],
        water: [78, 130, 170, 255],
        contour: [150, 102, 66, 102],
        contour_index: [118, 74, 42, 168],
        sun: [1.06, 0.96, 0.82],
        sky: [1.00, 1.08, 1.28],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 50.0, [1.000, 0.900, 0.720], [0.100, 0.130, 0.180], [0.200, 0.180, 0.150], [0.100, 0.070, 0.040])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// Black on paper: a monochrome sheet for printing or for when colour is noise. Grey relief, land cover as faint grey tints, every road in greys by weight, trails in solid black, water a cool grey.
pub fn ink() -> Theme {
    Theme {
        name: "Ink".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [188, 187, 180]), (600.0, [190, 188, 181]), (1200.0, [192, 190, 183]), (1900.0, [194, 192, 187]), (2600.0, [201, 199, 195])],
        sea: [172, 174, 180, 255],
        flat: [192, 190, 183, 255],
        bg: [30, 30, 30, 255],
        no_dem: [181, 180, 174, 255],
        land: [[0, 0, 0, 0], [186, 186, 184, 217], [188, 188, 187, 217], [184, 184, 182, 217], [184, 184, 182, 217], [172, 172, 171, 217], [179, 180, 182, 217], [192, 192, 188, 217], [180, 180, 180, 217], [198, 198, 200, 217], [179, 179, 178, 217], [180, 180, 180, 217], [184, 184, 182, 217], [158, 160, 165, 217]],
        line: [[0, 0, 0, 0], [24, 24, 24, 255], [32, 32, 32, 255], [44, 44, 44, 255], [58, 58, 58, 255], [72, 72, 72, 255], [88, 88, 88, 255], [112, 112, 112, 255], [92, 92, 92, 255], [0, 0, 0, 0], [40, 40, 40, 255], [140, 140, 140, 255], [104, 106, 114, 255], [120, 120, 120, 255], [120, 120, 120, 255], [120, 120, 120, 255], [120, 120, 120, 255], [120, 120, 120, 255]],
        water: [176, 178, 184, 255],
        contour: [128, 128, 128, 107],
        contour_index: [84, 84, 84, 184],
        sun: [0.86, 0.86, 0.86],
        sky: [1.36, 1.36, 1.36],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 50.0, [0.750, 0.750, 0.750], [0.120, 0.120, 0.120], [0.120, 0.120, 0.120], [0.040, 0.040, 0.040])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: LayerMask { hypso: false, ..ALL },
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
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
    Own(u8),
    Wild,
    Mvum(u8),
    Sun,
    Sky,
    /// A label colour, by [`LABEL_NAMES`] index.
    Label(u8),
    /// The labels' look: size, glow, smoothing, spacing.
    LabelLook,
    /// How many labels: density and repeat.
    LabelAmount,
    /// The lettering's typeface, an index into [`LABEL_FONTS`].
    LabelFont,
}

impl Field {
    /// Every field, in the order the editor lists them.
    pub fn all() -> Vec<Field> {
        let mut v: Vec<Field> = (0..5).map(Field::Hypso).collect();
        v.extend([Field::Sea, Field::Flat, Field::Bg, Field::NoDem]);
        v.extend((1..14).map(Field::Land));
        v.extend((1..18).map(Field::Line));
        v.extend((1..13).map(Field::Own));
        v.push(Field::Wild);
        v.extend((1..15).map(Field::Mvum));
        v.extend([Field::Water, Field::Contour, Field::ContourIndex, Field::Sun, Field::Sky]);
        v.extend([Field::LabelLook, Field::LabelAmount, Field::LabelFont]);
        v.extend((0..11).map(Field::Label));
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
            Field::Own(i) => mahere_tiles::landuse::own::NAMES[i as usize].to_string(),
            Field::Wild => "Wilderness".into(),
            Field::Mvum(i) => mahere_tiles::landuse::mvum::NAMES[i as usize].to_string(),
            Field::Sun => "Sun".into(),
            Field::Sky => "Sky".into(),
            Field::Label(i) => LABEL_NAMES[i as usize].to_string(),
            Field::LabelLook => "Label look".into(),
            Field::LabelAmount => "Label amount".into(),
            Field::LabelFont => "Label typeface".into(),
        }
    }

    /// Whether the field is a colour (three VSF RGB bytes) as opposed to a light (three channels, 0 to 3) or a single number.
    pub fn is_colour(self) -> bool {
        !matches!(self, Field::Sun | Field::Sky | Field::LabelLook | Field::LabelAmount | Field::LabelFont)
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
            Field::Own(i) => c(self.own[i as usize]),
            Field::Wild => c(self.wild),
            Field::Mvum(i) => c(self.mvum[i as usize]),
            Field::Sun => [self.sun[0], self.sun[1], self.sun[2], 0.0],
            Field::Sky => [self.sky[0], self.sky[1], self.sky[2], 0.0],
            Field::Label(i) => c(self.label[i as usize]),
            Field::LabelLook => self.label_look,
            Field::LabelAmount => [self.label_amount[0], self.label_amount[1], 0.0, 0.0],
            Field::LabelFont => [LABEL_FONTS.iter().position(|f| *f == self.label_font).unwrap_or(0) as f32, 0.0, 0.0, 0.0],
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
            Field::Own(i) => self.own[i as usize] = c,
            Field::Wild => self.wild = c,
            Field::Mvum(i) => self.mvum[i as usize] = c,
            Field::Sun => self.sun = [v[0].max(0.0), v[1].max(0.0), v[2].max(0.0)],
            Field::Sky => self.sky = [v[0].max(0.0), v[1].max(0.0), v[2].max(0.0)],
            Field::Label(i) => self.label[i as usize] = c,
            Field::LabelLook => self.label_look = [v[0].clamp(0.3, 2.0), v[1].clamp(0.0, 0.4), v[2].clamp(0.0, 2.0), v[3].clamp(0.0, 1.0)],
            Field::LabelAmount => self.label_amount = [v[0].clamp(0.0, 1.0), v[1].clamp(0.1, 1.0)],
            Field::LabelFont => self.label_font = LABEL_FONTS[(v[0].round().max(0.0) as usize).min(LABEL_FONTS.len() - 1)].to_string(),
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
        hypso: [(0.0, [30, 62, 112]), (600.0, [26, 56, 102]), (1200.0, [24, 48, 92]), (1900.0, [20, 40, 82]), (2600.0, [16, 34, 70])],
        sea: [18, 40, 80, 255],
        flat: [26, 56, 102, 255],
        bg: [10, 20, 40, 255],
        no_dem: [20, 40, 80, 255],
        land: [[0, 0, 0, 0], [40, 82, 136, 217], [44, 78, 132, 217], [40, 84, 138, 217], [38, 76, 128, 217], [30, 74, 124, 217], [36, 80, 140, 217], [50, 84, 134, 217], [44, 70, 118, 217], [70, 104, 150, 217], [42, 68, 112, 217], [46, 72, 120, 217], [50, 76, 124, 217], [22, 46, 92, 217]],
        line: [[0, 0, 0, 0], [255, 255, 255, 255], [246, 248, 255, 255], [232, 238, 250, 255], [214, 224, 242, 255], [196, 210, 232, 255], [178, 194, 220, 255], [150, 168, 200, 255], [204, 196, 160, 255], [255, 226, 120, 255], [200, 206, 220, 255], [160, 176, 204, 255], [140, 200, 240, 255], [170, 230, 210, 255], [180, 232, 190, 255], [170, 224, 170, 255], [160, 220, 220, 255], [230, 190, 240, 255]],
        water: [80, 136, 196, 255],
        contour: [140, 190, 235, 115],
        contour_index: [190, 225, 250, 191],
        sun: [0.90, 0.95, 1.05],
        sky: [1.10, 1.14, 1.24],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 45.0, [0.700, 0.800, 1.000], [0.150, 0.180, 0.240], [0.150, 0.180, 0.240], [0.040, 0.050, 0.070])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// An atlas plate of the 1890s: cream paper that browns toward the hills, woodland in dusky green, oxblood roads, umber rail, water in a faded blue, contours in sepia. A warm lamp over the page.
pub fn atlas() -> Theme {
    Theme {
        name: "Atlas".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [187, 175, 142]), (600.0, [180, 162, 127]), (1200.0, [167, 146, 111]), (1900.0, [151, 127, 98]), (2600.0, [136, 113, 87])],
        sea: [142, 168, 174, 255],
        flat: [191, 180, 153, 255],
        bg: [60, 48, 36, 255],
        no_dem: [178, 171, 151, 255],
        land: [[0, 0, 0, 0], [168, 171, 131, 217], [183, 173, 136, 217], [165, 170, 128, 217], [170, 165, 128, 217], [131, 144, 104, 217], [150, 165, 147, 217], [189, 179, 144, 217], [165, 157, 141, 217], [190, 189, 177, 217], [164, 152, 133, 217], [168, 158, 149, 217], [175, 164, 149, 217], [141, 160, 165, 217]],
        line: [[0, 0, 0, 0], [122, 28, 30, 255], [136, 40, 34, 255], [150, 54, 40, 255], [112, 70, 46, 255], [98, 76, 56, 255], [86, 74, 62, 255], [120, 108, 92, 255], [110, 82, 50, 255], [160, 36, 60, 255], [70, 52, 40, 255], [130, 112, 92, 255], [80, 120, 150, 255], [96, 130, 90, 255], [110, 140, 90, 255], [120, 124, 70, 255], [100, 130, 110, 255], [140, 100, 130, 255]],
        water: [124, 156, 168, 255],
        contour: [150, 112, 70, 115],
        contour_index: [118, 82, 46, 184],
        sun: [1.04, 0.98, 0.86],
        sky: [1.06, 1.08, 1.12],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 35.0, [1.000, 0.860, 0.660], [0.090, 0.110, 0.150], [0.170, 0.150, 0.120], [0.070, 0.055, 0.035])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// Volcanic: black ground that glows ember-orange toward the summits, water as cooled black glass with a blue edge, trails in bright yellow, roads in hot greys. A low red sun under a dark sky.
pub fn ember() -> Theme {
    Theme {
        name: "Ember".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [26, 20, 22]), (500.0, [56, 30, 26]), (1000.0, [108, 46, 28]), (1600.0, [172, 92, 36]), (2400.0, [220, 170, 84])],
        sea: [12, 16, 26, 255],
        flat: [42, 34, 34, 255],
        bg: [8, 6, 8, 255],
        no_dem: [34, 30, 30, 255],
        land: [[0, 0, 0, 0], [57, 53, 38, 217], [66, 55, 38, 217], [59, 57, 40, 217], [63, 53, 40, 217], [42, 48, 36, 217], [48, 55, 53, 217], [80, 66, 46, 217], [59, 55, 53, 217], [114, 112, 118, 217], [70, 61, 55, 217], [66, 59, 63, 217], [74, 66, 66, 217], [13, 17, 28, 217]],
        line: [[0, 0, 0, 0], [240, 110, 60, 255], [236, 130, 70, 255], [230, 150, 84, 255], [212, 170, 110, 255], [196, 186, 170, 255], [176, 172, 168, 255], [140, 136, 134, 255], [190, 140, 90, 255], [255, 230, 70, 255], [150, 150, 156, 255], [140, 130, 170, 255], [70, 130, 200, 255], [110, 170, 100, 255], [130, 180, 100, 255], [150, 160, 90, 255], [100, 170, 150, 255], [190, 130, 210, 255]],
        water: [26, 38, 60, 255],
        contour: [150, 110, 90, 76],
        contour_index: [200, 150, 110, 140],
        sun: [1.20, 0.80, 0.55],
        sky: [0.85, 0.84, 0.95],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 25.0, [1.000, 0.600, 0.350], [0.050, 0.040, 0.060], [0.120, 0.060, 0.040], [0.040, 0.020, 0.015])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// Monsoon country: emerald lowlands rising through saturated greens to ochre ridges, teal water, hot-pink trails, roads in deep plum. Warm hazy light.
pub fn monsoon() -> Theme {
    Theme {
        name: "Monsoon".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [53, 112, 68]), (500.0, [86, 131, 68]), (1000.0, [133, 135, 74]), (1600.0, [163, 125, 80]), (2400.0, [194, 180, 152])],
        sea: [34, 118, 136, 255],
        flat: [114, 142, 99, 255],
        bg: [20, 40, 36, 255],
        no_dem: [104, 118, 99, 255],
        land: [[0, 0, 0, 0], [93, 153, 86, 217], [153, 159, 93, 217], [102, 156, 90, 217], [109, 148, 93, 217], [47, 117, 70, 217], [63, 141, 132, 217], [176, 165, 117, 217], [132, 137, 124, 217], [180, 187, 187, 217], [137, 124, 102, 217], [131, 124, 137, 217], [148, 143, 137, 217], [34, 117, 132, 217]],
        line: [[0, 0, 0, 0], [96, 30, 80, 255], [110, 40, 92, 255], [124, 52, 104, 255], [100, 60, 96, 255], [86, 66, 90, 255], [72, 62, 80, 255], [110, 100, 116, 255], [120, 80, 60, 255], [255, 60, 150, 255], [50, 40, 60, 255], [120, 110, 160, 255], [20, 120, 150, 255], [40, 110, 70, 255], [70, 130, 70, 255], [90, 120, 50, 255], [50, 130, 110, 255], [150, 90, 170, 255]],
        water: [40, 128, 146, 255],
        contour: [60, 90, 70, 117],
        contour_index: [30, 60, 44, 184],
        sun: [1.06, 1.00, 0.84],
        sky: [1.00, 1.08, 1.14],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 50.0, [0.750, 0.750, 0.680], [0.140, 0.160, 0.180], [0.180, 0.190, 0.180], [0.050, 0.070, 0.040])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// An Admiralty chart: buff land kept flat and quiet, the water a pale white-blue that holds the eye, soundings-blue contours, black roads, magenta for every boundary as a chart draws its limits. Even light, no drama.
pub fn chart() -> Theme {
    Theme {
        name: "Chart".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [192, 183, 146]), (600.0, [192, 183, 146]), (1200.0, [192, 183, 146]), (1900.0, [192, 183, 146]), (2600.0, [192, 183, 146])],
        sea: [176, 204, 222, 255],
        flat: [192, 183, 146, 255],
        bg: [214, 222, 230, 255],
        no_dem: [181, 176, 150, 255],
        land: [[0, 0, 0, 0], [178, 178, 139, 217], [184, 178, 144, 217], [176, 178, 138, 217], [179, 174, 143, 217], [162, 168, 130, 217], [163, 178, 168, 217], [188, 182, 150, 217], [176, 171, 158, 217], [190, 194, 194, 217], [174, 166, 154, 217], [178, 170, 163, 217], [180, 174, 162, 217], [163, 182, 194, 217]],
        line: [[0, 0, 0, 0], [40, 40, 40, 255], [52, 52, 52, 255], [66, 66, 66, 255], [84, 84, 84, 255], [100, 100, 100, 255], [120, 120, 120, 255], [150, 150, 150, 255], [130, 100, 70, 255], [200, 30, 110, 255], [60, 60, 60, 255], [150, 140, 170, 255], [60, 130, 200, 255], [210, 60, 170, 255], [210, 60, 170, 255], [210, 60, 170, 255], [210, 60, 170, 255], [210, 60, 170, 255]],
        water: [162, 196, 220, 255],
        contour: [90, 140, 190, 102],
        contour_index: [50, 100, 160, 178],
        sun: [0.32, 0.32, 0.33],
        sky: [2.20, 2.22, 2.26],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 55.0, [0.500, 0.500, 0.480], [0.200, 0.200, 0.200], [0.200, 0.200, 0.200], [0.080, 0.080, 0.080])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: LayerMask { hypso: false, ..ALL },
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// Neon: near-black violet ground, ridges that light up magenta with height, cyan water, trails in electric yellow, roads in cool neon, contours as a faint blue grid. A magenta sun over a violet sky.
pub fn neon() -> Theme {
    Theme {
        name: "Neon".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [40, 26, 64]), (600.0, [62, 34, 92]), (1200.0, [100, 42, 124]), (1900.0, [156, 54, 148]), (2600.0, [214, 96, 184])],
        sea: [10, 36, 60, 255],
        flat: [50, 34, 76, 255],
        bg: [8, 6, 18, 255],
        no_dem: [40, 30, 60, 255],
        land: [[0, 0, 0, 0], [52, 90, 90, 217], [84, 64, 104, 217], [56, 92, 94, 217], [70, 72, 104, 217], [36, 82, 90, 217], [38, 90, 116, 217], [104, 84, 104, 217], [78, 70, 100, 217], [140, 140, 190, 217], [88, 64, 92, 217], [90, 60, 116, 217], [98, 70, 122, 217], [12, 44, 72, 217]],
        line: [[0, 0, 0, 0], [255, 90, 220, 255], [240, 110, 230, 255], [220, 130, 240, 255], [180, 140, 240, 255], [150, 150, 230, 255], [130, 140, 210, 255], [110, 110, 170, 255], [200, 120, 160, 255], [240, 255, 60, 255], [120, 120, 180, 255], [150, 110, 220, 255], [40, 220, 255, 255], [60, 240, 180, 255], [80, 240, 140, 255], [120, 230, 120, 255], [60, 230, 220, 255], [220, 120, 255, 255]],
        water: [18, 140, 196, 255],
        contour: [70, 100, 200, 102],
        contour_index: [110, 150, 255, 168],
        sun: [1.10, 0.70, 1.10],
        sky: [0.80, 0.80, 1.10],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 40.0, [1.000, 0.550, 1.000], [0.050, 0.100, 0.200], [0.100, 0.050, 0.140], [0.030, 0.020, 0.050])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// First light: rose-gold valleys rising through peach to lavender summits, water a dusty teal, roads in plum, trails in deep rose. A long low sun from the side and a cool sky.
pub fn dawn() -> Theme {
    Theme {
        name: "Dawn".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [172, 137, 116]), (600.0, [180, 146, 121]), (1200.0, [181, 155, 136]), (1900.0, [169, 151, 158]), (2600.0, [162, 151, 181])],
        sea: [116, 152, 166, 255],
        flat: [180, 155, 141, 255],
        bg: [50, 40, 60, 255],
        no_dem: [164, 150, 144, 255],
        land: [[0, 0, 0, 0], [160, 154, 120, 217], [177, 157, 126, 217], [158, 156, 122, 217], [167, 150, 127, 217], [139, 135, 113, 217], [147, 154, 150, 217], [182, 165, 137, 217], [163, 150, 147, 217], [182, 180, 187, 217], [160, 147, 139, 217], [165, 150, 157, 217], [172, 157, 154, 217], [113, 143, 150, 217]],
        line: [[0, 0, 0, 0], [120, 40, 90, 255], [134, 52, 100, 255], [150, 66, 110, 255], [120, 78, 110, 255], [108, 88, 112, 255], [96, 84, 104, 255], [130, 116, 130, 255], [130, 84, 70, 255], [190, 30, 90, 255], [70, 56, 76, 255], [140, 120, 170, 255], [60, 130, 170, 255], [90, 140, 100, 255], [110, 150, 100, 255], [120, 130, 80, 255], [80, 140, 130, 255], [160, 100, 170, 255]],
        water: [108, 150, 164, 255],
        contour: [170, 120, 120, 97],
        contour_index: [140, 86, 96, 163],
        sun: [1.10, 0.92, 0.78],
        sky: [0.96, 1.04, 1.26],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 18.0, [1.000, 0.750, 0.600], [0.080, 0.090, 0.160], [0.200, 0.140, 0.140], [0.050, 0.040, 0.040])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}


// The seven after Nick's ratings (2026-10-09): three natural landscapes dark enough to read like imagery, a candy one, and three darker paper maps whose road and trail colours come in families.

/// Natural ground as a satellite sees it: dark forest, straw grass, tan farmland, grey-brown rock, snow; lowlands olive darkening through earth to a pale summit. Lines light against the dark ground, highways in amber, trails in hot orange. A high sun and a modest sky, so relief reads as it does from orbit.
pub fn terra() -> Theme {
    Theme {
        name: "Terra".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [70, 86, 44]), (600.0, [96, 92, 52]), (1200.0, [116, 98, 68]), (1900.0, [132, 122, 106]), (2600.0, [204, 206, 208])],
        sea: [26, 50, 74, 255],
        flat: [96, 92, 72, 255],
        bg: [14, 16, 20, 255],
        no_dem: [92, 92, 84, 255],
        land: [[0, 0, 0, 0], [104, 112, 61, 217], [131, 121, 75, 217], [78, 97, 51, 217], [94, 95, 58, 217], [48, 71, 39, 217], [60, 82, 68, 217], [163, 151, 119, 217], [109, 100, 88, 217], [196, 199, 204, 217], [128, 116, 102, 217], [110, 107, 105, 217], [119, 114, 110, 217], [29, 53, 78, 217]],
        line: [[0, 0, 0, 0], [244, 190, 88, 255], [236, 172, 80, 255], [224, 160, 82, 255], [206, 194, 152, 255], [192, 186, 160, 255], [172, 172, 166, 255], [150, 150, 146, 255], [204, 150, 90, 255], [255, 122, 56, 255], [92, 92, 96, 255], [170, 150, 190, 255], [92, 160, 220, 255], [120, 200, 120, 255], [140, 210, 110, 255], [160, 190, 90, 255], [110, 200, 170, 255], [200, 150, 220, 255]],
        water: [32, 62, 92, 255],
        contour: [220, 200, 160, 71],
        contour_index: [240, 226, 190, 128],
        sun: [1.16, 1.08, 0.96],
        sky: [0.84, 0.90, 1.04],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 45.0, [1.000, 0.920, 0.780], [0.070, 0.090, 0.140], [0.130, 0.130, 0.120], [0.060, 0.050, 0.030])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// The far north from above: moss and lichen lowlands, grey scree, long snowfields, a steel sea; forest a dark blue-green, wetland rust. Lines in cool whites and sky blue, trails in coral. A low cold sun under a bright sky.
pub fn tundra() -> Theme {
    Theme {
        name: "Tundra".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [78, 92, 68]), (400.0, [96, 100, 78]), (800.0, [106, 104, 92]), (1300.0, [126, 128, 126]), (1800.0, [198, 204, 208])],
        sea: [42, 62, 80, 255],
        flat: [96, 100, 86, 255],
        bg: [16, 18, 22, 255],
        no_dem: [92, 94, 88, 255],
        land: [[0, 0, 0, 0], [102, 114, 76, 217], [119, 117, 85, 217], [82, 100, 71, 217], [95, 100, 78, 217], [44, 66, 56, 217], [109, 80, 54, 217], [146, 141, 121, 217], [114, 114, 110, 217], [194, 199, 204, 217], [128, 124, 116, 217], [109, 110, 112, 217], [119, 119, 121, 217], [44, 65, 82, 217]],
        line: [[0, 0, 0, 0], [236, 222, 180, 255], [232, 214, 170, 255], [226, 206, 160, 255], [206, 206, 196, 255], [190, 194, 190, 255], [172, 176, 176, 255], [150, 154, 156, 255], [196, 170, 120, 255], [255, 128, 100, 255], [84, 88, 96, 255], [160, 150, 190, 255], [120, 190, 236, 255], [120, 200, 150, 255], [140, 210, 130, 255], [160, 200, 100, 255], [110, 200, 190, 255], [190, 150, 220, 255]],
        water: [52, 80, 104, 255],
        contour: [210, 214, 206, 66],
        contour_index: [236, 238, 232, 122],
        sun: [1.00, 0.98, 0.96],
        sky: [1.00, 1.06, 1.18],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 22.0, [0.900, 0.900, 0.950], [0.100, 0.120, 0.160], [0.160, 0.170, 0.180], [0.080, 0.080, 0.080])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// Dry country from above: red earth, ochre grass, dark gallery forest along the water, salt-pale sand, basalt rock; lowlands rust rising through ochre to a bleached summit. Lines in cream and chalk, highways in pale gold, trails in bright cyan so they stand off the red. A hot white sun.
pub fn savanna() -> Theme {
    Theme {
        name: "Savanna".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [112, 70, 44]), (400.0, [128, 90, 52]), (900.0, [140, 110, 64]), (1500.0, [150, 128, 88]), (2200.0, [196, 186, 166])],
        sea: [32, 58, 80, 255],
        flat: [122, 86, 54, 255],
        bg: [18, 12, 10, 255],
        no_dem: [110, 90, 70, 255],
        land: [[0, 0, 0, 0], [136, 116, 65, 217], [146, 128, 78, 217], [82, 92, 46, 217], [109, 95, 53, 217], [53, 70, 36, 217], [71, 85, 60, 217], [180, 165, 128, 217], [95, 78, 68, 217], [201, 201, 201, 217], [136, 110, 92, 217], [116, 102, 95, 217], [128, 110, 100, 217], [34, 61, 82, 217]],
        line: [[0, 0, 0, 0], [250, 214, 110, 255], [244, 206, 110, 255], [236, 198, 112, 255], [230, 220, 190, 255], [216, 208, 186, 255], [196, 190, 176, 255], [172, 166, 156, 255], [220, 180, 120, 255], [60, 230, 240, 255], [70, 60, 56, 255], [180, 150, 190, 255], [100, 180, 230, 255], [150, 210, 110, 255], [170, 220, 100, 255], [190, 200, 80, 255], [120, 210, 170, 255], [210, 150, 220, 255]],
        water: [42, 74, 100, 255],
        contour: [240, 214, 170, 71],
        contour_index: [250, 234, 200, 128],
        sun: [1.20, 1.14, 1.02],
        sky: [0.86, 0.90, 1.02],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 55.0, [1.000, 0.850, 0.620], [0.080, 0.100, 0.150], [0.180, 0.150, 0.110], [0.090, 0.060, 0.030])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// Candy: lowlands in strawberry milk rising through mint and lemon to lavender and sugar-white summits, a bubblegum sea, forest in spearmint, farmland in banana. Roads in grape, cherry and cotton candy, trails in hot pink, contours in lilac. A bright sweet sun.
pub fn candy() -> Theme {
    Theme {
        name: "Candy".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [202, 158, 174]), (500.0, [158, 195, 172]), (1000.0, [204, 195, 146]), (1700.0, [174, 158, 204]), (2400.0, [209, 204, 211])],
        sea: [150, 204, 232, 255],
        flat: [201, 172, 185, 255],
        bg: [70, 40, 80, 255],
        no_dem: [188, 174, 181, 255],
        land: [[0, 0, 0, 0], [162, 194, 155, 217], [200, 188, 135, 217], [150, 194, 166, 217], [176, 190, 150, 217], [119, 182, 150, 217], [150, 187, 190, 217], [202, 187, 158, 217], [182, 170, 182, 217], [202, 202, 202, 217], [190, 166, 174, 217], [182, 163, 190, 217], [190, 168, 190, 217], [143, 179, 198, 217]],
        line: [[0, 0, 0, 0], [160, 60, 200, 255], [190, 70, 200, 255], [220, 80, 190, 255], [240, 100, 150, 255], [250, 130, 170, 255], [250, 160, 200, 255], [240, 180, 210, 255], [230, 150, 90, 255], [255, 40, 140, 255], [120, 90, 160, 255], [200, 160, 240, 255], [80, 170, 250, 255], [90, 210, 160, 255], [120, 220, 140, 255], [170, 220, 110, 255], [90, 210, 210, 255], [220, 140, 240, 255]],
        water: [124, 190, 232, 255],
        contour: [200, 160, 230, 102],
        contour_index: [170, 120, 220, 168],
        sun: [1.04, 1.00, 1.00],
        sky: [1.10, 1.08, 1.16],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 50.0, [1.000, 0.900, 0.950], [0.160, 0.160, 0.200], [0.200, 0.170, 0.190], [0.080, 0.070, 0.080])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// A survey sheet on kraft paper: a darker tan base than Atlas, relief in warm browns, forest in dull olive, water a slate blue. Lines by family: the highways a brick family from dark to light, the local roads a brown family, the trails a red family, rail and power in near-black, the boundaries a green family. Brown contours.
pub fn survey() -> Theme {
    Theme {
        name: "Survey".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [164, 147, 112]), (600.0, [158, 138, 100]), (1200.0, [149, 126, 91]), (1900.0, [138, 113, 82]), (2600.0, [126, 102, 74])],
        sea: [118, 140, 148, 255],
        flat: [167, 151, 117, 255],
        bg: [40, 32, 24, 255],
        no_dem: [156, 145, 123, 255],
        land: [[0, 0, 0, 0], [144, 147, 103, 217], [159, 149, 107, 217], [138, 144, 99, 217], [146, 140, 100, 217], [112, 123, 84, 217], [129, 142, 125, 217], [166, 155, 118, 217], [140, 133, 115, 217], [181, 179, 166, 217], [140, 127, 107, 217], [146, 136, 121, 217], [151, 140, 121, 217], [115, 133, 138, 217]],
        line: [[0, 0, 0, 0], [104, 24, 22, 255], [126, 34, 28, 255], [148, 48, 36, 255], [96, 60, 36, 255], [112, 76, 48, 255], [124, 92, 62, 255], [138, 110, 82, 255], [110, 70, 40, 255], [168, 30, 44, 255], [36, 30, 26, 255], [60, 52, 48, 255], [60, 100, 130, 255], [58, 104, 62, 255], [74, 118, 66, 255], [92, 112, 54, 255], [66, 112, 96, 255], [84, 92, 60, 255]],
        water: [98, 126, 140, 255],
        contour: [120, 86, 50, 122],
        contour_index: [90, 60, 30, 189],
        sun: [1.02, 0.96, 0.86],
        sky: [1.04, 1.06, 1.08],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 45.0, [0.850, 0.780, 0.660], [0.110, 0.110, 0.120], [0.140, 0.130, 0.110], [0.050, 0.040, 0.030])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// A woodcut print: grey-cream stock a shade darker than a page, relief in cool greys, forest in a soft grey-green, water a pale slate. Lines in one family of blacks: highways solid black, roads greys lightening as they shrink, trails in sepia red, boundaries in grey-green. Grey contours.
pub fn woodcut() -> Theme {
    Theme {
        name: "Woodcut".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [162, 155, 140]), (600.0, [155, 148, 135]), (1200.0, [146, 140, 130]), (1900.0, [135, 131, 122]), (2600.0, [124, 121, 115])],
        sea: [136, 146, 150, 255],
        flat: [164, 157, 142, 255],
        bg: [30, 30, 30, 255],
        no_dem: [153, 148, 137, 255],
        land: [[0, 0, 0, 0], [148, 150, 133, 217], [157, 152, 138, 217], [142, 148, 131, 217], [148, 146, 133, 217], [122, 130, 116, 217], [138, 146, 140, 217], [161, 157, 141, 217], [141, 140, 133, 217], [177, 177, 174, 217], [141, 138, 130, 217], [144, 142, 138, 217], [148, 148, 142, 217], [125, 133, 136, 217]],
        line: [[0, 0, 0, 0], [20, 18, 16, 255], [34, 32, 30, 255], [50, 48, 44, 255], [72, 70, 66, 255], [92, 90, 86, 255], [112, 110, 106, 255], [132, 130, 126, 255], [96, 76, 60, 255], [150, 50, 36, 255], [24, 22, 20, 255], [70, 66, 70, 255], [90, 104, 112, 255], [96, 112, 92, 255], [108, 122, 94, 255], [116, 118, 84, 255], [100, 118, 108, 255], [110, 100, 110, 255]],
        water: [118, 130, 136, 255],
        contour: [110, 106, 100, 107],
        contour_index: [70, 66, 62, 178],
        sun: [1.00, 1.00, 1.00],
        sky: [1.04, 1.04, 1.06],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 35.0, [1.000, 1.000, 1.000], [0.070, 0.070, 0.070], [0.100, 0.100, 0.100], [0.030, 0.030, 0.030])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

/// A gazetteer on sage paper: a grey-green base darker than a page, relief in muted greens to a dun summit, forest a deeper green, water a dusty blue. Lines by family: the highways an indigo family, the local roads a slate family, the trails a burgundy family, boundaries in olive. Green-grey contours.
pub fn gazetteer() -> Theme {
    Theme {
        name: "Gazetteer".to_string(),
        revision: BUILTIN_REVISION,
        edited: false,
        hypso: [(0.0, [140, 149, 121]), (600.0, [136, 141, 113]), (1200.0, [134, 134, 106]), (1900.0, [130, 125, 100]), (2600.0, [125, 115, 95])],
        sea: [118, 140, 152, 255],
        flat: [141, 151, 123, 255],
        bg: [28, 34, 30, 255],
        no_dem: [136, 140, 123, 255],
        land: [[0, 0, 0, 0], [136, 151, 119, 217], [151, 151, 119, 217], [126, 145, 115, 217], [134, 142, 115, 217], [101, 125, 93, 217], [120, 142, 131, 217], [158, 153, 126, 217], [134, 134, 124, 217], [176, 179, 174, 217], [133, 128, 113, 217], [136, 136, 129, 217], [140, 140, 133, 217], [115, 131, 140, 217]],
        line: [[0, 0, 0, 0], [44, 40, 110, 255], [58, 54, 128, 255], [76, 72, 146, 255], [78, 86, 104, 255], [94, 102, 118, 255], [110, 118, 132, 255], [128, 134, 146, 255], [104, 86, 64, 255], [138, 32, 64, 255], [40, 42, 48, 255], [84, 78, 96, 255], [70, 112, 140, 255], [96, 112, 60, 255], [110, 124, 66, 255], [122, 128, 56, 255], [92, 120, 92, 255], [112, 104, 70, 255]],
        water: [98, 126, 142, 255],
        contour: [104, 112, 88, 117],
        contour_index: [70, 78, 56, 184],
        sun: [1.00, 0.98, 0.92],
        sky: [1.04, 1.06, 1.10],
        probe: Some(crate::sh::Sh9::outdoor(315.0, 45.0, [0.800, 0.800, 0.720], [0.120, 0.130, 0.140], [0.150, 0.150, 0.140], [0.050, 0.060, 0.040])),
        own: OWN,
        wild: WILD,
        mvum: MVUM,
        layers: ALL,
        label: LABELS_LIGHT,
        label_look: LABEL_LOOK,
        label_amount: LABEL_AMOUNT,
        label_font: LABEL_FONTS[0].to_string(),
    }
}

pub const BUILTIN_NAMES: [&str; 20] = ["Trail", "Topo", "Night", "Alpine", "Desert", "Ink", "Blueprint", "Atlas", "Ember", "Monsoon", "Chart", "Neon", "Dawn", "Terra", "Tundra", "Savanna", "Candy", "Survey", "Woodcut", "Gazetteer"];

/// The built-in themes, as authored: VSF RGB, gamma 2. The renderers convert to the display at their one encode.
pub fn builtin() -> Vec<Theme> {
    let mut v = vec![trail(), topo(), night(), alpine(), desert(), ink(), blueprint(), atlas(), ember(), monsoon(), chart(), neon(), dawn(), terra(), tundra(), savanna(), candy(), survey(), woodcut(), gazetteer()];
    for t in &mut v {
        t.label = t.default_labels();
        lettering(t);
    }
    v
}

/// Each built-in style's own lettering (Nick 2026-10-14: theme the map's type and its colours too, Oxanium for Neon and the like): a typeface for every style, and for the strongest characters their own palette (in [`LABEL_NAMES`] order: places, summits, water, roads, trails, boundaries, areas, small marks, halo, route, route casing) and glow.
fn lettering(t: &mut Theme) {
    let (font, palette, glow): (&str, Option<[[u8; 4]; 11]>, Option<f32>) = match t.name.as_str() {
        // Glowing tubes on black: cyan water, magenta summits, amber roads, a violet bloom behind every letter.
        "Neon" => ("Oxanium", Some([[255, 236, 255, 255], [255, 96, 224, 255], [86, 236, 255, 255], [255, 214, 92, 255], [150, 255, 150, 255], [200, 146, 255, 255], [124, 255, 206, 255], [206, 206, 255, 255], [34, 0, 60, 210], [255, 64, 204, 255], [24, 0, 44, 255]]), Some(0.24)),
        // Drafting white on the blue sheet, lettered in a draughtsman's mono.
        "Blueprint" => ("Mahere Mono", Some([[236, 246, 255, 255], [196, 224, 255, 255], [156, 212, 255, 255], [222, 236, 255, 255], [186, 230, 255, 255], [166, 204, 244, 255], [176, 218, 255, 255], [206, 228, 255, 255], [8, 30, 72, 220], [255, 255, 255, 255], [8, 30, 72, 255]]), Some(0.16)),
        // Embers: warm amber and coals on the dark.
        "Ember" => ("Roboto Condensed", Some([[255, 214, 168, 255], [255, 156, 92, 255], [255, 196, 140, 255], [255, 186, 120, 255], [238, 140, 96, 255], [210, 120, 90, 255], [232, 170, 120, 255], [220, 180, 150, 255], [18, 6, 4, 210], [255, 120, 40, 255], [24, 8, 4, 255]]), Some(0.14)),
        // Pen on paper: every name in the same black ink, greys for what matters less.
        "Ink" => ("EB Garamond", Some([[16, 16, 16, 255], [36, 36, 36, 255], [52, 52, 52, 255], [24, 24, 24, 255], [46, 46, 46, 255], [70, 70, 70, 255], [58, 58, 58, 255], [80, 80, 80, 255], [250, 248, 242, 225], [0, 0, 0, 255], [255, 255, 255, 255]]), None),
        // An old atlas: ink-black towns, sepia mountains, blue water, red roads on cream.
        "Atlas" => ("EB Garamond", Some([[34, 26, 20, 255], [104, 62, 28, 255], [32, 78, 126, 255], [128, 36, 28, 255], [112, 56, 30, 255], [64, 90, 52, 255], [70, 96, 58, 255], [86, 74, 60, 255], [246, 236, 214, 215], [150, 30, 30, 255], [250, 242, 222, 255]]), None),
        // A carved block: deep brown capitals on the grain.
        "Woodcut" => ("Cinzel", Some([[38, 26, 18, 255], [70, 44, 24, 255], [40, 58, 72, 255], [52, 34, 22, 255], [76, 50, 30, 255], [86, 66, 46, 255], [74, 56, 38, 255], [90, 72, 54, 255], [236, 226, 206, 220], [120, 30, 20, 255], [240, 232, 214, 255]]), None),
        // Sugar: berry and bubblegum on mint.
        "Candy" => ("Fredoka", Some([[128, 24, 96, 255], [176, 40, 120, 255], [24, 96, 176, 255], [200, 50, 110, 255], [120, 60, 170, 255], [90, 120, 60, 255], [60, 130, 90, 255], [150, 80, 140, 255], [255, 246, 252, 220], [236, 60, 150, 255], [255, 255, 255, 255]]), Some(0.14)),
        // Ice: pale blue-white letters, cold blue water, on the dark.
        "Tundra" => ("Roboto Condensed", Some([[236, 244, 250, 255], [206, 226, 240, 255], [150, 206, 244, 255], [222, 232, 240, 255], [196, 216, 230, 255], [170, 196, 214, 255], [180, 216, 206, 255], [200, 214, 224, 255], [12, 20, 30, 210], [140, 210, 255, 255], [10, 18, 28, 255]]), None),
        // Sandstone: warm browns, desert-sky water.
        "Desert" => ("EB Garamond", Some([[52, 30, 16, 255], [120, 60, 24, 255], [30, 90, 130, 255], [110, 44, 24, 255], [126, 70, 34, 255], [96, 74, 44, 255], [100, 80, 46, 255], [104, 84, 60, 255], [248, 236, 214, 215], [170, 50, 20, 255], [250, 240, 222, 255]]), None),
        // A nautical chart: navy names and italic blue water on the pale sheet.
        "Chart" => ("EB Garamond", Some([[20, 34, 64, 255], [60, 50, 40, 255], [24, 70, 130, 255], [40, 40, 60, 255], [70, 60, 50, 255], [120, 40, 90, 255], [50, 90, 70, 255], [70, 76, 96, 255], [244, 246, 248, 220], [200, 40, 80, 255], [255, 255, 255, 255]]), None),
        "Topo" | "Alpine" | "Terra" => ("Roboto Condensed", None, None),
        "Survey" => ("Mahere Mono", None, None),
        "Gazetteer" | "Dawn" => ("EB Garamond", None, None),
        "Savanna" => ("Cinzel", None, None),
        _ => ("Open Sans", None, None),
    };
    t.label_font = font.to_string();
    if let Some(p) = palette {
        t.label = p;
    }
    if let Some(g) = glow {
        t.label_look[1] = g;
    }
}

impl Theme {
    /// The label palette that reads on this theme's ground: light letters on a dark halo where the lowland tint is dark, dark on a paper halo otherwise.
    pub fn default_labels(&self) -> [[u8; 4]; 11] {
        let c = self.hypso[1].1;
        let lum = 0.3 * c[0] as f32 + 0.59 * c[1] as f32 + 0.11 * c[2] as f32;
        if lum < 110.0 { LABELS_DARK } else { LABELS_LIGHT }
    }
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

    /// The theme's line colours with the classes added after its eighteen (2026-10-16), from its own palette so every theme, saved or built in, has them: a ferry the water deepened toward the rail ink, a lift in the rail's colour, a ridge no line at all (its name alone follows it), a cliff in the index contour's colour.
    pub fn lines(&self) -> [[u8; 4]; 22] {
        let mut out = [[0u8; 4]; 22];
        out[..18].copy_from_slice(&self.line);
        let rail = self.line[10];
        let mix = |a: u8, b: u8| ((a as u16 * 5 + b as u16 * 3) / 8) as u8;
        out[18] = [mix(self.water[0], rail[0]), mix(self.water[1], rail[1]), mix(self.water[2], rail[2]), 230];
        out[19] = [rail[0], rail[1], rail[2], 255];
        out[20] = [0, 0, 0, 0];
        out[21] = [self.contour_index[0], self.contour_index[1], self.contour_index[2], 220];
        out
    }

    pub fn style(&self) -> Style {
        Style { land: self.land, line: self.lines(), water: self.water, contour: self.contour, contour_index: self.contour_index, sea: self.sea, flat: self.flat, bg: self.bg, no_dem: self.no_dem, own: self.own, wild: self.wild, mvum: self.mvum }
    }
}
