//! The published cell (the flag day, Nick 2026-10-14): what a phone fetches, as three documents a cell, every field coding itself ([`crate::ac`]) and nothing wrapped around the file.
//!
//! - **base**: lines, land cover, water, ownership, motor vehicle use, names, routes, the geoid.
//! - **dem**: the terrain, on its own (Nick 2026-10-14: why not a layer of its own), so a view without terrain fetches none and a better elevation source replaces it alone.
//! - **img**: the image's red, green and blue (green and blue coded against red), fetched when the imagery shows.
//! - **nir**: the near infrared alone, coded on its own, so an infrared view needs nothing else.
//!
//! A cell missing a layer has no document for it. Recoding is exact: terrain and imagery were quantised when they were baked, and their coefficients are stored as they are.

use std::collections::HashMap;

use vsf::{VsfBuilder, VsfSection, VsfType};

use crate::{CellKey, CellPlanes, ac, decode_cell};

/// The layers a cell is published as.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Layer {
    Base,
    Dem,
    Img,
    Nir,
}

impl Layer {
    pub const ALL: [Layer; 4] = [Layer::Base, Layer::Dem, Layer::Img, Layer::Nir];

    pub fn dir(self) -> &'static str {
        match self {
            Layer::Base => "base",
            Layer::Dem => "dem",
            Layer::Img => "img",
            Layer::Nir => "nir",
        }
    }

    /// A published cell's path under the cells root: `<layer>/<cell>.vsf`.
    pub fn path(self, key: CellKey) -> String {
        format!("{}/{}.vsf", self.dir(), key.name())
    }
}

/// The image fields each image layer takes: its bands and their masks.
const IMG_FIELDS: [&str; 6] = ["red", "redmask", "green", "greenmask", "blue", "bluemask"];
const NIR_FIELDS: [&str; 2] = ["nir", "nirmask"];

/// A baked cell (the working form: one VSF document, zstd around it) as its published layers, each a document of its own; a layer the cell lacks is absent.
pub fn split(bytes: &[u8]) -> Result<HashMap<Layer, Vec<u8>>, String> {
    let plain: Vec<u8>;
    let data = if bytes.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
        plain = zstd::decode_all(bytes).map_err(|e| e.to_string())?;
        &plain[..]
    } else {
        bytes
    };
    let (header, end) = vsf::VsfHeader::decode(data).map_err(|e| format!("{e:?}"))?;
    let sections = header.sections(data, end).map_err(|e| format!("{e:?}"))?;
    let mut docs: HashMap<Layer, VsfBuilder> = HashMap::new();
    let mut put = |layer: Layer, s: VsfSection| {
        let b = docs.remove(&layer).unwrap_or_default();
        docs.insert(layer, b.add_section_direct(s));
    };
    for s in sections {
        if s.name != "img" {
            let layer = if s.name == "dem" { Layer::Dem } else { Layer::Base };
            let mut out = VsfSection::new(s.name.clone());
            for f in s.fields {
                let v = ac::code_field(&f.values, None).unwrap_or(f.values);
                out.add_field_multi(f.name, v);
            }
            put(layer, out);
            continue;
        }
        // The image: its shared fields (the loss steps, the tone) in both image layers, the bands split.
        let fields: Vec<(String, Vec<VsfType>)> = s.fields.into_iter().map(|f| (f.name, f.values)).collect();
        let shared: Vec<&(String, Vec<VsfType>)> = fields.iter().filter(|(n, _)| !IMG_FIELDS.contains(&n.as_str()) && !NIR_FIELDS.contains(&n.as_str())).collect();
        let red = fields.iter().find(|(n, _)| n == "red").and_then(|(_, v)| crate::pyr::from_vsf(v, crate::pyr::NCOEF));
        for (layer, names) in [(Layer::Img, &IMG_FIELDS[..]), (Layer::Nir, &NIR_FIELDS[..])] {
            if !names.iter().filter(|b| !b.ends_with("mask")).any(|b| fields.iter().any(|(n, _)| n == b)) {
                continue;
            }
            let mut out = VsfSection::new("img");
            for (n, v) in &shared {
                out.add_field_multi(n.clone(), v.clone());
            }
            for (n, v) in fields.iter().filter(|(n, _)| names.contains(&n.as_str())) {
                // Green and blue against red; red, the masks and the infrared alone.
                let reference = if layer == Layer::Img && (n == "green" || n == "blue") { red.as_deref() } else { None };
                out.add_field_multi(n.clone(), ac::code_field(v, reference).unwrap_or_else(|| v.clone()));
            }
            put(layer, out);
        }
    }
    docs.into_iter().map(|(layer, b)| b.build().map(|d| (layer, d)).map_err(|e| format!("{e:?}"))).collect()
}

/// A cell's planes from whichever of its published layers are at hand (the base, the terrain, the image, the infrared), merged.
pub fn decode_layers(base: Option<&[u8]>, dem: Option<&[u8]>, img: Option<&[u8]>, nir: Option<&[u8]>) -> Result<CellPlanes, String> {
    let mut out = match base {
        Some(b) => decode_cell(b)?,
        None => CellPlanes::default(),
    };
    if let Some(b) = dem {
        out.dem = decode_cell(b)?.dem;
    }
    if let Some(b) = img {
        out.img = decode_cell(b)?.img;
    }
    if let Some(b) = nir {
        if let Some(n) = decode_cell(b)?.img {
            match &mut out.img {
                Some(im) => im.nir = n.nir,
                None => out.img = Some(n),
            }
        }
    }
    Ok(out)
}
