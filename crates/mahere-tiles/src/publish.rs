//! The published cell (the flag day, Nick 2026-10-14): what a phone fetches, as three documents a cell, every field coding itself ([`crate::ac`]) and nothing wrapped around the file.
//!
//! - **base**: lines, land cover, water, ownership, motor vehicle use, names, routes, the geoid.
//! - **dem**: the terrain, on its own (Nick 2026-10-14: why not a layer of its own), so a view without terrain fetches none and a better elevation source replaces it alone.
//! - **img**: the image's red, green and blue (green and blue coded against red), fetched when the imagery shows.
//! - **nir**: the near infrared alone, coded on its own, so an infrared view needs nothing else.
//!
//! A cell missing a layer has no document for it; the base names each layer the cell has by its document's provenance hash (a `layers` section, `hp` values), so a reader never puts one publish's terrain or imagery under another's base. Recoding is exact: terrain and imagery were quantised when they were baked, and their coefficients are stored as they are.

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
    split_keeping(bytes, |_| None)
}

/// [`split`], keeping a layer's document as `kept` gives it when its contents are the new one's (only the header's time and hashes differ), so a republish of an unchanged layer leaves its bytes, its hash and the bucket's copy as they were (2026-10-17: a republish gave every layer a new hash at the same size, and a size-only upload left the bases naming layers the bucket did not hold).
pub fn split_keeping(bytes: &[u8], kept: impl Fn(Layer) -> Option<Vec<u8>>) -> Result<HashMap<Layer, Vec<u8>>, String> {
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
    // The other layers first, then the base naming each by its provenance hash.
    let mut out: HashMap<Layer, Vec<u8>> = HashMap::new();
    let mut links = VsfSection::new("layers");
    for layer in [Layer::Dem, Layer::Img, Layer::Nir] {
        if let Some(b) = docs.remove(&layer) {
            let mut d = b.build().map_err(|e| format!("{e:?}"))?;
            if let Some(old) = kept(layer).filter(|old| same_contents(old, &d)) {
                d = old;
            }
            links.add_field(layer.dir(), provenance(&d).ok_or("a layer without its provenance hash")?);
            out.insert(layer, d);
        }
    }
    let base = docs.remove(&Layer::Base).unwrap_or_default().add_section_direct(links);
    out.insert(Layer::Base, base.build().map_err(|e| format!("{e:?}"))?);
    Ok(out)
}

/// A published terrain document with its apron (the texels one step past each edge) replaced, every other field as it was: the apron quantised on the document's own base and step, coded as a publish codes it (2026-10-24: aprons filled from the seabed grid where a bake had no neighbour drew Antarctica's tile edges as contours).
pub fn with_apron(dem_doc: &[u8], apron: &[f32]) -> Result<Vec<u8>, String> {
    if apron.len() != crate::APRON {
        return Err(format!("an apron of {} texels, not {}", apron.len(), crate::APRON));
    }
    let fields = crate::section_fields(dem_doc, "dem").ok_or("no terrain section")?;
    let num = |n: &str| fields.get(n).and_then(|v| v.first()).and_then(crate::scalar_f32).ok_or_else(|| format!("the terrain without its {n}"));
    let (base, step) = (num("base")?, num("step")?);
    let q: Vec<u16> = apron.iter().map(|&e| if e.is_nan() { u16::MAX } else { (((e - base) / step).round() as i32).clamp(0, 65534) as u16 }).collect();
    let coded = VsfType::t_u4(vsf::types::Tensor::new(vec![4, 2, crate::TEX], q));
    let coded = ac::code_field(std::slice::from_ref(&coded), None).unwrap_or_else(|| vec![coded]);
    rebuild(dem_doc, |section, field| (section == "dem" && field == "apron").then(|| coded.clone()))
}

/// A base naming `doc` as its `layer`, every other field as it was.
pub fn relinked(base_doc: &[u8], layer: Layer, doc: &[u8]) -> Result<Vec<u8>, String> {
    let hp = provenance(doc).ok_or("a layer without its provenance hash")?;
    rebuild(base_doc, |section, field| (section == "layers" && field == layer.dir()).then(|| vec![hp.clone()]))
}

/// A document built again from its sections, a field's values replaced where `replace` gives them.
fn rebuild(doc: &[u8], replace: impl Fn(&str, &str) -> Option<Vec<VsfType>>) -> Result<Vec<u8>, String> {
    let (header, end) = vsf::VsfHeader::decode(doc).map_err(|e| format!("{e:?}"))?;
    let sections = header.sections(doc, end).map_err(|e| format!("{e:?}"))?;
    let mut b = VsfBuilder::new();
    let mut replaced = false;
    for s in sections {
        let mut out = VsfSection::new(s.name.clone());
        for f in s.fields {
            let v = match replace(&s.name, &f.name) {
                Some(v) => {
                    replaced = true;
                    v
                }
                None => f.values,
            };
            out.add_field_multi(f.name, v);
        }
        b = b.add_section_direct(out);
    }
    if !replaced {
        return Err("nothing to replace".into());
    }
    b.build().map_err(|e| format!("{e:?}"))
}

/// Whether two documents hold the same sections, byte for byte past their headers (which differ in their time and hashes alone).
pub fn same_contents(a: &[u8], b: &[u8]) -> bool {
    let body = |d: &[u8]| vsf::VsfHeader::decode(d).ok().map(|(_, end)| end);
    match (body(a), body(b)) {
        (Some(x), Some(y)) => a.len() == b.len() && a[x..] == b[y..],
        _ => false,
    }
}

/// A document's provenance hash (`hp`), from its header.
pub fn provenance(doc: &[u8]) -> Option<VsfType> {
    vsf::VsfHeader::decode(doc).ok().map(|(h, _)| h.provenance_hash)
}

/// Whether a layer's document is the one its base names (a base from before links, or a base naming none for that layer, takes any).
pub fn linked(base: &[u8], layer: Layer, doc: &[u8]) -> bool {
    let Some(fields) = crate::section_fields(base, "layers") else { return true };
    match fields.get(layer.dir()).and_then(|v| v.first()) {
        Some(want) => provenance(doc).as_ref() == Some(want),
        None => true,
    }
}

/// The name a base gives one of its layers, for an address no cache has seen an older copy under: the first half of the layer's provenance hash, in hex.
pub fn link_name(base: &[u8], layer: Layer) -> Option<String> {
    let fields = crate::section_fields(base, "layers")?;
    match fields.get(layer.dir())?.first()? {
        VsfType::hp(h) => Some(h.iter().take(16).map(|b| format!("{b:02x}")).collect()),
        _ => None,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A base's link names its layer by the first half of the layer's provenance hash; a layer it does not name has no name.
    #[test]
    fn a_link_names_its_layer() {
        let mut links = VsfSection::new("layers");
        links.add_field("dem", VsfType::hp((0u8..32).collect()));
        let base = VsfBuilder::new().add_section_direct(links).build().unwrap();
        assert_eq!(link_name(&base, Layer::Dem).as_deref(), Some("000102030405060708090a0b0c0d0e0f"));
        assert_eq!(link_name(&base, Layer::Img), None);
    }

    /// A republish keeps an unchanged layer's document and the base names it as before; a changed layer gets a document of its own.
    #[test]
    fn a_republish_keeps_an_unchanged_layer() {
        let cell = |v: u32| {
            let mut dem = VsfSection::new("dem");
            dem.add_field("elev", VsfType::u5(v));
            VsfBuilder::new().add_section_direct(dem).build().unwrap()
        };
        let first = split(&cell(7)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let again = split_keeping(&cell(7), |l| first.get(&l).cloned()).unwrap();
        assert_eq!(again[&Layer::Dem], first[&Layer::Dem]);
        assert_eq!(link_name(&again[&Layer::Base], Layer::Dem), link_name(&first[&Layer::Base], Layer::Dem));
        let changed = split_keeping(&cell(8), |l| first.get(&l).cloned()).unwrap();
        assert!(!same_contents(&changed[&Layer::Dem], &first[&Layer::Dem]));
        assert_ne!(link_name(&changed[&Layer::Base], Layer::Dem), link_name(&first[&Layer::Base], Layer::Dem));
    }
}
