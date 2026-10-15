//! A cell's fields context-coded through VSF's own codec (`vc`, vsf 0.10; Nick 2026-10-14: no zstd around the cells, every section coding itself, the codec in VSF where any reader finds it).
//!
//! What a cell codes, and how:
//! - A pyramid's coefficient stream (terrain, each image band) as a quadtree stream over the two face roots; green and blue against red, so their sizes and signs follow red's.
//! - A plane, one value per trixel (classes, coverages, names, routes, masks, the geoid), as a raster in memory order, 256 rows of 512 (each square's two halves side by side), so a trixel's neighbours are the other half of its square, the square to its left and the square above. It reads back in disk order, as the cell's readers take it.
//! - A table of 16-bit numbers (the terrain's apron) as a series of changes.
//!
//! A coded field is one value, `v` with encoding `c`; a field coded otherwise (Rice multi-values, a bare tensor) is never that, so old and new forms tell themselves apart.

use vsf::codec::{self, Model};
use vsf::types::{BitPackedTensor, Tensor};
use vsf::VsfType;

use crate::{TEX, TRI, disk_to_mem, mem_to_disk};

/// The quadtree roots of a cell's pyramid: its two faces.
const ROOTS: usize = 2;

/// The coded value of a field, if it is one.
fn coded(values: &[VsfType]) -> Option<&VsfType> {
    match values {
        [v @ VsfType::v(codec::ENCODING, _)] => Some(v),
        _ => None,
    }
}

/// A plane's values in memory order as the same element type, shaped as the raster the codec reads.
fn raster_of(v: &VsfType) -> Option<VsfType> {
    let shape = vec![TEX, 2 * TEX];
    Some(match v {
        VsfType::t_u3(t) if t.data.len() == TRI => VsfType::t_u3(Tensor::new(shape, disk_to_mem(&t.data))),
        VsfType::t_u4(t) if t.data.len() == TRI => VsfType::t_u4(Tensor::new(shape, disk_to_mem(&t.data))),
        VsfType::t_i4(t) if t.data.len() == TRI => VsfType::t_i4(Tensor::new(shape, disk_to_mem(&t.data))),
        VsfType::p(t) if t.bit_depth == 1 && t.shape.iter().product::<usize>() == TRI => {
            let bits: Vec<bool> = (0..TRI).map(|i| (t.data[i >> 3] >> (7 - (i & 7))) & 1 == 1).collect();
            VsfType::t_u0(Tensor::new(shape, disk_to_mem(&bits)))
        }
        _ => return None,
    })
}

/// A raster back as the plane it was: disk order, the cell's usual shapes (a mask a bit-packed run, the rest two faces).
fn plane_of(v: VsfType) -> Option<VsfType> {
    let faces = vec![2, TEX * TEX];
    Some(match v {
        VsfType::t_u3(t) if t.data.len() == TRI => VsfType::t_u3(Tensor::new(faces, mem_to_disk(&t.data))),
        VsfType::t_u4(t) if t.data.len() == TRI => VsfType::t_u4(Tensor::new(faces, mem_to_disk(&t.data))),
        VsfType::t_i4(t) if t.data.len() == TRI => VsfType::t_i4(Tensor::new(faces, mem_to_disk(&t.data))),
        VsfType::t_u0(t) if t.data.len() == TRI => {
            let disk = mem_to_disk(&t.data);
            let mut data = vec![0u8; TRI / 8];
            for (i, &b) in disk.iter().enumerate() {
                if b {
                    data[i >> 3] |= 0x80 >> (i & 7);
                }
            }
            VsfType::p(BitPackedTensor { bit_depth: 1, shape: vec![TRI], data })
        }
        _ => return None,
    })
}

/// A field's values coded, when they are a kind a cell codes (a pyramid, a plane, a table); None to keep them as they are. `reference` is the coefficients of the section's first band, for a later band of an image.
pub fn code_field(values: &[VsfType], reference: Option<&[i32]>) -> Option<Vec<VsfType>> {
    if let Some(c) = crate::pyr::from_vsf(values, crate::pyr::NCOEF) {
        let n = c.len();
        return codec::encode(&VsfType::t_i5(Tensor::new(vec![n], c)), Model::Pyramid { roots: ROOTS }, reference).ok().map(|v| vec![v]);
    }
    let [v] = values else { return None };
    if let Some(r) = raster_of(v) {
        return codec::encode(&r, Model::Raster, None).ok().map(|v| vec![v]);
    }
    match v {
        VsfType::t_u4(t) if t.data.len() >= 64 => codec::encode(v, Model::Series, None).ok().map(|v| vec![v]),
        _ => None,
    }
}

/// A coded plane or table back as the tensor the cell's readers take; None for anything else (a pyramid stays coded for [`coefs`]).
pub fn uncode_field(values: &[VsfType]) -> Option<VsfType> {
    let v = coded(values)?;
    // A pyramid's coefficients stay coded for the decoder that has the steps.
    if matches!(codec::model_of(v), Some(Model::Pyramid { .. }) | None) {
        return None;
    }
    let t = codec::decode(v, None).ok()?;
    match &t {
        VsfType::t_u3(x) if x.shape == [TEX, 2 * TEX] => plane_of(t),
        VsfType::t_u4(x) if x.shape == [TEX, 2 * TEX] => plane_of(t),
        VsfType::t_i4(x) if x.shape == [TEX, 2 * TEX] => plane_of(t),
        VsfType::t_u0(_) => plane_of(t),
        _ => Some(t),
    }
}

/// A pyramid's coefficients from its field, coded either way (the reference for a band coded against the first).
pub fn coefs(values: &[VsfType], reference: Option<&[i32]>) -> Option<Vec<i32>> {
    match coded(values) {
        Some(v) => match codec::decode(v, reference).ok()? {
            VsfType::t_i5(t) => Some(t.data),
            _ => None,
        },
        None => crate::pyr::from_vsf(values, crate::pyr::NCOEF),
    }
}
