//! Adaptive binary range coding with context models (Nick 2026-10-14: no zstd around the cells, every section coding itself): what a context knows predicts each bit, so the bits cost what they are worth.
//!
//! The range coder is LZMA's (32-bit range, carry propagation through a cached byte), with 15-bit probabilities that adapt by a 32nd per bit: a certain bit costs about a thousandth of a bit, so a plane of one value is a few dozen bytes.
//! A plane (one value per trixel) is coded in memory order, each trixel against three neighbours already coded: the one before it (the other half of its square, or the square to the left), the square to the left, and the square above. "Same as before", "same as above", "same as left" are adaptive bits under contexts of how those neighbours agree; anything else is a literal.
//! A pyramid's coefficient stream is coded level by level in quadtree order, each coefficient's "is zero" under its level, its parent's size, its place among its siblings and whether the sibling before was zero; the sign and size follow under their own contexts.

use vsf::VsfType;

use crate::{TEX, TRI, disk_to_mem, mem_to_disk};

const PROB_BITS: u32 = 15;
const PROB_ONE: u16 = 1 << PROB_BITS;
const ADAPT: u32 = 5;
const TOP: u32 = 1 << 24;

struct Encoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Encoder { low: 0, range: u32::MAX, cache: 0, cache_size: 1, out: Vec::new() }
    }

    fn shift_low(&mut self) {
        if self.low < 0xFF00_0000 || self.low > 0xFFFF_FFFF {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    fn bit(&mut self, p: &mut u16, bit: bool) {
        let bound = (self.range >> PROB_BITS) * *p as u32;
        if bit {
            self.low += bound as u64;
            self.range -= bound;
            *p -= *p >> ADAPT;
        } else {
            self.range = bound;
            *p += (PROB_ONE - *p) >> ADAPT;
        }
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        // The first byte the scheme writes is always the empty cache.
        self.out.remove(0);
        self.out
    }
}

struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    code: u32,
    range: u32,
}

impl<'a> Decoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut d = Decoder { data, pos: 0, code: 0, range: u32::MAX };
        for _ in 0..4 {
            d.code = (d.code << 8) | d.byte() as u32;
        }
        d
    }

    fn byte(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    fn bit(&mut self, p: &mut u16) -> bool {
        let bound = (self.range >> PROB_BITS) * *p as u32;
        let bit = if self.code < bound {
            self.range = bound;
            *p += (PROB_ONE - *p) >> ADAPT;
            false
        } else {
            self.code -= bound;
            self.range -= bound;
            *p -= *p >> ADAPT;
            true
        };
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.byte() as u32;
        }
        bit
    }

    /// Whether the decoder ran past the end of its bytes (a short or corrupt stream).
    fn overrun(&self) -> bool {
        self.pos > self.data.len() + 4
    }
}

/// An adaptive Exp-Golomb code for a whole number: the length in unary, then the bits under it, every bit under its own context.
struct Golomb {
    prefix: [u16; 33],
    suffix: Vec<u16>,
}

impl Golomb {
    fn new() -> Self {
        Golomb { prefix: [PROB_ONE / 2; 33], suffix: vec![PROB_ONE / 2; 33 * 32] }
    }

    fn put(&mut self, e: &mut Encoder, v: u32) {
        let w = v as u64 + 1;
        let n = 63 - w.leading_zeros();
        for i in 0..n {
            e.bit(&mut self.prefix[i as usize], true);
        }
        if n < 32 {
            e.bit(&mut self.prefix[n as usize], false);
        }
        for j in (0..n).rev() {
            e.bit(&mut self.suffix[(n as usize) * 32 + (n - 1 - j).min(31) as usize], (w >> j) & 1 == 1);
        }
    }

    fn get(&mut self, d: &mut Decoder) -> Option<u32> {
        let mut n = 0u32;
        while n < 32 && d.bit(&mut self.prefix[n as usize]) {
            n += 1;
        }
        let mut w: u64 = 1;
        for j in 0..n {
            w = (w << 1) | d.bit(&mut self.suffix[(n as usize) * 32 + j.min(31) as usize]) as u64;
        }
        u32::try_from(w - 1).ok()
    }
}

/// A plane (disk order, one value per trixel) as range-coded bytes.
pub fn encode_plane(disk: &[u32]) -> Vec<u8> {
    assert_eq!(disk.len(), TRI);
    let v = disk_to_mem(disk);
    let mut m = PlaneModel::new();
    let mut e = Encoder::new();
    for i in 0..TRI {
        let (a, l, u) = neighbours(&v, i);
        let c = PlaneModel::ctx(a, l, u);
        let x = v[i];
        e.bit(&mut m.same_a[c], x == a);
        if x == a {
            continue;
        }
        if u != a {
            e.bit(&mut m.same_u[c], x == u);
            if x == u {
                continue;
            }
        }
        if l != a && l != u {
            e.bit(&mut m.same_l[c], x == l);
            if x == l {
                continue;
            }
        }
        m.literal.put(&mut e, x);
    }
    e.finish()
}

/// A plane back from its bytes, in disk order; None if they run short.
pub fn decode_plane(bytes: &[u8]) -> Option<Vec<u32>> {
    let mut v = vec![0u32; TRI];
    let mut m = PlaneModel::new();
    let mut d = Decoder::new(bytes);
    for i in 0..TRI {
        let (a, l, u) = neighbours(&v, i);
        let c = PlaneModel::ctx(a, l, u);
        v[i] = if d.bit(&mut m.same_a[c]) {
            a
        } else if u != a && d.bit(&mut m.same_u[c]) {
            u
        } else if l != a && l != u && d.bit(&mut m.same_l[c]) {
            l
        } else {
            m.literal.get(&mut d)?
        };
    }
    (!d.overrun()).then(|| mem_to_disk(&v))
}

struct PlaneModel {
    same_a: [u16; 8],
    same_u: [u16; 8],
    same_l: [u16; 8],
    literal: Golomb,
}

impl PlaneModel {
    fn new() -> Self {
        PlaneModel { same_a: [PROB_ONE / 2; 8], same_u: [PROB_ONE / 2; 8], same_l: [PROB_ONE / 2; 8], literal: Golomb::new() }
    }

    fn ctx(a: u32, l: u32, u: u32) -> usize {
        (a == l) as usize | ((a == u) as usize) << 1 | ((l == u) as usize) << 2
    }
}

/// The trixel before (memory order: the other half of the square, or the square to the left), the same half to the left, the same half above; 0 off the plane.
#[inline(always)]
fn neighbours(v: &[u32], i: usize) -> (u32, u32, u32) {
    let a = if i > 0 { v[i - 1] } else { 0 };
    let tx = (i >> 1) % TEX;
    let l = if tx > 0 { v[i - 2] } else { 0 };
    let u = if i >= 2 * TEX { v[i - 2 * TEX] } else { 0 };
    (a, l, u)
}

const LEVELS: usize = crate::TEX_BITS as usize;

const fn level_offset(l: usize) -> usize {
    2 * ((1 << (2 * l)) - 1) / 3
}

/// The size classes a parent coefficient falls in, as a context for its children.
fn size_class(c: i32) -> usize {
    match c.unsigned_abs() {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        _ => 3,
    }
}

struct CoefModel {
    zero: Vec<u16>,
    sign: Vec<u16>,
    size: Vec<Golomb>,
    roots: Golomb,
}

impl CoefModel {
    fn new() -> Self {
        CoefModel { zero: vec![PROB_ONE / 2; (LEVELS + 1) * 4 * 4 * 4 * 5], sign: vec![PROB_ONE / 2; (LEVELS + 1) * 3], size: (0..(LEVELS + 1) * 4 * 4).map(|_| Golomb::new()).collect(), roots: Golomb::new() }
    }
}

/// A pyramid's coefficient stream (every level, coarse first, quadtree order within a level) as range-coded bytes.
pub fn encode_coefs(coef: &[i32]) -> Vec<u8> {
    encode_coefs_with(coef, None)
}

/// The same, with another plane's coefficients as a context (an image's bands light up in the same places): the reader must have that plane first.
pub fn encode_coefs_with(coef: &[i32], reference: Option<&[i32]>) -> Vec<u8> {
    assert_eq!(coef.len(), level_offset(LEVELS + 1));
    let mut m = CoefModel::new();
    let mut e = Encoder::new();
    for &c in &coef[..2] {
        m.roots.put(&mut e, zigzag(c));
    }
    for l in 1..=LEVELS {
        let (off, poff) = (level_offset(l), level_offset(l - 1));
        let mut prev = 0usize;
        for d in 0..level_offset(l + 1) - off {
            let parent = if l == 1 { 0 } else { size_class(coef[poff + d / 4]) };
            let c = coef[off + d];
            let r = reference.map_or(4, |r| size_class(r[off + d]));
            let z = (((l * 4 + parent) * 4 + d % 4) * 4 + prev) * 5 + r;
            e.bit(&mut m.zero[z], c == 0);
            if c != 0 {
                e.bit(&mut m.sign[l * 3 + reference.map_or(1, |r| (r[off + d].signum() + 1) as usize)], c < 0);
                m.size[(l * 4 + parent) * 4 + prev].put(&mut e, c.unsigned_abs() - 1);
            }
            prev = size_class(c);
        }
    }
    e.finish()
}

/// The coefficient stream back from its bytes; None if they run short.
pub fn decode_coefs(bytes: &[u8]) -> Option<Vec<i32>> {
    decode_coefs_with(bytes, None)
}

/// The coefficient stream back, coded against `reference`.
pub fn decode_coefs_with(bytes: &[u8], reference: Option<&[i32]>) -> Option<Vec<i32>> {
    let mut m = CoefModel::new();
    let mut d = Decoder::new(bytes);
    let mut coef = vec![0i32; level_offset(LEVELS + 1)];
    for c in coef.iter_mut().take(2) {
        *c = unzigzag(m.roots.get(&mut d)?);
    }
    for l in 1..=LEVELS {
        let (off, poff) = (level_offset(l), level_offset(l - 1));
        let mut prev = 0usize;
        for i in 0..level_offset(l + 1) - off {
            let parent = if l == 1 { 0 } else { size_class(coef[poff + i / 4]) };
            let r = reference.map_or(4, |r| size_class(r[off + i]));
            let z = (((l * 4 + parent) * 4 + i % 4) * 4 + prev) * 5 + r;
            if !d.bit(&mut m.zero[z]) {
                let neg = d.bit(&mut m.sign[l * 3 + reference.map_or(1, |r| (r[off + i].signum() + 1) as usize)]);
                let size = m.size[(l * 4 + parent) * 4 + prev].get(&mut d)? as i64 + 1;
                coef[off + i] = (if neg { -size } else { size }) as i32;
            }
            prev = size_class(coef[off + i]);
        }
    }
    (!d.overrun()).then_some(coef)
}

/// A short series (the terrain's apron, a table) as range-coded bytes: each value as its change from the one before.
pub fn encode_series(vals: &[u32]) -> Vec<u8> {
    let mut g = Golomb::new();
    let mut e = Encoder::new();
    let mut last = 0u32;
    for &v in vals {
        g.put(&mut e, zigzag(v.wrapping_sub(last) as i32));
        last = v;
    }
    e.finish()
}

/// A series of `n` values back from its bytes.
pub fn decode_series(bytes: &[u8], n: usize) -> Option<Vec<u32>> {
    let mut g = Golomb::new();
    let mut d = Decoder::new(bytes);
    let mut last = 0u32;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        last = last.wrapping_add(unzigzag(g.get(&mut d)?) as u32);
        out.push(last);
    }
    (!d.overrun()).then_some(out)
}

fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

fn unzigzag(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

// ==================== FIELDS ====================
// How a coded field sits in a VSF section (the cell format of the flag day, Nick 2026-10-14): two values, the range-coded bytes and a `t_u5` of what they were, so a reader can put a plane or a table back exactly as it was written and hand a pyramid's coefficients straight to the inverse.
// A field coded otherwise (Rice multi-values, a bare tensor) is never two values of this shape, so the old and new forms tell themselves apart.

/// A pyramid's coefficients.
const COEFS: u32 = 1;
/// A pyramid's coefficients coded against the section's first band (an image's red).
const COEFS_REF: u32 = 2;
/// One value per trixel: the meta goes on with the tensor's type and shape.
const PLANE: u32 = 3;
/// A short table of 16-bit numbers: type and shape follow.
const SERIES: u32 = 4;

const T_U8: u32 = 0;
const T_U16: u32 = 1;
const BITS: u32 = 2;
const T_I16: u32 = 3;

fn coded(bytes: Vec<u8>, meta: Vec<u32>) -> Vec<VsfType> {
    vec![VsfType::t_u3(vsf::types::Tensor::new(vec![bytes.len()], bytes)), VsfType::t_u5(vsf::types::Tensor::new(vec![meta.len()], meta))]
}

/// The coded bytes and their meta, if the values are a coded field (a one-dimensional tensor reads back as `v_`, so either form).
fn parts(values: &[VsfType]) -> Option<(&[u8], &[u32])> {
    let bytes: &[u8] = match values.first()? {
        VsfType::t_u3(b) => &b.data,
        VsfType::v_u3(b) => &b.data,
        _ => return None,
    };
    let meta: &[u32] = match values.get(1)? {
        VsfType::t_u5(m) => &m.data,
        VsfType::v_u5(m) => &m.data,
        _ => return None,
    };
    (values.len() == 2 && !meta.is_empty() && (COEFS..=SERIES).contains(&meta[0])).then_some((bytes, meta))
}

/// A field's values coded, when they are a kind the coder knows (a pyramid, a plane, a table); None to keep them as they are.
pub fn code_field(values: &[VsfType], reference: Option<&[i32]>) -> Option<Vec<VsfType>> {
    if let Some(c) = crate::pyr::from_vsf(values, crate::pyr::NCOEF) {
        return Some(match reference {
            Some(r) => coded(encode_coefs_with(&c, Some(r)), vec![COEFS_REF]),
            None => coded(encode_coefs(&c), vec![COEFS]),
        });
    }
    let [v] = values else { return None };
    let (ty, shape, vals): (u32, &[usize], Vec<u32>) = match v {
        VsfType::t_u3(t) => (T_U8, &t.shape, t.data.iter().map(|&x| x as u32).collect()),
        VsfType::t_u4(t) => (T_U16, &t.shape, t.data.iter().map(|&x| x as u32).collect()),
        VsfType::t_i4(t) => (T_I16, &t.shape, t.data.iter().map(|&x| x as u16 as u32).collect()),
        VsfType::p(t) if t.bit_depth == 1 => {
            let n: usize = t.shape.iter().product();
            (BITS, &t.shape, (0..n).map(|i| ((t.data[i >> 3] >> (7 - (i & 7))) & 1) as u32).collect())
        }
        _ => return None,
    };
    let mut meta: Vec<u32> = vec![0, ty];
    meta.extend(shape.iter().map(|&s| s as u32));
    if vals.len() == TRI {
        meta[0] = PLANE;
        Some(coded(encode_plane(&vals), meta))
    } else if vals.len() >= 64 && ty == T_U16 {
        meta[0] = SERIES;
        Some(coded(encode_series(&vals), meta))
    } else {
        None
    }
}

/// A coded plane or table back as the tensor it was written from; None for anything else (a pyramid stays coded for [`coefs`]).
pub fn uncode_field(values: &[VsfType]) -> Option<VsfType> {
    let (bytes, meta) = parts(values)?;
    if meta.len() < 2 || !(meta[0] == PLANE || meta[0] == SERIES) {
        return None;
    }
    let shape: Vec<usize> = meta[2..].iter().map(|&s| s as usize).collect();
    let n: usize = shape.iter().product();
    let vals = if meta[0] == PLANE { decode_plane(bytes)? } else { decode_series(bytes, n)? };
    if vals.len() != n {
        return None;
    }
    Some(match meta[1] {
        T_U8 => VsfType::t_u3(vsf::types::Tensor::new(shape, vals.iter().map(|&x| x as u8).collect())),
        T_U16 => VsfType::t_u4(vsf::types::Tensor::new(shape, vals.iter().map(|&x| x as u16).collect())),
        T_I16 => VsfType::t_i4(vsf::types::Tensor::new(shape, vals.iter().map(|&x| x as u16 as i16).collect())),
        BITS => {
            let mut data = vec![0u8; n.div_ceil(8)];
            for (i, &b) in vals.iter().enumerate() {
                if b != 0 {
                    data[i >> 3] |= 0x80 >> (i & 7);
                }
            }
            VsfType::p(vsf::types::BitPackedTensor { bit_depth: 1, shape, data })
        }
        _ => return None,
    })
}

/// Whether a field's coefficients were coded against the section's first band.
pub fn needs_reference(values: &[VsfType]) -> bool {
    parts(values).is_some_and(|(_, m)| m[0] == COEFS_REF)
}

/// A pyramid's coefficients from its field, coded either way (the reference for one coded against a first band).
pub fn coefs(values: &[VsfType], reference: Option<&[i32]>) -> Option<Vec<i32>> {
    match parts(values) {
        Some((b, m)) if m[0] == COEFS => decode_coefs(b),
        Some((b, m)) if m[0] == COEFS_REF => decode_coefs_with(b, Some(reference?)),
        Some(_) => None,
        None => crate::pyr::from_vsf(values, crate::pyr::NCOEF),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planes_round_trip() {
        let sparse: Vec<u32> = (0..TRI as u32).map(|i| if i % 513 < 3 { 5 + i % 3 } else { 0 }).collect();
        let dense: Vec<u32> = (0..TRI as u32).map(|i| 1 + (i / 4000) % 7).collect();
        let noisy: Vec<u32> = (0..TRI as u32).map(|i| i.wrapping_mul(2654435761) >> 16).collect();
        for p in [&sparse, &dense, &noisy, &vec![0u32; TRI], &vec![u32::MAX; TRI]] {
            assert_eq!(decode_plane(&encode_plane(p)).unwrap(), *p);
        }
        assert!(encode_plane(&vec![0u32; TRI]).len() < 64);
    }

    #[test]
    fn coefs_round_trip() {
        let n = level_offset(LEVELS + 1);
        let c: Vec<i32> = (0..n as i32).map(|i| if i % 7 == 0 { (i * 31) % 201 - 100 } else if i % 11 == 0 { i32::MIN + 1 } else { 0 }).collect();
        assert_eq!(decode_coefs(&encode_coefs(&c)).unwrap(), c);
        assert_eq!(decode_coefs(&encode_coefs(&vec![0; n])).unwrap(), vec![0; n]);
    }
}
