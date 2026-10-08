//! The pyramid codec: a cell's plane coded as its own quadtree, coarse first.
//!
//! A plane in disk order IS a quadtree — every four consecutive texels are the children of one parent triangle, recursively, up to the two face roots. The codec builds the pyramid of means bottom-up (the same rule the cross-cell pyramid uses) and then codes it top-down: the two roots raw, and every finer level as the difference between each texel and a prediction extrapolated from the reconstructed coarser level — the parent's value plus the parent plane's local gradient times the child's centroid offset. A sloped plane predicts exactly, so what remains is curvature and noise. Stored coarse-to-fine, the first N levels of the stream decode to a 2^N-wide version of the plane, so a reader can stop early.
//!
//! Lossy means the diffs are quantised with a per-level step and a small dead zone; the encoder predicts from what the decoder will reconstruct, so quantisation error never compounds. Smooth terrain and flat imagery collapse to runs of zeros, rough ground keeps its detail.
//!
//! Entropy coding is bit-packing in blocks: every block of BLOCK coefficients is zigzag-mapped and packed at the width its largest member needs, with one width byte per block. The packed bits are VSF `BitPackedTensor`s, one per width in use, so the file stays self-describing and whole-file zstd finishes the job.

use std::sync::OnceLock;

use vsf::VsfType;
use vsf::types::{BitPackedTensor, Tensor};

use crate::{TEX_BITS, TRI, tri_children};

/// Quadtree depth: the plane's leaves sit this many levels below the two face roots.
pub const LEVELS: usize = TEX_BITS as usize;

/// Coefficients per entropy block (and the granularity of the width table).
pub const BLOCK: usize = 16;

/// Texels at a level: two faces of 4^l.
#[inline]
const fn level_len(l: usize) -> usize {
    2 << (2 * l)
}

/// Offset of level `l` in the coefficient stream: every coarser level before it.
#[inline]
const fn level_offset(l: usize) -> usize {
    2 * ((1 << (2 * l)) - 1) / 3
}

/// Coefficients in a full stream: every level of the pyramid.
pub const NCOEF: usize = level_offset(LEVELS + 1);

/// Memory index of (tx, ty, half) on a 2^l grid.
#[inline(always)]
fn idx(l: usize, tx: usize, ty: usize, half: usize) -> usize {
    (((ty << l) | tx) << 1) | half
}

/// Disk (triangle path) order -> memory index, for the 2^l grid of level l.
fn order(l: usize) -> &'static [u32] {
    static ORDERS: OnceLock<Vec<Box<[u32]>>> = OnceLock::new();
    &ORDERS.get_or_init(|| {
        (0..=LEVELS)
            .map(|l| {
                let mut to_mem = vec![0u32; level_len(l)].into_boxed_slice();
                fn descend(l: usize, level: usize, tx: usize, ty: usize, half: usize, path: usize, out: &mut [u32]) {
                    if level == l {
                        out[path] = idx(l, tx, ty, half) as u32;
                        return;
                    }
                    for (d, (cx, cy, ch)) in tri_children(tx, ty, half).into_iter().enumerate() {
                        descend(l, level + 1, cx, cy, ch, (path << 2) | d, out);
                    }
                }
                for face in 0..2 {
                    descend(l, 0, 0, 0, face, face, &mut to_mem);
                }
                to_mem
            })
            .collect()
    })[l]
}

/// Per-level quantiser steps for the diffs, index 1..=LEVELS (0 unused); 1 everywhere is lossless.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Steps(pub [u32; LEVELS + 1]);

impl Steps {
    pub const LOSSLESS: Steps = Steps([1; LEVELS + 1]);

    /// Loss concentrated where it is invisible: `leaf` at the finest level, halving each level up until it reaches 1.
    pub fn tapered(leaf: u32) -> Steps {
        let mut s = [1u32; LEVELS + 1];
        let mut v = leaf.max(1);
        for l in (1..=LEVELS).rev() {
            s[l] = v;
            v = (v / 2).max(1);
        }
        Steps(s)
    }
}

/// Deadzone quantiser with a third-step rounding offset: small diffs round to zero a little more readily than a plain round, which is where the lossy savings live.
#[inline(always)]
fn quant(d: i32, step: u32) -> i32 {
    if step == 1 {
        return d;
    }
    let s = step as i64;
    let m = (d.unsigned_abs() as i64 + s / 3) / s;
    if d < 0 { -(m as i32) } else { m as i32 }
}

/// The pyramid of means, leaves first: level l is a 2^l grid in memory order, each parent the rounded mean of its four children.
fn means(leaves: &[i32]) -> Vec<Vec<i32>> {
    let mut levels: Vec<Vec<i32>> = vec![Vec::new(); LEVELS + 1];
    levels[LEVELS] = leaves.to_vec();
    for l in (0..LEVELS).rev() {
        let n = 1usize << l;
        let mut cur = vec![0i32; level_len(l)];
        let fine = &levels[l + 1];
        for ty in 0..n {
            for tx in 0..n {
                for half in 0..2 {
                    let sum: i64 = tri_children(tx, ty, half).iter().map(|&(cx, cy, ch)| fine[idx(l + 1, cx, cy, ch)] as i64).sum();
                    cur[idx(l, tx, ty, half)] = ((sum + 2) >> 2) as i32;
                }
            }
        }
        levels[l] = cur;
    }
    levels
}

/// Predict every texel of level `l` from the reconstructed level `l - 1`: the parent's value plus its plane's gradient (central differences along u and v within the same half, one-sided at the edges) times the child centroid's offset from the parent centroid, in twelfths of a parent texel.
fn predict(l: usize, coarse: &[i32], out: &mut [i32]) {
    let n = 1usize << (l - 1);
    let at = |tx: usize, ty: usize, half: usize| coarse[idx(l - 1, tx, ty, half)] as i64;
    for ty in 0..n {
        for tx in 0..n {
            for half in 0..2 {
                let p = at(tx, ty, half);
                // Differences over two texels (gradient × 2), one-sided at an edge (doubled to keep the scale).
                let du = match (tx > 0, tx + 1 < n) {
                    (true, true) => at(tx + 1, ty, half) - at(tx - 1, ty, half),
                    (false, true) => 2 * (at(tx + 1, ty, half) - p),
                    (true, false) => 2 * (p - at(tx - 1, ty, half)),
                    (false, false) => 0,
                };
                let dv = match (ty > 0, ty + 1 < n) {
                    (true, true) => at(tx, ty + 1, half) - at(tx, ty - 1, half),
                    (false, true) => 2 * (at(tx, ty + 1, half) - p),
                    (true, false) => 2 * (p - at(tx, ty - 1, half)),
                    (false, false) => 0,
                };
                // Child centroid offsets in sixths of a parent texel, in child order (apex, +v, +u, centre).
                let offs: [(i64, i64); 4] = if half == 0 { [(-1, -1), (-1, 2), (2, -1), (0, 0)] } else { [(1, 1), (-2, 1), (1, -2), (0, 0)] };
                for (k, (cx, cy, ch)) in tri_children(tx, ty, half).into_iter().enumerate() {
                    let (ou, ov) = offs[k];
                    // gradient · offset = (du / 2) · (ou / 6) + (dv / 2) · (ov / 6), rounded.
                    let pred = p + (du * ou + dv * ov + 6).div_euclid(12);
                    out[idx(l, cx, cy, ch)] = pred as i32;
                }
            }
        }
    }
}

/// Forward transform of a plane in memory order: returns the coefficient stream, quantised per `steps`.
pub fn forward(x: &[i32], steps: &Steps) -> Vec<i32> {
    assert_eq!(x.len(), TRI);
    let orig = means(x);
    let mut coef = vec![0i32; NCOEF];
    let mut recon = orig[0].clone();
    coef[0] = recon[order(0)[0] as usize];
    coef[1] = recon[order(0)[1] as usize];
    let mut pred = Vec::new();
    for l in 1..=LEVELS {
        let step = steps.0[l];
        pred.clear();
        pred.resize(level_len(l), 0);
        predict(l, &recon, &mut pred);
        let mut next = vec![0i32; level_len(l)];
        let out = &mut coef[level_offset(l)..level_offset(l + 1)];
        let ord = order(l);
        for g in 0..ord.len() / 4 {
            // Siblings share their curvature: each child after the first is nudged by half the mean residual of the siblings already coded.
            let mut bias = 0i32;
            for k in 0..4 {
                let (d, m) = (4 * g + k, ord[4 * g + k] as usize);
                let p = pred[m] + bias;
                let q = quant(orig[l][m] - p, step);
                out[d] = q;
                next[m] = p + q * step as i32;
                bias = sibling_bias(k, bias, next[m] - pred[m]);
            }
        }
        recon = next;
    }
    coef
}

/// The bias for child `k + 1` given child `k`'s bias and its residual against the plane prediction. The parent is the mean of its children and the plane predictions average to the parent, so the four residuals sum to about zero: each child expects minus the sum so far, shared over the children still to come. The last child is nearly free.
#[inline(always)]
fn sibling_bias(k: usize, bias: i32, residual: i32) -> i32 {
    // Sum of residuals r_0..r_k, recovered from the previous bias: sum_{k-1} = -bias · (4 - k).
    let sum = residual as i64 - bias as i64 * (4 - k as i64);
    let remaining = 3 - k as i64;
    if remaining == 0 { 0 } else { (-(sum + remaining.signum() * (remaining / 2) * sum.signum()) / remaining) as i32 }
}

/// Inverse transform to a plane in memory order. `levels` ≤ LEVELS stops early: the coarser result is replicated down to the leaves, which is what a progressive reader shows while the rest is in flight.
pub fn inverse(coef: &[i32], steps: &Steps, levels: usize) -> Vec<i32> {
    assert_eq!(coef.len(), NCOEF);
    let levels = levels.min(LEVELS);
    let mut recon = vec![0i32; 2];
    recon[order(0)[0] as usize] = coef[0];
    recon[order(0)[1] as usize] = coef[1];
    let mut pred = Vec::new();
    for l in 1..=levels {
        let step = steps.0[l] as i32;
        pred.clear();
        pred.resize(level_len(l), 0);
        predict(l, &recon, &mut pred);
        let diffs = &coef[level_offset(l)..level_offset(l + 1)];
        let ord = order(l);
        for g in 0..ord.len() / 4 {
            let mut bias = 0i32;
            for k in 0..4 {
                let (d, m) = (4 * g + k, ord[4 * g + k] as usize);
                let residual = bias + diffs[d] * step;
                pred[m] += residual;
                bias = sibling_bias(k, bias, residual);
            }
        }
        recon = std::mem::take(&mut pred);
    }
    if levels < LEVELS {
        // A leaf's ancestor at any level is its triangle path truncated: disk index shifted down two bits per level.
        let shift = 2 * (LEVELS - levels);
        let coarse = order(levels);
        let mut out = vec![0i32; TRI];
        for (d, &m) in order(LEVELS).iter().enumerate() {
            out[m as usize] = recon[coarse[d >> shift] as usize];
        }
        out
    } else {
        recon
    }
}

/// Fill holes before the transform so they cost nothing: each missing leaf takes the mean of the nearest ancestor that had any valid texel beneath it, so a hole's diffs are zero against its neighbours. Plane and mask in disk order.
pub fn fill_holes(x: &mut [i32], valid: &[bool]) {
    assert_eq!(x.len(), TRI);
    assert_eq!(valid.len(), TRI);
    // Bottom-up: (sum, count) per node, levels LEVELS..=0.
    let mut sums: Vec<Vec<(i64, u32)>> = Vec::with_capacity(LEVELS + 1);
    sums.push(x.iter().zip(valid).map(|(&v, &ok)| if ok { (v as i64, 1) } else { (0, 0) }).collect());
    for _ in 0..LEVELS {
        let prev = sums.last().unwrap();
        let next: Vec<(i64, u32)> = prev.chunks(4).map(|g| g.iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))).collect();
        sums.push(next);
    }
    // Top-down: the fill value at each node is its own mean, or its parent's fill.
    let mut fill: Vec<i32> = sums[LEVELS].iter().map(|&(s, n)| if n > 0 { (s / n as i64) as i32 } else { 0 }).collect();
    for d in (0..LEVELS).rev() {
        let level = &sums[d];
        let mut next = vec![0i32; level.len()];
        for (j, &(s, n)) in level.iter().enumerate() {
            next[j] = if n > 0 { (s / n as i64) as i32 } else { fill[j / 4] };
        }
        fill = next;
    }
    for i in 0..TRI {
        if !valid[i] {
            x[i] = fill[i];
        }
    }
}

#[inline(always)]
fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

#[inline(always)]
fn unzigzag(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

/// MSB-first, big-endian bit packing at a fixed width — the byte layout of a VSF `BitPackedTensor`, written here because the per-sample path matters for a loader thread.
fn pack_bits(width: u32, vals: &[u32]) -> Vec<u8> {
    let total_bits = vals.len() * width as usize;
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    let (mut acc, mut nbits, mut pos) = (0u64, 0u32, 0usize);
    for &v in vals {
        acc = (acc << width) | v as u64;
        nbits += width;
        while nbits >= 8 {
            nbits -= 8;
            out[pos] = (acc >> nbits) as u8;
            pos += 1;
        }
    }
    if nbits > 0 {
        out[pos] = (acc << (8 - nbits)) as u8;
    }
    out
}

fn unpack_bits(width: u32, count: usize, data: &[u8], out: &mut Vec<u32>) -> Option<()> {
    if (count * width as usize).div_ceil(8) > data.len() {
        return None;
    }
    let mask = if width == 32 { u32::MAX } else { (1u32 << width) - 1 };
    let (mut acc, mut nbits, mut pos) = (0u64, 0u32, 0usize);
    for _ in 0..count {
        while nbits < width {
            acc = (acc << 8) | data[pos] as u64;
            pos += 1;
            nbits += 8;
        }
        nbits -= width;
        out.push(((acc >> nbits) as u32) & mask);
    }
    Some(())
}

/// Largest Rice parameter a block may choose, and the unary length that instead means "the value follows in full".
const MAX_K: u32 = 24;
const ESCAPE: u32 = 24;

/// A coded plane as VSF values — Rice coding with a parameter per block: the parameter table, the unary quotients as a one-bit tensor, one remainder tensor per parameter in use (ascending, each self-describing its width), and a 32-bit tensor of escaped values if any quotient was too long.
pub fn to_vsf(coef: &[i32]) -> Vec<VsfType> {
    let zz: Vec<u32> = coef.iter().map(|&c| zigzag(c)).collect();
    let mut ks = Vec::with_capacity(zz.len().div_ceil(BLOCK));
    let mut unary: Vec<u32> = Vec::with_capacity(zz.len() * 2);
    let mut rem: Vec<Vec<u32>> = vec![Vec::new(); MAX_K as usize + 1];
    let mut escapes: Vec<u32> = Vec::new();
    for b in zz.chunks(BLOCK) {
        let cost = |k: u32| -> usize { b.iter().map(|&v| (k + 1 + (v >> k).min(ESCAPE + 32)) as usize).sum() };
        let k = (0..=MAX_K).min_by_key(|&k| cost(k)).unwrap();
        ks.push(k as u8);
        for &v in b {
            let q = v >> k;
            if q >= ESCAPE {
                unary.extend(std::iter::repeat_n(1u32, ESCAPE as usize));
                escapes.push(v);
            } else {
                unary.extend(std::iter::repeat_n(1u32, q as usize));
                unary.push(0);
                if k > 0 {
                    rem[k as usize].push(v & ((1 << k) - 1));
                }
            }
        }
    }
    let mut values = vec![
        VsfType::t_u3(Tensor::new(vec![ks.len()], ks)),
        VsfType::p(BitPackedTensor { bit_depth: 1, shape: vec![unary.len()], data: pack_bits(1, &unary) }),
    ];
    for (k, vals) in rem.iter().enumerate().skip(1) {
        if !vals.is_empty() {
            values.push(VsfType::p(BitPackedTensor { bit_depth: k as u8, shape: vec![vals.len()], data: pack_bits(k as u32, vals) }));
        }
    }
    if !escapes.is_empty() {
        values.push(VsfType::p(BitPackedTensor { bit_depth: 32, shape: vec![escapes.len()], data: pack_bits(32, &escapes) }));
    }
    values
}

/// The coefficient stream of `n` coefficients back from its VSF values, or None if they are not a coded plane of that size.
pub fn from_vsf(values: &[VsfType], n: usize) -> Option<Vec<i32>> {
    let ks: &[u8] = match values.first()? {
        VsfType::t_u3(t) => &t.data,
        _ => return None,
    };
    if ks.len() != n.div_ceil(BLOCK) {
        return None;
    }
    let unary = match values.get(1)? {
        VsfType::p(t) if t.bit_depth == 1 => t,
        _ => return None,
    };
    let unary_bits: usize = unary.shape.iter().product();
    if unary_bits.div_ceil(8) > unary.data.len() {
        return None;
    }
    let mut rem: Vec<Vec<u32>> = vec![Vec::new(); 33];
    for v in &values[2..] {
        if let VsfType::p(t) = v {
            if (1..=32).contains(&t.bit_depth) {
                let count: usize = t.shape.iter().product();
                let w = t.bit_depth as usize;
                rem[w].clear();
                unpack_bits(w as u32, count, &t.data, &mut rem[w])?;
            }
        }
    }
    let mut cursor = [0usize; 33];
    let mut bit = 0usize;
    let mut coef = Vec::with_capacity(n);
    for (b, &k) in ks.iter().enumerate() {
        let k = k as u32;
        if k > MAX_K {
            return None;
        }
        let len = BLOCK.min(n - b * BLOCK);
        for _ in 0..len {
            // Count the ones up to the terminating zero (or the escape length).
            let mut q = 0u32;
            while q < ESCAPE {
                if bit >= unary_bits {
                    return None;
                }
                let one = (unary.data[bit >> 3] >> (7 - (bit & 7))) & 1 == 1;
                bit += 1;
                if !one {
                    break;
                }
                q += 1;
            }
            let v = if q >= ESCAPE {
                let c = cursor[32];
                cursor[32] += 1;
                *rem[32].get(c)?
            } else if k == 0 {
                q
            } else {
                let c = cursor[k as usize];
                cursor[k as usize] += 1;
                (q << k) | *rem[k as usize].get(c)?
            };
            coef.push(unzigzag(v));
        }
    }
    Some(coef)
}

/// A validity mask as one bit per texel.
pub fn mask_vsf(valid: &[bool]) -> VsfType {
    let bits: Vec<u32> = valid.iter().map(|&v| v as u32).collect();
    VsfType::p(BitPackedTensor { bit_depth: 1, shape: vec![valid.len()], data: pack_bits(1, &bits) })
}

pub fn mask_from_vsf(v: &VsfType) -> Option<Vec<bool>> {
    let VsfType::p(t) = v else { return None };
    let n: usize = t.shape.iter().product();
    if t.bit_depth != 1 || n != TRI {
        return None;
    }
    let mut bits = Vec::with_capacity(n);
    unpack_bits(1, n, &t.data, &mut bits)?;
    Some(bits.into_iter().map(|b| b != 0).collect())
}

/// Encode a plane (memory order, holes filled) end to end.
pub fn encode(x: &[i32], steps: &Steps) -> Vec<VsfType> {
    to_vsf(&forward(x, steps))
}

/// Decode a plane (memory order) end to end, all levels.
pub fn decode(values: &[VsfType], steps: &Steps) -> Option<Vec<i32>> {
    Some(inverse(&from_vsf(values, NCOEF)?, steps, LEVELS))
}

/// The steps as a VSF value: one byte per level, finest last.
pub fn steps_vsf(s: &Steps) -> VsfType {
    VsfType::t_u3(Tensor::new(vec![LEVELS], (1..=LEVELS).map(|l| s.0[l].min(255) as u8).collect()))
}

pub fn steps_from_vsf(v: Option<&VsfType>) -> Steps {
    let mut s = Steps::LOSSLESS;
    if let Some(VsfType::t_u3(t)) = v {
        if t.data.len() == LEVELS {
            for l in 1..=LEVELS {
                s.0[l] = (t.data[l - 1] as u32).max(1);
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TEX, tri_idx};

    fn ramp() -> Vec<i32> {
        (0..TRI).map(|i| ((i * 7919) % 1000) as i32 - 500 + (i / 64) as i32).collect()
    }

    #[test]
    fn lossless_round_trip_and_progressive_means() {
        let x = ramp();
        let coef = forward(&x, &Steps::LOSSLESS);
        assert_eq!(inverse(&coef, &Steps::LOSSLESS, LEVELS), x);
        // Stopping two levels early gives every leaf its level-6 ancestor's mean (to within the rounding of nested means).
        let coarse = inverse(&coef, &Steps::LOSSLESS, LEVELS - 2);
        let m = means(&x);
        for ty in 0..TEX {
            for tx in 0..TEX {
                for half in 0..2 {
                    let v = coarse[tri_idx(tx, ty, half)];
                    let (cx, cy) = (tx >> 2, ty >> 2);
                    let candidates = [m[6][idx(6, cx, cy, 0)], m[6][idx(6, cx, cy, 1)]];
                    assert!(candidates.contains(&v), "({tx},{ty},{half}) = {v}, expected one of {candidates:?}");
                }
            }
        }
        let back = from_vsf(&to_vsf(&coef), NCOEF).unwrap();
        assert_eq!(back, coef);
    }

    #[test]
    fn a_plane_costs_nothing_past_the_roots() {
        // A tilted plane sampled at the centroids, f = 12u + 21v with the centroid a third or two thirds into the square: every level's prediction is exact, so nothing past the two roots carries anything.
        let x: Vec<i32> = (0..TRI).map(|i| { let (tx, ty, half) = ((i >> 1) & (TEX - 1), i >> 9, i & 1); 12 * tx as i32 + 21 * ty as i32 + 11 * (half as i32 + 1) }).collect();
        let coef = forward(&x, &Steps::LOSSLESS);
        let nonzero = coef[2..].iter().filter(|&&c| c != 0).count();
        assert!(nonzero < NCOEF / 50, "{nonzero} of {NCOEF} coefficients nonzero on a plane");
        assert_eq!(inverse(&coef, &Steps::LOSSLESS, LEVELS), x);
    }

    #[test]
    fn lossy_error_is_bounded_and_flat_runs_vanish() {
        let x = ramp();
        let steps = Steps::tapered(8);
        let coef = forward(&x, &steps);
        let y = inverse(&coef, &steps, LEVELS);
        let worst = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).max().unwrap();
        // Closed-loop prediction: a leaf's error is only its own level's quantiser, under two thirds of the step.
        assert!(worst <= 5, "worst error {worst}");
        let flat = vec![42i32; TRI];
        let c = forward(&flat, &steps);
        assert!(c[2..].iter().all(|&v| v == 0));
        assert_eq!(inverse(&c, &steps, LEVELS), flat);
        let v = to_vsf(&c);
        assert_eq!(v.len(), 3, "the parameter table, the unary bits, and one remainder stream for the roots' block");
    }

    #[test]
    fn packing_matches_vsf_bit_layout() {
        let vals: Vec<u32> = (0..1000u32).map(|i| i.wrapping_mul(2654435761) >> 19).collect();
        for w in [1u32, 3, 8, 13, 17, 32] {
            let mask = if w == 32 { u32::MAX } else { (1 << w) - 1 };
            let masked: Vec<u32> = vals.iter().map(|&v| v & mask).collect();
            let mine = pack_bits(w, &masked);
            let theirs = BitPackedTensor::pack_u32(w as u8, vec![masked.len()], &masked);
            assert_eq!(mine, theirs.data, "width {w}");
            let mut back = Vec::new();
            unpack_bits(w, masked.len(), &mine, &mut back).unwrap();
            assert_eq!(back, masked);
        }
    }

    #[test]
    fn holes_fill_from_neighbours() {
        let mut x = ramp();
        let mut valid = vec![true; TRI];
        for i in (0..TRI).step_by(5) {
            valid[i] = false;
            x[i] = 1 << 20;
        }
        fill_holes(&mut x, &valid);
        for i in (0..TRI).step_by(5) {
            let g = i / 4 * 4;
            let (lo, hi) = (g..g + 4).filter(|&j| valid[j]).fold((i32::MAX, i32::MIN), |(a, b), j| (a.min(x[j]), b.max(x[j])));
            assert!(x[i] >= lo && x[i] <= hi, "hole {i} filled with {} outside {lo}..{hi}", x[i]);
        }
    }
}
