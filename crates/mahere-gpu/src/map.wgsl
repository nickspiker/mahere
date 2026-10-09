// The #pagetable compositor on the GPU. One instance per screen block: the vertex stage places the block's quad and hands the fragment its position inside the block; the fragment walks position → diamond UV → page table → texel → the same compose the CPU loop does, per supersample. A second pass bins the supersampled image down 2×2 and lays the screen-space marks over it.

const NONE: u32 = 0xFFFFFFFFu;
// A cell the loader confirmed absent at the wanted depth: found by lookup, never referred to.
const ABSENT: u32 = 0xFFFFFFFEu;
// The coarsest depth a cell exists at (the global bake's root), the same as the planner's. A walk below it must stop: a fragment loop that never ends hangs the GPU and the driver resets it for every process on the device (2026-10-09, the phone's SystemUI went down with mahere).
const MIN_DEPTH: u32 = 0u;
const IMG_WATER: f32 = 0.03;
const TABLE_N: u32 = 16384u;
const ELEV_NODATA: u32 = 0xFFFFu;

const FLAG_DEM: u32 = 1u;
const FLAG_LINE: u32 = 2u;
const FLAG_LAND: u32 = 4u;
const FLAG_WATER: u32 = 8u;
const FLAG_IMG: u32 = 16u;
const FLAG_VEC: u32 = 14u;

const M_DEM: u32 = 1u;
const M_LAND: u32 = 2u;
const M_WATER: u32 = 4u;
const M_LINE: u32 = 8u;
const M_DEBUG: u32 = 16u;
const M_IMAGERY: u32 = 32u;
const M_CONTOURS: u32 = 64u;
const M_SLOPE: u32 = 128u;
const M_INFRARED: u32 = 256u;
const M_HYPSO: u32 = 512u;
const M_BOUND: u32 = 1024u;

// Where the style tables sit in the LUT buffer after the 4096 hypsometric rows.
const CLASS_BASE: u32 = 4096u;
const LAND_BASE: u32 = 4128u;
const CLASS_MAX: u32 = 17u;
const BOUNDARY_FIRST: u32 = 13u;
const WATERWAY_CLASS: u32 = 12u;

struct Uniforms {
    size: vec2<f32>,
    scale: f32,
    mask: u32,
    light: array<vec4<f32>, 8>,
    sun: vec4<f32>,
    contour: vec4<f32>,
    depths: vec4<u32>,
    offset: vec4<f32>,
    pin: vec4<f32>,
    line_hi: array<vec4<f32>, 8>,
    measure: vec4<f32>,
    style_water: vec4<f32>,
    style_contour: vec4<f32>,
    style_contour_index: vec4<f32>,
    style_flat: vec4<f32>,
    style_bg: vec4<f32>,
    style_no_dem: vec4<f32>,
    style_sea: vec4<f32>,
    // The display: VSF RGB to its primaries (rows), and x = 1 when highlights are compressed (else linear).
    display: array<vec4<f32>, 3>,
    tone: vec4<f32>,
    // Stored imagery byte to scene light (the cells' tone unrolled), 256 entries.
    img_table: array<vec4<f32>, 64>,
    // The globe's disk in render-target pixels: centre x, y, radius squared, and 1 when the limb is on the screen.
    globe: vec4<f32>,
};

// a: diamond, depth, cu, cv. b: dem slot, line slot, land+water slot, img slot. c: flags. d: base, step, eu, nu. e: ev, nv, inv det.
struct Ref { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<f32>, e: vec4<f32> };
// a: x, y, size, diamond. b: u0, v0. c: du/dx, dv/dx, du/dy, dv/dy. d: twist u, twist v.
struct Block { a: vec4<u32>, b: vec4<u32>, c: vec4<f32>, d: vec4<f32> };
struct Slot { tag: u32, cu: u32, cv: u32, index: u32 };

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> blocks: array<Block>;
@group(0) @binding(2) var<storage, read> refs: array<Ref>;
@group(0) @binding(3) var<storage, read> table: array<Slot>;
@group(0) @binding(4) var<storage, read> lut: array<u32>;
@group(0) @binding(5) var dem_tex: texture_2d_array<u32>;
@group(0) @binding(6) var line_tex: texture_2d_array<u32>;
@group(0) @binding(7) var lw_tex: texture_2d_array<u32>;
@group(0) @binding(8) var img_tex: texture_2d_array<u32>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) block: u32,
    @location(1) local: vec2<f32>,
};

@vertex
fn vs_map(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let b = blocks[ii];
    let x = f32(vi == 1u || vi == 3u || vi == 4u);
    let y = f32(vi == 2u || vi == 4u || vi == 5u);
    let size = f32(b.a.z);
    let local = vec2<f32>(x, y) * size;
    let screen = vec2<f32>(f32(b.a.x), f32(b.a.y)) + local;
    let ndc = vec2<f32>(screen.x * U.scale / U.size.x * 2.0 - 1.0, 1.0 - screen.y * U.scale / U.size.y * 2.0);
    var o: VOut;
    o.pos = vec4<f32>(ndc, 0.0, 1.0);
    o.block = ii;
    o.local = local;
    return o;
}

// The page table: a cell at (diamond, depth, grid) → its reference, or NONE. Linear probing over the same hash the planner used.
fn lookup(d: u32, depth: u32, u: u32, v: u32) -> u32 {
    let s = 30u - depth;
    let cu = u >> s;
    let cv = v >> s;
    var h = (cu * 0x9E3779B1u) ^ (cv * 0x85EBCA77u) ^ (depth * 0xC2B2AE3Du) ^ (d * 0x27D4EB2Fu);
    h = (h ^ (h >> 15u)) & (TABLE_N - 1u);
    let tag = d | (depth << 8u);
    for (var i = 0u; i < 8u; i++) {
        let e = table[(h + i) & (TABLE_N - 1u)];
        if (e.index == NONE) {
            return NONE;
        }
        if (e.tag == tag && e.cu == cu && e.cv == cv) {
            return e.index;
        }
    }
    return NONE;
}

// The probe: the cell at `want` or the nearest parent carrying any of `need`.
// Bounded by the depth count rather than open: no loop in this shader may run unbounded.
fn find(d: u32, want: u32, u: u32, v: u32, need: u32) -> u32 {
    var depth = want;
    for (var k = 0u; k <= 30u; k++) {
        let r = lookup(d, depth, u, v);
        if (r != NONE && r != ABSENT && (refs[r].c.x & need) != 0u) {
            return r;
        }
        if (depth <= MIN_DEPTH) {
            return NONE;
        }
        depth = depth - 1u;
    }
    return NONE;
}

// Triangle texel of a depth-30 UV in a cell at `depth`: the UV square, then the carry of the fractions.
fn tri(u: u32, v: u32, depth: u32) -> vec3<u32> {
    let s = 22u - depth;
    let m = (1u << s) - 1u;
    return vec3<u32>((u >> s) & 255u, (v >> s) & 255u, (((u & m) + (v & m)) >> s) & 1u);
}

fn dem_q(slot: u32, x: i32, y: i32) -> u32 {
    return textureLoad(dem_tex, vec2<i32>(x, y), i32(slot), 0).x;
}

fn diffq(m: u32, p: u32, c: u32) -> f32 {
    let mn = m == ELEV_NODATA;
    let pn = p == ELEV_NODATA;
    if (!mn && !pn) {
        return (f32(p) - f32(m)) * 0.5;
    }
    if (mn && !pn) {
        return f32(p) - f32(c);
    }
    if (!mn && pn) {
        return f32(c) - f32(m);
    }
    return 0.0;
}

struct Dem { ok: bool, elev: f32, n: vec3<f32>, depth: u32, slot: u32, base: f32, step: f32 };

// Elevation and normal at a UV: the wanted depth's cell, or parents while the texel is no-data (a merged cell carries elevation only inside the newer bake's footprint).
fn sample_dem(d: u32, u: u32, v: u32) -> Dem {
    var out: Dem;
    out.ok = false;
    var depth = U.depths.x;
    for (var k = 0u; k < 9u; k++) {
        let ri = find(d, depth, u, v, FLAG_DEM);
        if (ri == NONE) {
            return out;
        }
        let r = refs[ri];
        let rd = r.a.y;
        let t = tri(u, v, rd);
        let x = i32(2u * (t.x + 1u) + t.z);
        let y = i32(t.y + 1u);
        let slot = r.b.x;
        let q = dem_q(slot, x, y);
        if (q != ELEV_NODATA) {
            let step = r.d.y;
            let du = diffq(dem_q(slot, x - 2, y), dem_q(slot, x + 2, y), q) * step;
            let dv = diffq(dem_q(slot, x, y - 1), dem_q(slot, x, y + 1), q) * step;
            let eu = r.d.z;
            let nu = r.d.w;
            let ev = r.e.x;
            let nv = r.e.y;
            let inv = r.e.z;
            let ge = (du * nv - dv * nu) * inv;
            let gn = (eu * dv - ev * du) * inv;
            let s = inverseSqrt(1.0 + ge * ge + gn * gn);
            out.ok = true;
            out.elev = r.d.x + f32(q) * step;
            out.n = vec3<f32>(-ge * s, -gn * s, s);
            out.depth = rd;
            out.slot = slot;
            out.base = r.d.x;
            out.step = step;
            return out;
        }
        if (rd <= MIN_DEPTH) {
            return out;
        }
        depth = rd - 1u;
    }
    return out;
}

// The six centroid samples around a triangle vertex of the texel grid: the lower triangles of three squares and the upper triangles of three others, with no-data left out. Returns (sum, count).
fn dem_vertex(slot: u32, x: i32, y: i32) -> vec2<f32> {
    var sum = 0.0;
    var n = 0.0;
    let lows = array<vec2<i32>, 3>(vec2<i32>(x, y), vec2<i32>(x - 1, y), vec2<i32>(x, y - 1));
    let ups = array<vec2<i32>, 3>(vec2<i32>(x - 1, y - 1), vec2<i32>(x, y - 1), vec2<i32>(x - 1, y));
    for (var i = 0; i < 3; i++) {
        let l = lows[i];
        let ql = dem_q(slot, 2 * (l.x + 1), l.y + 1);
        if (ql != ELEV_NODATA) {
            sum += f32(ql);
            n += 1.0;
        }
        let p = ups[i];
        let qu = dem_q(slot, 2 * (p.x + 1) + 1, p.y + 1);
        if (qu != ELEV_NODATA) {
            sum += f32(qu);
            n += 1.0;
        }
    }
    return vec2<f32>(sum, n);
}

// Elevation interpolated across the texel: barycentric between the triangle's three vertices, each the mean of the six samples around it — a continuous surface, so contours drawn past the base depth are smooth instead of stepping along texel edges.
fn dem_smooth(d: Dem, u: u32, v: u32) -> f32 {
    let s = 22u - d.depth;
    let m = (1u << s) - 1u;
    let tx = i32((u >> s) & 255u);
    let ty = i32((v >> s) & 255u);
    let fu = f32(u & m) / f32(1u << s);
    let fv = f32(v & m) / f32(1u << s);
    var va: vec2<i32>;
    var vb: vec2<i32>;
    var vc: vec2<i32>;
    var la: f32;
    var lb: f32;
    var lc: f32;
    if (fu + fv < 1.0) {
        va = vec2<i32>(tx, ty);
        vb = vec2<i32>(tx + 1, ty);
        vc = vec2<i32>(tx, ty + 1);
        la = 1.0 - fu - fv;
        lb = fu;
        lc = fv;
    } else {
        va = vec2<i32>(tx + 1, ty + 1);
        vb = vec2<i32>(tx, ty + 1);
        vc = vec2<i32>(tx + 1, ty);
        la = fu + fv - 1.0;
        lb = 1.0 - fu;
        lc = 1.0 - fv;
    }
    let a = dem_vertex(d.slot, va.x, va.y);
    let b = dem_vertex(d.slot, vb.x, vb.y);
    let c = dem_vertex(d.slot, vc.x, vc.y);
    if (a.y == 0.0 || b.y == 0.0 || c.y == 0.0) {
        return d.elev;
    }
    return d.base + (la * a.x / a.y + lb * b.x / b.y + lc * c.x / c.y) * d.step;
}

fn light_eval(n: vec3<f32>) -> vec3<f32> {
    var m = array<f32, 10>(n.x * n.x, n.y * n.y, n.z * n.z, n.x * n.y, n.x * n.z, n.y * n.z, n.x, n.y, n.z, 1.0);
    var out = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c++) {
        var e = 0.0;
        for (var i = 0u; i < 10u; i++) {
            let k = c * 10u + i;
            e += U.light[k >> 2u][k & 3u] * m[i];
        }
        out[c] = e;
    }
    return out;
}

// Every authored colour is VSF RGB at gamma 2, quantised ×256: a byte is the light (b/256)². The style uniforms arrive already linear; the tables and constants are decoded here.
fn dec(c: vec3<f32>) -> vec3<f32> {
    let x = c / 256.0;
    return x * x;
}

fn unpack_rgb(p: u32) -> vec3<f32> {
    return dec(vec3<f32>(f32((p >> 16u) & 255u), f32((p >> 8u) & 255u), f32(p & 255u)));
}

fn img_light(b: u32) -> f32 {
    return U.img_table[b >> 2u][b & 3u];
}

// Opsin's highlight rail, (3x − x³)/2, clamped first because past 1 the cubic folds back.
fn rail(x: vec3<f32>) -> vec3<f32> {
    let c = clamp(x, vec3<f32>(0.0), vec3<f32>(1.0));
    return (3.0 * c - c * c * c) * 0.5;
}

// The one display encode: linear VSF RGB to the display's gamma-2 code values 0..255, truncated. Compressed: exposure 2/3 into the rail. Linear: exposure 1/2.8 straight, so the widest stored range (imagery's tone ceiling, 2.8 times paper white) reaches white unclipped. Then the square root.
fn to_display(x: vec3<f32>) -> vec3<f32> {
    let d = vec3<f32>(dot(U.display[0].xyz, x), dot(U.display[1].xyz, x), dot(U.display[2].xyz, x));
    let t = select(clamp(d * (1.0 / 2.8), vec3<f32>(0.0), vec3<f32>(1.0)), rail(d * (2.0 / 3.0)), U.tone.x > 0.5);
    return min(floor(sqrt(t) * 256.0), vec3<f32>(255.0));
}

// A line's magnitude against the largest of its class in view, 0..1.
fn line_scale(cls: u32, mag: u32) -> f32 {
    return clamp(f32(mag) / U.line_hi[cls >> 2u][cls & 3u], 0.0, 1.0);
}

fn line_colour(cls_id: u32, mag: u32) -> vec3<f32> {
    let cls = min(cls_id, CLASS_MAX);
    let c = unpack_rgb(lut[CLASS_BASE + cls]);
    if (cls == WATERWAY_CLASS) {
        // Water on a linear scale up to the largest magnitude in view: the biggest river on screen is full, a trickle a third.
        return c * (0.35 + 0.65 * line_scale(cls, mag));
    }
    return c;
}

// Coverage of a line as drawn: every class but water fades against the boldest of its class in view, so a lane next to a highway falls back and the same lane alone is full.
fn line_alpha(cls_id: u32, cov: u32, mag: u32) -> f32 {
    let cls = min(cls_id, CLASS_MAX);
    if (cls == WATERWAY_CLASS) {
        return f32(cov) / 255.0;
    }
    // Boundaries sit under the map, never competing with a road or a trail.
    if (cls >= BOUNDARY_FIRST) {
        return f32(cov) / 255.0 * 0.55;
    }
    return f32(cov) / 255.0 * (0.5 + 0.5 * line_scale(cls, mag));
}

fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

// What a sample is before the light touches it, and how what comes after folds: the final colour is `(base × light) × k + c`, every post-light layer (slope band, contour, lines, the debug tint) being an affine step that composes into one `k` and one `c`. Water sits between, with its own alpha, because its shade follows the sun. A G-buffer of these relights a frame without touching the page table.
struct Composed {
    base: vec3<f32>,
    ground: bool,
    n: vec3<f32>,
    k1: f32,
    c1: vec3<f32>,
    water: f32,
    k2: f32,
    c2: vec3<f32>,
};

// A lerp toward `col` by `a` after the light, folded into (k, c).
struct Fold { k: f32, c: vec3<f32> };

fn fold(f: Fold, col: vec3<f32>, a: f32) -> Fold {
    return Fold(f.k * (1.0 - a), f.c * (1.0 - a) + col * a);
}

fn compose(d: u32, u: u32, v: u32) -> Composed {
    let mask = U.mask;
    var out: Composed;
    out.base = U.style_no_dem.rgb;
    out.ground = false;
    out.n = vec3<f32>(0.0, 0.0, 1.0);
    out.k1 = 1.0;
    out.c1 = vec3<f32>(0.0);
    out.water = 0.0;
    out.k2 = 1.0;
    out.c2 = vec3<f32>(0.0);
    var contour_cov = 0.0;
    var contour_index = false;
    var band = vec3<f32>(0.0);
    var have_band = false;
    var found_depth = NONE;
    var is_sea = false;
    // The terrain sample feeds the tint and light (terrain on), and the contours and slope bands on their own.
    let want_dem = (mask & (M_DEM | M_CONTOURS | M_SLOPE)) != 0u;
    if (want_dem) {
        let s = sample_dem(d, u, v);
        if (s.ok) {
            found_depth = s.depth;
            let eq = u32(clamp((s.elev + 500.0) * 4.0, 0.0, 65534.0));
            if ((mask & M_DEM) != 0u) {
                out.n = s.n;
                // The open sea: exactly zero and dead flat, which is how a global DEM writes the ocean (below-sea-level land keeps its colour).
                if (abs(s.elev) < 0.75 && s.n.z > 0.9995) {
                    is_sea = true;
                    out.base = U.style_sea.rgb;
                } else if ((mask & M_HYPSO) != 0u) {
                    out.base = unpack_rgb(lut[eq >> 4u]);
                } else {
                    out.base = U.style_flat.rgb;
                }
                out.ground = true;
            }
            let nzn = max(s.n.z, 1e-4);
            let slope = sqrt(max(1.0 - nzn * nzn, 0.0)) / nzn;
            if ((mask & M_CONTOURS) != 0u && slope >= 0.02 && U.contour.x > 0.0) {
                let interval = U.contour.x;
                // Always the interpolated elevation, at every zoom: from the texels' own values a contour is a chain of triangle facets (Nick 2026-10-09).
                let elev_m = dem_smooth(s, u, v);
                let level = round(elev_m / interval);
                let d_m = abs(elev_m - level * interval);
                let d_px = d_m / (slope * U.contour.z);
                let li = i32(level);
                let every = i32(U.contour.y);
                contour_index = ((li % every) + every) % every == 0;
                let w = select(0.55, 0.9, contour_index);
                contour_cov = clamp(w + 0.5 - d_px, 0.0, 1.0);
            }
            if ((mask & M_SLOPE) != 0u) {
                let deg = degrees(atan(slope));
                if (deg >= 45.0) {
                    band = dec(vec3<f32>(150.0, 40.0, 200.0));
                    have_band = true;
                } else if (deg >= 35.0) {
                    band = dec(vec3<f32>(230.0, 50.0, 40.0));
                    have_band = true;
                } else if (deg >= 30.0) {
                    band = dec(vec3<f32>(250.0, 150.0, 40.0));
                    have_band = true;
                } else if (deg >= 25.0) {
                    band = dec(vec3<f32>(250.0, 220.0, 60.0));
                    have_band = true;
                }
            }
        } else if ((mask & M_DEM) != 0u) {
            out.base = U.style_bg.rgb;
        }
    }
    // Imagery has its own depth (a global 10 m layer sits several levels above the vector cells), so it is found on its own: the finest cell at or above the vector depth that carries any.
    // Imagery stands in for the lit ground where it has data and the terrain is not sea: the composite's own ocean pixels and the tiles' edges never show, the terrain's coastline does.
    var im = vec4<u32>(0u);
    if ((mask & (M_IMAGERY | M_INFRARED)) != 0u && !is_sea) {
        let ii = find(d, U.depths.y, u, v, FLAG_IMG);
        if (ii != NONE) {
            let ri = refs[ii];
            let ti = tri(u, v, ri.a.y);
            im = textureLoad(img_tex, vec2<i32>(i32(2u * ti.x + ti.z), i32(ti.y)), i32(ri.b.w), 0);
        }
    }
    let vi = find(d, U.depths.y, u, v, FLAG_VEC);
    if (vi == NONE && (mask & M_IMAGERY) != 0u && (im.x != 0u || im.y != 0u || im.z != 0u)) {
        out.ground = false;
        out.base = select(vec3<f32>(img_light(im.x), img_light(im.y), img_light(im.z)), vec3<f32>(img_light(im.w)), (mask & M_INFRARED) != 0u);
        return out;
    }
    if (vi != NONE) {
        let r = refs[vi];
        let flags = r.c.x;
        let t = tri(u, v, r.a.y);
        let xy = vec2<i32>(i32(2u * t.x + t.z), i32(t.y));
        var line = vec4<u32>(0u);
        var lw = vec4<u32>(0u);
        if ((flags & FLAG_LINE) != 0u) {
            line = textureLoad(line_tex, xy, i32(r.b.y), 0);
        }
        if ((flags & (FLAG_LAND | FLAG_WATER)) != 0u) {
            lw = textureLoad(lw_tex, xy, i32(r.b.z), 0);
        }
        let draw_line = (mask & M_LINE) != 0u && line.y != 0u && (line.x < BOUNDARY_FIRST || (mask & M_BOUND) != 0u);
        if ((mask & M_IMAGERY) != 0u && (im.x != 0u || im.y != 0u || im.z != 0u)) {
            // Imagery stands in for the ground: nothing lights it. True colour, or the near-infrared band as grey, unrolled to scene light.
            out.ground = false;
            if ((mask & M_INFRARED) != 0u) {
                out.base = vec3<f32>(img_light(im.w));
            } else {
                out.base = vec3<f32>(img_light(im.x), img_light(im.y), img_light(im.z));
            }
            // The water fill paints over imagery as it does over the ground, so a lake or a coast texel that is water in part matches the sea instead of showing the composite's own dark water: the composite's water taken out at the coverage, the theme's put in (the CPU's IMG_WATER).
            if ((mask & M_WATER) != 0u && lw.z != 0u) {
                let t = f32(lw.z) / 255.0;
                out.base = max(out.base - vec3<f32>(t * IMG_WATER), vec3<f32>(0.0)) + t * U.style_water.rgb;
            }
            if (draw_line) {
                { let f = fold(Fold(out.k2, out.c2), line_colour(line.x, line.z), line_alpha(line.x, line.y, line.z)); out.k2 = f.k; out.c2 = f.c; }
            }
            return out;
        }
        if ((mask & M_LAND) != 0u && lw.y != 0u) {
            out.base = lerp3(out.base, unpack_rgb(lut[LAND_BASE + min(lw.x, 13u)]), f32(lw.y) / 255.0 * 0.85);
        }
        if (have_band) {
            { let f = fold(Fold(out.k1, out.c1), band, 0.45); out.k1 = f.k; out.c1 = f.c; }
        }
        // Under the water: the contours and the waterway lines, so a lake covers the river running into it. Over it: every other line.
        if (contour_cov > 0.0) {
            let ink = select(U.style_contour, U.style_contour_index, contour_index);
            { let f = fold(Fold(out.k1, out.c1), ink.rgb, contour_cov * ink.a); out.k1 = f.k; out.c1 = f.c; }
        }
        if (draw_line && line.x == WATERWAY_CLASS) {
            { let f = fold(Fold(out.k1, out.c1), line_colour(line.x, line.z), line_alpha(line.x, line.y, line.z)); out.k1 = f.k; out.c1 = f.c; }
        }
        if ((mask & M_WATER) != 0u && lw.z != 0u) {
            out.water = f32(lw.z) / 255.0;
        }
        if (draw_line && line.x != WATERWAY_CLASS) {
            { let f = fold(Fold(out.k2, out.c2), line_colour(line.x, line.z), line_alpha(line.x, line.y, line.z)); out.k2 = f.k; out.c2 = f.c; }
        }
    } else {
        if (have_band) {
            { let f = fold(Fold(out.k1, out.c1), band, 0.45); out.k1 = f.k; out.c1 = f.c; }
        }
        if (contour_cov > 0.0) {
            let ink = select(U.style_contour, U.style_contour_index, contour_index);
            { let f = fold(Fold(out.k2, out.c2), ink.rgb, contour_cov * ink.a); out.k2 = f.k; out.c2 = f.c; }
        }
    }
    // Where the wanted terrain cell has not arrived at all (no reference and no absent marker), the texel is a random dark colour of its own, the CPU's loading_noise: a loading view shows the wanted triangles as a dark mesh.
    if (want_dem && found_depth != U.depths.x && lookup(d, U.depths.x, u, v) == NONE) {
        let s = 30u - U.depths.x - 8u;
        let m = (1u << s) - 1u;
        let half = (((u & m) + (v & m)) >> s) & 1u;
        var h = (d * 0x27D4EB2Fu) ^ (U.depths.x * 0xC2B2AE3Du) ^ ((u >> s) * 0x9E3779B1u) ^ ((v >> s) * 0x85EBCA77u) ^ (half * 0x165667B1u);
        h = h ^ (h >> 15u);
        h = h * 0x2C1B3C6Du;
        h = h ^ (h >> 12u);
        out.base = dec(vec3<f32>(f32((h & 255u) >> 2u), f32(((h >> 8u) & 255u) >> 2u), f32(((h >> 16u) & 255u) >> 2u)));
        out.ground = false;
        out.k1 = 1.0;
        out.c1 = vec3<f32>(0.0);
        out.water = 0.0;
        out.k2 = 1.0;
        out.c2 = vec3<f32>(0.0);
    }
    if ((mask & M_DEBUG) != 0u) {
        var tintd = vec3<f32>(-1.0);
        if (found_depth == NONE) {
            tintd = dec(vec3<f32>(255.0, 0.0, 255.0));
        } else if (found_depth + 2u <= U.depths.x) {
            tintd = dec(vec3<f32>(255.0, 40.0, 40.0));
        } else if (found_depth + 1u == U.depths.x) {
            tintd = dec(vec3<f32>(255.0, 170.0, 0.0));
        }
        if (tintd.x >= 0.0) {
            { let f = fold(Fold(out.k2, out.c2), tintd, 0.5); out.k2 = f.k; out.c2 = f.c; }
        }
    }
    return out;
}

// The light applied, all linear: the same arithmetic whether the pieces come from the page table or from the G-buffer.
fn shade(base: vec3<f32>, ground: bool, n: vec3<f32>, k1: f32, c1: vec3<f32>, water: f32, k2: f32, c2: vec3<f32>) -> vec3<f32> {
    var rgb = base;
    var diffuse = 1.0;
    if (ground) {
        diffuse = max(dot(n, U.sun.xyz), 0.0);
        // No cap: the display's highlight curve is the only place light meets white. The floor is physics (the sky's harmonics ring a little below zero facing away), not a clip.
        rgb = rgb * max(light_eval(n), vec3<f32>(0.0));
    }
    rgb = rgb * k1 + c1;
    if (water > 0.0) {
        let wr = U.style_water.rgb * (0.85 + 0.15 * diffuse);
        rgb = lerp3(rgb, wr, water);
    }
    return rgb * k2 + c2;
}

fn sample_uv(in: VOut) -> vec3<u32> {
    let b = blocks[in.block];
    let fx = in.local.x + U.offset.x;
    let fy = in.local.y + U.offset.y;
    let ou = b.c.x * fx + b.c.z * fy + b.d.x * fx * fy;
    let ov = b.c.y * fx + b.c.w * fy + b.d.y * fx * fy;
    return vec3<u32>(b.a.w, u32(i32(b.b.x) + i32(round(ou))), u32(i32(b.b.y) + i32(round(ov))));
}

// Off the globe, when its limb is on the screen.
fn off_globe(pos: vec2<f32>) -> bool {
    let d = pos - U.globe.xy;
    return U.globe.w > 0.5 && dot(d, d) > U.globe.z;
}

@fragment
fn fs_map(in: VOut) -> @location(0) vec4<f32> {
    if (off_globe(in.pos.xy)) {
        return vec4<f32>(to_display(U.style_bg.rgb) / 255.0, 1.0);
    }
    let s = sample_uv(in);
    let p = compose(s.x, s.y, s.z);
    return vec4<f32>(to_display(shade(p.base, p.ground, p.n, p.k1, p.c1, p.water, p.k2, p.c2)) / 255.0, 1.0);
}

// The map pass that writes the G-buffer instead of a colour: base + ground, normal + water, c1 + k1, c2 + k2. The relight pass then shades it.
struct GOut {
    @location(0) base: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) post1: vec4<f32>,
    @location(3) post2: vec4<f32>,
};

@fragment
fn fs_map_g(in: VOut) -> GOut {
    let s = sample_uv(in);
    var p = compose(s.x, s.y, s.z);
    if (off_globe(in.pos.xy)) {
        p.base = U.style_bg.rgb;
        p.ground = false;
        p.n = vec3<f32>(0.0, 0.0, 1.0);
        p.k1 = 1.0;
        p.c1 = vec3<f32>(0.0);
        p.water = 0.0;
        p.k2 = 1.0;
        p.c2 = vec3<f32>(0.0);
    }
    var g: GOut;
    // Linear light kept at gamma 2 so 8 bits hold the shadows; the base with 2.8 times headroom, the most unrolled imagery carries.
    g.base = vec4<f32>(sqrt(p.base / 2.8), select(0.0, 1.0, p.ground));
    g.normal = vec4<f32>(p.n * 0.5 + 0.5, p.water);
    g.post1 = vec4<f32>(sqrt(p.c1), p.k1);
    g.post2 = vec4<f32>(sqrt(p.c2), p.k2);
    return g;
}

// ==================== RELIGHT: the light alone, from the G-buffer ====================

@group(2) @binding(0) var g_base: texture_2d<f32>;
@group(2) @binding(1) var g_normal: texture_2d<f32>;
@group(2) @binding(2) var g_post1: texture_2d<f32>;
@group(2) @binding(3) var g_post2: texture_2d<f32>;

@fragment
fn fs_relight(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(pos.xy);
    let b = textureLoad(g_base, xy, 0);
    let nw = textureLoad(g_normal, xy, 0);
    let p1 = textureLoad(g_post1, xy, 0);
    let p2 = textureLoad(g_post2, xy, 0);
    let rgb = shade(b.rgb * b.rgb * 2.8, b.a > 0.5, normalize(nw.xyz * 2.0 - 1.0), p1.a, p1.rgb * p1.rgb, nw.w, p2.a, p2.rgb * p2.rgb);
    return vec4<f32>(to_display(rgb) / 255.0, 1.0);
}

// ==================== PRESENT: 2×2 bin plus the overlay ====================

@group(1) @binding(0) var map_tex: texture_2d<f32>;
@group(1) @binding(1) var overlay_tex: texture_2d<f32>;

@vertex
fn vs_present(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_present(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let s = i32(U.scale);
    let p = vec2<i32>(pos.xy) * s;
    // The bin averages the light the display will emit: each sample's code value (b/256)², then back to gamma 2. The round only recovers the stored integer from the texture's 0..1.
    var c = vec3<f32>(0.0);
    for (var j = 0; j < s; j++) {
        for (var i = 0; i < s; i++) {
            let e = round(textureLoad(map_tex, p + vec2<i32>(i, j), 0).rgb * 255.0) / 256.0;
            c += e * e;
        }
    }
    c = sqrt(c / f32(s * s));
    // The overlay is premultiplied display values: straight over.
    let o = textureLoad(overlay_tex, vec2<i32>(pos.xy), 0);
    var rgb = c * (1.0 - o.a) + o.rgb * (255.0 / 256.0);
    // The GPS pin: an accuracy ring and a crosshair, feathered over a pixel, in the engine's pin blue.
    if (U.pin.w > 0.5) {
        let d = pos.xy - U.pin.xy;
        let dist = length(d);
        let ring = clamp(1.0 - abs(dist - U.pin.z) + 0.45, 0.0, 1.0);
        let cross = select(0.0, clamp(1.6 - min(abs(d.x), abs(d.y)) + 0.5, 0.0, 1.0), max(abs(d.x), abs(d.y)) <= 9.0);
        let cov = max(ring, cross);
        rgb = mix(rgb, vec3<f32>(64.0, 156.0, 255.0) / 256.0, cov);
    }
    // The measurement: a line from origin to target and a crosshair on the target, feathered over a pixel.
    if (U.depths.w != 0u) {
        let o = U.measure.xy;
        let t = U.measure.zw;
        let ab = t - o;
        let len2 = max(dot(ab, ab), 1.0);
        let tt = clamp(dot(pos.xy - o, ab) / len2, 0.0, 1.0);
        let dline = length(pos.xy - (o + ab * tt));
        let line_cov = clamp(1.3 - dline + 0.5, 0.0, 1.0);
        let dt = pos.xy - t;
        let cross = select(0.0, clamp(1.6 - min(abs(dt.x), abs(dt.y)) + 0.5, 0.0, 1.0), max(abs(dt.x), abs(dt.y)) <= 12.0 && min(abs(dt.x), abs(dt.y)) <= 1.6 && length(dt) > 3.0);
        let cov = max(line_cov * 0.85, cross);
        rgb = mix(rgb, vec3<f32>(255.0, 196.0, 64.0) / 256.0, cov);
    }
    // Truncated to the code value, written exactly so the hardware's own rounding has nothing to do.
    return vec4<f32>(min(floor(rgb * 256.0), vec3<f32>(255.0)) / 255.0, 1.0);
}
