// The #pagetable compositor on the GPU. One instance per screen block: the vertex stage places the block's quad and hands the fragment its position inside the block; the fragment walks position → diamond UV → page table → texel → the same compose the CPU loop does, per supersample. A second pass bins the supersampled image down 2×2 and lays the screen-space marks over it.

const NONE: u32 = 0xFFFFFFFFu;
const MIN_DEPTH: u32 = 6u;
const TABLE_N: u32 = 4096u;
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
const M_CANOPY: u32 = 256u;

// Where the style tables sit in the LUT buffer after the 4096 hypsometric rows.
const CLASS_BASE: u32 = 4096u;
const LAND_BASE: u32 = 4112u;
const CLASS_MAX: u32 = 12u;
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
fn find(d: u32, want: u32, u: u32, v: u32, need: u32) -> u32 {
    var depth = want;
    loop {
        let r = lookup(d, depth, u, v);
        if (r != NONE && (refs[r].c.x & need) != 0u) {
            return r;
        }
        if (depth == MIN_DEPTH) {
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

struct Dem { ok: bool, elev: f32, n: vec3<f32>, depth: u32 };

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
            return out;
        }
        if (rd == MIN_DEPTH) {
            return out;
        }
        depth = rd - 1u;
    }
    return out;
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

fn unpack_rgb(p: u32) -> vec3<f32> {
    return vec3<f32>(f32((p >> 16u) & 255u), f32((p >> 8u) & 255u), f32(p & 255u));
}

fn line_colour(cls_id: u32, mag: u32) -> vec3<f32> {
    let cls = min(cls_id, CLASS_MAX);
    let c = unpack_rgb(lut[CLASS_BASE + cls]);
    if (cls == WATERWAY_CLASS) {
        return floor(c * (0.4 + 0.6 * f32(mag) / 255.0));
    }
    return c;
}

fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

fn compose(d: u32, u: u32, v: u32) -> vec3<f32> {
    let mask = U.mask;
    var tint = vec3<f32>(96.0, 100.0, 96.0);
    var diffuse = 1.0;
    var light = vec3<f32>(1.0);
    var have_ground = false;
    var contour_cov = 0.0;
    var contour_index = false;
    var band = vec3<f32>(0.0);
    var have_band = false;
    var found_depth = NONE;
    // The terrain sample feeds the tint and light (terrain on), and the contours and slope bands on their own.
    let want_dem = (mask & (M_DEM | M_CONTOURS | M_SLOPE)) != 0u;
    if (want_dem) {
        let s = sample_dem(d, u, v);
        if (s.ok) {
            found_depth = s.depth;
            let eq = u32(clamp((s.elev + 500.0) * 4.0, 0.0, 65534.0));
            if ((mask & M_DEM) != 0u) {
                diffuse = max(dot(s.n, U.sun.xyz), 0.0);
                light = clamp(light_eval(s.n), vec3<f32>(0.0), vec3<f32>(1.3));
                tint = unpack_rgb(lut[eq >> 4u]);
                have_ground = true;
            }
            let nzn = max(s.n.z, 1e-4);
            let slope = sqrt(max(1.0 - nzn * nzn, 0.0)) / nzn;
            if ((mask & M_CONTOURS) != 0u && slope >= 0.02 && U.contour.x > 0.0) {
                let interval = U.contour.x;
                let elev_m = f32(eq) * 0.25 - 500.0;
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
                    band = vec3<f32>(150.0, 40.0, 200.0);
                    have_band = true;
                } else if (deg >= 35.0) {
                    band = vec3<f32>(230.0, 50.0, 40.0);
                    have_band = true;
                } else if (deg >= 30.0) {
                    band = vec3<f32>(250.0, 150.0, 40.0);
                    have_band = true;
                } else if (deg >= 25.0) {
                    band = vec3<f32>(250.0, 220.0, 60.0);
                    have_band = true;
                }
            }
        } else if ((mask & M_DEM) != 0u) {
            tint = vec3<f32>(18.0, 20.0, 26.0);
        }
    }
    var rgb = tint;
    let vi = find(d, U.depths.y, u, v, FLAG_VEC);
    if (vi != NONE) {
        let r = refs[vi];
        let flags = r.c.x;
        let t = tri(u, v, r.a.y);
        let xy = vec2<i32>(i32(2u * t.x + t.z), i32(t.y));
        var line = vec4<u32>(0u);
        var lw = vec4<u32>(0u);
        var im = vec4<u32>(0u);
        if ((flags & FLAG_LINE) != 0u) {
            line = textureLoad(line_tex, xy, i32(r.b.y), 0);
        }
        if ((flags & (FLAG_LAND | FLAG_WATER)) != 0u) {
            lw = textureLoad(lw_tex, xy, i32(r.b.z), 0);
        }
        if ((flags & FLAG_IMG) != 0u) {
            im = textureLoad(img_tex, xy, i32(r.b.w), 0);
        }
        if ((mask & M_IMAGERY) != 0u) {
            if (im.x != 0u || im.y != 0u || im.z != 0u) {
                rgb = vec3<f32>(f32(im.z), f32(im.y), f32(im.x));
            }
            if ((mask & M_LINE) != 0u && line.y != 0u) {
                rgb = lerp3(rgb, line_colour(line.x, line.z), f32(line.y) / 255.0);
            }
            return floor(rgb);
        }
        if ((mask & M_LAND) != 0u && lw.y != 0u) {
            rgb = lerp3(rgb, unpack_rgb(lut[LAND_BASE + min(lw.x, 13u)]), f32(lw.y) / 255.0 * 0.85);
        }
        if ((mask & M_CANOPY) != 0u && im.w != 0u) {
            let c = min(f32(im.w) / 60.0, 1.0);
            let green = floor(vec3<f32>((1.0 - c) * 190.0 + c * 20.0, (1.0 - c) * 230.0 + c * 110.0, (1.0 - c) * 150.0 + c * 40.0));
            rgb = lerp3(rgb, green, 0.8);
        }
        if (have_ground) {
            rgb = rgb * light;
        }
        if (have_band) {
            rgb = lerp3(rgb, band, 0.45);
        }
        if ((mask & M_WATER) != 0u && lw.z != 0u) {
            let s = 0.85 + 0.15 * diffuse;
            let wr = vec3<f32>(26.0, 58.0, 82.0) * s;
            rgb = lerp3(rgb, wr, f32(lw.z) / 255.0);
        }
        if (contour_cov > 0.0) {
            let ink = select(vec3<f32>(92.0, 62.0, 34.0), vec3<f32>(64.0, 40.0, 18.0), contour_index);
            rgb = lerp3(rgb, ink, contour_cov * 0.85);
        }
        if ((mask & M_LINE) != 0u && line.y != 0u) {
            rgb = lerp3(rgb, line_colour(line.x, line.z), f32(line.y) / 255.0);
        }
    } else {
        if (have_ground) {
            rgb = rgb * light;
        }
        if (have_band) {
            rgb = lerp3(rgb, band, 0.45);
        }
        if (contour_cov > 0.0) {
            let ink = select(vec3<f32>(92.0, 62.0, 34.0), vec3<f32>(64.0, 40.0, 18.0), contour_index);
            rgb = lerp3(rgb, ink, contour_cov * 0.85);
        }
    }
    rgb = floor(rgb);
    if ((mask & M_DEBUG) != 0u) {
        var tintd = vec3<f32>(-1.0);
        if (found_depth == NONE) {
            tintd = vec3<f32>(255.0, 0.0, 255.0);
        } else if (found_depth + 2u <= U.depths.x) {
            tintd = vec3<f32>(255.0, 40.0, 40.0);
        } else if (found_depth + 1u == U.depths.x) {
            tintd = vec3<f32>(255.0, 170.0, 0.0);
        }
        if (tintd.x >= 0.0) {
            rgb = floor((rgb + tintd) * 0.5);
        }
    }
    return rgb;
}

@fragment
fn fs_map(in: VOut) -> @location(0) vec4<f32> {
    let b = blocks[in.block];
    let fx = in.local.x + U.offset.x;
    let fy = in.local.y + U.offset.y;
    let ou = b.c.x * fx + b.c.z * fy + b.d.x * fx * fy;
    let ov = b.c.y * fx + b.c.w * fy + b.d.y * fx * fy;
    let u = u32(i32(b.b.x) + i32(round(ou)));
    let v = u32(i32(b.b.y) + i32(round(ov)));
    let rgb = compose(b.a.w, u, v);
    return vec4<f32>(rgb / 255.0, 1.0);
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
    var c = vec4<f32>(0.0);
    for (var j = 0; j < s; j++) {
        for (var i = 0; i < s; i++) {
            c += textureLoad(map_tex, p + vec2<i32>(i, j), 0);
        }
    }
    c = c / f32(s * s);
    // The overlay is premultiplied: straight over.
    let o = textureLoad(overlay_tex, vec2<i32>(pos.xy), 0);
    var rgb = c.rgb * (1.0 - o.a) + o.rgb;
    // The GPS pin: an accuracy ring and a crosshair, feathered over a pixel, in the engine's pin blue.
    if (U.pin.w > 0.5) {
        let d = pos.xy - U.pin.xy;
        let dist = length(d);
        let ring = clamp(1.0 - abs(dist - U.pin.z) + 0.45, 0.0, 1.0);
        let cross = select(0.0, clamp(1.6 - min(abs(d.x), abs(d.y)) + 0.5, 0.0, 1.0), max(abs(d.x), abs(d.y)) <= 9.0);
        let cov = max(ring, cross);
        rgb = mix(rgb, vec3<f32>(64.0, 156.0, 255.0) / 255.0, cov);
    }
    return vec4<f32>(rgb, 1.0);
}
