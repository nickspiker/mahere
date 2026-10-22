// The #pagetable compositor on the GPU. One instance per screen block: the vertex stage places the block's quad and hands the fragment its position inside the block; the fragment walks position → diamond UV → page table → texel → the same compose the CPU loop does, per supersample. A second pass bins the supersampled image down 2×2 and lays the screen-space marks over it.

const NONE: u32 = 0xFFFFFFFFu;
// A cell the loader confirmed absent at the wanted depth: found by lookup, never referred to.
const ABSENT: u32 = 0xFFFFFFFEu;
// The coarsest depth a cell exists at (the global bake's root), the same as the planner's. A walk below it must stop: a fragment loop that never ends hangs the GPU and the driver resets it for every process on the device (2026-10-09, the phone's SystemUI went down with mahere).
const MIN_DEPTH: u32 = 0u;
const IMG_WATER: f32 = 0.03;
const INK_TAG_DEPTH: u32 = 64u;
// The selection's wash (the CPU's SEL_TINT and SEL_ALPHA).
const SEL_TINT: vec3<f32> = vec3<f32>(0.35, 0.6, 1.0);
const SEL_ALPHA: f32 = 0.35;
const SEL_EDGE_ALPHA: f32 = 0.85;

// Whether a selection texel is on the shape's edge: its coverage on one side of half, a neighbouring triangle's on the other (the CPU's sel_edge). The three neighbours of the lower triangle are the upper of its square and the uppers below and to the left; of the upper, the lower of its square and the lowers above and to the right. The cell's border is not an edge.
fn sel_edge(layer: i32, t: vec3<u32>, own: u32) -> bool {
    let inside = own >= 128u;
    let x = i32(t.x);
    let y = i32(t.y);
    var n0: vec2<i32>;
    var n1: vec2<i32>;
    var n2: vec2<i32>;
    if (t.z == 0u) {
        n0 = vec2<i32>(2 * x + 1, y);
        n1 = vec2<i32>(2 * x + 1, y - 1);
        n2 = vec2<i32>(2 * (x - 1) + 1, y);
    } else {
        n0 = vec2<i32>(2 * x, y);
        n1 = vec2<i32>(2 * x, y + 1);
        n2 = vec2<i32>(2 * (x + 1), y);
    }
    let lim = 256;
    if (n1.y >= 0 && n1.y < lim && (textureLoad(sel_tex, n1, layer, 0).x >= 128u) != inside) {
        return true;
    }
    if (n2.x >= 0 && n2.x < 2 * lim && (textureLoad(sel_tex, n2, layer, 0).x >= 128u) != inside) {
        return true;
    }
    return (textureLoad(sel_tex, n0, layer, 0).x >= 128u) != inside;
}
// The seabed, as the CPU's: the sea colour darkening to this much less light at this depth, the water a glaze over it.
const SEABED_FULL_DEPTH: f32 = 6000.0;
const SEABED_DARK: f32 = 0.64;
const SEABED_GLAZE: f32 = 0.35;
const TABLE_N: u32 = 16384u;
const ELEV_NODATA: u32 = 0xFFFFu;

const FLAG_DEM: u32 = 1u;
const FLAG_LINE: u32 = 2u;
const FLAG_LAND: u32 = 4u;
const FLAG_WATER: u32 = 8u;
const FLAG_IMG: u32 = 16u;
const FLAG_OWN: u32 = 64u;
const FLAG_MVUM: u32 = 128u;
// Line, land, water, and the overlays: whatever a vector cell may carry.
const FLAG_VEC: u32 = 206u;
// The reference's composite is baked and current: c.z and c.w are its slots.
const FLAG_COMP: u32 = 256u;

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
const M_OWN: u32 = 2048u;
const M_MVUM: u32 = 4096u;
const M_RAW: u32 = 8192u;
const M_GEOID: u32 = 32768u;
// The layers a pipeline is built for (2026-10-21: one shader for every layer kept a Pixel 3a's GPU at a third of a second a frame): the frame's mask is taken through it, so a layer outside it is false at compile time and its code leaves the pipeline, the registers with it. The host builds a pipeline per set of layers in use; left at its default, every layer is live.
override LAYERS: u32 = 4294967295u;
// A depth-0 texel's edge on the ground, metres (a 7054 km diamond edge over 256 texels): the geoid's gradient per texel into metres per metre.
const GEOID_TEXEL_M: f32 = 27555.0;

// Where the style tables sit in the LUT buffer after the 4096 hypsometric rows.
const CLASS_BASE: u32 = 4096u;
const LAND_BASE: u32 = 4128u;
const CLASS_MAX: u32 = 21u;
const BOUNDARY_FIRST: u32 = 13u;
const BOUNDARY_LAST: u32 = 17u;
const WATERWAY_CLASS: u32 = 12u;
// The overlays' tables, the CPU's OWN_FILL, and the wilderness flag on an ownership class.
const OWN_BASE: u32 = 4160u;
const WILD_AT: u32 = 4176u;
const MVUM_BASE: u32 = 4192u;
const OWN_FILL: f32 = 0.2;
const OWN_WILD: u32 = 16u;

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
    // The world's pole axis in view coordinates.
    pole: vec4<f32>,
    // The raw elevation view's range: lowest and highest metres in view.
    raw: vec4<f32>,
};

// a: diamond, depth, cu, cv. b: dem slot, line slot, land+water slot, img slot. c: flags. d: base, step, eu, nu. e: ev, nv, inv det.
struct Ref { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<f32>, e: vec4<f32> };
// a: x, y, size, diamond. b: u0, v0. c: du/dx, dv/dx, du/dy, dv/dy. d: twist u, twist v.
// e, f, g: the block's candidate references for the terrain, the vector planes and the imagery, NONE after the last.
// h: the block's user cells (ink, selection), finest first, NONE after the last.
struct Block { a: vec4<u32>, b: vec4<u32>, c: vec4<f32>, d: vec4<f32>, e: vec4<u32>, f: vec4<u32>, g: vec4<u32>, h: vec4<u32> };
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
@group(0) @binding(9) var ink_tex: texture_2d_array<u32>;
@group(0) @binding(10) var sel_tex: texture_2d_array<u32>;
@group(0) @binding(11) var ov_tex: texture_2d_array<u32>;
// The elevations this map pass draws, lowest and highest as quantised elevations: read back for the next frame's contour interval and raw elevation range (Nick 2026-10-10: the final gather, not a sampling).
// Then each line class's largest magnitude drawn, at 2 + class: the next frame's line weights, the final gather (Nick 2026-10-21) rather than the cells' own largest, which counted roads off the screen.
@group(0) @binding(12) var<storage, read_write> view_range: array<atomic<u32>, 34>;
// The geoid's height above the ellipsoid per depth-0 texel, centimetres offset by 32768, a layer per diamond; 0 where that diamond's cell has not arrived.
@group(0) @binding(13) var geoid_tex: texture_2d_array<u32>;
// A vector cell's static layers composited once (Nick 2026-10-21, a Pixel 3a drawing every layer every pixel every frame): the base before the light and the fold after it, slots in its reference's c.z and c.w ([`fs_bake`]).
@group(0) @binding(14) var comp0_tex: texture_2d_array<f32>;
@group(0) @binding(15) var comp1_tex: texture_2d_array<f32>;

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
    // A user cell's tag carries its depth plus INK_TAG_DEPTH; the grid is at the depth itself.
    let s = 30u - (depth % INK_TAG_DEPTH);
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

// The reference a pixel draws from: the block's candidate whose cell holds the UV, resolved by the plan once a block; a pixel none of them holds (a block touching more cells) searches the table as before (2026-10-21: a Pixel 3a's GPU spent a second a frame probing the table, several times a pixel).
fn pick(c: vec4<u32>, d: u32, u: u32, v: u32, want: u32, need: u32) -> u32 {
    for (var k = 0u; k < 4u; k++) {
        let r = c[k];
        if (r == NONE) {
            break;
        }
        let a = refs[r].a;
        let s = 30u - a.y;
        // Only a cell whose planes are on the GPU now: one still waiting for its upload has the flag cleared, and the search goes past it to a coarser cell, as before (2026-10-21: taken anyway, it drew as bare terrain, brown cells across the Pixel 3a's view).
        if (a.x == d && (u >> s) == a.z && (v >> s) == a.w && (refs[r].c.x & need) != 0u) {
            return r;
        }
    }
    return find(d, want, u, v, need);
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
fn sample_dem(d: u32, u: u32, v: u32, cand: vec4<u32>) -> Dem {
    var out: Dem;
    out.ok = false;
    var depth = U.depths.x;
    for (var k = 0u; k < 9u; k++) {
        // The block's candidate first; past a no-data texel the search goes on from the next depth up.
        var ri: u32;
        if (k == 0u) {
            ri = pick(cand, d, u, v, depth, FLAG_DEM);
        } else {
            ri = find(d, depth, u, v, FLAG_DEM);
        }
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

// The geoid at a depth-0 texel (x = 2 × column + half, y = row), metres; NaN-free: `ok` is false outside the cell or where its diamond has not arrived.
struct Geo { ok: bool, m: f32 };

fn geoid_texel(d: u32, x: i32, y: i32) -> Geo {
    if (x < 0 || y < 0 || x > 511 || y > 255) {
        return Geo(false, 0.0);
    }
    let q = textureLoad(geoid_tex, vec2<i32>(x, y), i32(d), 0).x;
    return Geo(q != 0u, (f32(q) - 32768.0) / 100.0);
}

// The six centroid samples about a vertex of the depth-0 texel grid (the terrain's dem_vertex, with no apron): (sum, count).
fn geoid_vertex(d: u32, x: i32, y: i32) -> vec2<f32> {
    var sum = 0.0;
    var n = 0.0;
    let lows = array<vec2<i32>, 3>(vec2<i32>(x, y), vec2<i32>(x - 1, y), vec2<i32>(x, y - 1));
    let ups = array<vec2<i32>, 3>(vec2<i32>(x - 1, y - 1), vec2<i32>(x, y - 1), vec2<i32>(x - 1, y));
    for (var i = 0; i < 3; i++) {
        let l = geoid_texel(d, 2 * lows[i].x, lows[i].y);
        if (l.ok) {
            sum += l.m;
            n += 1.0;
        }
        let p = geoid_texel(d, 2 * ups[i].x + 1, ups[i].y);
        if (p.ok) {
            sum += p.m;
            n += 1.0;
        }
    }
    return vec2<f32>(sum, n);
}

// The geoid interpolated across the depth-0 texel as the terrain is (barycentric between vertex means), and its gradient, metres per texel: (metres, gradient, 1 when found).
fn geoid_at(d: u32, u: u32, v: u32) -> vec3<f32> {
    let s = 22u;
    let m = (1u << s) - 1u;
    let tx = i32((u >> s) & 255u);
    let ty = i32((v >> s) & 255u);
    let fu = f32(u & m) / f32(1u << s);
    let fv = f32(v & m) / f32(1u << s);
    let lower = fu + fv < 1.0;
    let va = select(vec2<i32>(tx + 1, ty + 1), vec2<i32>(tx, ty), lower);
    let vb = select(vec2<i32>(tx, ty + 1), vec2<i32>(tx + 1, ty), lower);
    let vc = select(vec2<i32>(tx + 1, ty), vec2<i32>(tx, ty + 1), lower);
    let a = geoid_vertex(d, va.x, va.y);
    let b = geoid_vertex(d, vb.x, vb.y);
    let c = geoid_vertex(d, vc.x, vc.y);
    if (a.y == 0.0 || b.y == 0.0 || c.y == 0.0) {
        let own = geoid_texel(d, 2 * tx + select(1, 0, lower), ty);
        return vec3<f32>(own.m, 0.0, select(0.0, 1.0, own.ok));
    }
    let ga = a.x / a.y;
    let gb = b.x / b.y;
    let gc = c.x / c.y;
    // dem_smooth's weights: (1 − u − v, u, v) on the lower triangle, (u + v − 1, 1 − u, 1 − v) on the upper; the gradient along each texel axis from the same plane.
    let la = select(fu + fv - 1.0, 1.0 - fu - fv, lower);
    let lb = select(1.0 - fu, fu, lower);
    let lc = select(1.0 - fv, fv, lower);
    let g = la * ga + lb * gb + lc * gc;
    let du = select(ga - gb, gb - ga, lower);
    let dv = select(ga - gc, gc - ga, lower);
    return vec3<f32>(g, sqrt(du * du + dv * dv), 1.0);
}

// The geoid's contour interval at a zoom, metres: a 1-2-5 step near a metre per 150 m a pixel, from a tenth of a metre to ten.
fn geoid_interval(m_per_px: f32) -> f32 {
    let x = clamp(m_per_px / 150.0, 0.1, 10.0);
    let p = pow(10.0, floor(log2(x) / log2(10.0)));
    let m = x / p;
    return clamp(p * select(select(select(10.0, 5.0, m < 7.0), 2.0, m < 3.0), 1.0, m < 1.5), 0.1, 10.0);
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

// A styled table entry's opacity, its top byte as a fraction.
fn unpack_a(p: u32) -> f32 {
    return f32(p >> 24u) / 255.0;
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

// The coarsest view depth a line class draws at (the CPU's line_min_depth): interstates at every zoom, the great rivers too, streets from depth 9, so a continent zoomed out is not painted as lines.
fn line_min_depth(cls: u32, mag: u32) -> u32 {
    switch cls {
        case 1u: { return 0u; }
        case 2u: { return 2u; }
        case 3u: { return 4u; }
        case 4u: { return 6u; }
        case 5u: { return 7u; }
        case 6u, 8u, 9u: { return 9u; }
        case 7u: { return 10u; }
        case 10u: { return 4u; }
        case 11u: { return 8u; }
        case 12u: {
            if (mag >= 200u) { return 0u; }
            if (mag >= 128u) { return 3u; }
            if (mag >= 64u) { return 6u; }
            return 8u;
        }
        case 13u, 14u, 15u, 16u: { return 5u; }
        case 17u: { return 3u; }
        case 18u: { return 6u; }
        case 19u, 20u: { return 9u; }
        case 21u: { return 10u; }
        default: { return 0u; }
    }
}

// Coverage of a line as drawn: every class but water fades against the boldest of its class in view, so a lane next to a highway falls back and the same lane alone is full.
fn line_alpha(cls_id: u32, cov: u32, mag: u32) -> f32 {
    let cls = min(cls_id, CLASS_MAX);
    // The theme's opacity for the class, then the magnitude's.
    let a = unpack_a(lut[CLASS_BASE + cls]);
    if (cls == WATERWAY_CLASS) {
        return f32(cov) / 255.0 * a;
    }
    // Boundaries sit under the map, never competing with a road or a trail.
    if (cls >= BOUNDARY_FIRST && cls <= BOUNDARY_LAST) {
        return f32(cov) / 255.0 * 0.55 * a;
    }
    return f32(cov) / 255.0 * (0.5 + 0.5 * line_scale(cls, mag)) * a;
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
    // The sample took the ground's path (not the imagery's, not a bare cell): the slope band applies.
    banded: bool,
};

// A lerp toward `col` by `a` after the light, folded into (k, c).
struct Fold { k: f32, c: vec3<f32> };

fn fold(f: Fold, col: vec3<f32>, a: f32) -> Fold {
    return Fold(f.k * (1.0 - a), f.c * (1.0 - a) + col * a);
}

// A local east-north-up normal turned into the view frame at a fragment: the fragment's own direction on the ball is its up, the pole crossed with it its east (the CPU's view_normal). The identity at the disk's centre; across a full disk every normal faces the sun from where it stands.
fn view_normal(n: vec3<f32>, pos: vec2<f32>) -> vec3<f32> {
    let r = sqrt(U.globe.z);
    let x = (pos.x - U.globe.x) / r;
    let y = (U.globe.y - pos.y) / r;
    let up = vec3<f32>(x, y, sqrt(max(1.0 - x * x - y * y, 0.0)));
    var east = cross(U.pole.xyz, up);
    let len = length(east);
    if (len < 1e-6) {
        east = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        east = east / len;
    }
    let north = cross(up, east);
    return n.x * east + n.y * north + n.z * up;
}

// An ownership class's manager, without the wilderness flag.
fn own_mgr(c: u32) -> u32 {
    return select(c, c - OWN_WILD, c >= OWN_WILD);
}

// The ownership class of a neighbouring triangle (the selection's neighbours: the lower triangle's are the upper of its square and the uppers below and to the left; the upper's the lower of its square and the lowers above and to the right), or the texel's own where the neighbour is past the cell's edge, which is then no boundary.
fn own_at(layer: i32, t: vec3<u32>, k: u32, own: u32) -> u32 {
    let x = i32(t.x);
    let y = i32(t.y);
    var n: vec2<i32>;
    if (t.z == 0u) {
        n = select(select(vec2<i32>(2 * (x - 1) + 1, y), vec2<i32>(2 * x + 1, y - 1), k == 1u), vec2<i32>(2 * x + 1, y), k == 0u);
    } else {
        n = select(select(vec2<i32>(2 * (x + 1), y), vec2<i32>(2 * x, y + 1), k == 1u), vec2<i32>(2 * x, y), k == 0u);
    }
    if (n.x < 0 || n.y < 0 || n.x >= 512 || n.y >= 256) {
        return own;
    }
    return textureLoad(ov_tex, n, layer, 0).x;
}

// The overlays over the ground and the water, under the map's own lines (the CPU's overlays): land ownership as a wash with its boundary and the wilderness over it, then motor vehicle use as a band the road or trail sits on.
fn overlays(composed: Composed, ov: vec4<u32>, layer: i32, t: vec3<u32>) -> Composed {
    var out = composed;
    if (layer < 0) {
        return out;
    }
    let mask = U.mask & LAYERS;
    if ((mask & M_OWN) != 0u && ov.x != 0u && ov.y != 0u) {
        let m = own_mgr(ov.x);
        let col = lut[OWN_BASE + min(m, 12u)];
        let wild = lut[WILD_AT];
        let cov = f32(ov.y) / 255.0;
        { let f = fold(Fold(out.k2, out.c2), unpack_rgb(col), unpack_a(col) * OWN_FILL * cov); out.k2 = f.k; out.c2 = f.c; }
        if (ov.x >= OWN_WILD) {
            { let f = fold(Fold(out.k2, out.c2), unpack_rgb(wild), unpack_a(wild) * OWN_FILL * cov); out.k2 = f.k; out.c2 = f.c; }
        }
        var edge = false;
        var wild_edge = false;
        for (var k = 0u; k < 3u; k++) {
            let n = own_at(layer, t, k, ov.x);
            edge = edge || own_mgr(n) != m;
            wild_edge = wild_edge || (ov.x >= OWN_WILD && n < OWN_WILD);
        }
        if (edge) {
            { let f = fold(Fold(out.k2, out.c2), unpack_rgb(col), unpack_a(col)); out.k2 = f.k; out.c2 = f.c; }
        }
        if (wild_edge) {
            { let f = fold(Fold(out.k2, out.c2), unpack_rgb(wild), unpack_a(wild)); out.k2 = f.k; out.c2 = f.c; }
        }
    }
    if ((mask & M_MVUM) != 0u && ov.z != 0u && ov.w != 0u) {
        let col = lut[MVUM_BASE + min(ov.z, 14u)];
        { let f = fold(Fold(out.k2, out.c2), unpack_rgb(col), f32(ov.w) / 255.0 * unpack_a(col)); out.k2 = f.k; out.c2 = f.c; }
    }
    return out;
}

// The user's marks over everything, from the finest user cell here (every cell carries what coarser ones have): the pen's ink in its colour, the selection as a wash with its edge. Applied at every exit of the composite, over imagery too, as the CPU's user_marks is.
fn user_marks(composed: Composed, d: u32, u: u32, v: u32, cand: vec4<u32>) -> Composed {
    var out = composed;
    // A block none of whose corners lies in a user cell has nothing of the user's: nothing to look for (2026-10-21: every pixel walked a dozen depths probing for ink).
    if (cand.x == NONE) {
        return out;
    }
    // The block's user cell holding this pixel, finest first, as the plan resolved them; a pixel none holds walks the depths as before.
    var picked = NONE;
    for (var k = 0u; k < 4u; k++) {
        let r = cand[k];
        if (r == NONE) {
            break;
        }
        let a = refs[r].a;
        let s = 30u - a.y;
        if (a.x == d && (u >> s) == a.z && (v >> s) == a.w) {
            picked = r;
            break;
        }
    }
    // From the finest user cell here (every cell carries what coarser ones have): the pen's ink in its colour, the selection as a wash.
    var udepth = U.depths.x;
    for (var k = 0u; k <= 30u; k++) {
        var ui = picked;
        if (picked == NONE) {
            ui = lookup(d, udepth + INK_TAG_DEPTH, u, v);
        }
        if (ui != NONE && ui != ABSENT) {
            let r = refs[ui];
            let t = tri(u, v, r.a.y);
            let xy = vec2<i32>(i32(2u * t.x + t.z), i32(t.y));
            let px = textureLoad(ink_tex, xy, i32(r.b.x), 0);
            if (px.w != 0u) {
                // Premultiplied over, after the light: the ground kept by what the ink leaves, plus the ink's own light.
                let a = f32(px.w) / 255.0;
                let ink = dec(vec3<f32>(f32(px.x), f32(px.y), f32(px.z)));
                out.k2 = out.k2 * (1.0 - a);
                out.c2 = out.c2 * (1.0 - a) + ink;
            }
            let sp = textureLoad(sel_tex, xy, i32(r.b.y), 0);
            if (sp.x != 0u) {
                { let f = fold(Fold(out.k2, out.c2), SEL_TINT, f32(sp.x) / 255.0 * SEL_ALPHA); out.k2 = f.k; out.c2 = f.c; }
            }
            if (sel_edge(i32(r.b.y), t, sp.x)) {
                { let f = fold(Fold(out.k2, out.c2), SEL_TINT, SEL_EDGE_ALPHA); out.k2 = f.k; out.c2 = f.c; }
            }
            break;
        }
        if (udepth == 0u) {
            break;
        }
        udepth = udepth - 1u;
    }
    return out;
}

// The bake pass: a cell's static layers into its composite, the uniforms its own.
fn baking() -> bool {
    return U.tone.y > 0.5;
}

// The contour through a terrain sample: (coverage, 1 on an index line), none where the ground is near flat.
fn contour_at(s: Dem, u: u32, v: u32) -> vec2<f32> {
    let nzn = max(s.n.z, 1e-4);
    let slope = sqrt(max(1.0 - nzn * nzn, 0.0)) / nzn;
    if (slope < 0.02 || U.contour.x <= 0.0) {
        return vec2<f32>(0.0);
    }
    let interval = U.contour.x;
    // Always the interpolated elevation, at every zoom: from the texels' own values a contour is a chain of triangle facets (Nick 2026-10-09).
    let elev_m = dem_smooth(s, u, v);
    let level = round(elev_m / interval);
    let d_m = abs(elev_m - level * interval);
    let d_px = d_m / (slope * U.contour.z);
    let li = i32(level);
    let every = i32(U.contour.y);
    let index = ((li % every) + every) % every == 0;
    let w = select(0.55, 0.9, index);
    return vec2<f32>(clamp(w + 0.5 - d_px, 0.0, 1.0), select(0.0, 1.0, index));
}

// The slope band a terrain sample falls in: its colour, and 1 when there is one.
fn slope_band(s: Dem) -> vec4<f32> {
    let nzn = max(s.n.z, 1e-4);
    let deg = degrees(atan(sqrt(max(1.0 - nzn * nzn, 0.0)) / nzn));
    if (deg >= 45.0) {
        return vec4<f32>(dec(vec3<f32>(150.0, 40.0, 200.0)), 1.0);
    }
    if (deg >= 35.0) {
        return vec4<f32>(dec(vec3<f32>(230.0, 50.0, 40.0)), 1.0);
    }
    if (deg >= 30.0) {
        return vec4<f32>(dec(vec3<f32>(250.0, 150.0, 40.0)), 1.0);
    }
    if (deg >= 25.0) {
        return vec4<f32>(dec(vec3<f32>(250.0, 220.0, 60.0)), 1.0);
    }
    return vec4<f32>(0.0);
}

// The elevation range in view, gathered every fragment (about 0.3 ms at 3× on a desktop GPU; subsampling saved nothing); the load first, so most fragments never touch the atomic.
fn gather_range(elev: f32) {
    // The renderer's quantum: half-metre steps from 11 km below the sea (the CPU's ELEV_OFFSET and ELEV_STEP).
    let eq = u32(clamp((elev + 11000.0) * 2.0, 0.0, 65534.0));
    if (eq < atomicLoad(&view_range[0])) {
        atomicMin(&view_range[0], eq);
    }
    if (eq > atomicLoad(&view_range[1])) {
        atomicMax(&view_range[1], eq);
    }
}

// A line drawn: its magnitude into its class's largest in view.
fn gather_line(cls_id: u32, mag: u32) {
    let c = 2u + min(cls_id, 31u);
    if (mag > atomicLoad(&view_range[c])) {
        atomicMax(&view_range[c], mag);
    }
}

// The composite at a sample: ce, cf, cg the block's candidates for the terrain, the vector planes and the imagery, ch its user cells (NONE throughout when baking: the search finds them).
fn compose(d: u32, u: u32, v: u32, pos: vec2<f32>, ce: vec4<u32>, cf: vec4<u32>, cg: vec4<u32>, ch: vec4<u32>) -> Composed {
    let mask = U.mask & LAYERS;
    var out: Composed;
    out.base = U.style_no_dem.rgb;
    out.ground = false;
    out.n = vec3<f32>(0.0, 0.0, 1.0);
    out.k1 = 1.0;
    out.c1 = vec3<f32>(0.0);
    out.water = 0.0;
    out.k2 = 1.0;
    out.c2 = vec3<f32>(0.0);
    out.banded = false;
    var contour_cov = 0.0;
    var contour_index = false;
    // The theme's contour inks; the geoid's own, darker, over its pale wash.
    var contour_ink = U.style_contour;
    var contour_ink_index = U.style_contour_index;
    var band = vec3<f32>(0.0);
    var have_band = false;
    var found_depth = NONE;
    var is_sea = false;
    var elev_m = 0.0;
    // Sea level against the ellipsoid in place of the elevation (Nick 2026-10-19): the ground the geoid's height above WGS84, unlit, blue where sea level sits below the ellipsoid and orange above, full at a hundred metres; the contours drawn from it, at an interval that follows the zoom (ten metres at the globe, a metre at a city, a tenth up close), every fifth heavier.
    let geo = (mask & M_GEOID) != 0u;
    if (geo) {
        let g = geoid_at(d, u, v);
        if (g.z > 0.0) {
            let t = clamp(g.x / 100.0, -1.0, 1.0);
            let neutral = dec(vec3<f32>(236.0, 236.0, 232.0));
            out.base = select(lerp3(neutral, dec(vec3<f32>(40.0, 110.0, 230.0)), -t), lerp3(neutral, dec(vec3<f32>(236.0, 110.0, 40.0)), t), t >= 0.0);
            if ((mask & M_CONTOURS) != 0u) {
                contour_ink = vec4<f32>(dec(vec3<f32>(28.0, 28.0, 32.0)), 0.6);
                contour_ink_index = vec4<f32>(dec(vec3<f32>(28.0, 28.0, 32.0)), 0.9);
                let interval = geoid_interval(U.contour.z);
                let level = round(g.x / interval);
                let d_px = abs(g.x - level * interval) / max(g.y / GEOID_TEXEL_M * U.contour.z, 1e-6);
                let li = i32(level);
                contour_index = ((li % 5) + 5) % 5 == 0;
                contour_cov = clamp(select(0.55, 0.9, contour_index) + 0.5 - d_px, 0.0, 1.0);
            }
        }
    }
    // The terrain sample feeds the tint and light (terrain on), and the contours and slope bands on their own; the geoid stands in for all of it.
    let want_dem = !geo && (mask & (M_DEM | M_CONTOURS | M_SLOPE | M_RAW)) != 0u;
    if (want_dem) {
        let s = sample_dem(d, u, v, ce);
        if (s.ok) {
            found_depth = s.depth;
            elev_m = s.elev;
            let eq = u32(clamp((s.elev + 11000.0) * 2.0, 0.0, 65534.0));
            if (!baking()) {
                gather_range(s.elev);
            }
            // Raw elevation: grey across the range in view as a byte, unlit (the CPU's raw_grey, from the same quantum).
            if ((mask & M_RAW) != 0u) {
                let q = f32(eq) * 0.5 - 11000.0;
                let t = clamp((q - U.raw.x) / max(U.raw.y - U.raw.x, 0.5), 0.0, 1.0);
                out.base = dec(vec3<f32>(round(t * 255.0)));
                out.ground = false;
            }
            if ((mask & M_DEM) != 0u) {
                out.n = view_normal(s.n, pos);
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
            if ((mask & M_CONTOURS) != 0u) {
                let c = contour_at(s, u, v);
                contour_cov = c.x;
                contour_index = c.y > 0.5;
            }
            if ((mask & M_SLOPE) != 0u) {
                let b = slope_band(s);
                band = b.rgb;
                have_band = b.a > 0.5;
            }
        } else if ((mask & (M_DEM | M_RAW)) != 0u) {
            out.base = U.style_bg.rgb;
        }
    }
    // The vector cell first, for its water: a wet texel below sea level is seabed, tinted by depth, which the imagery leaves alone as it does the sea.
    let vi = pick(cf, d, u, v, U.depths.y, FLAG_VEC);
    var line = vec4<u32>(0u);
    var lw = vec4<u32>(0u);
    var ov = vec4<u32>(0u);
    var ov_layer = -1;
    var ov_t = vec3<u32>(0u);
    var vflags = 0u;
    if (vi != NONE) {
        let r = refs[vi];
        vflags = r.c.x;
        let t = tri(u, v, r.a.y);
        let xy = vec2<i32>(i32(2u * t.x + t.z), i32(t.y));
        if ((vflags & FLAG_LINE) != 0u) {
            line = textureLoad(line_tex, xy, i32(r.b.y), 0);
        }
        if ((vflags & (FLAG_LAND | FLAG_WATER)) != 0u) {
            lw = textureLoad(lw_tex, xy, i32(r.b.z), 0);
        }
        if ((vflags & (FLAG_OWN | FLAG_MVUM)) != 0u) {
            ov_layer = i32(r.c.y);
            ov_t = t;
            ov = textureLoad(ov_tex, xy, ov_layer, 0);
        }
    }
    var seabed = false;
    if (out.ground && !is_sea && elev_m < 0.0 && lw.z != 0u) {
        seabed = true;
        let t = clamp(-elev_m / SEABED_FULL_DEPTH, 0.0, 1.0);
        out.base = U.style_sea.rgb * (1.0 - SEABED_DARK * t);
    }
    // Imagery has its own depth (a global 10 m layer sits several levels above the vector cells), so it is found on its own: the finest cell at or above the vector depth that carries any.
    // Imagery stands in for the lit ground where it has data and the terrain is not sea: the composite's own ocean pixels and the tiles' edges never show, the terrain's coastline does.
    var im = vec4<u32>(0u);
    if ((mask & (M_IMAGERY | M_INFRARED)) != 0u && !is_sea && !seabed) {
        let ii = pick(cg, d, u, v, U.depths.y, FLAG_IMG);
        if (ii != NONE) {
            let ri = refs[ii];
            let ti = tri(u, v, ri.a.y);
            im = textureLoad(img_tex, vec2<i32>(i32(2u * ti.x + ti.z), i32(ti.y)), i32(ri.b.w), 0);
        }
    }
    if (vi == NONE && (mask & M_IMAGERY) != 0u && (im.x != 0u || im.y != 0u || im.z != 0u || im.w != 0u)) {
        out.ground = false;
        out.base = select(vec3<f32>(img_light(im.x), img_light(im.y), img_light(im.z)), vec3<f32>(img_light(im.w)), (mask & M_INFRARED) != 0u);
        if (contour_cov > 0.0) {
            let ink = select(contour_ink, contour_ink_index, contour_index);
            { let f = fold(Fold(out.k2, out.c2), ink.rgb, contour_cov * ink.a); out.k2 = f.k; out.c2 = f.c; }
        }
        return user_marks(out, d, u, v, ch);
    }
    if (vi != NONE) {
        let flags = vflags;
        let draw_line = (mask & M_LINE) != 0u && line.y != 0u && (line.x < BOUNDARY_FIRST || line.x > BOUNDARY_LAST || (mask & M_BOUND) != 0u) && U.depths.x >= line_min_depth(line.x, line.z);
        if (draw_line && !baking()) {
            gather_line(line.x, line.z);
        }
        if ((mask & M_IMAGERY) != 0u && (im.x != 0u || im.y != 0u || im.z != 0u || im.w != 0u)) {
            // Imagery stands in for the ground: nothing lights it. True colour, or the near-infrared band as grey, unrolled to scene light.
            out.ground = false;
            if ((mask & M_INFRARED) != 0u) {
                out.base = vec3<f32>(img_light(im.w));
            } else {
                out.base = vec3<f32>(img_light(im.x), img_light(im.y), img_light(im.z));
            }
            // Under the water, as over the ground: the waterway lines, into the base itself, so a lake covers the river running into it.
            if (draw_line && line.x == WATERWAY_CLASS) {
                out.base = lerp3(out.base, line_colour(line.x, line.z), line_alpha(line.x, line.y, line.z));
            }
            // The water fill paints over imagery as it does over the ground, so a lake or a coast texel that is water in part matches the sea instead of showing the composite's own dark water: the composite's water taken out at the coverage, the theme's put in (the CPU's IMG_WATER).
            if ((mask & M_WATER) != 0u && lw.z != 0u) {
                let t = f32(lw.z) / 255.0 * U.style_water.a * select(1.0, SEABED_GLAZE, seabed);
                if (baking()) {
                    // Baked apart from the imagery, so the waterway lines drawn live go under it: the frame takes the composite's water out and puts the theme's in.
                    out.water = t;
                } else {
                    out.base = max(out.base - vec3<f32>(t * IMG_WATER), vec3<f32>(0.0)) + t * U.style_water.rgb;
                }
            }
            // Contours over the imagery and the water, under the other lines.
            if (contour_cov > 0.0) {
                let ink = select(contour_ink, contour_ink_index, contour_index);
                { let f = fold(Fold(out.k2, out.c2), ink.rgb, contour_cov * ink.a); out.k2 = f.k; out.c2 = f.c; }
            }
            out = overlays(out, ov, ov_layer, ov_t);
            if (draw_line && line.x != WATERWAY_CLASS) {
                { let f = fold(Fold(out.k2, out.c2), line_colour(line.x, line.z), line_alpha(line.x, line.y, line.z)); out.k2 = f.k; out.c2 = f.c; }
            }
            return user_marks(out, d, u, v, ch);
        }
        // The slope band and the contours come under everything that follows, so a baked composite leaves them to the frame.
        out.banded = true;
        if ((mask & M_LAND) != 0u && lw.y != 0u) {
            out.base = lerp3(out.base, unpack_rgb(lut[LAND_BASE + min(lw.x, 13u)]), f32(lw.y) / 255.0 * unpack_a(lut[LAND_BASE + min(lw.x, 13u)]));
        }
        if (have_band) {
            { let f = fold(Fold(out.k1, out.c1), band, 0.45); out.k1 = f.k; out.c1 = f.c; }
        }
        // Under the water: the contours and the waterway lines, so a lake covers the river running into it. Over it: every other line.
        if (contour_cov > 0.0) {
            let ink = select(contour_ink, contour_ink_index, contour_index);
            { let f = fold(Fold(out.k1, out.c1), ink.rgb, contour_cov * ink.a); out.k1 = f.k; out.c1 = f.c; }
        }
        if (draw_line && line.x == WATERWAY_CLASS) {
            { let f = fold(Fold(out.k1, out.c1), line_colour(line.x, line.z), line_alpha(line.x, line.y, line.z)); out.k1 = f.k; out.c1 = f.c; }
        }
        if ((mask & M_WATER) != 0u && lw.z != 0u) {
            out.water = f32(lw.z) / 255.0 * U.style_water.a * select(1.0, SEABED_GLAZE, seabed);
        }
        out = overlays(out, ov, ov_layer, ov_t);
        if (draw_line && line.x != WATERWAY_CLASS) {
            { let f = fold(Fold(out.k2, out.c2), line_colour(line.x, line.z), line_alpha(line.x, line.y, line.z)); out.k2 = f.k; out.c2 = f.c; }
        }
    } else {
        if (have_band) {
            { let f = fold(Fold(out.k1, out.c1), band, 0.45); out.k1 = f.k; out.c1 = f.c; }
        }
        if (contour_cov > 0.0) {
            let ink = select(contour_ink, contour_ink_index, contour_index);
            { let f = fold(Fold(out.k2, out.c2), ink.rgb, contour_cov * ink.a); out.k2 = f.k; out.c2 = f.c; }
        }
    }
    // Where the wanted terrain cell has not arrived at all (no reference and no absent marker), the cell's two triangles each take a random dark colour of their own, the CPU's loading_noise, where nothing at any depth stands in; a coarser stand-in draws as it is.
    if (want_dem && !baking() && found_depth != U.depths.x && lookup(d, U.depths.x, u, v) == NONE) {
        let s = 30u - U.depths.x;
        let m = (1u << s) - 1u;
        let half = (((u & m) + (v & m)) >> s) & 1u;
        var h = (d * 0x27D4EB2Fu) ^ (U.depths.x * 0xC2B2AE3Du) ^ ((u >> s) * 0x9E3779B1u) ^ ((v >> s) * 0x85EBCA77u) ^ (half * 0x165667B1u);
        h = h ^ (h >> 15u);
        h = h * 0x2C1B3C6Du;
        h = h ^ (h >> 12u);
        let noise = dec(vec3<f32>(f32((h & 255u) >> 2u), f32(((h >> 8u) & 255u) >> 2u), f32(((h >> 16u) & 255u) >> 2u)));
        if (found_depth == NONE) {
            out.base = noise;
            out.ground = false;
            out.k1 = 1.0;
            out.c1 = vec3<f32>(0.0);
            out.water = 0.0;
            out.k2 = 1.0;
            out.c2 = vec3<f32>(0.0);
        }
    }
    out = user_marks(out, d, u, v, ch);
    // Detail: tinted by the depth of the terrain cell drawn, the CPU's DETAIL_COLOURS from depth 8 (the global base) to 14 (the finest bakes); magenta where there is none.
    if ((mask & M_DEBUG) != 0u) {
        var tintd = dec(vec3<f32>(255.0, 0.0, 255.0));
        if (found_depth != NONE) {
            let i = min(max(found_depth, 8u) - 8u, 6u);
            var c = vec3<f32>(230.0, 40.0, 40.0);
            switch i {
                case 0u: { c = vec3<f32>(40.0, 90.0, 230.0); }
                case 1u: { c = vec3<f32>(40.0, 160.0, 230.0); }
                case 2u: { c = vec3<f32>(40.0, 200.0, 160.0); }
                case 3u: { c = vec3<f32>(90.0, 210.0, 60.0); }
                case 4u: { c = vec3<f32>(220.0, 220.0, 40.0); }
                case 5u: { c = vec3<f32>(240.0, 140.0, 30.0); }
                default: {}
            }
            tintd = dec(c);
        }
        { let f = fold(Fold(out.k2, out.c2), tintd, 0.5); out.k2 = f.k; out.c2 = f.c; }
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
    let blk = blocks[in.block];
    let p = compose(s.x, s.y, s.z, in.pos.xy, blk.e, blk.f, blk.g, blk.h);
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
    let blk = blocks[in.block];
    var p = compose(s.x, s.y, s.z, in.pos.xy, blk.e, blk.f, blk.g, blk.h);
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

// ==================== COMPOSITE: a vector cell's static layers, baked once ====================

// What the light and the lines leave alone, folded: the frame's colour is `x × A + B + W × water`, x being the base lit (on the ground), with the slope band, the contours and the waterways over it, all live; then the other lines over everything (Nick 2026-10-21: lines baked in showed, and bought little). T0 the base at gamma 2 with 2.8 headroom and A; T1 B at gamma 2 and a byte of W (six bits), the ground (64) and whether the band applies (128, the ground's path; without it, imagery, whose water is laid on rather than mixed in).
struct BakeOut {
    @location(0) t0: vec4<f32>,
    @location(1) t1: vec4<f32>,
};

// A texel of the cell in U.raw (column, row, diamond, depth) at its triangle's centroid: the target is the cell's 512 × 256 texel layout.
@fragment
fn fs_bake(@builtin(position) pos: vec4<f32>) -> BakeOut {
    let px = u32(pos.x);
    let py = u32(pos.y);
    let depth = u32(U.raw.w);
    let s = 30u - depth;
    let ts = 22u - depth;
    let third = (1u << ts) / 3u;
    let off = select(third, 2u * third + 1u, (px & 1u) == 1u);
    let u = (u32(U.raw.x) << s) + ((px >> 1u) << ts) + off;
    let v = (u32(U.raw.y) << s) + (py << ts) + off;
    let none = vec4<u32>(NONE);
    let p = compose(u32(U.raw.z), u, v, pos.xy, none, none, none, none);
    // The ground's water covers what is under it; the imagery's is laid on (its own dark water taken out in the frame).
    let keep = select(1.0, 1.0 - p.water, p.banded);
    let a = p.k1 * keep * p.k2;
    let b = p.c1 * keep * p.k2 + p.c2;
    let w = p.water * p.k2;
    var o: BakeOut;
    o.t0 = vec4<f32>(sqrt(clamp(p.base / 2.8, vec3<f32>(0.0), vec3<f32>(1.0))), a);
    let bits = round(clamp(w, 0.0, 1.0) * 63.0) + select(0.0, 64.0, p.ground) + select(0.0, 128.0, p.banded);
    o.t1 = vec4<f32>(sqrt(clamp(b, vec3<f32>(0.0), vec3<f32>(1.0))), bits / 255.0);
    return o;
}

// The map pass over blocks whose vector cells are all baked: two loads for every static layer, the terrain sampled only for the light, the slope band and the contours. A shader of its own, small, so a phone's GPU keeps more fragments in flight than the full composite's registers allow.
@fragment
fn fs_fast(in: VOut) -> @location(0) vec4<f32> {
    if (off_globe(in.pos.xy)) {
        return vec4<f32>(to_display(U.style_bg.rgb) / 255.0, 1.0);
    }
    let s = sample_uv(in);
    let d = s.x;
    let u = s.y;
    let v = s.z;
    let blk = blocks[in.block];
    let mask = U.mask & LAYERS;
    let ci = pick(blk.f, d, u, v, U.depths.y, FLAG_COMP);
    if (ci == NONE) {
        return vec4<f32>(to_display(U.style_bg.rgb) / 255.0, 1.0);
    }
    let r = refs[ci];
    let t = tri(u, v, r.a.y);
    let xy = vec2<i32>(i32(2u * t.x + t.z), i32(t.y));
    let t0 = textureLoad(comp0_tex, xy, i32(r.c.z), 0);
    let t1 = textureLoad(comp1_tex, xy, i32(r.c.w), 0);
    let bits = u32(round(t1.a * 255.0));
    let ground = (bits & 64u) != 0u;
    let banded = (bits & 128u) != 0u;
    let w = f32(bits & 63u) / 63.0;
    var x = t0.rgb * t0.rgb * 2.8;
    var diffuse = 1.0;
    if ((mask & (M_DEM | M_CONTOURS | M_SLOPE)) != 0u) {
        let dem = sample_dem(d, u, v, blk.e);
        if (dem.ok) {
            gather_range(dem.elev);
            if (ground && (mask & M_DEM) != 0u) {
                let n = view_normal(dem.n, in.pos.xy);
                diffuse = max(dot(n, U.sun.xyz), 0.0);
                x = x * max(light_eval(n), vec3<f32>(0.0));
            }
            if (banded && (mask & M_SLOPE) != 0u) {
                let b = slope_band(dem);
                if (b.a > 0.5) {
                    x = lerp3(x, b.rgb, 0.45);
                }
            }
            if ((mask & M_CONTOURS) != 0u) {
                let c = contour_at(dem, u, v);
                if (c.x > 0.0) {
                    let ink = select(U.style_contour, U.style_contour_index, c.y > 0.5);
                    x = lerp3(x, ink.rgb, c.x * ink.a);
                }
            }
        }
    }
    // The lines, live: the waterways under the water, the rest over everything, weighted against the largest of each class the last frame drew.
    var line = vec4<u32>(0u);
    var draw_line = false;
    if ((mask & M_LINE) != 0u && (r.c.x & FLAG_LINE) != 0u) {
        line = textureLoad(line_tex, xy, i32(r.b.y), 0);
        draw_line = line.y != 0u && (line.x < BOUNDARY_FIRST || line.x > BOUNDARY_LAST || (mask & M_BOUND) != 0u) && U.depths.x >= line_min_depth(line.x, line.z);
    }
    if (draw_line) {
        gather_line(line.x, line.z);
    }
    if (draw_line && line.x == WATERWAY_CLASS) {
        x = lerp3(x, line_colour(line.x, line.z), line_alpha(line.x, line.y, line.z));
    }
    let wr = U.style_water.rgb * (0.85 + 0.15 * diffuse) - vec3<f32>(select(IMG_WATER, 0.0, banded));
    var col = x * t0.a + t1.rgb * t1.rgb + w * wr;
    if (draw_line && line.x != WATERWAY_CLASS) {
        col = lerp3(col, line_colour(line.x, line.z), line_alpha(line.x, line.y, line.z));
    }
    var marks: Composed;
    marks.k2 = 1.0;
    marks.c2 = vec3<f32>(0.0);
    marks = user_marks(marks, d, u, v, blk.h);
    col = col * marks.k2 + marks.c2;
    return vec4<f32>(to_display(col) / 255.0, 1.0);
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
@group(1) @binding(2) var preview_tex: texture_2d<f32>;
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
    // Truncated to the code value, written exactly so the hardware's own rounding has nothing to do. The route, the labels and then the top layer are blended over it in the same pass.
    return vec4<f32>(min(floor(c * 256.0), vec3<f32>(255.0)) / 255.0, 1.0);
}

// Premultiplied `top` of opacity `a` over what is gathered so far.
fn over(acc: vec4<f32>, top: vec3<f32>, a: f32) -> vec4<f32> {
    return vec4<f32>(top * a + acc.rgb * (1.0 - a), a + acc.a * (1.0 - a));
}

// The top layer, over the map and its labels: the overlay (the panel and the engine's marks), the GPS pin, the measurement, the selection being drawn. Premultiplied, blended over.
@fragment
fn fs_top(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let o = textureLoad(overlay_tex, vec2<i32>(pos.xy), 0);
    var acc = vec4<f32>(o.rgb * (255.0 / 256.0), o.a);
    // The GPS pin: an accuracy ring and a crosshair, feathered over a pixel, in the engine's pin blue.
    if (U.pin.w > 0.5) {
        let d = pos.xy - U.pin.xy;
        let dist = length(d);
        let ring = clamp(1.0 - abs(dist - U.pin.z) + 0.45, 0.0, 1.0);
        let cross = select(0.0, clamp(1.6 - min(abs(d.x), abs(d.y)) + 0.5, 0.0, 1.0), max(abs(d.x), abs(d.y)) <= 9.0);
        let cov = max(ring, cross);
        acc = over(acc, vec3<f32>(64.0, 156.0, 255.0) / 256.0, cov);
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
        acc = over(acc, vec3<f32>(255.0, 196.0, 64.0) / 256.0, cov);
    }
    // The selection being drawn: a wash where its closed curve covers, and its edge stronger, in display terms.
    let pxy = vec2<i32>(pos.xy);
    let pv = textureLoad(preview_tex, pxy, 0).r;
    let pdim = vec2<i32>(textureDimensions(preview_tex));
    var pedge = false;
    if (pxy.x > 0) { pedge = pedge || textureLoad(preview_tex, pxy - vec2<i32>(1, 0), 0).r != pv; }
    if (pxy.x + 1 < pdim.x) { pedge = pedge || textureLoad(preview_tex, pxy + vec2<i32>(1, 0), 0).r != pv; }
    if (pxy.y > 0) { pedge = pedge || textureLoad(preview_tex, pxy - vec2<i32>(0, 1), 0).r != pv; }
    if (pxy.y + 1 < pdim.y) { pedge = pedge || textureLoad(preview_tex, pxy + vec2<i32>(0, 1), 0).r != pv; }
    if (pv > 0.0) {
        acc = over(acc, sqrt(SEL_TINT), pv * SEL_ALPHA);
    }
    if (pedge) {
        acc = over(acc, sqrt(SEL_TINT), SEL_EDGE_ALPHA);
    }
    return acc;
}

// ==================== LABELS: a quad per letter, a quad per stretch of route ====================

@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_samp: sampler;

struct GlyphOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) text: vec4<f32>,
    @location(2) halo: vec4<f32>,
};

// Screen pixels to clip space.
fn clip_of(p: vec2<f32>) -> vec4<f32> {
    let screen = U.size / U.scale;
    return vec4<f32>(p.x / screen.x * 2.0 - 1.0, 1.0 - p.y / screen.y * 2.0, 0.0, 1.0);
}

// A letter: its quad about its centre (a.xy), half extents a.zw, turned by b.x (only what its glyph's rotation bin left over), sampling its atlas rectangle.
@vertex
fn vs_glyph(@builtin(vertex_index) vi: u32, @location(0) a: vec4<f32>, @location(1) b: vec4<f32>, @location(2) uv: vec4<f32>, @location(3) text: vec4<f32>, @location(4) halo: vec4<f32>) -> GlyphOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let corner = corners[vi];
    let local = corner * a.zw;
    let sn = sin(b.x);
    let cs = cos(b.x);
    var out: GlyphOut;
    out.pos = clip_of(a.xy + vec2<f32>(local.x * cs - local.y * sn, local.x * sn + local.y * cs));
    out.uv = mix(uv.xy, uv.zw, corner * 0.5 + 0.5);
    out.text = text;
    out.halo = halo;
    return out;
}

// Atlas texels per screen pixel: the labeller's ATLAS_DENSITY (Nick 2026-10-16: letters drawn into the still frame's 3× layer and binned with the map).
const ATLAS_DENSITY: f32 = 3.0;

// The letter over its halo, premultiplied (the atlas holds the letter's coverage in red, the halo's in green). Drawn into the map layer at its scale: a target pixel spans ATLAS_DENSITY / scale texels each way, box-averaged (one sample at 3×, nine at 1×), as binning would.
@fragment
fn fs_glyph(in: GlyphOut) -> @location(0) vec4<f32> {
    let n = max(i32(round(ATLAS_DENSITY / U.scale)), 1);
    let texel = 1.0 / vec2<f32>(textureDimensions(atlas_tex));
    var m = vec4<f32>(0.0);
    for (var j = 0; j < n; j++) {
        for (var i = 0; i < n; i++) {
            let off = (vec2<f32>(f32(i), f32(j)) + 0.5 - f32(n) * 0.5) * texel;
            m += textureSampleLevel(atlas_tex, atlas_samp, in.uv + off, 0.0);
        }
    }
    m /= f32(n * n);
    let ta = m.r * in.text.a;
    let ha = m.g * in.halo.a;
    // The colours come as the theme stores them and are encoded as the map's are, through the same highlights: a white halo is the map's white, not the display's.
    let text = to_display(dec(in.text.rgb * 255.0)) / 255.0;
    let halo = to_display(dec(in.halo.rgb * 255.0)) / 255.0;
    return vec4<f32>(text * ta + halo * ha * (1.0 - ta), ta + ha * (1.0 - ta));
}

struct SegOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) at: vec2<f32>,
    @location(1) ends: vec4<f32>,
    @location(2) half: f32,
    @location(3) colour: vec4<f32>,
};

// A stretch of the route: a quad around it a pixel past its half width, its coverage worked per pixel from the distance to the stretch.
@vertex
fn vs_segment(@builtin(vertex_index) vi: u32, @location(0) ends: vec4<f32>, @location(1) b: vec4<f32>, @location(2) colour: vec4<f32>) -> SegOut {
    let p0 = ends.xy;
    let p1 = ends.zw;
    let len = length(p1 - p0);
    var dir = vec2<f32>(1.0, 0.0);
    if (len > 1e-4) { dir = (p1 - p0) / len; }
    let nrm = vec2<f32>(-dir.y, dir.x);
    let ext = b.x + 1.0;
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    let p = mix(p0 - dir * ext, p1 + dir * ext, c.x) + nrm * c.y * ext;
    var out: SegOut;
    out.pos = clip_of(p);
    out.at = p;
    out.ends = ends;
    out.half = b.x;
    out.colour = colour;
    return out;
}

@fragment
fn fs_segment(in: SegOut) -> @location(0) vec4<f32> {
    let ab = in.ends.zw - in.ends.xy;
    let t = clamp(dot(in.at - in.ends.xy, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    let d = length(in.at - (in.ends.xy + ab * t));
    // Feathered over one pixel of the layer it is drawn into (a third of a screen pixel at 3×), the binning doing the rest, as for the map's own lines.
    let cov = clamp((in.half - d) * U.scale + 0.5, 0.0, 1.0) * in.colour.a;
    return vec4<f32>(in.colour.rgb * cov, cov);
}
