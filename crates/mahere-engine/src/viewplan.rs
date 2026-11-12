//! The view plan: what a GPU that finds every pixel's cell itself needs from the CPU (ferros's GPU map, mahere-ferros kernel.rs; ferros GPU-KISTS.md 0.2, "per frame: the camera, table changes").
//!
//! The kernel projects each pixel onto the icosahedron from the camera: the faces in view, each with the projective form of the encoder's (u, v) on it in view coordinates, the pixel's face the one whose centroid is nearest as the encoder chooses it.
//! It then finds its cell in a page table: for each role (the terrain, the vector planes) and each diamond in view, a box of the cell grid's squares at the role's wanted depth, each square naming the resident cell that stands in for it (the square's own, or the finest coarser one carrying the plane).
//! So there is no block lattice and no seam sub-blocks on the CPU (`plan_frame`'s, 6.5 ms of a frame at a town's scale on the Pro and 15.7 at the globe, 2026-11-12): what is left here is which faces are in view, the boxes, their stand-ins, and the desired set, from a lattice of points projected with no trigonometry.

use mahere_coord::{FACE_COUNT, face_map, morton_spread};
use mahere_tiles::{CellKey, ELEV_NODATA};
use rayon::prelude::*;

use crate::Camera;
use crate::plan::{NONE, desired_set};
use crate::raster::{BLOCK, MIN_DEPTH};
use crate::residency::{PRESENT_DEM, PRESENT_LAND, PRESENT_LINE, PRESENT_MVUM, PRESENT_OWN, PRESENT_WATER, Pool};

/// The spacing of the samples along the screen's edges, pixels: a diamond that pokes in at an edge thinner than this along it and holds no lattice point may go without squares of its own (its pixels then draw what its neighbour's clamped (u, v) reach).
pub const EDGE_STEP: usize = 1 << 3;

/// Depth-30 units in a diamond's side.
const UNITS: f64 = (1u64 << 30) as f64;
const TOP: i64 = (1 << 30) - 1;

/// The planes each role's stand-ins carry: the terrain, then the vector planes (`plan::FLAG_VEC`'s).
pub const ROLE_PLANES: [u8; 2] = [PRESENT_DEM, PRESENT_LINE | PRESENT_LAND | PRESENT_WATER | PRESENT_OWN | PRESENT_MVUM];

/// A face in view, everything in view coordinates (screen right, screen up, toward the viewer; the camera point at (0, 0, 1)): the encoder's search vector and the projective form of its (u, v), as `mahere_coord::FaceMap` has them.
#[derive(Clone, Copy, Debug)]
pub struct ViewFace {
    pub face: u8,
    pub diamond: u8,
    pub centroid: [f64; 3],
    pub gu: [f64; 3],
    pub gv: [f64; 3],
    pub n: [f64; 3],
}

impl ViewFace {
    /// The depth-30 (u, v) of view direction `p` on this face, toward zero and held inside the diamond, as the encoder quantises them.
    pub fn uv(&self, p: [f64; 3]) -> (u32, u32) {
        let d = dot(self.n, p);
        let q = |g: [f64; 3]| ((dot(g, p) / d * UNITS) as i64).clamp(0, TOP) as u32;
        (q(self.gu), q(self.gv))
    }
}

/// A diamond's box of squares for a role: the cell grid's corner (cu0, cv0) at the role's depth, its width and height in squares, and where its squares start in the role's `squares` (row by row).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CellBox {
    pub cu0: u32,
    pub cv0: u32,
    pub w: u32,
    pub h: u32,
    pub at: u32,
}

/// A cell standing in for squares, and whether one of them holds a point of the screen (not only the boxes' margins).
#[derive(Clone, Copy, Debug)]
pub struct StandIn {
    pub key: CellKey,
    pub on_screen: bool,
}

/// One role's page table.
#[derive(Clone, Debug, Default)]
pub struct RoleTable {
    pub depth: u8,
    /// By diamond: its box, None when the view shows none of that diamond.
    pub boxes: [Option<CellBox>; 10],
    /// Every box's squares: the index into `refs` of the cell standing in, [`NONE`] where no resident cell carries the role's planes.
    pub squares: Vec<u32>,
    pub refs: Vec<StandIn>,
}

impl RoleTable {
    /// The square depth-30 (u, v) of `diamond` lies in, as an index into `squares`; None outside its box.
    pub fn square(&self, diamond: u8, u: u32, v: u32) -> Option<usize> {
        let b = self.boxes[diamond as usize]?;
        let s = 30 - self.depth as u32;
        let (du, dv) = ((u >> s).wrapping_sub(b.cu0), (v >> s).wrapping_sub(b.cv0));
        (du < b.w && dv < b.h).then(|| (b.at + du + dv * b.w) as usize)
    }

    /// The cell standing in at depth-30 (u, v) of `diamond`.
    pub fn stand_in(&self, diamond: u8, u: u32, v: u32) -> Option<CellKey> {
        let i = self.squares[self.square(diamond, u, v)?];
        (i != NONE).then(|| self.refs[i as usize].key)
    }
}

pub struct ViewPlan {
    /// The faces in view, in the encoder's order (a direction on a seam goes to the first).
    pub faces: Vec<ViewFace>,
    /// The terrain's table at `dem_depth`, the vector planes' at `vec_depth`.
    pub roles: [RoleTable; 2],
    pub dem_depth: u8,
    pub vec_depth: u8,
    pub want: Vec<CellKey>,
    /// The quantised elevations at the lattice's points, lowest and highest (the next frame's contour interval); None where no terrain stands in.
    pub range: Option<(u16, u16)>,
    /// Screen to view: x = (px − centre x) / radius, y = (centre y − py) / radius, the camera point at the centre.
    pub centre: [f64; 2],
    pub radius: f64,
    /// The globe's disk (centre x, y, radius in pixels, whether the limb is on the screen) and the world's pole in view coordinates, as `FramePlan` has them.
    pub globe: [f32; 4],
    pub pole: [f32; 4],
    /// The points projected: the lattice's, the edges', the corners of faces on the screen.
    pub samples: usize,
    /// Where the plan's time went, µs: the faces and the points, the desired set, the boxes and their stand-ins, the elevation range.
    pub us: [u32; 4],
}

impl ViewPlan {
    /// The view direction of screen point (px, py), past the limb taken at the limb (`Camera::view_dir`).
    pub fn view_dir(&self, px: f64, py: f64) -> [f64; 3] {
        let mut x = (px - self.centre[0]) / self.radius;
        let mut y = (self.centre[1] - py) / self.radius;
        let rho2 = x * x + y * y;
        if rho2 >= 1.0 {
            let k = (1.0 - 1e-9) / rho2.sqrt();
            x *= k;
            y *= k;
        }
        [x, y, (1.0 - x * x - y * y).max(0.0).sqrt()]
    }

    /// The face (an index into `faces`) and depth-30 (u, v) under view direction `p`.
    pub fn locate(&self, p: [f64; 3]) -> (usize, u32, u32) {
        let mut best = 0;
        for (i, f) in self.faces.iter().enumerate().skip(1) {
            if dot(f.centroid, p) > dot(self.faces[best].centroid, p) {
                best = i;
            }
        }
        let (u, v) = self.faces[best].uv(p);
        (best, u, v)
    }
}

/// The faces whose triangles reach the screen: the camera point inside one, or within the screen's angular radius of one's edges or corners (the whole near hemisphere when the limb is on the screen), each in view coordinates.
fn faces_in_view(cam: &Camera, w: usize, h: usize) -> Vec<ViewFace> {
    let b = cam.basis();
    let view = |v: [f64; 3]| [dot(b[0], v), dot(b[1], v), dot(b[2], v)];
    let r = cam.radius();
    let (cx, cy) = Camera::centre(w, h);
    let rho = [(0.0, 0.0), (w as f64, 0.0), (0.0, h as f64), (w as f64, h as f64)].iter().map(|&(x, y)| ((x - cx) * (x - cx) + (y - cy) * (y - cy)).sqrt()).fold(0.0, f64::max);
    // The screen's farthest point from the camera point, as the sine and cosine of its angle on the ball (orthographic: the sine is its distance over the radius); a quarter turn when the limb is on the screen. The slack takes a point on an edge in.
    let sin_s = (rho / r).min(1.0);
    let cos_s = (1.0 - sin_s * sin_s).sqrt();
    let (sin_s, cos_s) = (sin_s + 1e-9, cos_s - 1e-9);
    let mut faces = Vec::with_capacity(FACE_COUNT);
    for f in 0..FACE_COUNT {
        let m = face_map(f);
        let v = m.verts.map(view);
        let turn = dot(cross(v[0], v[1]), v[2]).signum();
        // The camera point, (0, 0, 1), inside: on the inner side of all three edges' planes.
        let inside = (0..3).all(|k| cross(v[k], v[(k + 1) % 3])[2] * turn >= 0.0);
        let corner = v.iter().any(|p| p[2] >= cos_s);
        let edge = (0..3).any(|k| {
            let (p, q) = (v[k], v[(k + 1) % 3]);
            let e = normalize(cross(p, q));
            // The point of the edge's great circle nearest the camera point, and whether it lies between the edge's ends.
            let foot = [-e[2] * e[0], -e[2] * e[1], 1.0 - e[2] * e[2]];
            e[2].abs() <= sin_s && dot(cross(p, foot), e) >= 0.0 && dot(cross(foot, q), e) >= 0.0
        });
        if inside || corner || edge {
            faces.push(ViewFace { face: f as u8, diamond: m.diamond, centroid: view(m.centroid), gu: view(m.gu), gv: view(m.gv), n: view(m.n) });
        }
    }
    faces
}

/// Plan a view for a GPU that projects every pixel itself (`ViewPlan`): the faces in view, each role's boxes and stand-ins, the desired set and the elevation range.
pub fn plan_view(w: usize, h: usize, cam: &Camera, pool: &Pool, dem_depth: u8, vec_depth: u8) -> ViewPlan {
    let t0 = std::time::Instant::now();
    let r = cam.radius();
    let (cx, cy) = Camera::centre(w, h);
    let mut plan = ViewPlan {
        faces: faces_in_view(cam, w, h),
        roles: [RoleTable { depth: dem_depth, ..RoleTable::default() }, RoleTable { depth: vec_depth, ..RoleTable::default() }],
        dem_depth,
        vec_depth,
        want: Vec::new(),
        range: None,
        centre: [cx, cy],
        radius: r,
        globe: [cx as f32, cy as f32, r as f32, cam.limb_visible(w, h) as u32 as f32],
        pole: {
            let b = cam.basis();
            [b[0][2] as f32, b[1][2] as f32, b[2][2] as f32, 0.0]
        },
        samples: 0,
        us: [0; 4],
    };
    // The points: the 32 px lattice (the old plan's corners: the desired set and the elevation range come from them as before), every `EDGE_STEP` along the screen's edges, and each face's corners on the screen, so a diamond in view holds a point unless it is a sliver at an edge.
    let (bw, bh) = (w.div_ceil(BLOCK), h.div_ceil(BLOCK));
    let lattice = (bw + 1) * (bh + 1);
    let mut pts: Vec<(f64, f64)> = (0..lattice).map(|i| (((i % (bw + 1)) * BLOCK) as f64, ((i / (bw + 1)) * BLOCK) as f64)).collect();
    for x in (0..=w).step_by(EDGE_STEP) {
        pts.extend_from_slice(&[(x as f64, 0.0), (x as f64, h as f64)]);
    }
    for y in (0..=h).step_by(EDGE_STEP) {
        pts.extend_from_slice(&[(0.0, y as f64), (w as f64, y as f64)]);
    }
    let mut samples: Vec<(u8, u32, u32)> = pts
        .par_iter()
        .map(|&(px, py)| {
            let (f, u, v) = plan.locate(plan.view_dir(px, py));
            (plan.faces[f].diamond, u, v)
        })
        .collect();
    for f in &plan.faces {
        let m = face_map(f.face as usize);
        let b = cam.basis();
        for p in m.verts {
            let p = [dot(b[0], p), dot(b[1], p), dot(b[2], p)];
            let (sx, sy) = (cx + r * p[0], cy - r * p[1]);
            if p[2] > 0.0 && (0.0..=w as f64).contains(&sx) && (0.0..=h as f64).contains(&sy) {
                let (u, v) = f.uv(p);
                samples.push((f.diamond, u, v));
            }
        }
    }
    plan.samples = samples.len();
    let t1 = std::time::Instant::now();
    let raw = |&(d, u, v): &(u8, u32, u32)| (d as u64) << 60 | morton_spread(u as u64) << 1 | morton_spread(v as u64);
    plan.want = desired_set(samples.iter().map(raw), dem_depth.max(vec_depth)).into_iter().collect();
    let t2 = std::time::Instant::now();

    // The depths at which any resident cell carries each role's planes, a bit a depth: a stand-in's search skips the rest.
    let mut held = [0u64; 2];
    for (k, e) in &pool.map {
        for (role, need) in ROLE_PLANES.iter().enumerate() {
            if e.present & need != 0 {
                held[role] |= 1 << k.depth;
            }
        }
    }
    for (role, t) in plan.roles.iter_mut().enumerate() {
        let s = 30 - t.depth as u32;
        // Each diamond's squares the points fell in, then one more all round (a cell reaching the screen between points lies beside one holding a point: a cell is a hundred pixels or more across, `select_depth`), inside the diamond.
        let mut span: [Option<[u32; 4]>; 10] = [None; 10];
        for &(d, u, v) in &samples {
            let (cu, cv) = (u >> s, v >> s);
            let b = span[d as usize].get_or_insert([cu, cv, cu, cv]);
            *b = [b[0].min(cu), b[1].min(cv), b[2].max(cu), b[3].max(cv)];
        }
        let last = (1u32 << t.depth) - 1;
        let mut at = 0;
        for (d, sp) in span.iter().enumerate() {
            let Some([u0, v0, u1, v1]) = *sp else { continue };
            let (u0, v0, u1, v1) = (u0.saturating_sub(1), v0.saturating_sub(1), (u1 + 1).min(last), (v1 + 1).min(last));
            let b = CellBox { cu0: u0, cv0: v0, w: u1 - u0 + 1, h: v1 - v0 + 1, at };
            at += b.w * b.h;
            t.boxes[d] = Some(b);
        }
        // Each square's stand-in: its own cell at the role's depth, else the finest coarser one carrying the planes, as the CPU raster's probe climbs.
        t.squares = Vec::with_capacity(at as usize);
        let need = ROLE_PLANES[role];
        for (d, b) in t.boxes.iter().enumerate() {
            let Some(b) = b else { continue };
            for dv in 0..b.h {
                for du in 0..b.w {
                    let (cu, cv) = ((b.cu0 + du) as u64, (b.cv0 + dv) as u64);
                    let mut found = NONE;
                    for depth in (MIN_DEPTH..=t.depth).rev() {
                        if held[role] >> depth & 1 == 0 {
                            continue;
                        }
                        let up = (t.depth - depth) as u32;
                        let key = CellKey::from_grid(d as u8, depth, cu >> up, cv >> up);
                        if pool.map.get(&key).is_some_and(|e| e.present & need != 0) {
                            // A few dozen stand-ins a frame: a list, searched.
                            found = match t.refs.iter().position(|r| r.key == key) {
                                Some(i) => i as u32,
                                None => {
                                    t.refs.push(StandIn { key, on_screen: false });
                                    (t.refs.len() - 1) as u32
                                }
                            };
                            break;
                        }
                    }
                    t.squares.push(found);
                }
            }
        }
        for &(d, u, v) in &samples {
            let i = t.squares[t.square(d, u, v).expect("a point's own square is in its diamond's box")];
            if i != NONE {
                t.refs[i as usize].on_screen = true;
            }
        }
    }
    let t3 = std::time::Instant::now();

    // The elevation range at the lattice's points, through the terrain's stand-ins (the old plan's blocks' first corners).
    let (mut lo, mut hi) = (u16::MAX, 0u16);
    let dem = &plan.roles[0];
    for &(d, u, v) in &samples[..lattice] {
        let Some(key) = dem.stand_in(d, u, v) else { continue };
        let ts = 22 - key.depth as u32;
        let m = (1u32 << ts) - 1;
        let ti = ((((v >> ts) & 255) << 8 | ((u >> ts) & 255)) << 1 | (((u & m) + (v & m)) >> ts) & 1) as usize;
        if let Some(q) = pool.map.get(&key).and_then(|e| e.elev_q_at(ti)) {
            if q != ELEV_NODATA {
                lo = lo.min(q);
                hi = hi.max(q);
            }
        }
    }
    plan.range = (lo <= hi).then_some((lo, hi));
    let t4 = std::time::Instant::now();
    let us = |a: std::time::Instant, b: std::time::Instant| (b - a).as_micros() as u32;
    plan.us = [us(t0, t1), us(t1, t2), us(t2, t3), us(t3, t4)];
    plan
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = dot(v, v).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahere_coord::Coord;

    /// Every pixel's (diamond, u, v) through the plan against the encoder's own, and every pixel's square in its diamond's box, over views at every scale: a town, a seam, an icosahedron's corner, the globe with its limb.
    #[test]
    fn every_pixel_is_located_as_the_encoder_locates_it() {
        let (w, h) = (160usize, 120usize);
        let pool = Pool::default();
        // (lat, lon, ppd, bearing): a town; across the seam between diamonds 0 and 1 at 72° east; on the north pole, where five faces meet; the upper ring's corner at 26.57° N 0° E; the whole globe.
        for (lat, lon, ppd, bearing) in [(46.2, -121.5, 6000.0, 0.7), (40.0, 72.0, 600.0, 0.2), (90.0, 0.0, 3000.0, 0.0), (26.565, 0.0, 200.0, 1.1), (10.0, 30.0, 0.4, 0.0), (-30.0, 150.0, 0.6, 2.0)] {
            let cam = Camera::new(lat, lon, ppd, bearing);
            let d = crate::raster::select_depth(ppd, 0, 14);
            let plan = plan_view(w, h, &cam, &pool, d, d);
            let mut worst = 0i64;
            let mut seams = 0;
            for py in 0..h {
                for px in 0..w {
                    let (sx, sy) = (px as f64 + 0.5, py as f64 + 0.5);
                    if !cam.on_globe(sx, sy, w, h) {
                        continue;
                    }
                    // The encoder on the pixel's direction in the world (`screen_to_geo` without its trip through latitude and longitude, which the pole's clamp moves by a hundred metres).
                    let p = plan.view_dir(sx, sy);
                    let b = cam.basis();
                    let c = Coord::from_xyz([0, 1, 2].map(|k| p[0] * b[0][k] + p[1] * b[1][k] + p[2] * b[2][k]));
                    let (f, u, v) = plan.locate(p);
                    let dm = plan.faces[f].diamond;
                    if dm != c.diamond() {
                        // A pixel on a seam, within rounding of both diamonds.
                        seams += 1;
                        continue;
                    }
                    let (iu, iv) = c.uv();
                    worst = worst.max((u as i64 - iu as i64).abs()).max((v as i64 - iv as i64).abs());
                    for t in &plan.roles {
                        assert!(t.square(dm, u, v).is_some(), "({px}, {py}): no square for diamond {dm} at depth {}", t.depth);
                    }
                }
            }
            // The lat/lon round trip and the clamp at the pole cost the encoder's path its last few units; a pixel at depth 30 is far wider at every one of these scales.
            assert!(worst <= 1, "view ({lat}, {lon}, {ppd}): a pixel {worst} units from the encoder's (u, v)");
            assert!(seams < w, "view ({lat}, {lon}, {ppd}): {seams} pixels in another diamond than the encoder's");
            println!("view ({lat}, {lon}, ppd {ppd}): {} faces, {} points, boxes {:?}; worst {worst} units, {seams} seam pixels", plan.faces.len(), plan.samples, plan.roles[0].boxes.iter().flatten().map(|b| (b.w, b.h)).collect::<Vec<_>>());
        }
    }
}
