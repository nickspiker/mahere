//! Mahere's face on ferros (owner 2026-11-24, choosing from the circle proposal: "search, layers, locate, compass is good"; the scale on the arc, "between 1/2 and 1/4-ish of screen width"; the readout in the bottom cap; the arc pages ferros only; the recording ring; the bloom), in Photon's ferros language (photon src/ui/safe_layout.rs, render.rs draw_title_arc), everything inside the safe circle:
//! - The orb at the circle's top opens the root menu, the pages as names down the circle's axis. Back goes from a page to the root, from the root to the map ("Back is back, the orb is settings", owner 2026-07-04).
//! - Four beads on the arc through the orb, true circles with marks and no words (owner 2026-11-18): search and layers left of it, locate and the compass right.
//! - A page's title turns along the circle under the orb; a swipe brings the next page, following the finger, and on release snaps by half the remaining distance a frame, floored, one page at most (owner 2026-11-18, 2026-11-08).
//! - The scale is a stretch of an arc concentric with the circle, above the helm.
//! - A tap's readout fills the circle's bottom cap.
//! - The tool bead at the circle's foot blooms its tools along two arcs above it: the tap tools nearer, the ink tools beyond.
//! - While a track records, a ring round the orb fills with the hour's minutes, its distance and time under it.
//! Every shape is drawn from its distance to the pixel (a logistic edge, C∞) and composited under what is already painted, as fluor composites; sizes are in Photon's unit, the span over 32 at the app's scale.

use super::*;
use fluor::BlendMode;
use fluor::pixel::Blend;

/// The root menu's pages, in its order, and their names.
const ROOT: [(Page, &str); 8] = [(Page::Layers, "Layers"), (Page::Themes, "Theme"), (Page::Areas, "Areas"), (Page::Tracks, "Tracks"), (Page::Search, "Search"), (Page::Here, "Here"), (Page::Updates, "Updates"), (Page::Device, "Device")];

/// The pages a swipe turns through: the root's but the search, which takes the keyboard and so is opened, never passed through.
const RING: [Page; 7] = [Page::Layers, Page::Themes, Page::Areas, Page::Tracks, Page::Here, Page::Updates, Page::Device];

fn name_of(p: Page) -> &'static str {
    ROOT.iter().find(|(q, _)| *q == p).map_or("", |(_, n)| n)
}

/// What a bead in the bottom cap does.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum CapAct {
    Copy,
    Route,
    Close,
}

/// The circle's state and the places its last layout and paint found.
#[derive(Default)]
pub(crate) struct Round {
    /// The root menu is up (the orb opened it).
    pub root: bool,
    /// The open page's offset while a swipe turns it, pixels (left negative): the finger's while it is down, the snap's after.
    pub turn: f32,
    /// The page a released turn lands on, relative to the open one: -1, 0 or +1.
    settle: i32,
    /// Where a press on a page's body went down, for a swipe to measure from.
    press: Option<(f32, f32)>,
    /// The press became a swipe: the body follows the finger and takes no taps.
    turning: bool,
    /// The tool bead's tools are out.
    pub bloom: bool,
    /// Photon's unit: the span over 32 at the app's scale.
    u: f32,
    /// The beads beside the orb: search, layers, locate, the compass.
    beads: [(f32, f32, f32); 4],
    /// The tool bead at the circle's foot, and its bloom's beads.
    tool: (f32, f32, f32),
    /// Where the last paint drew it: raised above the cap while a readout shows.
    tool_drawn: (f32, f32, f32),
    bloom_beads: Vec<(f32, f32, f32, Pick)>,
    /// The root menu's rows: centre y, half height, the page.
    root_rows: Vec<(f32, f32, Page)>,
    /// The title's height on the arc, the pages' body top, and the lowest row a full-width row keeps its corners inside the circle.
    title_y: f32,
    body_top: f32,
    floor: f32,
    /// A row's side margin.
    margin: f32,
    /// The bottom cap's top while a readout shows, and its beads.
    cap_top: Option<f32>,
    cap_beads: Vec<(f32, f32, f32, CapAct)>,
    /// The tapped place, for the cap's beads.
    cap_at: (f64, f64),
}

/// A smooth cap (Photon safe_layout::soft_cap, ferros WINDOWS.md "Deck's turn"): `z` itself while well under `limit`, bending to `limit` with no kink.
fn soft_cap(z: f32, limit: f32) -> f32 {
    z / (1. + (z / limit).powi(8)).powf(1. / 8.)
}

/// How much of a pixel a shape covers, its centre `d` pixels outside the shape's edge (negative inside): a logistic, half at the edge, about a pixel wide, C∞.
fn cover(d: f32) -> f32 {
    1. / (1. + (4. * d).exp())
}

/// A colour at a fraction of its coverage: its opacity and darkness together, as fluor's layers hold them.
fn part(c: u32, k: f32) -> u32 {
    let b = |s: u32| ((((c >> s) & 0xFF) as f32 * k) as u32) << s;
    b(24) | b(16) | b(8) | b(0)
}

/// A shape under what is painted: `dist` the signed distance from a pixel's centre to its edge, over the box `[x0, y0, x1, y1]` it can reach.
fn shape(canvas: &mut Canvas, bx: [f32; 4], colour: u32, dist: impl Fn(f32, f32) -> f32) {
    let (w, h) = (canvas.width, canvas.height);
    // WHY: a shape near the frame's edge reaches past it (a bead at 400 %, the cap's arc); PROOF: an unclipped box indexes outside the buffer; the cast of a negative bound would wrap.
    let x0 = bx[0].floor().max(0.) as usize;
    let y0 = bx[1].floor().max(0.) as usize;
    let x1 = (bx[2].ceil().max(0.) as usize).min(w);
    let y1 = (bx[3].ceil().max(0.) as usize).min(h);
    for y in y0..y1 {
        for x in x0..x1 {
            let k = cover(dist(x as f32 + 0.5, y as f32 + 0.5));
            // Nothing a byte can hold: left as it is.
            if k * 256. < 1. {
                continue;
            }
            let i = y * w + x;
            canvas.pixels[i] = canvas.pixels[i].under(part(colour, k), BlendMode::Normal);
        }
    }
}

fn disc(canvas: &mut Canvas, x: f32, y: f32, r: f32, colour: u32) {
    shape(canvas, [x - r - 2., y - r - 2., x + r + 2., y + r + 2.], colour, |px, py| (px - x).hypot(py - y) - r);
}

fn ring(canvas: &mut Canvas, x: f32, y: f32, r: f32, half: f32, colour: u32) {
    let o = r + half + 2.;
    shape(canvas, [x - o, y - o, x + o, y + o], colour, |px, py| ((px - x).hypot(py - y) - r).abs() - half);
}

/// A line with round ends from `a` to `b`, `half` its half width.
fn capsule(canvas: &mut Canvas, a: (f32, f32), b: (f32, f32), half: f32, colour: u32) {
    let o = half + 2.;
    let bx = [a.0.min(b.0) - o, a.1.min(b.1) - o, a.0.max(b.0) + o, a.1.max(b.1) + o];
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    shape(canvas, bx, colour, |px, py| {
        let t = if len2 > 0. { (((px - a.0) * dx + (py - a.1) * dy) / len2).clamp(0., 1.) } else { 0. };
        (px - a.0 - dx * t).hypot(py - a.1 - dy * t) - half
    });
}

/// A convex polygon, its corners in either turn: the farthest of its edges' half planes.
fn polygon(canvas: &mut Canvas, pts: &[(f32, f32)], colour: u32) {
    let n = pts.len();
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    let (mut cx, mut cy) = (0., 0.);
    for p in pts {
        (x0, y0, x1, y1) = (x0.min(p.0), y0.min(p.1), x1.max(p.0), y1.max(p.1));
        cx += p.0 / n as f32;
        cy += p.1 / n as f32;
    }
    // Each edge's outward normal: the side away from the centroid.
    let edges: Vec<(f32, f32, f32)> = (0..n)
        .map(|i| {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            let (mut nx, mut ny) = (b.1 - a.1, a.0 - b.0);
            let l = nx.hypot(ny);
            (nx, ny) = (nx / l, ny / l);
            if (cx - a.0) * nx + (cy - a.1) * ny > 0. {
                (nx, ny) = (-nx, -ny);
            }
            (nx, ny, -(a.0 * nx + a.1 * ny))
        })
        .collect();
    shape(canvas, [x0 - 2., y0 - 2., x1 + 2., y1 + 2.], colour, |px, py| edges.iter().map(|&(nx, ny, c)| px * nx + py * ny + c).fold(f32::MIN, f32::max));
}

/// An arc of the circle about `(x, y)` with radius `r`, from the angle `mid - spread` to `mid + spread` (radians, clockwise from up), round ended.
#[allow(clippy::too_many_arguments)]
fn arc(canvas: &mut Canvas, x: f32, y: f32, r: f32, mid: f32, spread: f32, half: f32, colour: u32) {
    let at = |a: f32| (x + r * a.sin(), y - r * a.cos());
    let (e0, e1) = (at(mid - spread), at(mid + spread));
    let o = r + half + 2.;
    shape(canvas, [x - o, y - o, x + o, y + o], colour, |px, py| {
        let a = (px - x).atan2(y - py);
        let off = (a - mid + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        if off.abs() <= spread {
            ((px - x).hypot(py - y) - r).abs() - half
        } else {
            (px - e0.0).hypot(py - e0.1).min((px - e1.0).hypot(py - e1.1)) - half
        }
    });
}

/// What a bead shows.
#[derive(Clone, Copy)]
enum Mark {
    Search,
    Layers,
    Locate,
    /// The compass: the map's bearing, radians.
    Compass(f32),
    Tool(Pick),
    Cap(CapAct),
}

/// A bead's mark, drawn in `ink` inside a bead of radius `r` at `(x, y)`; the pen's colour for its own bead.
fn mark(canvas: &mut Canvas, text: &mut TextRenderer, m: Mark, x: f32, y: f32, r: f32, ink: u32, pen: u32) {
    let s = r * 0.5;
    let t = (r * 0.075).max(1.);
    match m {
        Mark::Search | Mark::Tool(Pick::Search) => {
            ring(canvas, x - s * 0.2, y - s * 0.2, s * 0.55, t, ink);
            capsule(canvas, (x + s * 0.2, y + s * 0.2), (x + s * 0.75, y + s * 0.75), t * 1.3, ink);
        }
        Mark::Layers => {
            // Two sheets: the top one whole, the edge of the one under it.
            let (hw, hh) = (s * 0.85, s * 0.42);
            let top = y - s * 0.25;
            for (a, b) in [((x - hw, top), (x, top - hh)), ((x, top - hh), (x + hw, top)), ((x + hw, top), (x, top + hh)), ((x, top + hh), (x - hw, top))] {
                capsule(canvas, a, b, t, ink);
            }
            let low = top + s * 0.55;
            capsule(canvas, (x - hw, low), (x, low + hh), t, ink);
            capsule(canvas, (x, low + hh), (x + hw, low), t, ink);
        }
        Mark::Locate => {
            disc(canvas, x, y, t * 1.6, ink);
            ring(canvas, x, y, s * 0.55, t, ink);
            for (dx, dy) in [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)] {
                capsule(canvas, (x + dx * s * 0.7, y + dy * s * 0.7), (x + dx * s * 1.05, y + dy * s * 1.05), t, ink);
            }
        }
        Mark::Compass(b) => {
            // North on the screen is (-sin B, -cos B): its half of the needle in the north colour.
            let (nx, ny) = (-b.sin(), -b.cos());
            let (px, py) = (-ny, nx);
            let (tip, base) = (s * 1.25, s * 0.32);
            polygon(canvas, &[(x + nx * tip, y + ny * tip), (x + px * base, y + py * base), (x - px * base, y - py * base)], NORTH);
            polygon(canvas, &[(x - nx * tip, y - ny * tip), (x + px * base, y + py * base), (x - px * base, y - py * base)], ink);
        }
        Mark::Tool(p) => match p {
            Pick::Info => {
                text.draw_text_center(canvas, "i", x, y, &TextStyle::new(r * 1.1, ink).weight(700), None, None);
            }
            Pick::Ruler => {
                let (a, b) = ((x - s * 0.95, y + s * 0.45), (x + s * 0.95, y - s * 0.45));
                for k in 1..4 {
                    let f = k as f32 / 4.;
                    let (mx, my) = (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f);
                    capsule(canvas, (mx, my), (mx - s * 0.17, my - s * 0.36), t * 0.8, ink);
                }
                capsule(canvas, a, b, t * 1.2, ink);
            }
            Pick::Directions => {
                capsule(canvas, (x - s * 0.7, y + s * 0.6), (x + s * 0.7, y - s * 0.6), t * 0.8, ink);
                disc(canvas, x - s * 0.7, y + s * 0.6, s * 0.28, ink);
                ring(canvas, x + s * 0.7, y - s * 0.6, s * 0.28, t, ink);
            }
            Pick::Navigate => {
                polygon(canvas, &[(x, y - s), (x + s * 0.7, y + s * 0.85), (x - s * 0.7, y + s * 0.85)], ink);
            }
            Pick::Pen => {
                capsule(canvas, (x - s * 0.35, y + s * 0.35), (x + s * 0.75, y - s * 0.75), t * 1.9, ink);
                polygon(canvas, &[(x - s * 0.85, y + s * 0.85), (x - s * 0.52, y + s * 0.18), (x - s * 0.18, y + s * 0.52)], ink);
            }
            Pick::Eraser => {
                let (c, sn) = (std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2);
                let rot = |dx: f32, dy: f32| (x + dx * c - dy * sn, y + dx * sn + dy * c);
                polygon(canvas, &[rot(-s * 0.9, -s * 0.42), rot(s * 0.9, -s * 0.42), rot(s * 0.9, s * 0.42), rot(-s * 0.9, s * 0.42)], ink);
            }
            Pick::Select => {
                // A dashed square: each side two dashes.
                let e = s * 0.8;
                for (a, b) in [((-e, -e), (e, -e)), ((e, -e), (e, e)), ((e, e), (-e, e)), ((-e, e), (-e, -e))] {
                    let lerp = |f: f32| (x + a.0 + (b.0 - a.0) * f, y + a.1 + (b.1 - a.1) * f);
                    capsule(canvas, lerp(0.), lerp(0.3), t, ink);
                    capsule(canvas, lerp(0.7), lerp(1.), t, ink);
                }
            }
            Pick::PenColour => disc(canvas, x, y, s * 0.8, pen),
            Pick::ClearInk | Pick::ClearSel => {
                capsule(canvas, (x - s * 0.6, y - s * 0.6), (x + s * 0.6, y + s * 0.6), t * 1.2, ink);
                capsule(canvas, (x - s * 0.6, y + s * 0.6), (x + s * 0.6, y - s * 0.6), t * 1.2, ink);
            }
            Pick::Search => {}
        },
        Mark::Cap(CapAct::Copy) => {
            // Two sheets, the front one whole.
            let e = s * 0.55;
            for (ox, oy) in [(s * 0.22, s * 0.22)] {
                let (cx, cy) = (x + ox, y + oy);
                for (a, b) in [((-e, -e), (e, -e)), ((e, -e), (e, e)), ((e, e), (-e, e)), ((-e, e), (-e, -e))] {
                    capsule(canvas, (cx + a.0, cy + a.1), (cx + b.0, cy + b.1), t, ink);
                }
            }
            let (cx, cy) = (x - s * 0.22, y - s * 0.22);
            capsule(canvas, (cx - e, cy + e * 0.5), (cx - e, cy - e), t, ink);
            capsule(canvas, (cx - e, cy - e), (cx + e * 0.5, cy - e), t, ink);
        }
        Mark::Cap(CapAct::Route) => {
            capsule(canvas, (x - s * 0.8, y), (x + s * 0.5, y), t * 1.2, ink);
            polygon(canvas, &[(x + s * 0.9, y), (x + s * 0.3, y - s * 0.5), (x + s * 0.3, y + s * 0.5)], ink);
        }
        Mark::Cap(CapAct::Close) => {
            capsule(canvas, (x - s * 0.55, y - s * 0.55), (x + s * 0.55, y + s * 0.55), t * 1.2, ink);
            capsule(canvas, (x - s * 0.55, y + s * 0.55), (x + s * 0.55, y - s * 0.55), t * 1.2, ink);
        }
    }
}

/// The compass needle's north half.
const NORTH: u32 = ink(0xE0_6A_3C, 255);
/// The recording ring.
const RECORD: u32 = ink(0xD2_4A_3C, 255);

/// A bead: its mark, then a thin rim, then its ground (topmost first). Lit, its ground is the accent.
#[allow(clippy::too_many_arguments)]
fn bead(canvas: &mut Canvas, text: &mut TextRenderer, m: Mark, (x, y, r): (f32, f32, f32), lit: bool, dim: bool, pen: u32) {
    let ink = if dim { hue::readout_dim() } else { hue::text() };
    mark(canvas, text, m, x, y, r, ink, pen);
    ring(canvas, x, y, r - r * 0.03, r * 0.03, part(hue::readout_dim(), 0.5));
    disc(canvas, x, y, r, if lit { hue::slider_fill() } else { hue::gear_bg() });
}

/// A line of text centred at `(x, y)`, turned by `theta` (radians, clockwise) about that point: drawn upright into a scratch buffer, then sampled turned (bilinear on fluor's opacity and darkness, both linear in coverage), at `fade` of its strength, under what is painted.
#[allow(clippy::too_many_arguments)]
fn turned_text(canvas: &mut Canvas, text: &mut TextRenderer, s: &str, x: f32, y: f32, theta: f32, style: &TextStyle, size: f32, fade: f32) {
    let tw = text.measure_text(s, style);
    let (sw, sh) = ((tw + size * 2.) as usize, (size * 2.) as usize);
    if sw == 0 || sh == 0 {
        return;
    }
    let mut scratch = vec![0u32; sw * sh];
    {
        let mut damage = Damage::new();
        let mut c = Canvas::new(&mut scratch, sw, sh, &mut damage);
        text.draw_text_center(&mut c, s, sw as f32 * 0.5, sh as f32 * 0.5, style, None, None);
    }
    let (sn, cs) = theta.sin_cos();
    let reach = (sw as f32).hypot(sh as f32) * 0.5 + 2.;
    let (w, h) = (canvas.width, canvas.height);
    // WHY: a title near a page's edge mid-turn reaches off the frame; PROOF: an unclipped box indexes outside the buffer.
    let (x0, y0) = ((x - reach).floor().max(0.) as usize, (y - reach).floor().max(0.) as usize);
    let (x1, y1) = (((x + reach).ceil().max(0.) as usize).min(w), ((y + reach).ceil().max(0.) as usize).min(h));
    let sample = |sx: f32, sy: f32| -> [f32; 4] {
        // The scratch is read where it is: outside it, nothing.
        let (fx, fy) = (sx - 0.5, sy - 0.5);
        let (ix, iy) = (fx.floor(), fy.floor());
        let (ax, ay) = (fx - ix, fy - iy);
        let mut out = [0f32; 4];
        for (dx, dy, k) in [(0., 0., (1. - ax) * (1. - ay)), (1., 0., ax * (1. - ay)), (0., 1., (1. - ax) * ay), (1., 1., ax * ay)] {
            let (px, py) = (ix + dx, iy + dy);
            if px < 0. || py < 0. || px >= sw as f32 || py >= sh as f32 {
                continue;
            }
            let p = scratch[py as usize * sw + px as usize];
            for (c, o) in out.iter_mut().enumerate() {
                *o += ((p >> (24 - 8 * c)) & 0xFF) as f32 * k;
            }
        }
        out
    };
    for py in y0..y1 {
        for px in x0..x1 {
            let (vx, vy) = (px as f32 + 0.5 - x, py as f32 + 0.5 - y);
            // The point upright: turned back by theta.
            let (ux, uy) = (vx * cs + vy * sn, -vx * sn + vy * cs);
            let v = sample(ux + sw as f32 * 0.5, uy + sh as f32 * 0.5);
            if v[0] * fade < 1. {
                continue;
            }
            let b = |c: usize, s: u32| ((v[c] * fade) as u32) << s;
            let i = py * w + px;
            canvas.pixels[i] = canvas.pixels[i].under(b(0, 24) | b(1, 16) | b(2, 8) | b(3, 0), BlendMode::Normal);
        }
    }
}

/// A style shrunk, smoothly, so `s` fits `width`.
fn fitted(text: &mut TextRenderer, s: &str, size: f32, colour: u32, width: f32, weight: u16) -> TextStyle<'static> {
    let natural = text.measure_text(s, &TextStyle::new(size, colour).weight(weight));
    let fit = if natural > 0. { soft_cap(size, size * width / natural) } else { size };
    TextStyle::new(fit, colour).weight(weight)
}

impl Panel {
    fn circle(&self) -> SafeCircle {
        self.safe.expect("the circle's layout runs only with a safe circle")
    }

    /// The circle's places for a `w` × `h` frame (`span` its harmonic mean): the orb (in [`Self::gear`]), the beads, the title and body of a page, the floor, the tool bead; and the Layers page's tree in the body.
    pub(super) fn layout_round(&mut self, c: SafeCircle, w: f32, _h: f32, span: f32) {
        let font = self.font;
        // The unit capped softly, as Photon caps it (OrbLayout::compute): the orb and two beads a side with their gaps, 2 + 2 · (3/8 + 2) units from the orb's centre to the outer bead's edge, held under three eighths of the width, so at 400 % the orb and the beads shrink together, with no kink where the cap takes over.
        let u = soft_cap(span / 32. * self.ru, w * (3. / 8.) / (2. + 2. * (3. / 8. + 2.)));
        let orb_r = 2. * u;
        let orb_y = c.top_at(c.cx) + u * 0.5 + orb_r;
        self.gear = (c.cx, orb_y, orb_r);
        // The beads on the arc through the orb's centre, concentric with the circle (Photon's OrbLayout): each a chord from the orb, the orb's radius, a gap and the beads between.
        let arc_r = c.cy - orb_y;
        let gap = u * (3. / 8.);
        let (d1, d2) = (orb_r + gap + u, orb_r + gap + u + 2. * u + gap);
        let at = |d: f32, side: f32| {
            let t = 2. * (d / (2. * arc_r)).asin() * side;
            (c.cx + arc_r * t.sin(), c.cy - arc_r * t.cos(), u)
        };
        self.round.beads = [at(d2, -1.), at(d1, -1.), at(d1, 1.), at(d2, 1.)];
        self.round.u = u;
        let m = font * 0.8;
        self.round.margin = m;
        self.round.floor = c.bottom_at(m).min(c.bottom_at(w - m)) - u / 4.;
        self.round.title_y = orb_y + orb_r + font * 1.1;
        self.round.body_top = self.round.title_y + font * 1.3;
        // The foot: the scale's arc lowest, its words over it, the tool bead over them.
        let foot = c.cy + c.r - u * 1.25;
        self.round.tool = (c.cx, foot - u * 3.3, u * 0.9);
        // A page's body: a column the frame's width less the margins, moved by the turn.
        let dx = self.round.turn;
        self.panel_w = w + dx;
        let row = font * 1.6;
        let body_top = self.round.body_top;
        // A page's own title row: under the arc's title for a page opened from another (it goes back), hidden above the body for a page the root opens (the arc names it).
        let nested = self.stack.len() > 1;
        self.page_gear = (c.cx, if nested { body_top - font * 0.9 } else { body_top - row * 1.2 - font * 0.9 }, 0.);
        self.layers_band = (body_top, self.round.floor);
        self.theme_row = (m + dx + font * 0.8, -1.0e6, row * 0.5);
        self.layout_layers(m + dx, w - 2. * m, body_top + font * 0.3 - self.layers_scroll, false);
    }

    /// Whether the circle is turning a page or its snap is still settling: the host keeps drawing frames while it is.
    pub fn animating(&self) -> bool {
        self.round.turning || self.round.turn != 0. || self.round.settle != 0
    }

    /// One frame of a released turn's snap: half the distance left, floored (owner 2026-11-18); at rest, the page it landed on is the open one.
    pub(super) fn step_turn(&mut self, w: usize) {
        let r = &mut self.round;
        if r.turning || (r.turn == 0. && r.settle == 0) {
            return;
        }
        let target = -(r.settle as f32) * w as f32;
        let left = target - r.turn;
        if left.abs() >= 1. {
            r.turn = target - (left / 2.).trunc();
            self.dirty = true;
            return;
        }
        if r.settle != 0 {
            let at = RING.iter().position(|p| *p == self.stack[0]).unwrap_or(0) as i32;
            self.stack = vec![RING[(at + r.settle).rem_euclid(RING.len() as i32) as usize]];
            self.layers_scroll = 0.;
            self.sliders.clear();
        }
        r.turn = 0.;
        r.settle = 0;
        self.dirty = true;
    }

    /// The screen the circle's controls covered at its last paint, for the labeller to leave clear.
    pub(super) fn round_keep_out(&self) -> Vec<[f32; 4]> {
        let (w, h) = (self.w as f32, self.h as f32);
        if self.open {
            return vec![[0., 0., w, h]];
        }
        let (gx, gy, gr) = self.gear;
        let u = self.round.u;
        let mut out = vec![[gx - gr, gy - gr, gx + gr, gy + gr + u * 1.6]];
        for &(x, y, r) in self.round.beads.iter() {
            out.push([x - r, y - r, x + r, y + r]);
        }
        let (tx, ty, tr) = self.round.tool_drawn;
        out.push([tx - tr, ty - tr, tx + tr, ty + tr]);
        for &(x, y, r, _) in &self.round.bloom_beads {
            out.push([x - r, y - r, x + r, y + r]);
        }
        match self.round.cap_top {
            Some(t) => out.push([0., t, w, h]),
            None => out.push([w * 0.25, h - u * 2.4, w * 0.75, h]),
        }
        out
    }

    /// Paint the circle's screen: the orb, then the map's controls, the root menu or a page.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn paint_round(&mut self, mask: LayerMask, ctl: Controls, r: &Readouts, measure: Option<&MeasureView>, themes: &[Theme], _areas: &AreasView, orb: Option<&dyn Fn(usize) -> Option<Vec<[u8; 4]>>>) -> &[u32] {
        // The orb first: its pixels are set, not composited, so nothing may be under them yet.
        self.paint_orb(orb, false);
        let (w, h) = (self.w, self.h);
        let mut buf = std::mem::take(&mut self.buf);
        let mut damage = Damage::new();
        let mut canvas = Canvas::new(&mut buf, w, h, &mut damage);
        self.copy_btns.clear();
        self.pills.begin();
        self.strip_close = None;
        self.card_top = None;
        self.route_pills.clear();
        self.search_rows.clear();
        self.search_acts.clear();
        self.search_field = None;
        self.notice_btn = None;
        self.locate_btn = None;
        self.tool_btn = None;
        self.round.root_rows.clear();
        self.round.bloom_beads.clear();
        self.round.cap_beads.clear();
        self.round.cap_top = None;
        self.paint_recording(&mut canvas, &ctl);
        if !self.open {
            self.paint_round_map(&mut canvas, &ctl, r, measure);
        } else if self.round.root {
            self.paint_root(&mut canvas);
        } else {
            self.paint_round_page(&mut canvas, mask, ctl, r, themes);
        }
        drop(canvas);
        self.buf = buf;
        &self.buf
    }

    /// While a track records: a ring round the orb filling with the hour's minutes (full at the hour, empty again after), and the distance and time under the orb.
    fn paint_recording(&mut self, canvas: &mut Canvas, ctl: &Controls) {
        if !ctl.record {
            return;
        }
        let (gx, gy, gr) = self.gear;
        let u = self.round.u;
        let (rr, half) = (gr + u * 0.28, u * 0.07);
        let line = match &self.tracks.live {
            Some(n) if n.fixes > 0 => format!("{} \u{b7} {}", length_text(n.metres), duration_text(n.seconds)),
            _ => "waiting for a fix".to_string(),
        };
        let secs = self.tracks.live.as_ref().map_or(0., |n| n.seconds);
        let f = (secs.rem_euclid(3600.) / 3600.) as f32;
        let y = gy + gr + u * 1.9;
        let style = fitted(&mut self.text, &line, self.font * 0.8, hue::text(), u * 4., 600);
        // With a page open its title is there: the ring alone then.
        if !self.open {
            self.text.draw_text_center(canvas, &line, gx, y, &style, None, None);
        }
        if f > 0. {
            arc(canvas, gx, gy, rr, f * std::f32::consts::PI, f * std::f32::consts::PI, half, RECORD);
        }
        ring(canvas, gx, gy, rr, half, part(RECORD, 0.3));
    }

    /// The map's controls: the bloom, the tool bead, the beads, and the readout in the bottom cap or the scale on its arc.
    fn paint_round_map(&mut self, canvas: &mut Canvas, ctl: &Controls, r: &Readouts, measure: Option<&MeasureView>) {
        let c = self.circle();
        let u = self.round.u;
        let pen = pen_swatch(ctl.pen_rgba);
        let current = Pick::current(ctl);
        let (tx, ty, tr) = self.round.tool;
        // The readout raises the tool bead above the cap.
        let cap = measure.map(|m| self.cap_height(m));
        let ty = match cap {
            Some(hh) => ty.min(c.cy + c.r - u * 1.3 - hh - tr - u * 0.4),
            None => ty,
        };
        self.round.tool_drawn = (tx, ty, tr);
        if self.round.bloom {
            // The tap tools on an arc nearer the bead, the ink tools beyond, each fanned over the upper half.
            let inner = [Pick::Info, Pick::Ruler, Pick::Directions, Pick::Navigate];
            let mut outer = vec![Pick::Pen, Pick::Eraser, Pick::Select];
            match current {
                Pick::Pen => outer.push(Pick::PenColour),
                Pick::Eraser => outer.push(Pick::ClearInk),
                Pick::Select => outer.push(Pick::ClearSel),
                _ => {}
            }
            let br = u * 0.75;
            for (picks, rad, spread) in [(&inner[..], u * 2.7, 1.05f32), (&outer[..], u * 4.7, 1.1f32)] {
                let n = picks.len();
                for (i, &p) in picks.iter().enumerate() {
                    let a = if n > 1 { -spread + 2. * spread * i as f32 / (n - 1) as f32 } else { 0. };
                    self.round.bloom_beads.push((tx + rad * a.sin(), ty - rad * a.cos(), br, p));
                }
            }
            let beads = self.round.bloom_beads.clone();
            for (x, y, br, p) in beads {
                bead(canvas, &mut self.text, Mark::Tool(p), (x, y, br), p == current, false, pen);
            }
        }
        bead(canvas, &mut self.text, Mark::Tool(current), (tx, ty, tr), current != Pick::Info, false, pen);
        let marks = [Mark::Search, Mark::Layers, Mark::Locate, Mark::Compass(r.heading_deg.to_radians() as f32)];
        for (k, m) in marks.into_iter().enumerate() {
            bead(canvas, &mut self.text, m, self.round.beads[k], false, k == 2 && !r.has_fix, pen);
        }
        match measure {
            Some(m) => self.paint_cap(canvas, ctl, m),
            None => self.paint_scale_arc(canvas, r),
        }
    }

    /// The bottom cap's height for a readout: its lines and its row of beads.
    fn cap_height(&self, m: &MeasureView) -> f32 {
        let (font, u) = (self.font, self.round.u);
        let lines = if m.route {
            1 + m.info.len().min(ROUTE_LINES)
        } else if m.ruler {
            2 + m.info.len()
        } else {
            m.info.len() + 2
        };
        lines as f32 * font * 1.1 + font * 0.6 + u * 1.6
    }

    /// What a tap found, in the circle's bottom cap (the region under a chord, its lower edge the circle): the lines centred, each fitted to the chord at its height, a row of beads under them (copy the place, route there, close), then the cap's ground.
    fn paint_cap(&mut self, canvas: &mut Canvas, ctl: &Controls, m: &MeasureView) {
        let c = self.circle();
        let (font, u, w) = (self.font, self.round.u, self.w as f32);
        let bead_y = c.cy + c.r - u * 1.3;
        let top = bead_y - self.cap_height(m) + u * 0.6;
        self.round.cap_top = Some(top);
        self.card_top = Some(top);
        self.round.cap_at = m.target;
        let mut y = top + font * 0.9;
        let line = |panel: &mut Panel, canvas: &mut Canvas, y: &mut f32, s: &str, size: f32, colour: u32, weight: u16, copy: Option<String>| panel.cap_line(canvas, y, s, size, colour, weight, copy);
        if m.route {
            // The way of travel, and for directions the swap, on the cap's first row.
            let (pw, ph) = (font * 3.4, font * 1.25);
            let row_w = pw * 3. + font * 0.7 + if ctl.route == 1 { pw + font * 0.4 } else { 0. };
            let x0 = c.cx - row_w * 0.5;
            self.travel.set_font_size(font * 0.8);
            self.travel.set_selected(ctl.travel.max(1) as usize - 1);
            self.travel.set_rect(x0, y - ph * 0.5, pw * 3. + font * 0.7, ph);
            self.travel.render_content_into(canvas, &mut self.text, None, None);
            if ctl.route == 1 {
                let r = (x0 + pw * 3. + font * 1.1, y - ph * 0.5, pw, ph);
                self.pills.draw(canvas, &mut self.text, r, "Swap", false, font * 0.8, None);
                self.route_pills.push((r, 4));
            }
            y += font * 1.4;
            for (k, l) in m.info.iter().take(ROUTE_LINES).enumerate() {
                let (size, colour) = if k == 0 { (font * 0.95, hue::text()) } else { (font * 0.8, hue::readout()) };
                line(self, canvas, &mut y, l, size, colour, 400, None);
            }
        } else if m.ruler {
            let dist = length_text(m.distance_m);
            line(self, canvas, &mut y, &format!("{dist}   {}\u{b0}", m.bearing_deg.round() as i64), font * 0.95, hue::text(), 600, None);
            let e = |v: Option<f32>| v.map_or("\u{2014}".to_string(), |e| format!("{} m", e.round() as i64));
            let rise = match (m.elev_origin, m.elev_target) {
                (Some(a), Some(b)) => format!("   {:+} m", (b - a).round() as i64),
                _ => String::new(),
            };
            line(self, canvas, &mut y, &format!("{} \u{2192} {}{rise}", e(m.elev_origin), e(m.elev_target)), font * 0.8, hue::readout(), 400, None);
            for l in &m.info {
                line(self, canvas, &mut y, l, font * 0.8, hue::readout(), 400, None);
            }
        } else {
            // What is there, the first line strongest; the place in degrees and as its words, each a tap to copy.
            for (k, l) in m.info.iter().enumerate() {
                let (size, colour, weight) = if k == 0 { (font * 1.05, hue::text(), 600) } else { (font * 0.8, hue::readout(), 400) };
                line(self, canvas, &mut y, l, size, colour, weight, None);
            }
            let (lat, lon) = m.target;
            line(self, canvas, &mut y, &degrees_text(lat, lon), font * 0.8, hue::readout(), 400, Some(format!("{lat:.6}, {lon:.6}")));
            let words = vsf::types::WorldCoord::from_lat_lon(lat, lon).to_words();
            line(self, canvas, &mut y, &words, font * 0.8, hue::readout_dim(), 400, Some(words.clone()));
        }
        // The beads, centred under the lines.
        let br = u * 0.65;
        let step = br * 2. + u * 0.5;
        let acts = [CapAct::Copy, CapAct::Route, CapAct::Close];
        for (k, a) in acts.into_iter().enumerate() {
            let x = c.cx + (k as f32 - 1.) * step;
            self.round.cap_beads.push((x, bead_y, br, a));
            bead(canvas, &mut self.text, Mark::Cap(a), (x, bead_y, br), false, false, 0);
        }
        // The cap's ground: inside the circle, below its top.
        shape(canvas, [0., top - 2., w, self.h as f32], hue::panel_bg(), |px, py| ((px - c.cx).hypot(py - c.cy) - c.r).max(top - py));
    }

    /// One line of the cap at `*y`, centred, fitted to the circle's chord there less the margins; with `copy` a tap on it copies that. `*y` moves to the next line.
    #[allow(clippy::too_many_arguments)]
    fn cap_line(&mut self, canvas: &mut Canvas, y: &mut f32, s: &str, size: f32, colour: u32, weight: u16, copy: Option<String>) {
        let c = self.circle();
        let w = self.w as f32;
        // WHY: the chord's half is a root of r² − dy², negative under no line the cap lays (all inside the circle), but its edge rows at the circle's foot round to it; PROOF: a NaN width would size the text NaN.
        let half = (c.r * c.r - (*y - c.cy) * (*y - c.cy)).max(0.).sqrt();
        let chord = (2. * half).min(w) - 2. * self.round.margin;
        let style = fitted(&mut self.text, s, size, colour, chord, weight);
        let tw = self.text.measure_text(s, &style);
        self.text.draw_text_center(canvas, s, c.cx, *y, &style, None, None);
        if let Some(what) = copy {
            self.copy_btns.push(((c.cx - tw * 0.5, *y - size * 0.6, tw, size * 1.2), what));
        }
        *y += self.font * 1.1;
    }

    /// The scale: a stretch of an arc concentric with the circle above the helm, a round distance (1, 2, 3 or 5 of a power of ten metres) as long as it is, chosen to keep it between a quarter and a half of the width; its length over it, the contour interval over that.
    fn paint_scale_arc(&mut self, canvas: &mut Canvas, r: &Readouts) {
        if !(r.m_per_px > 0.) {
            return;
        }
        let c = self.circle();
        let (font, u, w) = (self.font, self.round.u, self.w as f64);
        let rad = c.r - u * 1.25;
        // The candidates around three eighths of the width; the one nearest it, as a ratio.
        let aim = w * 0.375 * r.m_per_px;
        let p = 10f64.powf(aim.log10().floor());
        let mut best = (f64::MAX, 0.);
        for scale in [p * 0.1, p, p * 10.] {
            for k in [1., 2., 3., 5.] {
                let d = scale * k;
                let off = (d / aim).ln().abs();
                if off < best.0 {
                    best = (off, d);
                }
            }
        }
        let d = best.1;
        let len = (d / r.m_per_px) as f32;
        let spread = len * 0.5 / rad;
        arc(canvas, c.cx, c.cy, rad, std::f32::consts::PI, spread, (u * 0.07).max(1.), hue::text());
        let low = c.cy + rad;
        self.text.draw_text_center(canvas, &length_text(d), c.cx, low - u * 0.78, &TextStyle::new(font * 0.85, hue::text()).weight(600), None, None);
        if let Some(cm) = r.contour_major_m {
            self.text.draw_text_center(canvas, &format!("contours {}", interval_text(cm)), c.cx, low - u * 1.5, &TextStyle::new(font * 0.65, hue::readout()), None, None);
        }
        // A soft ground under the words and the arc, so they read over any map.
        shape(canvas, [c.cx as f32 - len * 0.6 - u, low - u * 2.1, c.cx + len * 0.6 + u, low + u], part(hue::panel_bg(), 0.55), |px, py| {
            let (dx, dy) = ((px - c.cx) / (len * 0.55 + u * 0.5), (py - (low - u * 0.75)) / (u * 1.15));
            (dx.hypot(dy) - 1.) * u
        });
    }

    /// The root menu: the pages' names down the circle's axis, centred in the body, the open page's lit; the map dimmed under it all.
    fn paint_root(&mut self, canvas: &mut Canvas) {
        let c = self.circle();
        let (font, u, w, h) = (self.font, self.round.u, self.w as f32, self.h as f32);
        // A row a finger can take: about seven millimetres on the Pro at its own scale.
        let row = font * 2.8;
        let (gy, gr) = (self.gear.1, self.gear.2);
        let (top, bottom) = (gy + gr + u, self.round.floor);
        let y0 = (top + bottom) * 0.5 - row * ROOT.len() as f32 * 0.5;
        let open = self.stack[0];
        for (k, &(p, name)) in ROOT.iter().enumerate() {
            let y = y0 + row * (k as f32 + 0.5);
            self.round.root_rows.push((y, row * 0.5, p));
            self.text.draw_text_center(canvas, name, c.cx, y, &TextStyle::new(font * 1.4, hue::text()).weight(600), None, None);
            if p == open {
                let half = font * 4.;
                capsule(canvas, (c.cx - half, y), (c.cx + half, y), row * 0.36, part(hue::text(), 0.1));
            }
        }
        paint::fill_rect(canvas, 0, 0, w as isize, h as isize, hue::panel_bg(), None, None);
    }

    /// A page: its title on the arc (the neighbours' fading in as a swipe turns), its body and the neighbour's while turning, the map dimmed under it all.
    fn paint_round_page(&mut self, canvas: &mut Canvas, mask: LayerMask, ctl: Controls, r: &Readouts, themes: &[Theme]) {
        let c = self.circle();
        let (font, w, h) = (self.font, self.w as f32, self.h as f32);
        let open = self.stack[0];
        let at = RING.iter().position(|p| *p == open);
        // The titles: each a page's width of arc apart, faded as cos² of its offset (gone, with no slope, half a page out).
        let arc_r = c.cy - self.round.title_y;
        let step = w / arc_r;
        let turned = -self.round.turn / w;
        for k in -1i32..=1 {
            let o = k as f32 - turned;
            if o.abs() >= 0.5 {
                continue;
            }
            let p = match at {
                Some(i) => RING[(i as i32 + k).rem_euclid(RING.len() as i32) as usize],
                None if k == 0 => open,
                None => continue,
            };
            let name = name_of(p);
            let a = (std::f32::consts::PI * o).cos().powi(2);
            let theta = o * step;
            let (x, y) = (c.cx + arc_r * theta.sin(), c.cy - arc_r * theta.cos());
            let style = fitted(&mut self.text, name, font * 1.35, hue::text(), w * 0.75, 700);
            turned_text(canvas, &mut self.text, name, x, y, theta, &style, font * 1.35, a);
        }
        // The neighbour coming in, then the open page (its rows the ones taps find).
        let dx = self.round.turn;
        if dx != 0. {
            if let Some(i) = at {
                let k = if dx < 0. { 1 } else { -1 };
                let p = RING[(i as i32 + k).rem_euclid(RING.len() as i32) as usize];
                let ndx = dx + if dx < 0. { w } else { -w };
                self.paint_body(canvas, p, ndx, mask, ctl, r, themes);
            }
        }
        let p = self.page();
        if p == Page::Layers && dx != 0. {
            // Laid out again at its own offset, after a neighbour's layout borrowed the tree.
            let m = self.round.margin;
            self.layout_layers(m + dx, w - 2. * m, self.round.body_top + font * 0.3 - self.layers_scroll, false);
        }
        self.paint_body(canvas, p, dx, mask, ctl, r, themes);
        paint::fill_rect(canvas, 0, 0, w as isize, h as isize, hue::panel_bg(), None, None);
    }

    /// One page's body in the column moved by `dx`.
    #[allow(clippy::too_many_arguments)]
    fn paint_body(&mut self, canvas: &mut Canvas, p: Page, dx: f32, mask: LayerMask, ctl: Controls, r: &Readouts, themes: &[Theme]) {
        let (font, w, h) = (self.font, self.w as f32, self.h);
        let m = self.round.margin;
        let (ox, cw) = (m + dx, w - 2. * m);
        let gear = self.page_gear;
        let current = themes.get(ctl.theme);
        match p {
            Page::Layers => {
                if dx != 0. && self.page() != Page::Layers {
                    self.layout_layers(ox, cw, self.round.body_top + font * 0.3, false);
                }
                self.paint_layers(canvas, mask, ctl, r, themes, false);
            }
            Page::Themes => self.theme_rows = Self::paint_themes(canvas, &mut self.text, &mut self.pills, &mut self.theme_list, font, ox, cw, gear, themes, ctl.theme, false),
            Page::Edit => {
                if let Some(t) = current {
                    let (rows, extent) = Self::paint_edit(canvas, &mut self.text, font, ox, cw, gear, h, t, self.edit_scroll);
                    self.edit_rows = rows;
                    self.edit_extent = extent;
                }
            }
            Page::Field(f) => Self::paint_field(canvas, &mut self.text, font, ox, cw, gear, f, &mut self.sliders, self.field_vals, current),
            Page::Pen => Self::paint_pen(canvas, &mut self.text, font, ox, cw, gear, &mut self.sliders, self.field_vals, if self.colour_of_track { "Track colour" } else { "Pen colour" }),
            Page::Areas => self.area_rows = Self::paint_areas(canvas, &mut self.text, &mut self.pills, &mut self.area_list, font, ox, cw, gear, &self.areas, false),
            Page::Area(i) => self.area_rows = Self::paint_area(canvas, &mut self.text, &mut self.pills, font, ox, cw, gear, &self.areas, i, &mut self.sliders, self.field_vals[0] as u8),
            Page::Tracks => self.track_rows = Self::paint_tracks(canvas, &mut self.text, &mut self.track_list, font, ox, cw, gear, &self.tracks, ctl.record, false),
            Page::Track(i) => self.track_rows = Self::paint_track(canvas, &mut self.text, &mut self.pills, font, ox, cw, gear, &self.tracks, i),
            Page::Updates => self.update_rows = Self::paint_updates(canvas, &mut self.text, &mut self.pills, &mut self.auto_check, font, ox, cw, gear, &self.updates, false),
            Page::Search => {
                let (fx, fy, fw, fh) = (ox + font * 0.9, self.round.body_top + font * 0.6, cw - font * 1.8, font * 1.7);
                let hits = &mut self.hits;
                let tb = self.search_box.get_or_insert_with(|| Textbox::new(hits, fx + fw * 0.5, fy + fh * 0.5, fw, fh, font));
                tb.set_rect(fx + fw * 0.5, fy + fh * 0.5, fw, fh);
                tb.set_font_size(font, &mut self.text);
                if tb.chars.iter().collect::<String>() != self.query {
                    tb.set_text(&self.query, &mut self.text);
                }
                tb.set_focused(true);
                let search_gear = (gear.0, self.round.body_top - font * 0.9 - font * 1.6 * 1.3 - font * 0.6 + font * 0.6, 0.);
                let card = Self::paint_search_page(canvas, &mut self.text, &mut self.pills, &mut self.search_list, tb, font, ox, cw, self.round.floor as usize, search_gear, &self.found, self.picked, self.search_pending, self.query.trim().is_empty(), false);
                self.search_field = Some(card.field);
                self.search_rows = card.rows;
                self.search_acts = card.acts;
            }
            Page::Here => {
                self.paint_readouts(canvas, ox + font * 0.9, ox + cw - font * 0.9, self.round.body_top + font * 0.6, r);
            }
            Page::Device => {
                self.cache_max = r.cache_max;
                self.slider = (ox + font * 0.9, self.round.body_top + font * 2.0, cw - font * 1.8);
                Self::paint_slider(canvas, &mut self.text, font, self.slider, &ctl, r);
                let small = TextStyle::new(font * 0.82, hue::readout());
                let dim = TextStyle::new(font * 0.72, hue::readout_dim());
                let mut y = self.slider.1 + font * 1.6;
                for (label, value) in [("scale", format!("{:.2} m/px", r.m_per_px)), ("frame", format!("{:.1} ms  \u{b7}  {} cells", r.frame_ms, r.resident))] {
                    self.text.draw_text_left(canvas, label, ox + font * 0.9, y, &dim, None, None);
                    self.text.draw_text_right(canvas, &value, ox + cw - font * 0.9, y, &small, None, None);
                    y += font * 1.15;
                }
            }
        }
    }

    /// A tap on the circle's screen: Some(taken) when the circle answered it, None for a page's body, which the pages' own taps answer.
    pub(super) fn tap_round(&mut self, x: f32, y: f32, ctl: &mut Controls) -> Option<bool> {
        let near = |(bx, by, br): (f32, f32, f32), k: f32| (x - bx).powi(2) + (y - by).powi(2) <= (br * k).powi(2);
        // A turn in flight takes no taps.
        if self.animating() {
            return Some(true);
        }
        self.round.press = None;
        if self.round.bloom {
            self.round.bloom = false;
            self.dirty = true;
            if let Some(&(_, _, _, p)) = self.round.bloom_beads.iter().find(|&&(bx, by, br, _)| near((bx, by, br), 1.15)) {
                self.pick(p, ctl);
                self.round.root = false;
            }
            return Some(true);
        }
        if near(self.gear, 1.1) {
            match (self.open, self.round.root) {
                (false, _) => {
                    self.open = true;
                    self.round.root = true;
                }
                (true, true) => {
                    self.open = false;
                    self.round.root = false;
                }
                (true, false) => self.to_root(),
            }
            self.dirty = true;
            return Some(true);
        }
        if !self.open {
            for (k, &b) in self.round.beads.iter().enumerate() {
                if !near(b, 1.15) {
                    continue;
                }
                match k {
                    0 => self.open_page(Page::Search, ctl),
                    1 => self.open_page(Page::Layers, ctl),
                    2 => ctl.locate = true,
                    _ => ctl.north = true,
                }
                self.dirty = true;
                return Some(true);
            }
            if near(self.round.tool_drawn, 1.15) {
                self.round.bloom = true;
                self.dirty = true;
                return Some(true);
            }
            if let Some(&(_, _, _, a)) = self.round.cap_beads.iter().find(|&&(bx, by, br, _)| near((bx, by, br), 1.15)) {
                let (lat, lon) = self.round.cap_at;
                match a {
                    CapAct::Copy => self.copied = Some(format!("{lat:.6}, {lon:.6}")),
                    CapAct::Route => self.search_act = Some(SearchAct::Go(lat, lon)),
                    CapAct::Close => self.clear_measure = true,
                }
                self.dirty = true;
                return Some(true);
            }
            if let Some((_, what)) = self.copy_btns.iter().find(|(r, _)| x >= r.0 && x <= r.0 + r.2 && y >= r.1 && y <= r.1 + r.3) {
                self.copied = Some(what.clone());
                return Some(true);
            }
            if let Some(&(_, b)) = self.route_pills.iter().find(|(r, _)| x >= r.0 && x <= r.0 + r.2 && y >= r.1 && y <= r.1 + r.3) {
                if b == 4 {
                    self.swap = true;
                }
                self.dirty = true;
                return Some(true);
            }
            if self.round.cap_top.is_some() && self.travel.click_at(x, y) {
                if let Some(i) = self.travel.take_change() {
                    ctl.travel = i as u8 + 1;
                }
                self.dirty = true;
                return Some(true);
            }
            // The cap is the readout's, not the map's.
            return Some(self.round.cap_top.is_some_and(|t| y >= t));
        }
        if self.round.root {
            let Some(&(_, _, p)) = self.round.root_rows.iter().find(|&&(cy, hh, _)| (y - cy).abs() <= hh) else { return Some(false) };
            self.open_page(p, ctl);
            self.dirty = true;
            return Some(true);
        }
        // A page: its title on the arc opens the root; the body is the page's, and a press there may become a swipe.
        if y < self.round.body_top {
            self.to_root();
            self.dirty = true;
            return Some(true);
        }
        self.round.press = Some((x, y));
        None
    }

    /// Open `p` as a page of its own (from the root or a bead), its search under way when it is the search.
    fn open_page(&mut self, p: Page, ctl: &mut Controls) {
        if self.page() == Page::Search && p != Page::Search {
            self.pop();
        }
        self.open = true;
        self.round.root = false;
        self.stack = vec![p];
        self.layers_scroll = 0.;
        self.sliders.clear();
        if p == Page::Search {
            ctl.tool = 0;
            ctl.ruler = false;
            ctl.search = true;
        }
    }

    /// Up to the root menu, a search ended on the way.
    fn to_root(&mut self) {
        if self.page() == Page::Search {
            self.pop();
        }
        self.round.root = true;
        self.layers_press = None;
        self.press = None;
    }

    /// A move while pressed on a page's body: a sideways move turns the page (Some(true)); None leaves it to the page.
    pub(super) fn drag_round(&mut self, x: f32, y: f32) -> Option<bool> {
        let (px, py) = self.round.press?;
        if !self.open || self.round.root || self.stack.len() != 1 || self.page() == Page::Search {
            return None;
        }
        let (dx, dy) = (x - px, y - py);
        if !self.round.turning {
            if dx.abs() < self.font * 0.6 || dx.abs() < 2. * dy.abs() {
                return None;
            }
            // A swipe: what the press began on the page is let go.
            self.round.turning = true;
            self.layers_press = None;
            self.press = None;
            self.held = None;
            self.slider_held = false;
        }
        self.round.turn = dx;
        self.dirty = true;
        Some(true)
    }

    /// The finger lifted: a turn past a sixth of the width settles on the next page that way, a shorter one back where it was. True when a turn ended.
    pub(super) fn release_round(&mut self) -> bool {
        self.round.press = None;
        if !self.round.turning {
            return false;
        }
        self.round.turning = false;
        let w = self.w as f32;
        self.round.settle = if self.round.turn.abs() > w / 6. { if self.round.turn < 0. { 1 } else { -1 } } else { 0 };
        self.dirty = true;
        true
    }

    /// Back on the circle: the bloom folded, then a page's parent, then the root, then the map; with the map showing, the readout closed. False when there was nothing to go back from.
    pub(super) fn back_round(&mut self) -> bool {
        self.layers_press = None;
        self.press = None;
        self.dirty = true;
        if self.round.bloom {
            self.round.bloom = false;
            return true;
        }
        if !self.open {
            if self.round.cap_top.take().is_some() {
                self.clear_measure = true;
                return true;
            }
            return false;
        }
        if self.round.root {
            self.round.root = false;
            self.open = false;
            return true;
        }
        if self.stack.len() > 1 {
            self.pop();
        } else {
            self.to_root();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Pro's frame and circle: the beads inside the circle at every scale from a quarter to four times, clear of each other and of the orb.
    #[test]
    fn the_beads_stay_inside_the_circle_at_every_scale() {
        let c = SafeCircle { cx: 672., cy: 118.56 + 1436.72, r: 1436.72 };
        for ru in [0.25, 0.5, 1., 2., 4.] {
            let mut p = Panel::new();
            p.set_safe(Some(c));
            p.set_ru(ru);
            p.layout(1344, 2992);
            let (gx, gy, gr) = p.gear;
            for &(x, y, r) in &p.round.beads {
                assert!((x - c.cx).hypot(y - c.cy) + r < c.r, "bead inside the circle at {ru}");
                assert!((x - gx).hypot(y - gy) >= gr + r, "bead clear of the orb at {ru}");
                assert!(x - r >= 0. && x + r <= 1344., "bead on the glass at {ru}");
            }
        }
    }

    /// A released turn snaps by half the distance left a frame, floored, and lands on the next page.
    #[test]
    fn a_turn_snaps_by_halves_onto_the_next_page() {
        let mut p = Panel::new();
        p.set_safe(Some(SafeCircle { cx: 672., cy: 1555.28, r: 1436.72 }));
        p.layout(1344, 2992);
        p.open = true;
        p.round.turn = -400.;
        p.round.settle = 1;
        let mut frames = 0;
        while p.animating() {
            p.step_turn(1344);
            frames += 1;
            assert!(frames < 64, "the snap ends");
        }
        assert!(p.page() == Page::Themes, "Layers turned left brings Theme");
        assert_eq!(p.round.turn, 0.);
    }
}

impl Panel {
    /// Hold a page's turn at `px` as a finger would, for a headless check of the titles and bodies mid-swipe.
    pub fn show_turn(&mut self, px: f32) {
        self.round.turning = px != 0.;
        self.round.turn = px;
        self.dirty = true;
    }
}
