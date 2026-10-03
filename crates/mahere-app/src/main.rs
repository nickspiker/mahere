//! mahere desktop app — fluor host, CPU rendering.
//!
//! v0 smoke test: an equirectangular world view colored by mahere-coord's 10
//! icosahedral diamonds, checker-shaded by Morton quadtree cells at two
//! depths, with Seattle's depth-7 cell highlighted. Every pixel runs the real
//! encoder, so this window is a visual test of the coordinate system before
//! any map data flows.

use fluor::coord::Coord as Px;
use fluor::event::{CursorIcon, Event as FEvent};
use fluor::host::app::{Context, EventResponse, FluorApp, run_app};
use fluor::paint::pack_argb;
use mahere_coord::Coord;

/// One distinct base color per diamond, visible RGB.
const DIAMOND_RGB: [[u8; 3]; 10] = [
    [86, 139, 191],
    [191, 134, 86],
    [107, 168, 112],
    [168, 107, 158],
    [191, 180, 86],
    [86, 180, 180],
    [191, 97, 97],
    [120, 120, 190],
    [150, 168, 92],
    [168, 128, 168],
];

struct MahereApp {
    /// Cached packed world image at the current viewport size.
    cache: Vec<u32>,
    cache_w: usize,
    cache_h: usize,
    seattle: Coord,
}

impl MahereApp {
    fn new() -> Self {
        MahereApp {
            cache: Vec::new(),
            cache_w: 0,
            cache_h: 0,
            seattle: Coord::from_lat_lon(47.6062, -122.3321),
        }
    }

    /// Rasterize the world into the cache: letterboxed 2:1 equirectangular,
    /// one `Coord::from_lat_lon` per map pixel.
    fn rebuild(&mut self, w: usize, h: usize) {
        let bg = pack_argb(14, 16, 19, 255);
        self.cache.clear();
        self.cache.resize(w * h, bg);
        self.cache_w = w;
        self.cache_h = h;
        if w == 0 || h == 0 {
            return;
        }
        let map_w = w.min(h * 2);
        let map_h = map_w / 2;
        let x0 = (w - map_w) / 2;
        let y0 = (h - map_h) / 2;
        let seattle_cell = self.seattle.cell(7);
        for py in 0..map_h {
            let lat = 90. - (py as f64 + 0.5) / map_h as f64 * 180.;
            let row = &mut self.cache[(y0 + py) * w + x0..(y0 + py) * w + x0 + map_w];
            for (px, out) in row.iter_mut().enumerate() {
                let lon = (px as f64 + 0.5) / map_w as f64 * 360. - 180.;
                let c = Coord::from_lat_lon(lat, lon);
                let [mut r, mut g, mut b] = DIAMOND_RGB[c.diamond() as usize];
                // Checker shading from Morton cells at depths 3 and 6 makes
                // the quadtree nesting visible.
                let (iu, iv) = c.uv();
                let coarse = ((iu >> 27) ^ (iv >> 27)) & 1;
                let fine = ((iu >> 24) ^ (iv >> 24)) & 1;
                let scale = 200 + 25 * coarse + 30 * fine; // 200..=255 of 255
                r = ((r as u64 * scale) / 255) as u8;
                g = ((g as u64 * scale) / 255) as u8;
                b = ((b as u64 * scale) / 255) as u8;
                if seattle_cell.contains(c) {
                    (r, g, b) = (255, 70, 70);
                }
                *out = pack_argb(r, g, b, 255);
            }
        }
    }
}

impl FluorApp for MahereApp {
    type UserEvent = ();

    fn title(&self) -> &str {
        "mahere"
    }

    fn init(&mut self, _ctx: &mut Context) {}

    fn on_resize(&mut self, _width: u32, _height: u32, _ctx: &mut Context) {
        // Invalidate; render() rebuilds at the new size.
        self.cache_w = 0;
        self.cache_h = 0;
    }

    fn on_event(&mut self, event: &FEvent, _ctx: &mut Context) -> EventResponse {
        match event {
            // Host defaults are right for everything v0 does: background
            // drag moves the window, close request exits.
            _ => EventResponse::Pass,
        }
    }

    fn render(&mut self, target: &mut [u32], ctx: &mut Context) {
        let w = ctx.viewport.width_px as usize;
        let h = ctx.viewport.height_px as usize;
        if (w, h) != (self.cache_w, self.cache_h) {
            self.rebuild(w, h);
        }
        let n = (w * h).min(target.len()).min(self.cache.len());
        target[..n].copy_from_slice(&self.cache[..n]);
    }

    fn cursor_for(&self, _x: Px, _y: Px, _ctx: &Context) -> CursorIcon {
        CursorIcon::Default
    }
}

fn main() {
    run_app(MahereApp::new()).expect("fluor event loop failed");
}
