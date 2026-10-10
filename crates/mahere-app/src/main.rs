//! mahere desktop app — a thin fluor frontend over mahere-engine's MapCore.
//! Chrome, desktop input mapping, and the bg-layer blit live here; all map state and rendering live in the engine (shared with the Android shell).

use fluor::coord::Coord as Px;
use fluor::event::{CursorIcon, ElementState, Event as FEvent, MouseButton, MouseScrollDelta};
use fluor::geom::Viewport;
use fluor::host::app::{Context, EventResponse, FluorApp, run_app};
use fluor::host::chrome::{self, HIT_NONE, ResizeEdge};
use fluor::host::chrome_widget::DefaultChrome;
use fluor::paint::{Clip, HitId, pack_argb};
use mahere_engine::residency::{CellStore, DirStore, HttpStore, TieredStore, DEFAULT_CELLS_URL};
use mahere_engine::{Camera, MapCore};
use mahere_panel::{Controls, Panel, Readouts};
use std::time::Instant;

struct MahereApp {
    map: MapCore,
    chrome: DefaultChrome,
    dragging: bool,
    /// Pixels the cursor travelled since the press: a few at release is a click on the map, a measurement.
    travel: f64,
    /// Right-button drag: rotate about the screen centre.
    rotating: bool,
    last_cursor: (f64, f64),
    store: Option<std::sync::Arc<mahere_store::FlatStorage>>,
    cells: Option<std::sync::Arc<mahere_store::VaultCells>>,
    last_save: Instant,
    panel: Panel,
    cache_budget: u64,
}

impl MahereApp {
    /// The editor's requests: a built-in is copied before it is edited, a field change is live, and closing the editor writes the theme to the vault.
    fn apply_theme_edits(&mut self) {
        for e in self.panel.take_theme_edits() {
            match e {
                mahere_panel::ThemeEdit::Set(f, v) => self.map.edit_theme(f, v),
                mahere_panel::ThemeEdit::Done | mahere_panel::ThemeEdit::Duplicate | mahere_panel::ThemeEdit::Reset => {
                    if e == mahere_panel::ThemeEdit::Duplicate {
                        self.map.duplicate_theme();
                    }
                    if e == mahere_panel::ThemeEdit::Reset && !self.map.reset_theme() {
                        continue;
                    }
                    if let Some(store) = &self.store {
                        mahere_store::themes::save(store, self.map.current_theme());
                    }
                }
                mahere_panel::ThemeEdit::Delete => {
                    if let (Some(name), Some(store)) = (self.map.delete_theme(), &self.store) {
                        mahere_store::themes::delete(store, &name);
                    }
                }
            }
        }
    }

    fn new(map: MapCore) -> Self {
        let mut hit_counter: HitId = HIT_NONE;
        // The app orb: the paper-craft relief map (icon.jpeg, cropped to its disc), bundled as fluor's 256×256 VSF orb. A decode failure is non-fatal — the chrome just draws no orb.
        let orb = fluor::host::icon::Icon::from_vsf_bytes(include_bytes!("../assets/mahere_orb.vsf")).ok();
        let chrome = DefaultChrome::new(
            Viewport::new(1280, 800),
            "mahere",
            orb,
            Some("drag pan · right-drag rotate · wheel zoom · A/D W/S sun · Q/E rotate · R home · 1-4 layers · 5 imagery · 6 contours · 7 slope · 8 infrared".to_string()),
            &mut hit_counter,
        );
        MahereApp {
            map,
            chrome,
            dragging: false,
            travel: 0.0,
            rotating: false,
            last_cursor: (0., 0.),
            store: None,
            cells: None,
            last_save: Instant::now(),
            panel: Panel::new(),
            cache_budget: 2 << 30,
        }
    }

    fn dims(ctx: &Context) -> (usize, usize) {
        (ctx.viewport.width_px as usize, ctx.viewport.height_px as usize)
    }
}

impl FluorApp for MahereApp {
    type UserEvent = ();

    fn title(&self) -> &str {
        "mahere"
    }

    /// The OS window / taskbar icon: the same orb the chrome draws.
    fn window_icon(&self) -> Option<&fluor::host::icon::Icon> {
        self.chrome.app_icon.as_ref()
    }

    fn init(&mut self, ctx: &mut Context) {
        self.chrome.resize(ctx.viewport);
    }

    /// Pinch zooms the MAP, not fluor's UI scale.
    fn owns_zoom_gesture(&self) -> bool {
        true
    }

    fn on_zoom(&mut self, factor: f32, anchor_x: Px, anchor_y: Px, ctx: &mut Context) {
        let (w, h) = Self::dims(ctx);
        self.map.zoom_about(
            (factor as f64).clamp(0.5, 2.0),
            anchor_x as f64,
            anchor_y as f64,
            w,
            h,
        );
        ctx.window.request_redraw();
    }

    fn on_resize(&mut self, width: u32, height: u32, ctx: &mut Context) {
        self.chrome.resize(ctx.viewport);
        self.chrome.set_full_edge(ctx.is_maximized);
        self.map.camera_moved(width as usize, height as usize);
    }

    fn hit_test_map(&self) -> Option<(&[HitId], usize, usize)> {
        let (w, h) = self.chrome.dims();
        Some((self.chrome.hit_test_map(), w, h))
    }

    /// Hover tints are applied by the host's overlay pass from this table.
    fn overlay_deltas(&mut self) -> Vec<u32> {
        let mut t = vec![0u32; 5];
        for id in [
            self.chrome.min_btn.id(),
            self.chrome.max_btn.id(),
            self.chrome.close_btn.id(),
        ] {
            if self.chrome.hit_at(self.last_cursor.0 as Px, self.last_cursor.1 as Px) == id {
                if let Some(d) = self.chrome.hover_colour_for(id) {
                    t[id as usize] = d;
                }
            }
        }
        t
    }

    fn on_event(&mut self, event: &FEvent, ctx: &mut Context) -> EventResponse {
        let (w, h) = Self::dims(ctx);
        match event {
            FEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left } => {
                // ALL cursor positions come from host-tracked ctx.cursor_*:
                // raw event coords arrive in logical points on macOS.
                let (cx, cy) = (ctx.cursor_x, ctx.cursor_y);
                self.last_cursor = (cx as f64, cy as f64);
                let hit = self.chrome.hit_at(cx, cy);
                if hit != HIT_NONE {
                    return if hit == self.chrome.close_btn.id() {
                        EventResponse::Close
                    } else if hit == self.chrome.min_btn.id() {
                        EventResponse::Minimize
                    } else if hit == self.chrome.max_btn.id() {
                        EventResponse::ToggleMaximized
                    } else {
                        EventResponse::Handled
                    };
                }
                let edge = chrome::get_resize_edge(ctx.viewport, cx, cy);
                if edge != ResizeEdge::None {
                    return EventResponse::StartResize(edge);
                }
                if cy < chrome::strip_height(ctx.viewport) {
                    return EventResponse::StartWindowDrag;
                }
                let mut mask = self.map.layers();
                let mut ctl = Controls { real_sun: self.map.real_sun, real_light: self.map.real_light, follow_heading: self.map.follow_heading, lock_to_fix: self.map.lock_to_fix, cache_budget: self.cache_budget, theme: self.map.theme, compressed: self.map.compressed() };
                let was_open = self.panel.is_open();
                if self.panel.tap(cx as f32, cy as f32, w, h, &mut mask, &mut ctl, &self.map.themes) {
                    self.cache_budget = ctl.cache_budget;
                    self.apply_theme_edits();
                    if ctl.theme != self.map.theme {
                        self.map.set_theme(ctl.theme);
                    }
                    if self.panel.take_clear_measure() {
                        self.map.clear_measure();
                    }
                    self.map.set_layers(mask);
                    if ctl.real_sun != self.map.real_sun {
                        self.map.set_real_sun(ctl.real_sun);
                    }
                    if ctl.real_light != self.map.real_light {
                        self.map.set_real_light(ctl.real_light);
                    }
                    if ctl.compressed != self.map.compressed() {
                        self.map.set_compressed(ctl.compressed);
                    }
                    if ctl.follow_heading != self.map.follow_heading {
                        self.map.set_follow_heading(ctl.follow_heading);
                    }
                    if ctl.lock_to_fix != self.map.lock_to_fix {
                        self.map.set_lock_to_fix(ctl.lock_to_fix);
                    }
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }
                // With the panel open the map takes no gesture: a click on it closes the panel and nothing else.
                if was_open {
                    self.panel.close();
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }
                self.dragging = true;
                self.travel = 0.0;
                EventResponse::Handled
            }
            FEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left } => {
                self.panel.release();
                // A click that never travelled is a tap: a new measurement to that point.
                if self.dragging && self.travel < 6.0 {
                    self.map.tap(ctx.cursor_x as f64, ctx.cursor_y as f64, w, h);
                    ctx.window.request_redraw();
                }
                self.dragging = false;
                EventResponse::Handled
            }
            FEvent::MouseInput { state, button: MouseButton::Right } => {
                self.rotating = *state == ElementState::Pressed;
                self.last_cursor = (ctx.cursor_x as f64, ctx.cursor_y as f64);
                EventResponse::Handled
            }
            FEvent::CursorMoved { .. } => {
                let (hx, hy) = (ctx.cursor_x, ctx.cursor_y);
                if self.chrome.set_hover(self.chrome.hit_at(hx, hy)) {
                    ctx.window.request_redraw();
                }
                let (x, y) = (hx as f64, hy as f64);
                let mut ctl = Controls { real_sun: self.map.real_sun, real_light: self.map.real_light, follow_heading: self.map.follow_heading, lock_to_fix: self.map.lock_to_fix, cache_budget: self.cache_budget, theme: self.map.theme, compressed: self.map.compressed() };
                if self.panel.drag(hx, hy, &mut ctl) {
                    self.cache_budget = ctl.cache_budget;
                    self.apply_theme_edits();
                    ctx.window.request_redraw();
                    self.last_cursor = (x, y);
                    return EventResponse::Handled;
                }
                if self.dragging {
                    let (dx, dy) = (x - self.last_cursor.0, y - self.last_cursor.1);
                    self.travel += dx.abs() + dy.abs();
                    self.map.pan(dx, dy, w, h);
                    ctx.window.request_redraw();
                } else if self.rotating {
                    // Angle swept by the cursor around the screen centre, so the map turns with the hand instead of at a fixed rate.
                    let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
                    let a0 = (self.last_cursor.1 - cy).atan2(self.last_cursor.0 - cx);
                    let a1 = (y - cy).atan2(x - cx);
                    let mut da = a1 - a0;
                    if da > std::f64::consts::PI {
                        da -= std::f64::consts::TAU;
                    } else if da < -std::f64::consts::PI {
                        da += std::f64::consts::TAU;
                    }
                    self.map.rotate_view(-da);
                    self.map.camera_moved(w, h);
                    ctx.window.request_redraw();
                }
                self.last_cursor = (x, y);
                EventResponse::Handled
            }
            FEvent::CursorLeft => {
                self.dragging = false;
                self.rotating = false;
                if self.chrome.set_hover(HIT_NONE) {
                    ctx.window.request_redraw();
                }
                EventResponse::Handled
            }
            FEvent::Focused(focused) => {
                if self.chrome.set_focused(*focused) {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }
            FEvent::MouseWheel { delta } => {
                // Trackpads deliver Pixels with momentum; clamp per event.
                let notches = (match delta {
                    MouseScrollDelta::Lines(_, y) => *y,
                    MouseScrollDelta::Pixels(_, y) => *y / 120.,
                } as f64)
                    .clamp(-3.0, 3.0);
                self.map.zoom_about(
                    1.18_f64.powf(notches),
                    ctx.cursor_x as f64,
                    ctx.cursor_y as f64,
                    w,
                    h,
                );
                ctx.window.request_redraw();
                EventResponse::Handled
            }
            FEvent::KeyboardInput { event } => {
                if event.state != ElementState::Pressed {
                    return EventResponse::Pass;
                }
                let handled = match event.text.as_deref() {
                    Some("a") => {
                        self.map.adjust_sun(-15.0, 0.0);
                        true
                    }
                    Some("d") => {
                        self.map.adjust_sun(15.0, 0.0);
                        true
                    }
                    Some("w") => {
                        self.map.adjust_sun(0.0, 5.0);
                        true
                    }
                    Some("s") => {
                        self.map.adjust_sun(0.0, -5.0);
                        true
                    }
                    Some("q") => {
                        self.map.rotate_view(15f64.to_radians());
                        self.map.camera_moved(w, h);
                        true
                    }
                    Some("e") => {
                        self.map.rotate_view(-15f64.to_radians());
                        self.map.camera_moved(w, h);
                        true
                    }
                    Some("r") => {
                        self.map.go_home(w, h);
                        true
                    }
                    // The layer filter: 1 terrain, 2 land cover, 3 water, 4 lines; 0 shows residency (fallback tint).
                    Some(k @ ("1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "0")) => {
                        let mut m = self.map.layers();
                        match k {
                            "1" => m.dem = !m.dem,
                            "2" => m.land = !m.land,
                            "3" => m.water = !m.water,
                            "4" => m.line = !m.line,
                            "5" => m.imagery = !m.imagery,
                            "6" => m.contours = !m.contours,
                            "7" => m.slope = !m.slope,
                            "8" => m.infrared = !m.infrared,
                            _ => m.debug = !m.debug,
                        }
                        self.map.set_layers(m);
                        true
                    }
                    _ => false,
                };
                if handled {
                    self.chrome.set_status_text(Some(format!(
                        "sun {:.0}° az / {:.0}° alt · contour {:.0} m · R home",
                        self.map.sun_az, self.map.sun_alt, self.map.contour_interval
                    )));
                    ctx.window.request_redraw();
                    EventResponse::Handled
                } else {
                    EventResponse::Pass
                }
            }
            _ => EventResponse::Pass,
        }
    }

    fn wake_at(&self) -> Option<Instant> {
        if self.map.converged() { None } else { Some(Instant::now()) }
    }

    fn tick(&mut self, ctx: &mut Context) -> bool {
        let (w, h) = Self::dims(ctx);
        // Session persistence rides the refinement ticks: at most one vault write per few seconds, only while the view is actually changing.
        if self.last_save.elapsed().as_secs() >= 1 {
            self.last_save = Instant::now();
            self.chrome.set_status_text(Some(format!(
                "{:.1} ms/frame · sun {:.0}°/{:.0}° · Q/E rotate · R home",
                self.map.last_frame_ms, self.map.sun_az, self.map.sun_alt
            )));
            if let Some(store) = &self.store {
                let c = &self.map.cam;
                let _ = mahere_store::save_session(
                    store,
                    &mahere_store::Session {
                        lat: c.lat(),
                        lon: c.lon(),
                        ppd: c.ppd,
                        bearing: c.bearing(),
                        sun_az: self.map.sun_az as f64,
                        sun_alt: self.map.sun_alt as f64,
                    },
                );
            }
        }
        self.map.tick(w, h)
    }

    fn render(&mut self, target: &mut [u32], ctx: &mut Context) {
        let (w, h) = Self::dims(ctx);
        if self.map.needs_render(w, h) {
            self.map.render(w, h);
            self.chrome.invalidate_bg();
        }
        // The panel over the map, then fluor composites front-to-back: the map is the chrome group's BACKGROUND layer, never painted straight over `target`.
        let c = self.map.cam;
        let mut heading = c.bearing().to_degrees().rem_euclid(360.0);
        if heading > 180.0 {
            heading -= 360.0;
        }
        let used = self.cells.as_ref().map_or(0, |c| c.cached_bytes());
        let readouts = Readouts { lat: c.lat(), lon: c.lon(), elev: self.map.elevation_at(c.lat(), c.lon()), heading_deg: heading, m_per_px: 111_320.0 / c.ppd, frame_ms: self.map.last_frame_ms, resident: self.map.pool().map.len(), phone_heading: None, cache_used: used, cache_max: used + (8u64 << 30) };
        let measure = self.map.measure_view(w, h, Panel::strip_samples(w));
        let themes = self.map.themes.clone();
        self.panel.set_ru(ctx.viewport.ru);
        self.panel.paint(w, h, self.map.layers(), Controls { real_sun: self.map.real_sun, real_light: self.map.real_light, follow_heading: self.map.follow_heading, lock_to_fix: self.map.lock_to_fix, cache_budget: self.cache_budget, theme: self.map.theme, compressed: self.map.compressed() }, &readouts, measure.as_ref(), &themes, Some(&|n: usize| self.map.orb(n)));
        let mut map = self.map.canvas.clone();
        self.panel.composite_rgb(&mut map, w, h);
        self.chrome.rasterize_bg(ctx.damage, |c| {
            let n = c.pixels.len().min(map.len());
            for (out, &rgb) in c.pixels[..n].iter_mut().zip(&map[..n]) {
                *out = pack_argb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8, 255);
            }
        });
        self.chrome.rasterize_perimeter(target, w, h, ctx.clip_mask);
        self.chrome.rasterize_chrome(ctx.damage, ctx.text, ctx.clip_mask);
        let clip = ctx.damage_clip;
        self.chrome
            .flatten_into(target, w, h, Some(Clip::new(clip.x0, clip.y0, clip.x1, clip.y1)));
    }

    fn cursor_for(&self, x: Px, y: Px, ctx: &Context) -> CursorIcon {
        let hit = self.chrome.hit_at(x, y);
        if self.chrome.owns_hit(hit) && hit != self.chrome.app_icon_btn.id() {
            return CursorIcon::Pointer;
        }
        match chrome::get_resize_edge(ctx.viewport, x, y) {
            ResizeEdge::Top | ResizeEdge::Bottom => CursorIcon::NsResize,
            ResizeEdge::Left | ResizeEdge::Right => CursorIcon::EwResize,
            ResizeEdge::TopLeft | ResizeEdge::BottomRight => CursorIcon::NwseResize,
            ResizeEdge::TopRight | ResizeEdge::BottomLeft => CursorIcon::NeswResize,
            ResizeEdge::None => CursorIcon::Default,
        }
    }
}

fn main() {
    let vault = mahere_store::open(None).ok();
    // A local bake (argv[1], or data/cells) serves directly; otherwise cells come from the bucket through the vault, like the phone.
    let local = std::env::args().nth(1).map(std::path::PathBuf::from).unwrap_or_else(|| "data/cells".into());
    let mut cells: Option<std::sync::Arc<mahere_store::VaultCells>> = None;
    let store: std::sync::Arc<dyn CellStore> = if local.is_dir() {
        eprintln!("cells: {}", local.display());
        std::sync::Arc::new(DirStore(local))
    } else {
        eprintln!("cells: {} (vault-cached: {})", DEFAULT_CELLS_URL, vault.is_some());
        let remote = std::sync::Arc::new(HttpStore::new(DEFAULT_CELLS_URL));
        match &vault {
            Some(v) => {
                let cache = std::sync::Arc::new(mahere_store::VaultCells::new(v.clone()));
                cells = Some(cache.clone());
                std::sync::Arc::new(TieredStore::new(cache, remote))
            }
            None => remote,
        }
    };
    let session = vault.as_ref().and_then(|s| mahere_store::load_session(s));
    let cam = match session {
        Some(s) => Camera::new(s.lat, s.lon, s.ppd, s.bearing),
        None => Camera::new(46.2000, -122.1900, 12_000.0, 0.0),
    };
    let mut map = MapCore::new(store, cam);
    if let Some(s) = session {
        map.sun_az = s.sun_az as f32;
        map.sun_alt = s.sun_alt as f32;
    }
    if let Some(v) = &vault {
        map.set_themes(mahere_store::themes::load_all(v));
    }
    let mut app = MahereApp::new(map);
    app.store = vault;
    app.cells = cells;
    run_app(app).expect("fluor event loop failed");
}
