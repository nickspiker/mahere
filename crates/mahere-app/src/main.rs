//! mahere desktop app — a thin fluor frontend over mahere-engine's MapCore.
//! Chrome, desktop input mapping, and the bg-layer blit live here; all map
//! state and rendering live in the engine (shared with the Android shell).

use fluor::coord::Coord as Px;
use fluor::event::{CursorIcon, ElementState, Event as FEvent, MouseButton, MouseScrollDelta};
use fluor::geom::Viewport;
use fluor::host::app::{Context, EventResponse, FluorApp, run_app};
use fluor::host::chrome::{self, HIT_NONE, ResizeEdge};
use fluor::host::chrome_widget::DefaultChrome;
use fluor::paint::{Clip, HitId, pack_argb};
use mahere_engine::{Camera, MapCore, PPD_REF};
use std::time::Instant;

struct MahereApp {
    map: MapCore,
    chrome: DefaultChrome,
    dragging: bool,
    last_cursor: (f64, f64),
    store: Option<std::sync::Arc<mahere_store::FlatStorage>>,
    last_save: Instant,
}

impl MahereApp {
    fn new(map: MapCore) -> Self {
        let mut hit_counter: HitId = HIT_NONE;
        let chrome = DefaultChrome::new(
            Viewport::new(1280, 800),
            "mahere",
            None,
            Some("drag pan · wheel zoom · A/D W/S sun · Q/E rotate · R home".to_string()),
            &mut hit_counter,
        );
        MahereApp {
            map,
            chrome,
            dragging: false,
            last_cursor: (0., 0.),
            store: None,
            last_save: Instant::now(),
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
                self.dragging = true;
                EventResponse::Handled
            }
            FEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left } => {
                self.dragging = false;
                EventResponse::Handled
            }
            FEvent::CursorMoved { .. } => {
                let (hx, hy) = (ctx.cursor_x, ctx.cursor_y);
                if self.chrome.set_hover(self.chrome.hit_at(hx, hy)) {
                    ctx.window.request_redraw();
                }
                let (x, y) = (hx as f64, hy as f64);
                if self.dragging {
                    let (dx, dy) = (x - self.last_cursor.0, y - self.last_cursor.1);
                    self.map.pan(dx, dy, w, h);
                    ctx.window.request_redraw();
                }
                self.last_cursor = (x, y);
                EventResponse::Handled
            }
            FEvent::CursorLeft => {
                self.dragging = false;
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
                        self.map.set_bearing(self.map.cam.bearing + 15f64.to_radians());
                        self.map.camera_moved(w, h);
                        true
                    }
                    Some("e") => {
                        self.map.set_bearing(self.map.cam.bearing - 15f64.to_radians());
                        self.map.camera_moved(w, h);
                        true
                    }
                    Some("r") => {
                        self.map.go_home(w, h);
                        true
                    }
                    _ => false,
                };
                if handled {
                    self.chrome.set_status_text(Some(format!(
                        "sun {:.0}° az / {:.0}° alt · R home",
                        self.map.sun_az, self.map.sun_alt
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
        // Session persistence rides the refinement ticks: at most one vault
        // write per few seconds, only while the view is actually changing.
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
                        lat: c.lat,
                        lon: c.lon,
                        ppd: c.ppd,
                        bearing: c.bearing,
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
        // fluor composites front-to-back: the map is the chrome group's
        // BACKGROUND layer, never painted straight over `target`.
        let map = &self.map.canvas;
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
    let cells = std::env::args().nth(1).unwrap_or_else(|| "data/cells".into());
    eprintln!("cells: {cells}");
    let store = std::sync::Arc::new(mahere_engine::residency::DirStore(cells.into()));

    let vault = mahere_store::open(None).ok();
    let session = vault.as_ref().and_then(|s| mahere_store::load_session(s));
    let cam = match session {
        Some(s) => Camera { lat: s.lat, lon: s.lon, ppd: s.ppd, bearing: s.bearing },
        None => Camera { lat: 46.2024, lon: -121.4909, ppd: PPD_REF, bearing: 0.0 },
    };
    let mut map = MapCore::new(store, cam);
    if let Some(s) = session {
        map.sun_az = s.sun_az as f32;
        map.sun_alt = s.sun_alt as f32;
    }
    let mut app = MahereApp::new(map);
    app.store = vault;
    run_app(app).expect("fluor event loop failed");
}
