// A window, not a console, for a release build on Windows (2026-10-16, the first Windows build).
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
//! mahere desktop app — a thin fluor frontend over mahere-engine's MapCore.
//! Chrome, desktop input mapping, and the bg-layer blit live here; all map state and rendering live in the engine (shared with the Android shell).

mod gpu;

use fluor::coord::Coord as Px;
use fluor::event::{CursorIcon, ElementState, Event as FEvent, MouseButton, MouseScrollDelta};
use fluor::geom::Viewport;
use fluor::host::app::{Context, EventResponse, FluorApp, run_app};
use fluor::host::chrome::{self, HIT_NONE, ResizeEdge};
use fluor::host::chrome_widget::DefaultChrome;
use fluor::paint::{Clip, HitId, pack_argb};
use mahere_engine::residency::DEFAULT_CELLS_URL;
use mahere_engine::{Camera, MapCore};
use mahere_panel::{Controls, Panel, Readouts};
use std::time::Instant;

struct MahereApp {
    map: MapCore,
    chrome: DefaultChrome,
    /// The app orb as decoded, put back in the chrome when no light mode shows its ball there; and the light version the ball was last drawn for.
    app_orb: Option<fluor::host::icon::Icon>,
    orb_light: Option<u64>,
    dragging: bool,
    /// A stroke of the pen or a selection in progress.
    drawing: bool,
    /// Pixels the cursor travelled since the press: a few at release is a click on the map, a measurement.
    travel: f64,
    /// Right-button drag: rotate about the screen centre.
    rotating: bool,
    last_cursor: (f64, f64),
    store: Option<std::sync::Arc<mahere_store::FlatStorage>>,
    /// The UI scale kept from the last launch, handed to fluor once; the scale last kept.
    zoom_restore: Option<f32>,
    kept_ru: f32,
    cells: Option<std::sync::Arc<mahere_store::VaultCells>>,
    last_save: Instant,
    panel: Panel,
    /// The map's lettering, and when the camera last moved: labels are laid out once it has rested a quarter second, and hidden while it moves.
    labeller: mahere_panel::labels::Labeller,
    label_cam: ((u64, u64, u64, u64), Instant),
    /// Profiling (MAHERE_SPIN=move pans a pixel a frame, =still redraws in place): each frame part's milliseconds summed, frames counted, when last reported.
    spin: Option<bool>,
    prof: [f32; 6],
    prof_n: u32,
    prof_at: Instant,
    prof_flat: f32,
    cache_budget: u64,
    areas: mahere_engine::areas::Areas,
    /// Updates (after Photon's): a verified build is swapped in for this one, which then starts again as it.
    updater: mahere_update::Updater,
    /// The window's span (fluor's harmonic mean of width and height) the map's scale was last set for: a resize scales the map by the change, as fluor scales everything else (Nick 2026-10-17).
    map_span: f64,
    /// The map on the GPU (the phone's compositor, read back into fluor's buffer); None draws it on the CPU.
    gpu: Option<gpu::Gpu>,
}

impl MahereApp {
    /// The window's perimeter and chrome over the background, flattened into `target`; with profiling on, the frame's parts reported every three seconds.
    fn finish_frame(&mut self, target: &mut [u32], ctx: &mut Context, w: usize, h: usize, t: [Instant; 4]) {
        let [t0, t1, t2, t3] = t;
        let t4 = Instant::now();
        self.chrome.rasterize_perimeter(target, w, h, ctx.clip_mask);
        self.chrome.rasterize_chrome(ctx.damage, ctx.text, ctx.clip_mask);
        let t4b = Instant::now();
        let clip = ctx.damage_clip;
        match &self.gpu {
            // On fluor's device the map is its layer under the chrome: the scratch is the chrome alone.
            Some(g) if g.layer => self.chrome.flatten_chrome_into(target, w, h, Some(Clip::new(clip.x0, clip.y0, clip.x1, clip.y1))),
            // The chrome alone, then the map's frame under it straight from the readback, one pass over the damage (2026-10-25: the chrome group's background layer cost a copy into it, and the group a composite and a blit, every frame).
            Some(g) if g.frame.len() == w * h => {
                self.chrome.flatten_chrome_into(target, w, h, Some(Clip::new(clip.x0, clip.y0, clip.x1, clip.y1)));
                let (x0, x1) = (clip.x0.max(0) as usize, (clip.x1.max(0) as usize).min(w));
                // Where nothing of the chrome is, the map's pixel as it is; where the chrome is opaque, the chrome; only its anti-aliased edges take fluor's under kernel.
                for y in (clip.y0.max(0) as usize)..(clip.y1.max(0) as usize).min(h) {
                    let r = y * w + x0..y * w + x1;
                    let (dst, src) = (&mut target[r.clone()], &g.frame[r]);
                    for i in 0..dst.len() {
                        match dst[i] >> 24 {
                            0 => dst[i] = src[i],
                            255 => {}
                            _ => fluor::paint::flatten_premult(&mut dst[i..i + 1], &src[i..i + 1]),
                        }
                    }
                }
            }
            _ => self.chrome.flatten_into(target, w, h, Some(Clip::new(clip.x0, clip.y0, clip.x1, clip.y1))),
        }
        if self.spin.is_some() {
            let t5 = Instant::now();
            let ms = |a: Instant, b: Instant| (b - a).as_secs_f32() * 1000.0;
            for (k, v) in [ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5), ms(t0, t5)].into_iter().enumerate() {
                self.prof[k] += v;
            }
            self.prof_n += 1;
            self.prof_flat += t4b.elapsed().as_secs_f32() * 1000.0;
            if self.prof_at.elapsed().as_secs() >= 3 {
                eprintln!("  flatten alone {:.1} ms", self.prof_flat / self.prof_n.max(1) as f32);
                self.prof_flat = 0.0;
                let n = self.prof_n.max(1) as f32;
                let p = self.prof.map(|v| v / n);
                if let Some(g) = &self.gpu {
                    eprintln!("  map parts (last frame): plan + sync {:.1} ms, overlay {:.1}, submit and poll {:.1}, landing a frame {:.1}; GPU passes {:.1}", g.parts[0], g.parts[1], g.parts[2], g.parts[3], g.parts[4]);
                }
                eprintln!("desktop {w}x{h}: {} frames, per frame before the panel {:.1} ms, panel paint {:.1}, map (gpu frame) {:.1}, into fluor's layer {:.1}, chrome + flatten {:.1}, render total {:.1}", self.prof_n, p[0], p[1], p[2], p[3], p[4], p[5]);
                self.prof = [0.0; 6];
                self.prof_n = 0;
                self.prof_at = Instant::now();
            }
        }
    }

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
                mahere_panel::ThemeEdit::CaptureLight | mahere_panel::ThemeEdit::ClearLight => {
                    if e == mahere_panel::ThemeEdit::ClearLight {
                        self.map.clear_light();
                    } else if !self.map.capture_light() {
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

    fn new(map: MapCore, areas: mahere_engine::areas::Areas) -> Self {
        let mut hit_counter: HitId = HIT_NONE;
        // The app orb: the paper-craft relief map (icon.jpeg, cropped to its disc), bundled as fluor's 256×256 VSF orb. A decode failure is non-fatal — the chrome just draws no orb.
        let orb = fluor::host::icon::Icon::from_vsf_bytes(mahere_panel::APP_ORB_VSF).ok();
        let chrome = DefaultChrome::new(
            Viewport::new(1280, 800),
            mahere_panel::APP_NAME,
            orb,
            Some("drag pan · right-drag rotate · wheel zoom · A/D W/S sun · Q/E rotate · R home · 1-4 layers · 5 imagery · 6 contours · 7 slope · 8 infrared".to_string()),
            &mut hit_counter,
        );
        let app_orb = chrome.app_icon.clone();
        MahereApp {
            map,
            chrome,
            app_orb,
            orb_light: None,
            dragging: false,
            drawing: false,
            travel: 0.0,
            rotating: false,
            last_cursor: (0., 0.),
            store: None,
            zoom_restore: None,
            kept_ru: 1.0,
            cells: None,
            last_save: Instant::now(),
            panel: Panel::new(),
            labeller: mahere_panel::labels::Labeller::new(),
            label_cam: ((0, 0, 0, 0), Instant::now()),
            spin: std::env::var("MAHERE_SPIN").ok().map(|v| v == "move"),
            prof: [0.0; 6],
            prof_n: 0,
            prof_at: Instant::now(),
            prof_flat: 0.0,
            cache_budget: 2 << 30,
            areas,
            updater: mahere_update::Updater::new(mahere_store::user_dir()),
            map_span: 0.0,
            gpu: None,
        }
    }

    /// What a tap or a release on the panel changed, applied to the map.
    fn apply_controls(&mut self, ctl: Controls, mask: mahere_engine::LayerMask) {
        self.cache_budget = ctl.cache_budget;
        self.apply_theme_edits();
        self.apply_area_acts();
        self.apply_update_acts();
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
        self.map.set_travel(travel_of(ctl.travel));
        self.map.set_tool(match ctl.tool {
            1 => mahere_engine::ink::Tool::Pen,
            3 => mahere_engine::ink::Tool::Erase,
            2 => mahere_engine::ink::Tool::Select,
            _ => mahere_engine::ink::Tool::None,
        });
        self.map.set_pen_colour(ctl.pen_rgba);
        if ctl.clear_ink {
            self.map.clear_ink();
        }
        if ctl.clear_sel {
            self.map.clear_selection();
        }
        self.map.ruler = ctl.ruler;
        // Auto fits each view; off, the interval holds at what the panel gives (the slider's step, or the interval in use when Auto was unticked).
        if ctl.contour_auto {
            if self.map.contour_hold.is_some() {
                self.map.hold_contours(false);
            }
        } else if ctl.contour_interval > 0.0 {
            if self.map.contour_hold != Some(ctl.contour_interval) {
                self.map.contour_hold = Some(ctl.contour_interval);
                self.map.mark_dirty();
            }
        } else if self.map.contour_hold.is_none() {
            self.map.hold_contours(true);
        }
        if ctl.locate {
            let (w, h) = self.chrome.dims();
            self.map.locate(w, h);
        }
    }

    /// What the Updates page asked, and what it shows now.
    fn apply_update_acts(&mut self) {
        use mahere_panel::UpdateAct;
        use mahere_update::Channel;
        for act in self.panel.take_update_acts() {
            match act {
                UpdateAct::Check => self.updater.check(),
                UpdateAct::Get(i) => self.updater.get(if i == 0 { Channel::Release } else { Channel::Dev }),
                UpdateAct::Auto(on) => self.updater.auto = on,
            }
        }
        let view = updates_view(&self.updater);
        self.panel.set_updates(view);
    }

    /// What the areas pages asked. A share is a file under data/user/share, its path printed.
    fn apply_area_acts(&mut self) {
        use mahere_panel::AreaAct;
        for act in self.panel.take_area_acts() {
            match act {
                AreaAct::Save => {
                    let depth = self.map.area_depth();
                    if let Some(i) = self.areas.save_selection(&self.map.user, depth) {
                        self.panel.show_area(i, depth);
                    }
                }
                AreaAct::SetDepth(i, d) => self.areas.set_depth(i, d),
                AreaAct::Download(i) => self.areas.download(i, false),
                AreaAct::Refresh(i) => self.areas.download(i, true),
                AreaAct::Cancel(i) => self.areas.cancel(i),
                AreaAct::Share(i) => {
                    if let Some((name, bytes)) = self.areas.share_bytes(i) {
                        let dir = mahere_store::user_dir().join("share");
                        let _ = std::fs::create_dir_all(&dir);
                        let path = dir.join(format!("{}.vsf.zst", name.replace(' ', "-")));
                        match std::fs::write(&path, &bytes) {
                            Ok(()) => eprintln!("area shared: {}", path.display()),
                            Err(e) => eprintln!("area share failed: {e}"),
                        }
                    }
                }
                AreaAct::Show(i) => {
                    if let Some(e) = self.areas.list.get(i) {
                        let planes = e.area.cells.clone();
                        self.map.show_selection(planes);
                        self.panel.close();
                    }
                }
                AreaAct::Delete(i) => self.areas.delete(i),
            }
        }
    }

    /// The map keeps a device of its own (or the CPU): MAHERE_CPU=1 or MAHERE_READBACK=1.
    #[cfg(not(target_os = "windows"))]
    fn own_device() -> bool {
        std::env::var("MAHERE_CPU").as_deref() == Ok("1") || std::env::var("MAHERE_READBACK").as_deref() == Ok("1")
    }

    fn dims(ctx: &Context) -> (usize, usize) {
        (ctx.viewport.width_px as usize, ctx.viewport.height_px as usize)
    }
}

impl FluorApp for MahereApp {
    /// The UI scale kept from the last launch, once, then the panel's interface size when the slider sets it: fluor applies either as its own zoom, as its Cmd/Ctrl keys and scroll do.
    fn take_zoom_request(&mut self) -> Option<f32> {
        self.zoom_restore.take().or_else(|| self.panel.take_ru_request())
    }

    type UserEvent = ();

    fn title(&self) -> &str {
        mahere_panel::APP_NAME
    }

    /// The OS window / taskbar icon: the same orb the chrome draws.
    fn window_icon(&self) -> Option<&fluor::host::icon::Icon> {
        self.chrome.app_icon.as_ref()
    }

    fn init(&mut self, ctx: &mut Context) {
        self.chrome.resize(ctx.viewport);
        self.map.marks_top = chrome::strip_height(ctx.viewport) as f32;
        // fluor's device came first (gpu_ready) where it presents on the GPU; otherwise a headless one of the map's own.
        if self.gpu.is_none() {
            self.gpu = gpu::Gpu::new();
        }
        if self.gpu.is_some() {
            // The cells go to the GPU as they are; the CPU's own unpacked copies are not kept.
            self.map.set_gpu_only();
        }
        self.map_span = ctx.viewport.span as f64;
    }

    /// The map on fluor's device: the features and limits it needs (MAHERE_CPU=1 the CPU raster, MAHERE_READBACK=1 the headless device and its readback).
    #[cfg(not(target_os = "windows"))]
    fn gpu_request(&self, adapter: &wgpu::Adapter) -> Option<fluor::host::gpu::GpuRequest> {
        if Self::own_device() {
            return None;
        }
        Some(fluor::host::gpu::GpuRequest { features: mahere_gpu::wanted_features(adapter), limits: gpu::Gpu::limits(adapter) })
    }

    #[cfg(not(target_os = "windows"))]
    fn gpu_ready(&mut self, g: &fluor::host::gpu::Gpu) {
        // The Mac always makes fluor's device: a plain one when the map asked for nothing, which the map can't draw on.
        if !Self::own_device() {
            self.gpu = Some(gpu::Gpu::on(g.device.clone(), g.queue.clone(), &g.adapter));
        }
    }

    #[cfg(not(target_os = "windows"))]
    fn under_layer(&mut self) -> Option<fluor::host::gpu::UnderLayer> {
        self.gpu.as_ref()?.under_layer()
    }

    /// With the map under fluor's chrome on the GPU, the scratch holds the chrome alone: only its own changes are damage. Otherwise the map changes every pixel.
    fn damage_rect(&mut self, viewport: fluor::Viewport) -> Option<fluor::canvas::PixelRect> {
        if self.gpu.as_ref().is_some_and(|g| g.layer) {
            return self.chrome.chrome_damage_rect();
        }
        Some(fluor::canvas::PixelRect::new(0, 0, viewport.width_px as usize, viewport.height_px as usize))
    }

    fn on_resize(&mut self, width: u32, height: u32, ctx: &mut Context) {
        self.chrome.resize(ctx.viewport);
        // The compass and the zoom below the window buttons, which sit where the phone has its status bar.
        self.map.marks_top = chrome::strip_height(ctx.viewport) as f32;
        self.chrome.set_full_edge(ctx.is_maximized);
        // The map scales with the window about its centre, so the same ground fills it; a change of fluor's own zoom (Ctrl +/-) leaves the span, and the map, as they were.
        let span = ctx.viewport.span as f64;
        if self.map_span > 0.0 && span > 0.0 && span != self.map_span {
            let (w, h) = (width as usize, height as usize);
            self.map.zoom_about(span / self.map_span, w as f64 / 2.0, h as f64 / 2.0, w, h);
        }
        if span > 0.0 {
            self.map_span = span;
        }
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
                // The orb is the settings: it shows and hides the panel, as Photon's does.
                if hit != HIT_NONE && hit == self.chrome.app_icon_btn.id() {
                    self.panel.toggle();
                    self.chrome.invalidate_bg();
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }
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
                let mut ctl = Controls { real_sun: self.map.real_sun, real_light: self.map.real_light, follow_heading: self.map.follow_heading, lock_to_fix: self.map.lock_to_fix, cache_budget: self.cache_budget, theme: self.map.theme, compressed: self.map.compressed(), tool: self.map.tool as u8, pen_rgba: self.map.pen_colour, travel: travel_code(self.map.travel), record: false, clear_ink: false, clear_sel: false, locate: false, ruler: self.map.ruler, contour_auto: self.map.contour_hold.is_none(), contour_interval: self.map.contour_hold.unwrap_or(self.map.contour_interval) };
                let was_open = self.panel.is_open();
                if self.panel.tap(cx as f32, cy as f32, w, h, &mut mask, &mut ctl, &self.map.themes) {
                    self.apply_controls(ctl, mask);
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
                // With a tool on, the left button draws (the right button still turns the map).
                if self.map.tool != mahere_engine::ink::Tool::None {
                    self.map.begin_stroke(cx as f64, cy as f64, 0.5, w, h);
                    self.drawing = true;
                    ctx.window.request_redraw();
                }
                EventResponse::Handled
            }
            FEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left } => {
                // A press on the layer list acts when it lifts without having scrolled.
                let mut mask = self.map.layers();
                let mut ctl = Controls { real_sun: self.map.real_sun, real_light: self.map.real_light, follow_heading: self.map.follow_heading, lock_to_fix: self.map.lock_to_fix, cache_budget: self.cache_budget, theme: self.map.theme, compressed: self.map.compressed(), tool: self.map.tool as u8, pen_rgba: self.map.pen_colour, travel: travel_code(self.map.travel), record: false, clear_ink: false, clear_sel: false, locate: false, ruler: self.map.ruler, contour_auto: self.map.contour_hold.is_none(), contour_interval: self.map.contour_hold.unwrap_or(self.map.contour_interval) };
                if self.panel.release(&mut mask, &mut ctl) {
                    self.apply_controls(ctl, mask);
                    ctx.window.request_redraw();
                }
                if self.drawing {
                    self.map.end_stroke(w, h);
                    self.drawing = false;
                    self.dragging = false;
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }
                // A click that never travelled is a tap: a new measurement to that point.
                if self.dragging && self.travel < 6.0 {
                    // A label under the pointer names what it letters first.
                    let label = self.labeller.hit(&self.map.cam, w, h, ctx.cursor_x as f32, ctx.cursor_y as f32).cloned();
                    self.map.tap_label(ctx.cursor_x as f64, ctx.cursor_y as f64, w, h, label.as_ref());
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
                let mut ctl = Controls { real_sun: self.map.real_sun, real_light: self.map.real_light, follow_heading: self.map.follow_heading, lock_to_fix: self.map.lock_to_fix, cache_budget: self.cache_budget, theme: self.map.theme, compressed: self.map.compressed(), tool: self.map.tool as u8, pen_rgba: self.map.pen_colour, travel: travel_code(self.map.travel), record: false, clear_ink: false, clear_sel: false, locate: false, ruler: self.map.ruler, contour_auto: self.map.contour_hold.is_none(), contour_interval: self.map.contour_hold.unwrap_or(self.map.contour_interval) };
                if self.panel.drag(hx, hy, &mut ctl) {
                    self.cache_budget = ctl.cache_budget;
                    self.map.set_pen_colour(ctl.pen_rgba);
                    self.apply_theme_edits();
                    self.apply_area_acts();
                    self.apply_update_acts();
                    ctx.window.request_redraw();
                    self.last_cursor = (x, y);
                    return EventResponse::Handled;
                }
                if self.drawing {
                    self.map.extend_stroke(x, y, 0.5, w, h);
                    ctx.window.request_redraw();
                } else if self.dragging {
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
                // Trackpads deliver Pixels with momentum; clamp per event. A Mac turns Shift and scroll sideways: either axis counts.
                let (dx, dy) = match delta {
                    MouseScrollDelta::Lines(x, y) => (*x, *y),
                    MouseScrollDelta::Pixels(x, y) => (*x / 120., *y / 120.),
                };
                let notches = (if dy != 0.0 { dy } else { dx } as f64).clamp(-3.0, 3.0);
                // Nick 2026-10-25: over the open panel the scroll moves its list; on the map with the panel open, or with Shift, it sizes the map's lettering; otherwise it zooms the map (Cmd/Ctrl and scroll, the interface size, is fluor's own and never arrives here).
                if self.panel.covers(ctx.cursor_x, ctx.cursor_y) {
                    if self.panel.wheel(-(notches as f32) * self.panel.font_px() * 2.0) {
                        ctx.window.request_redraw();
                    }
                    return EventResponse::Handled;
                }
                if self.panel.is_open() || ctx.modifiers.shift_key() {
                    let t = self.panel.label_scale() * fluor::geom::zoom_step_factor(notches as f32 * 2.0);
                    self.panel.set_label_scale(t);
                    if let Some(store) = &self.store {
                        let _ = mahere_store::save_label_scale(store, self.panel.label_scale());
                    }
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }
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
                // Escape steps back: the panel one page up, then closed, then a measurement cleared.
                if event.logical_key == fluor::event::Key::Named(fluor::event::NamedKey::Escape) {
                    if self.panel.back() {
                        self.chrome.invalidate_bg();
                        ctx.window.request_redraw();
                        return EventResponse::Handled;
                    }
                    if self.map.has_measure() {
                        self.map.clear_measure();
                        ctx.window.request_redraw();
                        return EventResponse::Handled;
                    }
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
        // A download or a font pack on its way: a frame now and then until it lands.
        if self.areas.downloading() || self.panel.fonts_due() {
            return Some(Instant::now() + std::time::Duration::from_millis(200));
        }
        if self.map.converged() { None } else { Some(Instant::now()) }
    }

    fn tick(&mut self, ctx: &mut Context) -> bool {
        if let Some(moving) = self.spin {
            let (w, h) = Self::dims(ctx);
            if moving {
                self.map.pan(1.0, 0.0, w, h);
            }
            self.map.mark_dirty();
            ctx.window.request_redraw();
        }
        let (w, h) = Self::dims(ctx);
        // The UI scale kept whenever it changes (fluor's Cmd/Ctrl keys and scroll).
        if (ctx.viewport.ru - self.kept_ru).abs() > 1e-3 {
            self.kept_ru = ctx.viewport.ru;
            if let Some(store) = &self.store {
                let _ = mahere_store::save_ui_scale(store, self.kept_ru);
            }
        }
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
        // Updates: what the threads found, onto the page; a verified build swapped in and started in this one's place.
        let updates_changed = self.updater.poll();
        if updates_changed {
            let view = updates_view(&self.updater);
            self.panel.set_updates(view);
            self.chrome.invalidate_bg();
        }
        if let Some(staged) = self.updater.take_ready() {
            match mahere_update::install_desktop(&staged) {
                Ok(exe) => {
                    eprintln!("update: starting {}", exe.display());
                    let args: Vec<String> = std::env::args().skip(1).collect();
                    #[cfg(unix)]
                    {
                        use std::os::unix::process::CommandExt;
                        let e = std::process::Command::new(&exe).args(&args).exec();
                        eprintln!("update: restart failed: {e}");
                    }
                    #[cfg(not(unix))]
                    {
                        if std::process::Command::new(&exe).args(&args).spawn().is_ok() {
                            std::process::exit(0);
                        }
                    }
                }
                Err(e) => self.updater.status = format!("Update failed, nothing changed: {e}"),
            }
        }
        // A download's progress repaints the panel as it goes.
        let areas_changed = self.areas.poll() || self.areas.downloading() || updates_changed || self.updater.busy() || self.panel.fonts_due();
        if areas_changed {
            self.chrome.invalidate_bg();
        }
        // Labels waiting for the camera to rest keep the ticks coming until they are drawn.
        let labels_due = self.labeller.poll() || self.labeller.pending(&self.map, w, h);
        self.map.tick(w, h) || areas_changed || labels_due || self.gpu.as_ref().is_some_and(|g| g.wants_frame())
    }

    fn render(&mut self, target: &mut [u32], ctx: &mut Context) {
        let (w, h) = Self::dims(ctx);
        let t0 = Instant::now();
        if self.gpu.is_some() {
            // Drawn below, once the panel is painted: it goes over the map on the GPU.
        } else if self.map.needs_render(w, h) {
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
        let readouts = Readouts { lat: c.lat(), lon: c.lon(), gps_height: self.map.gps_sea_level(), elev: self.map.elevation_at(c.lat(), c.lon()), heading_deg: heading, m_per_px: 111_320.0 / c.ppd, frame_ms: self.map.last_frame_ms, resident: self.map.pool().map.len(), phone_heading: None, cache_used: used, cache_max: used + (8u64 << 30), has_fix: self.map.gps.is_some(), contour_major_m: self.map.contour_major() };
        let measure = self.map.measure_view(w, h, Panel::strip_samples(w));
        let themes = self.map.themes.clone();
        self.panel.set_ru(ctx.viewport.ru);
        // The chrome draws the orb and the name; the panel lays out under it and draws none of its own.
        self.panel.set_host_orb(self.chrome.orb_geometry().map(|(x, y, r)| (x as f32, y as f32, r as f32)));
        // With a light mode on, the orb is the light as a ball, redrawn when the light changes; otherwise the app orb.
        let light = (self.map.real_sun || self.map.real_light).then(|| self.map.light_version());
        if light != self.orb_light {
            self.orb_light = light;
            let ball = light.and_then(|_| self.map.orb(256));
            self.chrome.app_icon = match ball {
                Some(img) => Some(fluor::host::icon::Icon {
                    width: 256,
                    height: 256,
                    pixels: img.iter().map(|p| if p[3] == 0 { 0xFFFF_FFFF } else { 0xFF00_0000 | ((255 - p[0] as u32) << 16) | ((255 - p[1] as u32) << 8) | (255 - p[2] as u32) }).collect(),
                }),
                None => self.app_orb.clone(),
            };
            self.chrome.invalidate_chrome();
        }
        let t1 = Instant::now();
        let areas = self.areas.view(self.map.user.has_selection());
        self.panel.paint(w, h, self.map.layers(), Controls { real_sun: self.map.real_sun, real_light: self.map.real_light, follow_heading: self.map.follow_heading, lock_to_fix: self.map.lock_to_fix, cache_budget: self.cache_budget, theme: self.map.theme, compressed: self.map.compressed(), tool: self.map.tool as u8, pen_rgba: self.map.pen_colour, travel: travel_code(self.map.travel), record: false, clear_ink: false, clear_sel: false, locate: false, ruler: self.map.ruler, contour_auto: self.map.contour_hold.is_none(), contour_interval: self.map.contour_hold.unwrap_or(self.map.contour_interval) }, &readouts, measure.as_ref(), &themes, &areas, None);
        let t2 = Instant::now();
        // On the GPU: the map, its labels, the marks and the panel drawn together, the newest frame landed in fluor's own pixels, copied into the chrome's background layer as it is.
        if let Some(g) = self.gpu.as_mut() {
            g.render(&mut self.map, &mut self.labeller, self.panel.label_font_px(), &self.panel, self.panel.pixels(), w, h);
            let t3 = Instant::now();
            self.finish_frame(target, ctx, w, h, [t0, t1, t2, t3]);
            return;
        }
        let mut map = self.map.canvas.clone();
        // The labels, laid out a quarter second after the camera stops, drawn every frame letter by letter over the map and under the panel.
        let c = self.map.cam;
        let cam_key = (c.lat().to_bits(), c.lon().to_bits(), c.ppd.to_bits(), c.bearing().to_bits());
        if cam_key != self.label_cam.0 {
            self.label_cam = (cam_key, Instant::now());
        }
        // On the GPU the labels are in the frame already.
        if self.gpu.is_none() && mahere_panel::labels::Labeller::wanted(&self.map) {
            let at_rest = self.label_cam.1.elapsed().as_millis() >= 250;
            self.labeller.ask(&mut self.map, w, h, self.panel.label_font_px(), at_rest);
            self.labeller.draw_cpu(&mut map, &self.map.cam, w, h);
        }
        self.panel.composite_rgb(&mut map, w, h);
        let t3 = Instant::now();
        self.chrome.rasterize_bg(ctx.damage, |c| {
            let n = c.pixels.len().min(map.len());
            for (out, &rgb) in c.pixels[..n].iter_mut().zip(&map[..n]) {
                *out = pack_argb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8, 255);
            }
        });
        self.finish_frame(target, ctx, w, h, [t0, t1, t2, t3]);
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
    // The program checks its own signature before anything else (after Photon's): a published build must carry the author's; `mahere verify` checks and says so, for an install script.
    let verify_only = std::env::args().nth(1).as_deref() == Some("verify");
    if let Err(e) = mahere_update::verify_self() {
        eprintln!("Mahere's own signature failed: {e}");
        eprintln!("The download was cut short or damaged, the disk is failing, or the program was changed. Download it again from https://holdmyoscilloscope.com/mahere");
        std::process::exit(1);
    }
    if verify_only {
        println!("OK");
        return;
    }
    let vault = mahere_store::open(None).ok();
    // A local bake (argv[1], or data/cells) serves directly; otherwise cells come from the bucket through the vault, like the phone.
    let local = std::env::args().nth(1).map(std::path::PathBuf::from).unwrap_or_else(|| "data/cells".into());
    if local.is_dir() {
        eprintln!("cells: {}", local.display());
    } else {
        eprintln!("cells: {} (kept in the vault)", DEFAULT_CELLS_URL);
    }
    let (store, layer_stores, cells) = mahere_store::cell_stores(local.is_dir().then_some(local.as_path()), vault.clone());
    // The plain cell files a Mahere of 2026-10-18 to 2026-10-25 kept move into the vault on a thread of their own.
    if let Some(c) = &cells {
        c.take_files(&mahere_store::cells_dir(None));
    }
    let mut session = vault.as_ref().and_then(|s| mahere_store::load_session(s));
    // MAHERE_AT=lat,lon,ppd[,bearing°]: open there instead of the saved view (a place to look at again, a report to reproduce).
    if let Some(at) = std::env::var("MAHERE_AT").ok() {
        let v: Vec<f64> = at.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if v.len() >= 3 {
            session = Some(mahere_store::Session { lat: v[0], lon: v[1], ppd: v[2], bearing: v.get(3).copied().unwrap_or(0.0).to_radians(), sun_az: 315.0, sun_alt: 40.0 });
        }
    }
    // Without a saved view, a fresh install's whole globe, anywhere and turned any way ([`MapCore::fresh_globe`]; the desktop has no GPS).
    let fresh = session.is_none();
    let cam = match session {
        Some(s) => Camera::new(s.lat, s.lon, s.ppd, s.bearing),
        None => Camera::new(0.0, 0.0, 1.0, 0.0),
    };
    let mut areas = mahere_engine::areas::Areas::new(store.clone());
    if let Some(l) = &layer_stores {
        areas.set_layer_stores(vec![l.dem.clone(), l.img.clone()]);
    }
    let mut map = MapCore::layered(store, layer_stores, cam);
    if fresh {
        map.fresh_globe(None);
    }
    // The user's own cells, as files beside the data; the areas as files under them.
    let user_dir = mahere_store::user_dir();
    let _ = std::fs::create_dir_all(user_dir.join("areas"));
    areas.set_store(std::sync::Arc::new(DirAreas(user_dir.join("areas"))));
    map.set_user_store(std::sync::Arc::new(DirUser(user_dir)));
    // MAHERE_AREA=<file>: a shared area document taken in at start.
    if let Some(bytes) = std::env::var("MAHERE_AREA").ok().and_then(|f| std::fs::read(f).ok()) {
        match areas.import(&bytes) {
            Some(i) => eprintln!("area imported: {}", areas.list[i].area.name),
            None => eprintln!("MAHERE_AREA: not an area document"),
        }
    }
    if let Some(s) = session {
        map.sun_az = s.sun_az as f32;
        map.sun_alt = s.sun_alt as f32;
    }
    if let Some(v) = &vault {
        map.set_themes(mahere_store::themes::load_all(v));
    }
    let mut app = MahereApp::new(map, areas);
    // Font packs (Nick 2026-10-16): names in scripts the binary lacks draw in their own face once its pack, signed by the author, is fetched.
    let fonts = mahere_panel::fonts::FontPacks::new(mahere_store::VaultFonts::new(vault.clone()), mahere_update::AUTHOR_PUBKEY);
    // Each renderer loads only the packs its own text wants; a document is let go once one has loaded it, read again from the vault for another that wants it later.
    fonts.set_renderers(1);
    app.panel.set_fonts(fonts.clone());
    app.labeller.set_fonts(fonts);
    // The UI scale from the last launch, through fluor's own zoom (its Cmd/Ctrl keys and scroll); the text size over it.
    if let Some(ru) = vault.as_ref().and_then(|v| mahere_store::load_ui_scale(v)) {
        app.zoom_restore = Some(ru);
        app.kept_ru = ru;
    }
    if let Some(t) = vault.as_ref().and_then(|v| mahere_store::load_label_scale(v)) {
        app.panel.set_label_scale(t);
    }
    app.store = vault;
    app.cells = cells;
    run_app(app).expect("fluor event loop failed");
}

/// The user's areas as files in a directory, one per area by its name.
struct DirAreas(std::path::PathBuf);

impl DirAreas {
    fn path(&self, name: &str) -> std::path::PathBuf {
        self.0.join(format!("{}.vsf.zst", name.replace(' ', "-")))
    }
}

impl mahere_engine::areas::AreaStore for DirAreas {
    fn put(&self, name: &str, bytes: &[u8]) {
        let _ = std::fs::write(self.path(name), bytes);
    }

    fn delete(&self, name: &str) {
        let _ = std::fs::remove_file(self.path(name));
    }

    fn load_all(&self) -> Vec<Vec<u8>> {
        let Ok(dir) = std::fs::read_dir(&self.0) else { return Vec::new() };
        dir.flatten().filter_map(|e| std::fs::read(e.path()).ok()).collect()
    }
}

/// The user's cells as files in a directory, one per cell by its case-free name ([`mahere_tiles::CellKey::file_name`]).
struct DirUser(std::path::PathBuf);

impl DirUser {
    fn path(&self, key: mahere_tiles::CellKey) -> std::path::PathBuf {
        self.0.join(format!("{}.vsf.zst", key.file_name()))
    }
}

impl mahere_engine::ink::UserStore for DirUser {
    fn put(&self, key: mahere_tiles::CellKey, bytes: &[u8]) {
        let _ = std::fs::write(self.path(key), bytes);
    }

    fn delete(&self, key: mahere_tiles::CellKey) {
        let _ = std::fs::remove_file(self.path(key));
    }

    /// Every cell here; one still under its bucket name (before 2026-10-25) is renamed to the case-free one as it is read.
    fn load_all(&self) -> Vec<(mahere_tiles::CellKey, Vec<u8>)> {
        let Ok(dir) = std::fs::read_dir(&self.0) else { return Vec::new() };
        dir.flatten()
            .filter(|e| e.path().is_file())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let key = match mahere_tiles::CellKey::from_file_name(&name) {
                    Some(k) => k,
                    None => {
                        let k = mahere_tiles::CellKey::from_name(&name)?;
                        let _ = std::fs::rename(e.path(), self.path(k));
                        k
                    }
                };
                let bytes = std::fs::read(self.path(key)).ok()?;
                Some((key, bytes))
            })
            .collect()
    }
}

/// The panel's navigation code (0 off, 1 car, 2 bike, 3 foot) and the engine's travel, each way.
fn travel_of(code: u8) -> Option<mahere_engine::route::Travel> {
    use mahere_engine::route::Travel;
    match code {
        1 => Some(Travel::Car),
        2 => Some(Travel::Bike),
        3 => Some(Travel::Foot),
        _ => None,
    }
}

fn travel_code(t: Option<mahere_engine::route::Travel>) -> u8 {
    use mahere_engine::route::Travel;
    match t {
        Some(Travel::Car) => 1,
        Some(Travel::Bike) => 2,
        Some(Travel::Foot) => 3,
        None => 0,
    }
}

/// The Updates page's view of an updater.
fn updates_view(u: &mahere_update::Updater) -> mahere_panel::UpdatesView {
    use mahere_update::{Channel, Known};
    let line = |c: Channel| -> (String, String, bool) {
        let what = c.label();
        // A build's name is "0.<count>+<hash>": the number big, the hash small.
        let split = |name: &str| -> (String, String) {
            let mut parts = name.splitn(2, '+');
            (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string())
        };
        match u.known(c) {
            Known::Unchecked | Known::Checking => (capital(what), "checking\u{2026}".into(), false),
            Known::Failed(_) => (capital(what), "unavailable".into(), false),
            Known::Has(None) => (capital(what), "none for this computer yet".into(), false),
            Known::Has(Some(r)) if u.offer(c).is_none() => {
                let (n, h) = split(&r.name);
                (format!("{} {n}", capital(what)), format!("{h} \u{b7} this build"), false)
            }
            Known::Has(Some(r)) => {
                let (n, h) = split(&r.name);
                (format!("Get {what} {n}"), format!("{h} \u{b7} tap to install"), !u.busy())
            }
        }
    };
    mahere_panel::UpdatesView {
        this: mahere_update::our_name().to_string(),
        channels: [line(Channel::Release), line(Channel::Dev)],
        status: u.status.clone(),
        progress: u.progress.map(|(d, t)| if t > 0 { d as f32 / t as f32 } else { 0.0 }),
        auto: u.auto,
    }
}

fn capital(s: &str) -> String {
    let mut c = s.chars();
    c.next().map_or(String::new(), |f| f.to_uppercase().collect::<String>() + c.as_str())
}
