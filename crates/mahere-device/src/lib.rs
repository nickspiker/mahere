//! Mahere on a touch device: the app Android and ferros both run (mahere-android's JNI shell, mahere-ferros's process), the shared MapCore under fluor's touch shells, plus the raw two-finger gesture solve and the GPS fix.
//!
//! Gesture philosophy (Nick's spec): no gesture recognizers, no slop radius, no focal-point abstraction. Two fingers are two point correspondences; with rotation locked the solve is scale = finger-distance ratio and the geographic midpoint pinned under the screen midpoint — applied per raw touch event, so the map locks to the fingers from the first event.
//! Rotation joins when the camera grows a bearing.
//!
//! Single-finger events forward into the platform's fluor shell (which synthesizes MouseInput/CursorMoved); the shell's synthetic MouseWheel (meant for scrolly apps) is swallowed — zoom belongs to the two-finger solve.
//!
//! The map draws on the platform's GPU where it has one (`G`: Android's Vulkan host) and on the CPU raster otherwise (ferros, whose GPU is not a wgpu adapter, and Android without Vulkan): the map, its lettering and the panel composed here, as the desktop does on its CPU path.

use fluor::coord::Coord as Px;
use fluor::event::{CursorIcon, ElementState, Event as FEvent, MouseButton};
use fluor::host::EventResponse;
use fluor::host::app::{Context, FluorApp};
use fluor::paint::pack_argb;
use mahere_engine::{Camera, GpsFix, MapCore};
use mahere_panel::{Controls, Panel, Readouts};
use std::time::Instant;

/// The platform's GPU map host, as the app needs it: the lettering it laid out, to name what a tap lands on. Android's Vulkan host is one; a platform with none uses [`NoGpu`].
pub trait MapGpu {
    fn labeller(&self) -> &mahere_panel::labels::Labeller;
}

/// No GPU map host: the map draws on the CPU raster (ferros: its Mali is driven by its own gpu process, not a wgpu adapter).
pub enum NoGpu {}

impl MapGpu for NoGpu {
    fn labeller(&self) -> &mahere_panel::labels::Labeller {
        match *self {}
    }
}

pub struct TwoFinger {
    /// The ground under the screen midpoint at the gesture's start, which the gesture keeps there.
    pub geo_mid: (f64, f64),
    pub d0: f64,
    /// The view at the gesture's start: every update is built from it.
    pub cam0: Camera,
    /// Screen angle of the finger vector at capture.
    pub alpha0: f64,
}

pub struct DeviceApp<G: MapGpu = NoGpu> {
    pub map: MapCore,
    /// The way of travel routes take (1 car, 2 bike, 3 foot), kept while no route mode is on; whether the search card is up, and whether the keyboard was last asked up for it.
    pub travel_pref: u8,
    pub searching: bool,
    pub keyboard_up: bool,
    /// The search running on its own thread; the worker's cache of cells' names.
    pub search_rx: Option<std::sync::mpsc::Receiver<(Vec<mahere_engine::search::Found>, Vec<mahere_tiles::addr::Found>)>>,
    /// Address documents for the geocoder: the vault's copies, else the bucket's (Nick 2026-10-27).
    pub docs: Option<std::sync::Arc<mahere_store::VaultDocs>>,
    /// The query typed while a search ran, for the next (Nick 2026-10-26: "search sticks hard" — every keystroke started a search and they queued behind one another).
    pub search_next: Option<String>,
    pub search_cache: std::sync::Arc<std::sync::Mutex<mahere_engine::labels::NamesCache>>,
    /// A route pin held by the finger.
    pub pin_drag: Option<mahere_engine::Pin>,
    pub w: usize,
    pub h: usize,
    pub dragging: bool,
    /// Pixels the finger has travelled since the press: under a dozen at release is a tap.
    pub travel: f64,
    /// A second finger touched during this press.
    pub had_two: bool,
    /// A finger down on the map with the panel open: the panel closes when it lifts, unless a second finger came (a pinch sizes the interface or the lettering, Nick 2026-10-26) or it travelled.
    pub close_on_lift: bool,
    pub pinched: bool,
    /// A stroke of the pen or a selection in progress; the touch that is drawing is a stylus; its pressure; when a stylus last touched, since while the pen is out fingers only navigate.
    pub drawing: bool,
    pub stylus: bool,
    /// The stylus is erasing: its eraser end, or its side button held.
    pub erasing: bool,
    pub pressure: f32,
    pub stylus_seen: Option<Instant>,
    /// What the phone says about its pen being out (Samsung's slot broadcast, or a stylus hover); None when it has said nothing, and the last stylus touch decides.
    pub pen_out: Option<bool>,
    pub last_cursor: (f64, f64),
    pub two: Option<TwoFinger>,
    /// Re-anchor the next CursorMoved instead of panning (finger handoff after a pinch would otherwise jump by the stale delta).
    pub suppress_move: bool,
    pub centered_once: bool,
    pub store: Option<std::sync::Arc<mahere_store::FlatStorage>>,
    /// The dated cell cache, flushed at the pause moment.
    pub cells_cache: Option<std::sync::Arc<mahere_store::VaultCells>>,
    pub recorder: Option<std::sync::Arc<std::sync::Mutex<mahere_store::TrackRecorder>>>,
    /// Recording a track (Nick 2026-10-13): fixes go to the recorder only while this is on, and the activity runs the track service, which owns the GPS meanwhile.
    pub recording: bool,
    pub fixes_since_save: u32,
    pub persist: Persist,
    /// The GPU path, created on the first draw; None after a failure means the CPU present is in use.
    pub gpu: Option<G>,
    pub gpu_failed: bool,
    /// Draws that panicked inside wgpu; the host is remade each time, until a few in a row.
    pub gpu_panics: u32,
    /// The latest short bracket from the front camera (its stop, the binned frame): the long frame's clipped bins take their light from it.
    pub probe_short: Option<(u32, mahere_engine::probe::Binned)>,
    /// The UI's scale, pinched while the panel is open; the last finger distance of such a pinch.
    pub ui_ru: f32,
    pub ui_pinch: Option<f64>,
    /// Where the first finger came down, and whether a pinch with the panel open began there on the panel (the interface size) or on the map (the map's lettering).
    pub first_down: (f64, f64),
    pub pinch_on_panel: bool,
    pub panel: Panel,
    /// Font packs (Nick 2026-10-16), shared by the panel and the GPU host's labels.
    pub fonts: std::sync::Arc<mahere_panel::fonts::FontPacks>,
    /// Bytes the cell cache may hold; purged to it at the pause moment.
    pub cache_budget: u64,
    pub data_dir: Option<String>,
    /// The saved areas, and a file written for the share sheet, taken by the touch that asked for it.
    pub areas: mahere_engine::areas::Areas,
    pub share_file: Option<String>,
    /// When the panel last repainted for a download's progress: a few times a second, not every frame.
    pub areas_shown: Instant,
    /// Updates (after Photon's): the channels' builds, a download, and a verified APK waiting for the activity to install.
    pub updater: mahere_update::Updater,
    pub install: Option<String>,
    /// The lettering on the CPU path (with a GPU host, the host letters the map), and when the camera last moved: labels are laid out a quarter second after it rests.
    pub labeller: mahere_panel::labels::Labeller,
    pub label_cam: ((u64, u64, u64, u64), Instant),
    /// The map with its lettering and the panel over it, the CPU path's frame.
    frame: Vec<u32>,
}

/// Vault writes, off the UI thread. The vault serialises every write through one queue, and the loader fills it with cells while the map streams: a settings save on the UI thread waited behind hundreds of them and the system killed the app as unresponsive. Touch, location and pause now hand their writes here and return at once.
pub struct Persist {
    pub tx: std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>,
    /// The latest settings not yet written; a burst of changes (a slider drag) collapses into one write.
    pub pending_settings: std::sync::Arc<std::sync::Mutex<Option<mahere_store::Settings>>>,
}

impl Persist {
    pub fn new() -> Persist {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
        std::thread::Builder::new()
            .name("mahere-persist".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            })
            .expect("persist thread");
        Persist { tx, pending_settings: Default::default() }
    }

    pub fn run(&self, job: impl FnOnce() + Send + 'static) {
        let _ = self.tx.send(Box::new(job));
    }

    pub fn settings(&self, store: std::sync::Arc<mahere_store::FlatStorage>, s: mahere_store::Settings) {
        let was_pending = self.pending_settings.lock().unwrap().replace(s).is_some();
        if !was_pending {
            let pending = self.pending_settings.clone();
            self.run(move || {
                let latest = pending.lock().unwrap().take();
                if let Some(s) = latest {
                    let _ = mahere_store::save_settings(&store, &s);
                }
            });
        }
    }
}

/// A gigabyte of cells until the slider says otherwise.
pub const DEFAULT_CACHE_BUDGET: u64 = 1 << 30;

/// Free bytes on the volume holding `dir`, or none if it cannot be asked.
#[cfg(target_os = "android")]
fn free_bytes(dir: &str) -> Option<u64> {
    let c = std::ffi::CString::new(dir).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    Some(st.f_bavail as u64 * st.f_frsize as u64)
}

/// No volume on ferros: the cells live in the vault, whose room the kernel keeps (a free-space call is to come); the cache slider's top is what the cache holds.
#[cfg(not(target_os = "android"))]
fn free_bytes(_dir: &str) -> Option<u64> {
    None
}

impl<G: MapGpu> DeviceApp<G> {
    /// A search of the view for `query` over the cells the device keeps, on a thread of its own.
    /// One search at a time: a query typed while one runs waits as the next, the keystrokes between folded into it.
    pub fn start_search(&mut self, query: String) {
        if self.search_rx.is_some() {
            self.search_next = Some(query);
            return;
        }
        let cells = self.cells_cache.clone();
        let docs = self.docs.clone();
        let mut job = self.map.search_job(&query, self.w, self.h, Vec::new());
        let cache = self.search_cache.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // An address typed: its places from its ZIP's document, or the ZIPs near the view.
            let addresses = docs.map(|d| mahere_engine::geocode::geocode(&job.query, (job.cam.lat(), job.cam.lon()), &*d)).unwrap_or_default();
            let Some(cells) = cells else {
                let _ = tx.send((Vec::new(), addresses));
                return;
            };
            // The cells the device keeps, listed on the worker: the index is long on a phone that has looked around.
            job.cached = cells.keys(mahere_tiles::publish::Layer::Base);
            let mut cache = cache.lock().unwrap();
            // A worker's cache of names: bounded, the phone's memory being what it is.
            if cache.len() > 1024 {
                cache.clear();
            }
            let _ = tx.send((mahere_engine::search::run(&job, &mut cache, &mahere_panel::labels::tier), addresses));
        });
        self.search_rx = Some(rx);
    }

    /// The places a finished search found, onto the card: what each is and how far from the view's centre.
    pub fn poll_search(&mut self) -> bool {
        let Some(rx) = &self.search_rx else { return false };
        let Ok((found, addresses)) = rx.try_recv() else { return false };
        self.search_rx = None;
        if let Some(next) = self.search_next.take() {
            self.start_search(next);
        }
        let (clat, clon) = (self.map.cam.lat(), self.map.cam.lon());
        let m_lon = 111_320.0 * clat.to_radians().cos();
        // Typed addresses first: the address as found, how it was found, how far.
        let mut rows: Vec<mahere_panel::SearchRow> = addresses
            .into_iter()
            .map(|f| {
                let d = (((f.lat - clat) * 111_320.0).powi(2) + ((f.lon - clon) * m_lon).powi(2)).sqrt();
                mahere_panel::SearchRow { name: f.label, detail: format!("{} \u{b7} {}", f.detail, mahere_panel::length_text(d)), lat: f.lat, lon: f.lon }
            })
            .collect();
        let names: Vec<mahere_panel::SearchRow> = found
            .into_iter()
            .map(|f| {
                let n = mahere_tiles::Named { kind: f.kind, rank: f.rank, name: f.name.clone(), reference: f.reference.clone() };
                let d = (((f.lat - clat) * 111_320.0).powi(2) + ((f.lon - clon) * m_lon).powi(2)).sqrt();
                let name = if f.reference.is_empty() || f.reference == f.name { f.name.clone() } else if f.name.is_empty() { f.reference.clone() } else { format!("{} ({})", f.name, f.reference) };
                mahere_panel::SearchRow { name, detail: format!("{} \u{b7} {}", mahere_engine::named_kind(&n), mahere_panel::length_text(d)), lat: f.lat, lon: f.lon }
            })
            .collect();
        rows.extend(names);
        self.panel.set_found(rows);
        true
    }

    pub fn controls(&self) -> Controls {
        Controls { real_sun: self.map.real_sun, real_light: self.map.real_light, follow_heading: self.map.follow_heading, lock_to_fix: self.map.lock_to_fix, cache_budget: self.cache_budget, theme: self.map.theme, compressed: self.map.compressed(), tool: self.map.tool as u8, pen_rgba: self.map.pen_colour, travel: self.travel_pref, route: match self.map.route_mode {
            mahere_engine::RouteMode::Off => 0,
            mahere_engine::RouteMode::Directions => 1,
            mahere_engine::RouteMode::Navigate => 2,
        }, search: self.searching, record: self.recording, clear_ink: false, clear_sel: false, locate: false, ruler: self.map.ruler, contour_auto: self.map.contour_hold.is_none(), contour_interval: self.map.contour_hold.unwrap_or(self.map.contour_interval) }
    }

    /// A pinch with the panel open: the finger distance's ratio to the last event's scales the interface when it began on the panel, the map's lettering when it began on the map (Nick 2026-10-25: by where the first finger came down).
    pub fn ui_pinch(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        let d = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt().max(1.0);
        match self.ui_pinch {
            None => self.pinch_on_panel = self.panel.covers(self.first_down.0 as f32, self.first_down.1 as f32),
            Some(prev) if self.pinch_on_panel => {
                self.ui_ru = (self.ui_ru * (d / prev) as f32).clamp(mahere_panel::RU_MIN, mahere_panel::RU_MAX);
                self.panel.set_ru(self.ui_ru);
            }
            Some(prev) => {
                let t = self.panel.label_scale() * (d / prev) as f32;
                self.panel.set_label_scale(t);
            }
        }
        self.ui_pinch = Some(d);
    }

    /// The editor's requests: a built-in is copied before it is edited, a field change is live, and closing the editor writes the theme to the vault.
    pub fn apply_theme_edits(&mut self) {
        for e in self.panel.take_theme_edits() {
            let save_current = |me: &Self| {
                if let Some(store) = &me.store {
                    let (store, t) = (store.clone(), me.map.current_theme().clone());
                    me.persist.run(move || mahere_store::themes::save(&store, &t));
                }
            };
            match e {
                mahere_panel::ThemeEdit::Set(f, v) => self.map.edit_theme(f, v),
                mahere_panel::ThemeEdit::Done => save_current(self),
                mahere_panel::ThemeEdit::Duplicate => {
                    self.map.duplicate_theme();
                    save_current(self);
                    self.save_settings();
                }
                mahere_panel::ThemeEdit::Reset => {
                    if self.map.reset_theme() {
                        save_current(self);
                    }
                }
                mahere_panel::ThemeEdit::CaptureLight => {
                    if self.map.capture_light() {
                        save_current(self);
                    }
                }
                mahere_panel::ThemeEdit::ClearLight => {
                    self.map.clear_light();
                    save_current(self);
                }
                mahere_panel::ThemeEdit::Delete => {
                    if let Some(name) = self.map.delete_theme() {
                        if let Some(store) = &self.store {
                            let store = store.clone();
                            self.persist.run(move || mahere_store::themes::delete(&store, &name));
                        }
                        self.save_settings();
                    }
                }
            }
        }
    }

    /// Apply what the panel changed and keep it.
    pub fn apply_controls(&mut self, ctl: Controls, mask: mahere_engine::LayerMask) {
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
        self.cache_budget = ctl.cache_budget;
        if ctl.theme != self.map.theme {
            self.map.set_theme(ctl.theme);
        }
        // Routes (Nick 2026-10-25): directions between two pins or navigation from the fix, by the way of travel kept.
        self.travel_pref = ctl.travel.max(1);
        self.searching = ctl.search;
        let mode = match ctl.route {
            1 => mahere_engine::RouteMode::Directions,
            2 => mahere_engine::RouteMode::Navigate,
            _ => mahere_engine::RouteMode::Off,
        };
        self.map.set_route_mode(mode);
        let travel = if mode == mahere_engine::RouteMode::Off { None } else { travel_of(self.travel_pref) };
        if travel != self.map.travel {
            self.map.set_travel(travel);
            self.map.reroute();
        }
        if self.panel.take_swap() {
            self.map.swap_pins();
        }
        if !self.searching && self.map.found_at.take().is_some() {
            self.map.mark_dirty();
        }
        if let Some(act) = self.panel.take_search_act() {
            use mahere_panel::SearchAct;
            match act {
                SearchAct::Show(lat, lon) => {
                    self.map.found_at = Some((lat, lon));
                    self.map.go_to(lat, lon, None, self.w, self.h);
                }
                SearchAct::Go(lat, lon) => self.map.set_pin(mahere_engine::Pin::Dest, (lat, lon)),
                SearchAct::From(lat, lon) => self.map.set_pin(mahere_engine::Pin::A, (lat, lon)),
                SearchAct::To(lat, lon) => self.map.set_pin(mahere_engine::Pin::B, (lat, lon)),
            }
        }
        if ctl.record != self.recording {
            self.recording = ctl.record;
            // Each recording a track of its own.
            if let (true, Some(rec)) = (self.recording, &self.recorder) {
                let rec = rec.clone();
                self.persist.run(move || rec.lock().unwrap().begin());
            }
        }
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
            self.map.locate(self.w, self.h);
        }
        self.save_settings();
    }

    /// What the areas pages asked: save the selection, change an area's detail, download, share, show, delete.
    /// What the Updates page asked: check, get a channel's build, automatic on or off.
    pub fn apply_update_acts(&mut self) {
        use mahere_panel::UpdateAct;
        use mahere_update::Channel;
        for act in self.panel.take_update_acts() {
            match act {
                UpdateAct::Check => self.updater.check(),
                UpdateAct::Get(i) => self.updater.get(if i == 0 { Channel::Release } else { Channel::Dev }),
                UpdateAct::Auto(on) => {
                    self.updater.auto = on;
                    self.save_settings();
                }
            }
        }
        let view = self.updates_view();
        self.panel.set_updates(view);
    }

    /// The Updates page's view of the updater.
    pub fn updates_view(&self) -> mahere_panel::UpdatesView {
        use mahere_update::{Channel, Known};
        let u = &self.updater;
        let line = |c: Channel| -> (String, String, bool) {
            let what = c.label();
            // The version as people read it, nothing else (Nick 2026-10-26: no build numbers).
            let split = |r: &mahere_update::Row| -> (String, String) { (mahere_update::short_name(&r.name), String::new()) };
            match u.known(c) {
                Known::Unchecked | Known::Checking => (capital(what), "checking\u{2026}".into(), false),
                Known::Failed(_) => (capital(what), "unavailable".into(), false),
                Known::Has(None) => (capital(what), "none for this device yet".into(), false),
                Known::Has(Some(r)) if u.offer(c).is_none() => {
                    let (n, _) = split(r);
                    (format!("{} {n}", capital(what)), "this build".into(), false)
                }
                Known::Has(Some(r)) => {
                    let (n, _) = split(r);
                    (format!("Get {what} {n}"), "tap to install".into(), !u.busy())
                }
            }
        };
        mahere_panel::UpdatesView {
            this: mahere_update::short_name(mahere_update::our_name()),
            channels: [line(Channel::Release), line(Channel::Dev)],
            status: u.status.clone(),
            progress: u.progress.map(|(d, t)| if t > 0 { d as f32 / t as f32 } else { 0.0 }),
            auto: u.auto,
        }
    }

    pub fn apply_area_acts(&mut self) {
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
                    if let (Some(dir), Some((name, bytes))) = (&self.data_dir, self.areas.share_bytes(i)) {
                        let share = std::path::PathBuf::from(dir).join("share");
                        let _ = std::fs::create_dir_all(&share);
                        let path = share.join(format!("{}.vsf.zst", name.replace(' ', "-")));
                        match std::fs::write(&path, &bytes) {
                            Ok(()) => self.share_file = Some(path.to_string_lossy().to_string()),
                            Err(e) => eprintln!("share: writing {}: {e}", path.display()),
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

    /// A shared area document taken in: kept, and opened.
    pub fn open_area(&mut self, bytes: &[u8]) {
        if let Some(i) = self.areas.import(bytes) {
            let depth = self.areas.list[i].area.depth;
            self.panel.show_area(i, depth);
        } else {
            eprintln!("area: not an area document ({} bytes)", bytes.len());
        }
    }

    pub fn save_settings(&self) {
        if let Some(store) = &self.store {
            let s = mahere_store::Settings { cache_budget: self.cache_budget, layer_bits: self.map.layers().bits() as u64, real_sun: self.map.real_sun, real_light: self.map.real_light, follow_heading: self.map.follow_heading, lock_to_fix: self.map.lock_to_fix, theme: self.map.theme as u64, compressed: self.map.compressed(), ui_ru: self.ui_ru, tool: self.map.tool as u8, pen_rgba: Some(self.map.pen_colour), travel: self.travel_pref, auto_update: self.updater.auto, record: self.recording, ruler: self.map.ruler, contour_hold: self.map.contour_hold };
            self.persist.settings(store.clone(), s);
        }
    }

    /// Whether the camera has rested a quarter second, when labels are laid out; a camera that moved since the last look starts the quarter second again.
    fn label_rest(&mut self) -> bool {
        let c = self.map.cam;
        let cam_key = (c.lat().to_bits(), c.lon().to_bits(), c.ppd.to_bits(), c.bearing().to_bits());
        if cam_key != self.label_cam.0 {
            self.label_cam = (cam_key, Instant::now());
        }
        self.label_cam.1.elapsed().as_millis() >= 250
    }

    /// What the cache holds and the most the slider may allow: what is held plus the free space.
    pub fn cache_stats(&self) -> (u64, u64) {
        let used = self.cells_cache.as_ref().map_or(0, |c| c.cached_bytes());
        let free = self.data_dir.as_deref().and_then(free_bytes).unwrap_or(0);
        (used, used + free)
    }

    pub fn two_begin(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        // A second finger means a gesture, never a tap, however little the first one travelled; a stroke the first finger began is undone, the two fingers navigate, and the one left afterwards keeps navigating.
        if self.drawing {
            self.map.cancel_stroke();
            self.drawing = false;
        }
        self.had_two = true;
        // The ground under the screen midpoint, which the gesture keeps there: the mean of the two fingers' latitudes and longitudes is nowhere near it at a globe zoom, where the fingers can straddle a pole or the antimeridian (Nick 2026-10-09: the map jumped a region away on the second finger).
        let geo_mid = self.map.cam.screen_to_geo((x0 + x1) * 0.5, (y0 + y1) * 0.5, self.w, self.h);
        let d0 = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt().max(1.0);
        self.two = Some(TwoFinger {
            geo_mid,
            d0,
            cam0: self.map.cam,
            alpha0: (y1 - y0).atan2(x1 - x0),
        });
        self.dragging = false;
    }

    /// Full 4-DOF similarity solve: two finger correspondences determine pan+rotate+zoom. Scale from the distance ratio, the turn from the finger-vector angle's change, then the ground that was under the screen midpoint pinned under it.
    pub fn two_update(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        let Some(t) = &self.two else { return };
        let d = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt().max(1.0);
        let alpha = (y1 - y0).atan2(x1 - x0);
        // From the view at the gesture's start every time, so nothing accumulates: the scale, the turn by the fingers' angle (a rotation of the ball about the view axis, never a bearing), then the anchor.
        let (geo_mid, cam0, d0, alpha0) = (t.geo_mid, t.cam0, t.d0, t.alpha0);
        self.map.cam = cam0;
        self.map.set_ppd(cam0.ppd * d / d0);
        self.map.rotate_view(alpha0 - alpha);
        let (mx, my) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        self.map.place_anchor(geo_mid.0, geo_mid.1, mx, my, self.w, self.h);
    }

    pub fn two_end(&mut self) {
        if self.two.take().is_some() {
            // Keep panning with whichever finger remains, re-anchored.
            self.dragging = true;
            self.suppress_move = true;
        }
    }

    pub fn on_gps(&mut self, fix: GpsFix) {
        if !self.centered_once {
            self.centered_once = true;
            let bearing = self.map.cam.bearing();
            self.map.cam.look_at(fix.lat, fix.lon, bearing);
            self.map.camera_moved(self.w, self.h);
        }
        self.map.set_gps(fix);
        // The session (camera and sun) refreshed every tenth fix; fixes are recorded by the track service's path while recording.
        self.fixes_since_save += 1;
        if self.fixes_since_save >= 10 {
            self.fixes_since_save = 0;
            self.save_session();
        }
    }

    pub fn save_session(&self) {
        if let Some(store) = &self.store {
            let c = &self.map.cam;
            let (route_mode, travel, pin_a, pin_b, nav_to) = self.map.route_state();
            let s = mahere_store::Session { lat: c.lat(), lon: c.lon(), ppd: c.ppd, bearing: c.bearing(), sun_az: self.map.sun_az as f64, sun_alt: self.map.sun_alt as f64, route_mode, travel, pin_a, pin_b, nav_to, window: None };
            let store = store.clone();
            self.persist.run(move || {
                let _ = mahere_store::save_session(&store, &s);
            });
        }
    }
}

impl<G: MapGpu> FluorApp for DeviceApp<G> {
    type UserEvent = ();

    fn title(&self) -> &str {
        mahere_panel::APP_NAME
    }

    fn init(&mut self, _ctx: &mut Context) {}

    fn on_resize(&mut self, width: u32, height: u32, _ctx: &mut Context) {
        self.w = width as usize;
        self.h = height as usize;
        self.map.camera_moved(self.w, self.h);
    }

    fn on_event(&mut self, event: &FEvent, ctx: &mut Context) -> EventResponse {
        match event {
            FEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left } => {
                // The panel first: the gear and its rows take the tap; the map gets the rest.
                let mut mask = self.map.layers();
                let mut ctl = self.controls();
                let was_open = self.panel.is_open();
                if self.panel.tap(ctx.cursor_x as f32, ctx.cursor_y as f32, self.w, self.h, &mut mask, &mut ctl, &self.map.themes) {
                    self.apply_controls(ctl, mask);
                    self.apply_theme_edits();
                    self.apply_area_acts();
                self.apply_update_acts();
                    self.apply_update_acts();
                    if self.panel.take_clear_measure() {
                        self.map.clear_measure();
                    }
                    self.dragging = false;
                    return EventResponse::Handled;
                }
                // With the panel open the map takes no gesture: a tap on it closes the panel and nothing else, decided when the finger lifts.
                if was_open {
                    self.close_on_lift = true;
                    self.pinched = false;
                    self.travel = 0.0;
                    self.last_cursor = (ctx.cursor_x as f64, ctx.cursor_y as f64);
                    self.dragging = false;
                    return EventResponse::Handled;
                }
                // A route pin under the finger is picked up and moved, not the map.
                if self.two.is_none() {
                    if let Some(pin) = self.map.pin_at(ctx.cursor_x as f64, ctx.cursor_y as f64, self.w, self.h) {
                        self.pin_drag = Some(pin);
                        return EventResponse::Handled;
                    }
                }
                if self.two.is_none() {
                    self.dragging = true;
                    self.travel = 0.0;
                    self.had_two = false;
                    self.last_cursor = (ctx.cursor_x as f64, ctx.cursor_y as f64);
                    // The S Pen always writes on the map: the tool picked, or the pen with none (Nick 2026-10-13).
                    // A finger draws only with a tool on, and not while the pen is out (a pen touched in the last few seconds), when fingers only navigate (Nick 2026-10-10).
                    use mahere_engine::ink::Tool;
                    let pen_out = self.pen_out.unwrap_or_else(|| self.stylus_seen.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(3)));
                    let job = match (self.stylus, self.map.tool) {
                        _ if self.stylus && self.erasing => Tool::Erase,
                        (true, Tool::None) => Tool::Pen,
                        (true, t) => t,
                        (false, _) if pen_out => Tool::None,
                        (false, t) => t,
                    };
                    if job != Tool::None {
                        self.map.begin_stroke_as(job, ctx.cursor_x as f64, ctx.cursor_y as f64, self.pressure as f64, self.w, self.h);
                        self.drawing = true;
                        ctx.window.request_redraw();
                    }
                }
                EventResponse::Handled
            }
            FEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left } => {
                // A press on the layer list acts when it lifts without having scrolled.
                let mut mask = self.map.layers();
                let mut ctl = self.controls();
                if self.panel.release(&mut mask, &mut ctl) {
                    self.apply_controls(ctl, mask);
                }
                self.apply_area_acts();
                self.apply_update_acts();
                if std::mem::take(&mut self.close_on_lift) {
                    if !self.pinched && self.travel < 12.0 {
                        self.panel.close();
                        self.save_settings();
                    }
                    self.pinched = false;
                    return EventResponse::Handled;
                }
                if self.drawing {
                    self.map.end_stroke(self.w, self.h);
                    self.drawing = false;
                    self.dragging = false;
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }
                // A pin put down: the route asked for again from where it lies.
                if self.pin_drag.take().is_some() {
                    self.map.reroute();
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }
                // A press and release that never travelled is a tap: a new measurement to that point.
                if self.dragging && self.two.is_none() && !self.had_two && self.travel < 12.0 {
                    // A label under the finger names what it letters first.
                    let label = self.gpu.as_ref().and_then(|g| g.labeller().hit(&self.map.cam, self.w, self.h, ctx.cursor_x as f32, ctx.cursor_y as f32).cloned());
                    self.map.tap_label(ctx.cursor_x as f64, ctx.cursor_y as f64, self.w, self.h, label.as_ref());
                }
                self.dragging = false;
                EventResponse::Handled
            }
            FEvent::CursorMoved { .. } => {
                let (x, y) = (ctx.cursor_x as f64, ctx.cursor_y as f64);
                let mut ctl = self.controls();
                if self.panel.drag(x as f32, y as f32, &mut ctl) {
                    let mask = self.map.layers();
                    self.apply_controls(ctl, mask);
                    self.apply_theme_edits();
                    self.apply_area_acts();
                self.apply_update_acts();
                    self.apply_update_acts();
                    self.last_cursor = (x, y);
                    return EventResponse::Handled;
                }
                if self.close_on_lift {
                    self.travel += (x - self.last_cursor.0).abs() + (y - self.last_cursor.1).abs();
                } else if let Some(pin) = self.pin_drag {
                    self.map.move_pin(pin, x, y, self.w, self.h);
                    ctx.window.request_redraw();
                } else if self.suppress_move {
                    self.suppress_move = false;
                } else if self.drawing {
                    self.map.extend_stroke(x, y, self.pressure as f64, self.w, self.h);
                    ctx.window.request_redraw();
                } else if self.dragging && self.two.is_none() && !self.panel.is_open() {
                    let (dx, dy) = (x - self.last_cursor.0, y - self.last_cursor.1);
                    self.travel += dx.abs() + dy.abs();
                    if self.travel >= 12.0 {
                        self.map.pan(dx, dy, self.w, self.h);
                        ctx.window.request_redraw();
                    }
                }
                self.last_cursor = (x, y);
                EventResponse::Handled
            }
            // The shell synthesizes MouseWheel from touch-drags for scrolly apps; swallowed — zoom is the two-finger solve's job.
            FEvent::MouseWheel { .. } => EventResponse::Handled,
            // The soft keyboard types into the search.
            FEvent::Ime(fluor::event::Ime::Commit(t)) if self.searching => {
                for c in t.chars() {
                    match c {
                        '\n' => self.panel.enter(),
                        c => self.panel.type_text(&c.to_string()),
                    }
                }
                let (ctl, mask) = (self.controls(), self.map.layers());
                self.apply_controls(ctl, mask);
                EventResponse::Handled
            }
            // Escape steps back, as Android's back button does (`nativeBack`): ferros's back tap arrives as Escape (ferros WINDOWS.md "The helm"). The panel one page up, then closed; with it closed already, nothing more here.
            FEvent::KeyboardInput { event } if event.state == ElementState::Pressed && event.logical_key == fluor::event::Key::Named(fluor::event::NamedKey::Escape) => {
                if self.panel.back() { EventResponse::Handled } else { EventResponse::Pass }
            }
            FEvent::KeyboardInput { event } if self.searching && event.state == ElementState::Pressed => {
                use fluor::event::{Key, NamedKey};
                match &event.logical_key {
                    Key::Named(NamedKey::Backspace) => self.panel.backspace(),
                    Key::Named(NamedKey::Enter) => {
                        self.panel.enter();
                        let (ctl, mask) = (self.controls(), self.map.layers());
                        self.apply_controls(ctl, mask);
                    }
                    _ => return EventResponse::Pass,
                }
                EventResponse::Handled
            }
            _ => EventResponse::Pass,
        }
    }

    /// The CPU path draws on change only: a cell, the lettering or a font pack arriving, the camera, a touch (`tick` and the input say so).
    /// A frame every pass while the residency was unsettled redrew the same screen over and over while cells came in tens of seconds apart (ferros, 2026-11-11: frames of seconds, the four cores and the heap's one lock taken from the loaders and the network).
    /// A GPU host draws while unsettled as before: its frame is cheap.
    fn wake_at(&self) -> Option<Instant> {
        if self.gpu.is_none() || self.map.converged() { None } else { Some(Instant::now()) }
    }

    fn tick(&mut self, ctx: &mut Context) -> bool {
        if let Some(what) = self.panel.take_copy() {
            ctx.window.set_clipboard(&what);
        }
        // The search is on while its page is: closing the panel any way ends it (and takes the keyboard down).
        if self.searching != self.panel.is_searching() {
            let (mut ctl, mask) = (self.controls(), self.map.layers());
            ctl.search = self.panel.is_searching();
            self.apply_controls(ctl, mask);
        }
        if let Some(q) = self.panel.take_query() {
            self.start_search(q);
        }
        let found = self.poll_search();
        // On the CPU path the frame is this app's to draw, the panel and the lettering in it: what the GPU host polls before its frame is polled here (a download's progress a few times a second, a font pack arriving, the updater's news, labels laid out).
        let mut more = false;
        if self.gpu.is_none() {
            let finished = self.areas.poll();
            if self.panel.fonts_arrived() {
                self.panel.invalidate();
            }
            if self.panel.is_open() && (finished || (self.areas.downloading() && self.areas_shown.elapsed().as_millis() >= 250)) {
                self.panel.invalidate();
                self.areas_shown = Instant::now();
            }
            if self.updater.poll() {
                let view = self.updates_view();
                self.panel.set_updates(view);
            }
            // The lettering is asked for here as well as in the frame, so a view come to rest (a quarter second after the camera stopped) or cells arrived are lettered without a frame drawn to ask; the layout landing is the frame.
            if mahere_panel::labels::Labeller::wanted(&self.map) {
                let at_rest = self.label_rest();
                self.labeller.ask(&mut self.map, self.w, self.h, self.panel.label_font_px(), at_rest);
            }
            more = self.panel.take_dirty() || self.labeller.poll();
        }
        // A search under way is no frame: its result landing is (`found`).
        self.map.tick(self.w, self.h) || found || more
    }

    /// The keyboard up while the search page is, down when it goes (fluor's one-shot ask, read after every touch and every frame).
    fn wants_keyboard(&mut self) -> Option<bool> {
        if self.searching == self.keyboard_up {
            return None;
        }
        self.keyboard_up = self.searching;
        Some(self.searching)
    }

    fn render(&mut self, target: &mut [u32], ctx: &mut Context) {
        let w = ctx.viewport.width_px as usize;
        let h = ctx.viewport.height_px as usize;
        if self.map.needs_render(w, h) {
            self.map.render(w, h);
        }
        // The map's frame, then its lettering letter by letter over it (laid out a quarter second after the camera rests), then the panel over both: the desktop's CPU path.
        self.frame.clear();
        self.frame.extend_from_slice(&self.map.canvas);
        let c = self.map.cam;
        let at_rest = self.label_rest();
        if mahere_panel::labels::Labeller::wanted(&self.map) {
            self.labeller.ask(&mut self.map, w, h, self.panel.label_font_px(), at_rest);
            self.labeller.draw_cpu(&mut self.frame, &self.map.cam, w, h);
        }
        let mut heading = c.bearing().to_degrees().rem_euclid(360.0);
        if heading > 180.0 {
            heading -= 360.0;
        }
        let cache = self.cache_stats();
        let readouts = Readouts { lat: c.lat(), lon: c.lon(), gps_height: self.map.gps_sea_level(), elev: self.map.elevation_at(c.lat(), c.lon()), heading_deg: heading, m_per_px: 111_320.0 / c.ppd, frame_ms: self.map.last_frame_ms, resident: self.map.pool().map.len(), phone_heading: self.map.have_rotation.then(|| self.map.true_heading()), cache_used: cache.0, cache_max: cache.1, has_fix: self.map.gps.is_some(), contour_major_m: self.map.contour_major() };
        let measure = self.map.measure_view(w, h, Panel::strip_samples(w));
        let (mask, ctl, areas) = (self.map.layers(), self.controls(), self.areas.view(self.map.user.has_selection()));
        let map = &self.map;
        let orb = |n: usize| map.orb(n);
        self.panel.paint(w, h, mask, ctl, &readouts, measure.as_ref(), &map.themes, &areas, Some(&orb));
        self.panel.composite_rgb(&mut self.frame, w, h);
        // ANativeWindow RGBA_8888 lands R in the low byte: swap R/B vs the desktop path or Puget Sound renders brown and the pin orange. fluor's ferros host takes fluor's own order, as the desktop does.
        let n = (w * h).min(target.len()).min(self.frame.len());
        for (out, &rgb) in target[..n].iter_mut().zip(&self.frame[..n]) {
            #[cfg(target_os = "android")]
            {
                *out = pack_argb(rgb as u8, (rgb >> 8) as u8, (rgb >> 16) as u8, 255);
            }
            #[cfg(not(target_os = "android"))]
            {
                *out = pack_argb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8, 255);
            }
        }
    }

    fn cursor_for(&self, _x: Px, _y: Px, _ctx: &Context) -> CursorIcon {
        CursorIcon::Default
    }
}


/// The panel's way of travel (1 car, 2 bike, 3 foot) as the engine's.
fn travel_of(code: u8) -> Option<mahere_engine::route::Travel> {
    use mahere_engine::route::Travel;
    match code {
        1 => Some(Travel::Car),
        2 => Some(Travel::Bike),
        3 => Some(Travel::Foot),
        _ => None,
    }
}


fn capital(s: &str) -> String {
    let mut c = s.chars();
    c.next().map_or(String::new(), |f| f.to_uppercase().collect::<String>() + c.as_str())
}

/// A verified update the activity should install now, once; null when there is none.

impl<G: MapGpu> DeviceApp<G> {
    /// The app as a device starts it: the view, settings, themes, areas and fonts from `store` (the platform's vault: Android's under its files directory, ferros's the process's vault scope), and `dir`, the platform's own directory where it has one (Android's files directory: hand-pushed `cells-local`, the plain cell files of 2026-10-18 to 2026-10-25, the share sheet's files, staged updates; ferros has none).
    pub fn open(width: u32, height: u32, store: Option<std::sync::Arc<mahere_store::FlatStorage>>, dir: Option<&str>) -> DeviceApp<G> {
        // How long each step of the start takes, in the log (2026-10-18: 74 s of black screen before the first cell).
        let t0 = std::time::Instant::now();
        let lap = |what: &str| eprintln!("start: {what} at {} ms", t0.elapsed().as_millis());
        lap("vault open");
        // Cells stream from the bucket through the vault; `cells-local` (pushed by hand) overrides for offline development.
        let local = dir.map(|d| std::path::PathBuf::from(format!("{d}/cells-local"))).filter(|l| l.is_dir());
        let (cell_store, layer_stores, cells_cache) = mahere_store::cell_stores(local.as_deref(), store.clone());
        // The plain cell files of 2026-10-18 to 2026-10-25 move into the vault on a thread of their own.
        if let (Some(c), Some(d)) = (&cells_cache, dir) {
            c.take_files(&mahere_store::cells_dir(Some(d)));
        }
        lap("cell stores");
        let session = store.as_ref().and_then(|s| mahere_store::load_session(s));
        let settings = store.as_ref().and_then(|s| mahere_store::load_settings(s));
        let travel_pref = settings.as_ref().map_or(1, |s| s.travel.max(1));
        lap("session and settings");
        // A saved view only if it is a view (finite, on the globe, at a zoom the map has); without one, a fresh install's whole globe ([`MapCore::fresh_globe`]), which the first GPS fix turns to face.
        let restored = match session {
            Some(s) if s.lat.is_finite() && s.lon.is_finite() && s.ppd.is_finite() && s.bearing.is_finite() && s.lat.abs() <= 90.0 && s.lon.abs() <= 360.0 && (1.0..=4_000_000.0).contains(&s.ppd) => {
                Some(Camera::new(s.lat, s.lon, s.ppd, s.bearing))
            }
            _ => None,
        };
        let cam = restored.unwrap_or_else(|| Camera::new(0.0, 0.0, 1.0, 0.0));
        eprintln!("camera: {:.5} {:.5} ppd {:.1} bearing {:.3}", cam.lat(), cam.lon(), cam.ppd, cam.bearing());
        let mut areas = mahere_engine::areas::Areas::new(cell_store.clone());
        // Address documents: the vault's copies, else the bucket's; an area's download takes the ZIPs it covers.
        let docs = std::sync::Arc::new(mahere_store::VaultDocs::new(store.clone()));
        areas.set_docs(docs.clone());
        if let Some(l) = &layer_stores {
            areas.set_layer_stores(vec![l.dem.clone(), l.img.clone()]);
        }
        let mut map = MapCore::layered(cell_store, layer_stores, cam);
        if restored.is_none() {
            map.fresh_globe(None);
        }
        // The route as it stood last session.
        if let Some(s) = session {
            map.restore_route(s.route_mode, s.travel, s.pin_a, s.pin_b, s.nav_to);
        }
        if let Some(v) = &store {
            map.set_user_store(std::sync::Arc::new(mahere_store::VaultUser::new(v.clone())));
            areas.set_store(std::sync::Arc::new(mahere_store::VaultAreas::new(v.clone())));
        }
        // The session's sun is not restored: the phone has no way to move the map's own sun, so it is always the default, and older sessions hold the real sun's last place in it.
        let _ = session.map(|s| (s.sun_az, s.sun_alt));
        // Themes from the vault (the built-ins written on first launch), then the saved choice, then the saved layers over the theme's defaults.
        lap("map, ink and areas");
        if let Some(v) = &store {
            let themes = mahere_store::themes::load_all(v);
            lap("themes read");
            map.set_themes(themes);
        }
        lap("themes");
        let mut ui_ru = 1.0f32;
        if let Some(s) = settings {
            map.set_theme(s.theme as usize);
            map.set_layers(mahere_engine::LayerMask::from_bits(s.layer_bits as u32));
            map.set_real_sun(s.real_sun);
            map.set_real_light(s.real_light);
            ui_ru = s.ui_ru;
            map.set_compressed(s.compressed);
            map.set_follow_heading(s.follow_heading);
            map.set_lock_to_fix(s.lock_to_fix);
            map.ruler = s.ruler;
            map.contour_hold = s.contour_hold;
            map.set_tool(match s.tool {
                1 => mahere_engine::ink::Tool::Pen,
                3 => mahere_engine::ink::Tool::Erase,
                2 => mahere_engine::ink::Tool::Select,
                _ => mahere_engine::ink::Tool::None,
            });
            if let Some(c) = s.pen_rgba {
                map.set_pen_colour(c);
            }
        }
        let recorder = store.clone().map(|s| std::sync::Arc::new(std::sync::Mutex::new(mahere_store::TrackRecorder::new(s))));
        let fonts = mahere_panel::fonts::FontPacks::new(mahere_store::VaultFonts::new(store.clone()), mahere_update::AUTHOR_PUBKEY);
        // Each renderer loads only the packs its own text wants; a document is let go once one has loaded it, read again from the vault for another that wants it later.
        fonts.set_renderers(1);
        let mut app = DeviceApp {
            map,
            travel_pref,
            searching: false,
            keyboard_up: false,
            search_rx: None,
            search_next: None,
            search_cache: Default::default(),
            docs: Some(docs.clone()),
            pin_drag: None,
            w: width as usize,
            h: height as usize,
            dragging: false,
            travel: 0.0,
            had_two: false,
            close_on_lift: false,
            pinched: false,
            drawing: false,
            stylus: false,
            erasing: false,
            pressure: 0.5,
            stylus_seen: None,
            pen_out: None,
            last_cursor: (0., 0.),
            two: None,
            suppress_move: false,
            // A restored session IS the view; don't let the first fix yank it. A fresh globe turns to face the first fix, keeping its zoom.
            centered_once: restored.is_some(),
            store,
            cells_cache,
            recorder,
            recording: settings.is_some_and(|s| s.record),
            fixes_since_save: 0,
            persist: Persist::new(),
            gpu: None,
            gpu_failed: false,
            gpu_panics: 0,
                probe_short: None,
                ui_ru,
                ui_pinch: None,
                first_down: (0.0, 0.0),
                pinch_on_panel: true,
            panel: Panel::new(),
            fonts,
            cache_budget: settings.map_or(DEFAULT_CACHE_BUDGET, |s| if s.cache_budget == 0 { DEFAULT_CACHE_BUDGET } else { s.cache_budget }),
            data_dir: dir.map(str::to_string),
            areas,
            share_file: None,
            areas_shown: Instant::now(),
            updater: {
                let mut u = mahere_update::Updater::new(std::path::PathBuf::from(dir.unwrap_or_default()));
                u.auto = settings.is_none_or(|s| s.auto_update);
                u
            },
            install: None,
            labeller: mahere_panel::labels::Labeller::new(),
            label_cam: ((0, 0, 0, 0), Instant::now()),
            frame: Vec::new(),
        };
        app.panel.set_ru(ui_ru);
        // A touch screen: the interface size is the pinch on the panel.
        app.panel.set_touch(true);
        if let Some(t) = app.store.as_ref().and_then(|v| mahere_store::load_label_scale(v)) {
            app.panel.set_label_scale(t);
        }
        app.panel.set_fonts(app.fonts.clone());
        // The CPU path's lettering in the faces the packs bring, as the GPU host's.
        app.labeller.set_fonts(app.fonts.clone());
        app
    }
}
