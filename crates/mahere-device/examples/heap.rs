// What Mahere's heap holds on a ferros device, measured on the desktop: the device app at the Pro's panel (1344 x 2992), composed as the device's CPU path composes it (the map, its lettering over it, the panel over both), with a counting allocator under it. It sized the ferros heap Mahere declared until 2026-11-23, when the owner ruled out fixed ceilings (a process gets memory as it uses it, up to what the device has); it still says what Mahere uses.
// cargo run --release -p mahere-device --example heap -- <cells dir> [lat lon]
// The cells dir is a local bake (data/cells); with "" the map streams from the bucket (a ferros device reads the same cells from its vault).
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static NOW: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let n = NOW.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(n, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        NOW.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, size) };
        if !q.is_null() {
            if size > l.size() {
                let n = NOW.fetch_add(size - l.size(), Ordering::Relaxed) + size - l.size();
                PEAK.fetch_max(n, Ordering::Relaxed);
            } else {
                NOW.fetch_sub(l.size() - size, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static A: Counting = Counting;

const MIB: f64 = (1u64 << 20) as f64;

/// The process's threads (Linux): on ferros each one's stack comes from the heap, 256 KiB unless it asks for more (ferros-rust std, DEFAULT_MIN_STACK_SIZE).
fn threads() -> usize {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find_map(|l| l.strip_prefix("Threads:").map(|n| n.trim().parse().unwrap_or(0)))).unwrap_or(0)
}

fn line(what: &str, app: &mahere_device::DeviceApp) {
    let (cells, _, bytes) = app.map.pool().held();
    println!("{what}: heap {:.1} MiB now, {:.1} MiB peak; {cells} cells resident holding {:.1} MiB of planes; {} threads", NOW.load(Ordering::Relaxed) as f64 / MIB, PEAK.load(Ordering::Relaxed) as f64 / MIB, bytes as f64 / MIB, threads());
    // The planes by kind: the CPU's packed terrain, the GPU's copy of it, the vector layers, the imagery, the geoid.
    let class = |c: &Option<mahere_tiles::ClassCell>| c.as_ref().map_or(0, |c| c.class.len() + c.cov.len() + c.mag.len() + c.uses.len());
    let mut k = [0usize; 6];
    for e in app.map.pool().map.values() {
        k[0] += e.dem.as_ref().map_or(0, |d| d.texel.len() * 8);
        k[1] += e.dem_q.as_ref().map_or(0, |d| d.tex.len() * 2);
        k[2] += class(&e.line) + class(&e.land) + class(&e.own) + class(&e.mvum) + e.water.as_ref().map_or(0, |w| w.cov.len());
        k[3] += e.img.as_ref().map_or(0, |i| i.red.len() + i.green.len() + i.blue.len() + i.nir.len());
        k[4] += e.geoid.as_ref().map_or(0, |g| g.len() * 2);
        k[5] += (e.img.is_some()) as usize;
    }
    let m = |b: usize| b as f64 / MIB;
    println!("    terrain packed {:.1} MiB, terrain as a GPU uploads it {:.1}, lines, land, water, ownership and MVUM {:.1}, imagery {:.1} ({} cells), geoid {:.1}", m(k[0]), m(k[1]), m(k[2]), m(k[3]), k[5], m(k[4]));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (w, h) = (1344usize, 2992usize);
    // DeviceApp::open takes a local bake from `<dir>/cells-local`, as Android's hand-pushed cells.
    let dir = std::env::temp_dir().join(format!("mahere-heap-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    if let Some(cells) = a.get(1).filter(|s| !s.is_empty()) {
        std::os::unix::fs::symlink(std::fs::canonicalize(cells).unwrap(), dir.join("cells-local")).unwrap();
    }
    let at = |i: usize, d: f64| a.get(i).map_or(d, |s| s.parse().expect("a number"));
    let (lat, lon) = (at(2, 46.2), at(3, -121.5));
    // fluor's ferros host: its scratch frame and clip mask at the frame's size, and its own text renderer.
    let mut frame = vec![0u32; w * h];
    let clip = vec![255u8; w * h];
    let shell_text = fluor::text::TextRenderer::new();
    let mut app: mahere_device::DeviceApp = mahere_device::DeviceApp::open(w as u32, h as u32, None, Some(dir.to_str().unwrap()));
    app.w = w;
    app.h = h;
    line("opened (the shell's frame, clip mask and text with it)", &app);
    let themes = app.map.themes.clone();
    let mut draw = |app: &mut mahere_device::DeviceApp, frames: usize| {
        for _ in 0..frames {
            app.map.tick(w, h);
            app.map.render(w, h);
            let letters = mahere_panel::labels::Labeller::wanted(&app.map);
            if letters {
                app.labeller.ask(&mut app.map, w, h, app.panel.label_font_px(), true);
            }
            // The map, its lettering, then the panel over them, as the device's CPU path composes a frame.
            frame.clear();
            frame.extend_from_slice(&app.map.canvas);
            if letters {
                app.labeller.draw_cpu(&mut frame, &app.map.cam, w, h);
            }
            let (mask, ctl) = (app.map.layers(), app.controls());
            let areas = mahere_engine::areas::AreasView::default();
            app.panel.paint(w, h, mask, ctl, &mahere_panel::Readouts::default(), None, &themes, &areas, None);
            app.panel.composite_rgb(&mut frame, w, h);
            if app.map.converged() && !app.labeller.pending(&app.map, w, h) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
    };
    // The views: the globe a fresh install opens on, then a region, a valley, a trail's scale, the same with the panel open, and back out.
    draw(&mut app, 256);
    line("the globe", &app);
    for (what, ppd) in [("a region (2^11 px a degree)", 2048.0), ("a valley (2^14)", 16384.0), ("a trail (2^17)", 131072.0)] {
        app.map.cam = mahere_engine::Camera::new(lat, lon, ppd, 0.0);
        app.map.camera_moved(w, h);
        draw(&mut app, 2048);
        line(what, &app);
    }
    app.panel.toggle();
    draw(&mut app, 64);
    line("the panel open", &app);
    app.panel.toggle();
    for (what, dlat) in [("panned a screen north", 0.02), ("panned two screens north", 0.04)] {
        app.map.cam = mahere_engine::Camera::new(lat + dlat, lon, 131072.0, 0.0);
        app.map.camera_moved(w, h);
        draw(&mut app, 2048);
        line(what, &app);
    }
    app.map.cam = mahere_engine::Camera::new(lat, lon, 2048.0, 0.0);
    app.map.camera_moved(w, h);
    draw(&mut app, 2048);
    line("back out to the region", &app);
    // The imagery on (the most a view holds: every cell's bands beside its terrain), at the region and at a trail.
    let mut mask = app.map.layers();
    mask.imagery = true;
    app.map.set_layers(mask);
    for (what, ppd) in [("the region in imagery", 2048.0), ("a trail in imagery", 131072.0)] {
        app.map.cam = mahere_engine::Camera::new(lat, lon, ppd, 0.0);
        app.map.camera_moved(w, h);
        draw(&mut app, 2048);
        line(what, &app);
    }
    drop((shell_text, clip));
    let _ = std::fs::remove_dir_all(&dir);
}
