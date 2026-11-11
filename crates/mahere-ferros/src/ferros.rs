//! The ferros side of Mahere: the grants, the vault's secret, the GPS fix, and the touch server's contacts into the device app's two-finger solve, the rest going to fluor's ferros shell as its pointer.

use fluor::host::ferros::{Caps, Contact, FerrosShell, run_app_with};
use mahere_device::{DeviceApp, NoGpu};
use mahere_engine::GpsFix;
use std::sync::{Arc, Mutex};

/// The grant slots, in the spawn's order (slot 0 is the console): the display endpoint, the touch endpoint, this process's surface, the ira. The vault scope (`--store`) and the entropy follow, found by kete and ferros_rt by asking each grant; then `--use gps`, when the person grants it, and `--use udp`, the ip server (the network: the cells from the bucket behind the vault).
const DISPLAY: u64 = 1;
const TOUCH: u64 = 2;
const SURFACE: u64 = 3;
const IRA: u64 = 4;
const GPS: u64 = 7;
const UDP: u64 = 8;

type Shell = FerrosShell<DeviceApp<NoGpu>>;

/// The kernel, as this process reaches it (ferros KERNEL.md, the process ABI): the call, the ira key, the grants page. Inlined as fluor's and kete's are: ferros_sys is no_std and not a published crate.
pub(crate) mod sys {
    use core::arch::asm;

    pub const ERR: u64 = u64::MAX;
    const CALL: u64 = 1;
    const IRA: u64 = 23;

    #[inline(always)]
    fn svc(n: u64, a0: u64, a1: u64, a2: u64, a3: u64, a4: u64) -> u64 {
        let r: u64;
        unsafe { asm!("svc #0", inlateout("x0") a0 => r, in("x1") a1, in("x2") a2, in("x3") a3, in("x4") a4, in("x8") n, options(nostack)) };
        r
    }

    /// The arc base (x28, the kernel's, never written by compiled code on this target).
    #[inline(always)]
    fn arc_base() -> u64 {
        let b: u64;
        unsafe { asm!("mov {}, x28", out(reg) b, options(nomem, nostack, preserves_flags)) };
        b
    }

    #[inline(always)]
    fn off<T: ?Sized>(p: *const T) -> u64 {
        (p as *const u8 as u64).wrapping_sub(arc_base())
    }

    /// The grants page is arc offset 0: the handle table in its first 512 bytes, the u32 count at G#200, the 32-byte tokens after it (ferros SANDBOX.md, "Handles").
    pub fn grant_count() -> u64 {
        unsafe { ((arc_base() + 0x200) as *const u32).read_volatile() as u64 }
    }

    fn cap(slot: u64) -> u64 {
        0x204 + 32 * slot
    }

    pub fn call(slot: u64, msg: &[u8], reply: &mut [u8]) -> u64 {
        svc(CALL, cap(slot), off(msg.as_ptr()), msg.len() as u64, off(reply.as_mut_ptr()), reply.len() as u64)
    }

    /// A 32-byte key for `domain` rooted in the device's ira (ferros_sys `ira_key`): the same on every boot, never the ira itself; None without an Ira grant or an ira this boot.
    pub fn ira_key(slot: u64, domain: &str) -> Option<[u8; 32]> {
        let mut out = [0u8; 32];
        let r = svc(IRA, cap(slot), off(domain.as_ptr()), domain.len() as u64, off(out.as_mut_ptr()), 0);
        (r == 32).then_some(out)
    }
}

pub fn run() {
    std::panic::set_hook(Box::new(|info| {
        let at = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())).unwrap_or_default();
        println!("mahere: PANIC at {at}: {info}");
    }));
    println!("mahere: {} on ferros, {} grants", env!("CARGO_PKG_VERSION"), sys::grant_count());
    // The vault: this process's scope of the kernel's, kete's secret for it the device's ira key for Mahere (nothing kept beside it: there is no beside).
    let store = match sys::ira_key(IRA, "mahere vault") {
        Some(secret) => match mahere_store::open_with_secret(secret) {
            Ok(s) => Some(s),
            Err(e) => {
                println!("mahere: {e}; the map runs without its vault (no session, settings, areas, or cells kept)");
                None
            }
        },
        None => {
            println!("mahere: no ira key at slot {IRA} (spawn with --ira); the map runs without its vault");
            None
        }
    };
    // The network before the app opens its cell stores: with it the cells come from the vault, then the bucket over plain HTTP, kept in the vault as they come (as on every other platform); without it from the vault alone.
    if sys::grant_count() > UDP {
        match crate::net::IpNet::open(UDP) {
            Ok(net) => {
                mahere_engine::plain_http::set_dial(Arc::new(net));
                println!("mahere: the network at slot {UDP}; cells from {} behind the vault", mahere_engine::residency::DEFAULT_CELLS_URL);
            }
            Err(e) => println!("mahere: no network: {e}; cells from the vault alone"),
        }
    } else {
        println!("mahere: no network grant (slot {UDP}); cells from the vault alone");
    }
    // The size comes from the shell at once (`on_resize` as it is made): the panel's, the frame being the whole panel.
    let app: DeviceApp<NoGpu> = DeviceApp::open(0, 0, store, None);
    let fixes: Arc<Mutex<Option<GpsFix>>> = Arc::new(Mutex::new(None));
    if sys::grant_count() > GPS {
        let fixes = fixes.clone();
        std::thread::Builder::new().name("mahere-gps".into()).spawn(move || gps(fixes)).expect("gps thread");
    } else {
        println!("mahere: no gps grant; the map does not know where the device is");
    }
    let mut touch = Touch { fingers: Vec::with_capacity(16) };
    let caps = Caps { display: DISPLAY, touch: TOUCH, surface: SURFACE };
    let r = run_app_with(app, caps, 1.0, None, move |shell: &mut Shell, contacts: &[Contact]| {
        let fix = fixes.lock().unwrap().take();
        if let Some(f) = fix {
            shell.app().on_gps(f);
            shell.request_redraw();
        }
        for c in contacts {
            touch.contact(shell, c);
        }
    });
    match r {
        Ok(()) => println!("mahere: closed"),
        Err(e) => println!("mahere: {e}"),
    }
}

/// The `gps` endpoint's fix, once a second: op 0 FIX, 33 bytes (ferros payloads/gps): valid, the fix count, its age, latitude and longitude in 2⁻²³ degrees, altitude (m), error (m), speed and track. A new fix (its count moved on) goes to the frame loop. The altitude goes nowhere: its datum (ellipsoid or sea level) is the chip's and not yet known.
fn gps(fixes: Arc<Mutex<Option<GpsFix>>>) {
    let mut last = 0u32;
    let mut r = [0u8; 33];
    loop {
        let n = sys::call(GPS, &[0], &mut r);
        if n == sys::ERR {
            println!("mahere: the gps endpoint refused the call; no more fixes");
            return;
        }
        let w = |k: usize| [r[k], r[k + 1], r[k + 2], r[k + 3]];
        let count = u32::from_le_bytes(w(1));
        if n >= 33 && r[0] == 1 && count != last {
            last = count;
            let deg = |v: i32| v as f64 / (1u64 << 23) as f64;
            let fix = GpsFix { lat: deg(i32::from_le_bytes(w(9))), lon: deg(i32::from_le_bytes(w(13))), accuracy_m: u32::from_le_bytes(w(21)) as f32, height_m: None, v_accuracy_m: None };
            *fixes.lock().unwrap() = Some(fix);
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

/// What a touch event is to the two-finger solve, in Android's MotionEvent terms (mahere-android's `nativeOnTouch` is the other caller of the same app methods).
#[derive(Clone, Copy, PartialEq)]
enum Action {
    Down,
    PointerDown,
    Move,
    PointerUp,
    Up,
}

/// The fingers down, in the order they came: the first two are the solve's.
struct Touch {
    fingers: Vec<(u8, f64, f64)>,
}

impl Touch {
    /// One contact record (phase 0 enter, 1 move, 2 leave; frame pixels) as the Android shell's touch: two fingers to the solve (or, with the panel open, to the interface's size or the lettering's), one finger to fluor's shell as the pointer.
    fn contact(&mut self, shell: &mut Shell, c: &Contact) {
        let (x, y) = (c.x as f64, c.y as f64);
        let before = self.fingers.len();
        let (action, count) = match c.phase {
            0 => {
                self.fingers.retain(|f| f.0 != c.id);
                self.fingers.push((c.id, x, y));
                (if self.fingers.len() == 1 { Action::Down } else { Action::PointerDown }, self.fingers.len())
            }
            1 => {
                let Some(f) = self.fingers.iter_mut().find(|f| f.0 == c.id) else { return };
                (f.1, f.2) = (x, y);
                (Action::Move, before)
            }
            _ => {
                let Some(k) = self.fingers.iter().position(|f| f.0 == c.id) else { return };
                self.fingers[k] = (c.id, x, y);
                (if before >= 2 { Action::PointerUp } else { Action::Up }, before)
            }
        };
        let (x0, y0) = (self.fingers[0].1, self.fingers[0].2);
        let (x1, y1) = if count >= 2 { (self.fingers[1].1, self.fingers[1].2) } else { (x0, y0) };
        if c.phase == 2 {
            self.fingers.retain(|f| f.0 != c.id);
        }
        let app = shell.app();
        if action == Action::Down {
            app.first_down = (x0, y0);
        }
        // With the panel open, two fingers scale the interface or the map's lettering, by where the first came down; never the map.
        if app.panel.is_open() {
            if count >= 2 && action != Action::PointerUp {
                app.pinched = true;
                app.ui_pinch(x0, y0, x1, y1);
                shell.request_redraw();
                return;
            }
            if count >= 2 || app.ui_pinch.is_some() {
                app.ui_pinch = None;
                if app.pinch_on_panel {
                    app.save_settings();
                } else if let Some(store) = &app.store {
                    let _ = mahere_store::save_label_scale(store, app.panel.label_scale());
                }
                return;
            }
            pointer(shell, action, x0, y0);
            return;
        }
        if count >= 2 {
            if action == Action::PointerUp {
                app.two_end();
            } else if app.two.is_none() {
                app.two_begin(x0, y0, x1, y1);
            } else if action == Action::Move {
                app.two_update(x0, y0, x1, y1);
            }
            shell.request_redraw();
            return;
        }
        if app.two.is_some() {
            // The last finger of a pinch lifted or moved: the gesture closes first.
            app.two_end();
        }
        pointer(shell, action, x0, y0);
    }
}

/// One finger to fluor's shell as its pointer: pressed, moved, released.
fn pointer(shell: &mut Shell, action: Action, x: f64, y: f64) {
    let phase = match action {
        Action::Down => 0,
        Action::Move => 1,
        Action::Up => 2,
        // Only one finger reaches the pointer: a second finger's coming and going is the solve's.
        Action::PointerDown | Action::PointerUp => unreachable!("a two-finger action at the pointer"),
    };
    shell.on_contacts(&[Contact { id: 0, phase, x: x as u16, y: y as u16 }]);
}
