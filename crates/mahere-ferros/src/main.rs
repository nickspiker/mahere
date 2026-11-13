//! mahere on ferros: the touch-device app (mahere-device, the one Android runs) as a ferros process, a kist. fluor's ferros host draws it into the display server's app surface and reads the touch server; what is ferros's is here: every contact into the two-finger solve or the pointer, the fix from the `gps` endpoint, the network (connections over the ip server, for the cells from the bucket), and the vault: the process's scope of the kernel's, its kete secret derived from the device's ira, nothing kept anywhere else.
//!
//! Spawned (ferros docs/STD-TARGET.md, tools/mahere-build.sh) with, in this slot order: `--use display --use touch --dma surface --ira --store <scope> --entropy`, then `--use gps` when the person lets the map know where they are (the fix is theirs: ferros GPS.md, never before attest), then `--use udp`, the network; then `--use gpu --dma layers --dma surface`, the GPU service, a block of the layer arena of its own and a second surface, for the map drawn on the GPU (`gpu.rs`; without them the CPU raster draws it). As a kist (ferros tools/mahere-kist.sh) the slots are its wants in that order, a want the person said no to an empty slot, and then the kernel's three GPU slots, which every kist is handed on a development build and none is asked for (ferros GPU-KISTS.md plan step 1, 2026-11-12).

#[cfg(target_os = "ferros")]
mod ferros;
#[cfg(target_os = "ferros")]
mod net;
#[cfg(any(target_os = "ferros", test))]
mod names;
#[cfg(target_os = "ferros")]
mod gpu;
#[cfg(any(target_os = "ferros", test))]
mod kernel;
#[cfg(any(target_os = "ferros", test))]
mod arena;
#[cfg(any(target_os = "ferros", test))]
mod letters;
#[cfg(test)]
mod tests;

/// The heap as Mahere uses it, counted at every allocation (two atomics on top of the runtime's allocator): the bytes held now and the most held, said on the console beside the arc's size so what fills the arc can be told from what the allocator keeps (ferros, 2026-11-12: the arc reached 538 MiB at the globe with the pool's cells at 190, and an 11.9 MB allocation failed at the 640 MiB ceiling on the 8a).
#[cfg(target_os = "ferros")]
pub mod heap {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    pub struct Counting;
    static HELD: AtomicUsize = AtomicUsize::new(0);
    static MOST: AtomicUsize = AtomicUsize::new(0);

    fn grew(n: usize) {
        let now = HELD.fetch_add(n, Relaxed) + n;
        MOST.fetch_max(now, Relaxed);
    }

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let p = unsafe { System.alloc(l) };
            if !p.is_null() {
                grew(l.size());
            }
            p
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            let p = unsafe { System.alloc_zeroed(l) };
            if !p.is_null() {
                grew(l.size());
            }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) };
            HELD.fetch_sub(l.size(), Relaxed);
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            let q = unsafe { System.realloc(p, l, n) };
            if !q.is_null() {
                HELD.fetch_sub(l.size(), Relaxed);
                grew(n);
            }
            q
        }
    }

    /// The bytes held now and the most held since the last call (the most then starts again from now).
    pub fn take() -> (usize, usize) {
        let now = HELD.load(Relaxed);
        (now, MOST.swap(now, Relaxed))
    }
}

#[cfg(target_os = "ferros")]
#[global_allocator]
static HEAP: heap::Counting = heap::Counting;

#[cfg(target_os = "ferros")]
fn main() {
    ferros::run();
}

#[cfg(not(target_os = "ferros"))]
fn main() {
    eprintln!("mahere-ferros is Mahere as a ferros process: build it with the ferros repo's tools/mahere-build.sh, run it on a ferros device");
    std::process::exit(1);
}
