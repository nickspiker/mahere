//! mahere on ferros: the touch-device app (mahere-device, the one Android runs) as a ferros process, a kist. fluor's ferros host draws it into the display server's app surface and reads the touch server; what is ferros's is here: every contact into the two-finger solve or the pointer, the fix from the `gps` endpoint, and the vault — the process's scope of the kernel's, its kete secret derived from the device's ira, nothing kept anywhere else.
//!
//! Spawned (ferros docs/STD-TARGET.md, tools/mahere-build.sh) with, in this slot order: `--use display --use touch --dma surface --ira --store <scope> --entropy`, and `--use gps` last when the person lets the map know where they are (the fix is theirs: ferros GPS.md, never before attest).

#[cfg(target_os = "ferros")]
mod ferros;

#[cfg(target_os = "ferros")]
fn main() {
    ferros::run();
}

#[cfg(not(target_os = "ferros"))]
fn main() {
    eprintln!("mahere-ferros is Mahere as a ferros process: build it with the ferros repo's tools/mahere-build.sh, run it on a ferros device");
    std::process::exit(1);
}
