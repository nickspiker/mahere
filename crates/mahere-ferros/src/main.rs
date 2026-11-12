//! mahere on ferros: the touch-device app (mahere-device, the one Android runs) as a ferros process, a kist. fluor's ferros host draws it into the display server's app surface and reads the touch server; what is ferros's is here: every contact into the two-finger solve or the pointer, the fix from the `gps` endpoint, the network (connections over the ip server, for the cells from the bucket), and the vault: the process's scope of the kernel's, its kete secret derived from the device's ira, nothing kept anywhere else.
//!
//! Spawned (ferros docs/STD-TARGET.md, tools/mahere-build.sh) with, in this slot order: `--use display --use touch --dma surface --ira --store <scope> --entropy`, then `--use gps` when the person lets the map know where they are (the fix is theirs: ferros GPS.md, never before attest), then `--use udp`, the network; then `--use gpu --dma layers`, the GPU service and the layer arena, for the map drawn on the GPU (`gpu.rs`; without them the CPU raster draws it). As a kist (ferros tools/mahere-kist.sh) the slots are its wants in that order, a want the person said no to an empty slot.

#[cfg(target_os = "ferros")]
mod ferros;
#[cfg(target_os = "ferros")]
mod net;
#[cfg(target_os = "ferros")]
mod gpu;
#[cfg(any(target_os = "ferros", test))]
mod kernel;
#[cfg(any(target_os = "ferros", test))]
mod arena;
#[cfg(test)]
mod tests;
// The addresses the connections dial, pure over its lookup: built for the host's tests too.
#[cfg(any(target_os = "ferros", test))]
mod names;

#[cfg(target_os = "ferros")]
fn main() {
    ferros::run();
}

#[cfg(not(target_os = "ferros"))]
fn main() {
    eprintln!("mahere-ferros is Mahere as a ferros process: build it with the ferros repo's tools/mahere-build.sh, run it on a ferros device");
    std::process::exit(1);
}
