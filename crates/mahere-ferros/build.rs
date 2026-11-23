fn main() {
    // The heap a ferros build links in (ferros_rt/ferros.ld's __ferros_heap): a ceiling the process grows to, not a reservation, once ferros docs/ARCS.md step 2 is built (until then the kernel gives the arc all of it at spawn). Measured with mahere-device's `heap` example (the device app at the Pro's 1344 x 2992, composed as ferros composes it, a counting allocator under it, cells from the bucket around Mount Adams, 2026-11-11):
    // - The frames, 34.5 MiB: fluor's ferros shell's frame (1344 x 2992 x 4 bytes, 15.3 MiB) and its clip mask (a byte a pixel, 3.8 MiB), the map's canvas (15.3 MiB). Nothing else is a full screen since the panel and the lettering go straight into the shell's frame.
    // - The rest of the app without a cell, 30 MiB steady and 37 at its peak: the lettering's glyph atlases (8 MiB each, two held and a third while a new one is handed over), three text renderers with their faces, the themes.
    // - Font packs as the text asks for them, about 37 MiB at the globe (Chinese for Hong Kong 11.4 MiB, Simplified 17 MiB, a dozen small scripts).
    // - The decoded cells, the bulk, as many as the view holds, with no budget of their own: 300 MiB at a trail's scale (188 cells), 392 at a region's (259 cells), 422 with imagery on (terrain packed for the CPU 184 MiB, the same terrain as a GPU uploads it 47, lines, land, water, ownership and MVUM 154, imagery 34, the geoid 2.5).
    // - In flight: cells decoded but not yet drained, the loaders' buffers. The measured peak of all of it was 576.4 MiB.
    // - Thread stacks, from the heap on ferros (256 KiB each, ferros-rust std): about 28 threads (the 20 cell loaders, 4 rayon workers, persist, gps, lettering, font packs), 7 MiB.
    // So 576 + 7 = 583 MiB, and the ceiling the next sum of powers of two above it: 2^29 + 2^27 = 640 MiB.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("ferros") {
        println!("cargo:rustc-link-arg=--defsym=__ferros_heap={}", (1u64 << 29) + (1 << 27));
    }
}
