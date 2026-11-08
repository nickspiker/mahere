fn main() {
    // The heap a ferros build links in (ferros_rt/ferros.ld's __ferros_heap): the full-panel frame three times over (the shell's scratch, the map's canvas and the frame the lettering and panel go into: 1344 × 2992 × 4 bytes, 16 MiB each), the decoded cells, the panel, the fonts. The arc rounds the image plus this up to a power of two and ferros_rt's allocator takes the slack past it too.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("ferros") {
        println!("cargo:rustc-link-arg=--defsym=__ferros_heap={}", (1u64 << 27) + (1 << 25));
    }
}
