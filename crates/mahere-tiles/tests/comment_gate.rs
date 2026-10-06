//! The wrapped-comment gate runs with `cargo test`: scripts/comment-gate.sh over the whole workspace, zero baseline. A failure lists file:line of every comment line that continues onto the next; join them (one thought per line).
use std::process::Command;

#[test]
fn no_hard_wrapped_comments_anywhere() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = Command::new("sh").arg(root.join("scripts/comment-gate.sh")).output().expect("run comment-gate.sh");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
