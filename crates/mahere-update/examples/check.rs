//! `check`: both channels' manifests as this build reads them.
fn main() {
    for c in [mahere_update::Channel::Release, mahere_update::Channel::Dev] {
        match mahere_update::fetch(c) {
            Ok(rows) => {
                for r in rows {
                    println!("{} {} {} {} (code {}) {} bytes {}", c.label(), r.platform, r.arch, r.version.iter().map(|n| n.to_string()).collect::<Vec<_>>().join("."), r.build, r.size, r.name);
                }
            }
            Err(e) => println!("{}: {e}", c.label()),
        }
    }
}
