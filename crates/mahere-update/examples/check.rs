//! `check`: both channels' manifests as this build reads them.
fn main() {
    for c in [mahere_update::Channel::Release, mahere_update::Channel::Dev] {
        match mahere_update::fetch(c) {
            Ok(rows) => {
                for r in rows {
                    println!("{} {} {} build {} {} bytes {}", c.label(), r.platform, r.arch, r.build, r.size, r.name);
                }
            }
            Err(e) => println!("{}: {e}", c.label()),
        }
    }
}
