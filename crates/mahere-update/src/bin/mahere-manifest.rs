//! The publish side of updates. Usage:
//!   mahere-manifest hash <file>                       its BLAKE3 and size, for a script
//!   mahere-manifest sign <file>                       a desktop binary signed in place (the author's signature over its BLAKE3, appended)
//!   mahere-manifest write <channel> <out> [--merge <existing manifest>] --row <platform> <arch> <build> <name> <url> <b3 hex> <size> [--row ...]
//!   mahere-manifest promote <dev manifest> <build> <out> [--merge <existing release manifest>]   every row of that build from dev into release
//! A new row replaces the merged manifest's row for the same platform and architecture; the others are kept. Signed with the author's key ($MAHERE_SIGNING_KEY, else the key Photon's builds are signed with), and read back as the app reads it before anything is written.
use mahere_update::{Channel, Row};

fn fail(msg: &str) -> ! {
    eprintln!("mahere-manifest: {msg}");
    std::process::exit(1)
}

fn key() -> [u8; 32] {
    let path = std::env::var("MAHERE_SIGNING_KEY").ok().or_else(|| ["/mnt/Harbor/Code/keys/photon-signing-key", "/mnt/Chiton/MEGA/Code/keys/photon-signing-key"].iter().map(|s| s.to_string()).find(|p| std::path::Path::new(p).exists())).unwrap_or_else(|| fail("no signing key: set MAHERE_SIGNING_KEY"));
    std::fs::read(&path).unwrap_or_else(|e| fail(&format!("{path}: {e}"))).try_into().unwrap_or_else(|_| fail("the signing key must be exactly 32 bytes"))
}

fn unhex(s: &str) -> [u8; 32] {
    let v: Vec<u8> = (0..s.len()).step_by(2).filter_map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok()).collect();
    v.try_into().unwrap_or_else(|_| fail("a hash is 64 hex digits"))
}

/// The rows of a manifest file, or none when it is missing or unreadable (a first publish).
fn rows_of(path: &str, channel: Channel) -> Vec<Row> {
    match std::fs::read(path) {
        Ok(b) => mahere_update::parse(&b, channel).unwrap_or_else(|e| {
            eprintln!("mahere-manifest: {path} unreadable ({e}), starting fresh");
            Vec::new()
        }),
        Err(_) => Vec::new(),
    }
}

/// `new` over `old`: a platform and architecture `new` has replaces `old`'s.
fn merge(mut new: Vec<Row>, old: Vec<Row>) -> Vec<Row> {
    for r in old {
        if !new.iter().any(|n| n.platform == r.platform && n.arch == r.arch) {
            new.push(r);
        }
    }
    new.sort_by(|a, b| (&a.platform, &a.arch).cmp(&(&b.platform, &b.arch)));
    new
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| a.get(i).cloned().unwrap_or_else(|| fail("missing arguments; see the usage at the top of mahere-manifest.rs"));
    let merged_from = a.iter().position(|x| x == "--merge").map(|i| arg(i + 1));
    match a.first().map(String::as_str) {
        Some("hash") => {
            let b = std::fs::read(arg(1)).unwrap_or_else(|e| fail(&e.to_string()));
            println!("{} {}", mahere_update::hex(blake3::hash(&b).as_bytes()), b.len());
        }
        Some("sign") => {
            mahere_update::sign_binary(std::path::Path::new(&arg(1)), &key()).unwrap_or_else(|e| fail(&e));
            println!("signed {}", arg(1));
        }
        Some("write") => {
            let channel = Channel::parse(&arg(1)).unwrap_or_else(|| fail("channel is release or dev"));
            let mut rows = Vec::new();
            for (i, x) in a.iter().enumerate() {
                if x == "--row" {
                    rows.push(Row { platform: arg(i + 1), arch: arg(i + 2), build: arg(i + 3).parse().unwrap_or_else(|_| fail("build is a number")), name: arg(i + 4), url: arg(i + 5), hash: unhex(&arg(i + 6)), size: arg(i + 7).parse().unwrap_or_else(|_| fail("size is a number")) });
                }
            }
            let rows = merge(rows, merged_from.map_or(Vec::new(), |p| rows_of(&p, channel)));
            let bytes = mahere_update::write(channel, &rows, &key()).unwrap_or_else(|e| fail(&e));
            std::fs::write(arg(2), bytes).unwrap_or_else(|e| fail(&e.to_string()));
            for r in &rows {
                println!("{} {} {} {} build {}", channel.label(), r.platform, r.arch, r.name, r.build);
            }
        }
        Some("promote") => {
            let build: u64 = arg(2).parse().unwrap_or_else(|_| fail("build is a number"));
            let dev = rows_of(&arg(1), Channel::Dev);
            let chosen: Vec<Row> = dev.into_iter().filter(|r| r.build == build).collect();
            if chosen.is_empty() {
                fail(&format!("the dev manifest has no row of build {build}"));
            }
            let rows = merge(chosen, merged_from.map_or(Vec::new(), |p| rows_of(&p, Channel::Release)));
            let bytes = mahere_update::write(Channel::Release, &rows, &key()).unwrap_or_else(|e| fail(&e));
            std::fs::write(arg(3), bytes).unwrap_or_else(|e| fail(&e.to_string()));
            for r in &rows {
                println!("release {} {} {} build {}", r.platform, r.arch, r.name, r.build);
            }
        }
        _ => fail("hash, sign, write or promote"),
    }
}
