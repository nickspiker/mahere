//! Updates (Nick 2026-10-13: "steal the update button from Photon", a few platforms): two channels, release and dev, each a VSF manifest in the bucket signed with the author's key, one section per platform and architecture naming the build, where it lives, its size and its BLAKE3. The app checks a channel, takes its own platform's row, downloads the build, checks it against the signed hash, and hands it on to be installed (Android's package installer, or the desktop binary swapped in place). Version is the build number, the repository's commit count, the same everywhere.
use std::io::Read;
use std::path::{Path, PathBuf};

use vsf::VsfType;

/// The author's public key: Nick Spiker's, the one Photon's builds are signed with. A manifest signed by any other key is refused.
pub const AUTHOR_PUBKEY: [u8; 32] = [
    0xdf, 0xf3, 0xaf, 0x0c, 0x12, 0x7c, 0x0b, 0xeb, 0xe5, 0x39, 0xc4, 0x21, 0xda, 0x37, 0x99, 0x3a, 0x51, 0x7b, 0xfd, 0x78, 0xd2, 0xf5, 0xee, 0x49, 0x1d, 0x52, 0xfb, 0xf6, 0x16, 0x44, 0x47, 0x47,
];

/// Where the manifests and builds live.
pub const BASE: &str = "https://brobdingnagian.holdmyoscilloscope.com/mahere";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    Release,
    Dev,
}

impl Channel {
    pub fn label(self) -> &'static str {
        match self {
            Channel::Release => "release",
            Channel::Dev => "dev",
        }
    }

    pub fn section(self) -> &'static str {
        match self {
            Channel::Release => "manifest.mahere.release",
            Channel::Dev => "manifest.mahere.development",
        }
    }

    pub fn url(self) -> String {
        format!("{BASE}/manifest-{}.vsf", self.label())
    }

    pub fn parse(s: &str) -> Option<Channel> {
        match s {
            "release" => Some(Channel::Release),
            "dev" | "development" => Some(Channel::Dev),
            _ => None,
        }
    }
}

/// One build in a manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub platform: String,
    pub arch: String,
    /// Android's version code, minor times a thousand plus patch: what builds from before the version numbers compared (they take any release whose build is not theirs).
    pub build: u64,
    /// The version, major, minor and patch (Nick 2026-10-15: Photon's numbering): a release is x.y.0, a dev build carries its patch. [0, 0, 0] for a row from before.
    pub version: [u64; 3],
    /// What a person reads, "0.216.0+457980b".
    pub name: String,
    pub url: String,
    pub hash: [u8; 32],
    pub size: u64,
}

/// A version from its text, "0.216.1" or "0.216.1+457980b".
pub fn parse_version(s: &str) -> Option<[u64; 3]> {
    let core = s.split('+').next()?;
    let mut it = core.split('.').map(|p| p.parse::<u64>().ok());
    let v = [it.next()??, it.next()??, it.next()??];
    it.next().is_none().then_some(v)
}

/// This build's version, the workspace's own: a release x.y.0, a dev build x.y.patch, and the tree between publishes at the next dev patch, so a local build never wears a published version.
pub fn our_version() -> [u64; 3] {
    parse_version(env!("CARGO_PKG_VERSION")).unwrap_or([0, 0, 0])
}

/// Android's version code for a version: minor times a thousand plus patch, rising with every build.
pub fn version_code(v: [u64; 3]) -> u64 {
    v[0] * 1_000_000 + v[1] * 1000 + v[2]
}

/// This build's number, stamped by the publish scripts (MAHERE_VERSION_CODE); 0 for a local build, which no channel ever counts as current.
pub fn our_build() -> u64 {
    option_env!("MAHERE_VERSION_CODE").and_then(|s| s.parse().ok()).unwrap_or(0)
}

/// This build's name, "0.178+457980b", or "local" for a build no script stamped.
pub fn our_name() -> &'static str {
    option_env!("MAHERE_VERSION_NAME").unwrap_or("local")
}

/// This build's platform and architecture as the manifests name them.
pub fn our_platform() -> (&'static str, &'static str) {
    let platform = if cfg!(target_os = "android") {
        "Android"
    } else if cfg!(target_os = "macos") {
        "macOS"
    } else if cfg!(target_os = "windows") {
        "Windows"
    } else {
        "Linux"
    };
    let arch = if cfg!(target_arch = "aarch64") { "arm64" } else { "x86_64" };
    (platform, arch)
}

/// A manifest's rows for a channel, after its signature is checked against the author's key. Fields are read by name and type, never by position.
pub fn parse(bytes: &[u8], channel: Channel) -> Result<Vec<Row>, String> {
    let (header, end) = vsf::verification::read_verified(bytes, Some(AUTHOR_PUBKEY)).map_err(|e| format!("manifest not signed by the author: {e}"))?;
    let sections = header.sections(bytes, end).map_err(|e| format!("manifest sections: {e}"))?;
    let mut rows = Vec::new();
    for s in sections.iter().filter(|s| s.name == channel.section()) {
        let first = |name: &str| s.get_fields(name).first().and_then(|f| f.values.first()).cloned();
        let text = |name: &str| match first(name) {
            Some(VsfType::x(t) | VsfType::nu(t)) => Some(t),
            _ => None,
        };
        let num = |name: &str| first(name).and_then(|v| match v {
            VsfType::z(n) => Some(n as u64),
            other => other.as_usize().map(|n| n as u64),
        });
        let hash = match first("hash") {
            Some(VsfType::hb(h)) if h.len() == 32 => h.as_slice().try_into().ok(),
            _ => None,
        };
        let (Some(platform), Some(arch), Some(url), Some(hash), Some(build)) = (text("platform"), text("arch"), text("url"), hash, num("build")) else { continue };
        // Major is left out while 0 and patch when 0 (a release), as Photon writes them; a row with no minor is from before the version numbers, older than any with one.
        let version = match num("minor") {
            Some(minor) => [num("major").unwrap_or(0), minor, num("patch").unwrap_or(0)],
            None => [0, 0, 0],
        };
        rows.push(Row { platform, arch, build, version, name: text("name").unwrap_or_default(), url, hash, size: num("size").unwrap_or(0) });
    }
    Ok(rows)
}

/// This platform's row.
pub fn our_row(rows: &[Row]) -> Option<Row> {
    let (p, a) = our_platform();
    rows.iter().find(|r| r.platform == p && r.arch == a).cloned()
}

fn agent(timeout_s: u64) -> ureq::Agent {
    ureq::AgentBuilder::new().timeout_connect(std::time::Duration::from_secs(10)).timeout_read(std::time::Duration::from_secs(timeout_s)).build()
}

/// A channel's manifest from the bucket, past any cache (a fresh query each time), checked and read.
pub fn fetch(channel: Channel) -> Result<Vec<Row>, String> {
    let bust = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let resp = agent(30).get(&format!("{}?v={bust}", channel.url())).set("Cache-Control", "no-cache").call().map_err(|e| format!("{} manifest: {e}", channel.label()))?;
    let mut bytes = Vec::new();
    resp.into_reader().take(4 << 20).read_to_end(&mut bytes).map_err(|e| format!("manifest read: {e}"))?;
    parse(&bytes, channel)
}

/// This platform's build on a channel, if the channel has one.
pub fn check(channel: Channel) -> Result<Option<Row>, String> {
    fetch(channel).map(|rows| our_row(&rows))
}

/// Download a build to `dest` and check it against the signed hash; nothing is kept unless it matches. A file already there with the right hash is taken as it is (a second tap does not fetch it again). `progress(done, total)` as it comes in.
pub fn download(row: &Row, dest: &Path, progress: &dyn Fn(u64, u64)) -> Result<(), String> {
    if let Ok(have) = std::fs::read(dest) {
        if blake3::hash(&have).as_bytes() == &row.hash {
            progress(have.len() as u64, have.len() as u64);
            return Ok(());
        }
    }
    // The URL changes exactly when the content does, so a CDN edge can never hand back an older build under a new hash.
    let url = format!("{}?v={}", row.url, hex(&row.hash));
    let resp = agent(600).get(&url).set("Cache-Control", "no-cache").call().map_err(|e| format!("download: {e}"))?;
    let total = resp.header("Content-Length").and_then(|v| v.parse().ok()).unwrap_or(row.size);
    let mut reader = resp.into_reader();
    let mut out = Vec::with_capacity(total as usize);
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("download: {e}"))?;
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
        progress(out.len() as u64, total);
    }
    if blake3::hash(&out).as_bytes() != &row.hash {
        return Err("the download does not match the signed manifest".into());
    }
    let part = dest.with_extension("part");
    std::fs::write(&part, &out).map_err(|e| format!("save: {e}"))?;
    std::fs::rename(&part, dest).map_err(|e| format!("save: {e}"))?;
    Ok(())
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A desktop binary's own signature checked, without running it (after Photon's): its last 64 bytes are the author's Ed25519 signature over the BLAKE3 of everything before them.
pub fn verify_file(path: &Path) -> Result<(), String> {
    use ed25519_dalek::Verifier;
    let mut data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if data.len() < 64 {
        return Err("too small to carry a signature".into());
    }
    let sig = data.split_off(data.len() - 64);
    if sig.iter().all(|&b| b == 0) {
        return Err("not signed".into());
    }
    let sig = ed25519_dalek::Signature::from_bytes(sig.as_slice().try_into().map_err(|_| "bad signature")?);
    let key = ed25519_dalek::VerifyingKey::from_bytes(&AUTHOR_PUBKEY).map_err(|e| format!("author key: {e}"))?;
    key.verify(blake3::hash(&data).as_bytes(), &sig).map_err(|_| "the signature does not match: the file is damaged or was changed".to_string())
}

/// This program's own signature, checked as it starts. A local build (no build number stamped) carries none and is let through, so a build from source runs; a published build must be signed by the author.
pub fn verify_self() -> Result<(), String> {
    if our_build() == 0 {
        return Ok(());
    }
    verify_file(&std::env::current_exe().map_err(|e| format!("where am I: {e}"))?)
}

/// Sign a desktop binary in place: the author's signature over its BLAKE3 appended (the publish tool's side).
#[cfg(feature = "publish")]
pub fn sign_binary(path: &Path, key: &[u8; 32]) -> Result<(), String> {
    use ed25519_dalek::Signer;
    let mut data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let k = ed25519_dalek::SigningKey::from_bytes(key);
    if k.verifying_key().to_bytes() != AUTHOR_PUBKEY {
        return Err("that key is not the author's".into());
    }
    let sig = k.sign(blake3::hash(&data).as_bytes());
    data.extend_from_slice(&sig.to_bytes());
    std::fs::write(path, &data).map_err(|e| format!("write: {e}"))?;
    verify_file(path)
}

/// A desktop build swapped in for the running one: the verified file (its signature checked again) made executable and renamed over the program (Windows first moves the running one aside). Returns the program's path, for the caller to start again.
#[cfg(not(target_os = "android"))]
pub fn install_desktop(staged: &Path) -> Result<PathBuf, String> {
    // Its own signature as well as the manifest's hash: nothing is swapped in that the author did not sign.
    verify_file(staged)?;
    let exe = std::env::current_exe().map_err(|e| format!("where am I: {e}"))?;
    // After an earlier swap the running image's path reads "... (deleted)".
    let exe = PathBuf::from(exe.to_string_lossy().trim_end_matches(" (deleted)"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(staged, std::fs::Permissions::from_mode(0o755)).map_err(|e| format!("chmod: {e}"))?;
    }
    #[cfg(windows)]
    {
        let old = exe.with_extension("old.exe");
        let _ = std::fs::remove_file(&old);
        std::fs::rename(&exe, &old).map_err(|e| format!("move aside: {e}"))?;
    }
    std::fs::rename(staged, &exe).map_err(|e| format!("swap in: {e}"))?;
    Ok(exe)
}

#[cfg(feature = "publish")]
/// Build and sign a manifest of a channel's rows (the publish tool's side), and check the result reads back through [`parse`] as the app will read it.
pub fn write(channel: Channel, rows: &[Row], key: &[u8; 32]) -> Result<Vec<u8>, String> {
    let signer = ed25519_dalek::SigningKey::from_bytes(key).verifying_key().to_bytes().to_vec();
    let mut b = vsf::VsfBuilder::new().creation_time_oscillations(vsf::eagle_time_oscillations()).signed_only(VsfType::ke(signer));
    for r in rows {
        let mut s = vsf::VsfSection::new(channel.section());
        s.add_field("platform", VsfType::x(r.platform.clone()));
        s.add_field("arch", VsfType::x(r.arch.clone()));
        s.add_field("build", VsfType::z(r.build as usize));
        if r.version[0] > 0 {
            s.add_field("major", VsfType::z(r.version[0] as usize));
        }
        s.add_field("minor", VsfType::z(r.version[1] as usize));
        if r.version[2] > 0 {
            s.add_field("patch", VsfType::z(r.version[2] as usize));
        }
        s.add_field("name", VsfType::x(r.name.clone()));
        s.add_field("size", VsfType::z(r.size as usize));
        s.add_field("url", VsfType::nu(r.url.clone()));
        s.add_field("hash", VsfType::hb(r.hash.to_vec()));
        b = b.add_section_direct(s);
    }
    let signed = vsf::verification::sign_file(b.build().map_err(|e| format!("build: {e}"))?, key)?;
    let back = parse(&signed, channel)?;
    if back != rows {
        return Err(format!("the manifest does not read back: wrote {} rows, read {}", rows.len(), back.len()));
    }
    Ok(signed)
}

#[cfg(all(test, feature = "publish"))]
mod tests {
    use super::*;

    #[test]
    fn a_manifest_signed_by_another_key_is_refused() {
        let row = Row { platform: "Android".into(), arch: "arm64".into(), build: 216000, version: [0, 216, 0], name: "0.216.0+457980b".into(), url: format!("{BASE}/android/mahere-178.apk"), hash: [7; 32], size: 1234 };
        // Any key but the author's: the manifest is well formed but not his.
        let other = [9u8; 32];
        let bytes = write(Channel::Dev, std::slice::from_ref(&row), &other);
        assert!(bytes.is_err(), "a manifest by another key must not read back");
    }
}

#[cfg(all(test, feature = "publish"))]
mod signed_tests {
    use super::*;

    /// With the author's key on this machine, a manifest written and signed reads back whole.
    #[test]
    fn a_manifest_signed_by_the_author_reads_back() {
        let Ok(k) = std::fs::read("/mnt/Harbor/Code/keys/photon-signing-key") else { return };
        let key: [u8; 32] = k.try_into().unwrap();
        let rows = vec![Row { platform: "Android".into(), arch: "arm64".into(), build: 216000, version: [0, 216, 0], name: "0.216.0+457980b".into(), url: format!("{BASE}/android/mahere-178.apk"), hash: [7; 32], size: 1234 }];
        let bytes = write(Channel::Release, &rows, &key).unwrap();
        assert_eq!(parse(&bytes, Channel::Release).unwrap(), rows);
        assert!(parse(&bytes, Channel::Dev).unwrap().is_empty(), "a release manifest has no dev rows");
    }
}

/// What is known of a channel.
#[derive(Clone, Debug, PartialEq)]
pub enum Known {
    Unchecked,
    Checking,
    Failed(String),
    /// The channel's build for this platform, if it has one.
    Has(Option<Row>),
}

enum Event {
    Checked(Channel, Result<Option<Row>, String>),
    Progress(u64, u64),
    Downloaded(PathBuf),
    Failed(String),
}

/// How long after starting the first automatic check waits, and how often one comes after.
const FIRST_CHECK: std::time::Duration = std::time::Duration::from_secs(60);
const EVERY: std::time::Duration = std::time::Duration::from_secs(4 * 3600);

/// The app's side: both channels checked on threads of their own, a build downloaded and verified, and the file handed to the host to install. With `auto` on, the release channel is checked a minute after start and every four hours, and a different release build is taken where the host says the network is free to (unmetered).
pub struct Updater {
    pub release: Known,
    pub dev: Known,
    /// One line of what is happening: downloading, ready, failed.
    pub status: String,
    /// Bytes in and expected, while downloading.
    pub progress: Option<(u64, u64)>,
    pub auto: bool,
    /// Whether the network is unmetered, as the host last said.
    pub unmetered: bool,
    dir: PathBuf,
    tx: std::sync::mpsc::Sender<Event>,
    rx: std::sync::mpsc::Receiver<Event>,
    busy: bool,
    ready: Option<PathBuf>,
    started: std::time::Instant,
    last_auto: Option<std::time::Instant>,
    /// The channel the running download is for.
    getting: Option<Channel>,
}

impl Updater {
    /// `dir`: where a downloaded build is kept until it is installed.
    pub fn new(dir: PathBuf) -> Updater {
        let (tx, rx) = std::sync::mpsc::channel();
        Updater { release: Known::Unchecked, dev: Known::Unchecked, status: String::new(), progress: None, auto: true, unmetered: !cfg!(target_os = "android"), dir, tx, rx, busy: false, ready: None, started: std::time::Instant::now(), last_auto: None, getting: None }
    }

    pub fn known(&self, c: Channel) -> &Known {
        match c {
            Channel::Release => &self.release,
            Channel::Dev => &self.dev,
        }
    }

    fn known_mut(&mut self, c: Channel) -> &mut Known {
        match c {
            Channel::Release => &mut self.release,
            Channel::Dev => &mut self.dev,
        }
    }

    /// Check both channels.
    pub fn check(&mut self) {
        for c in [Channel::Release, Channel::Dev] {
            if *self.known(c) == Known::Checking {
                continue;
            }
            *self.known_mut(c) = Known::Checking;
            let tx = self.tx.clone();
            std::thread::spawn(move || {
                let _ = tx.send(Event::Checked(c, check(c)));
            });
        }
    }

    /// The channel's build, when it is not the one running: what "Get" would install (older too: changing channel is a tap).
    pub fn offer(&self, c: Channel) -> Option<&Row> {
        match self.known(c) {
            Known::Has(Some(r)) if r.version != our_version() => Some(r),
            _ => None,
        }
    }

    /// The release to take on its own: a release (patch 0) strictly newer than the running build, never a dev build and never a step back (Nick 2026-10-15: it tried to put release 211 over dev 215).
    pub fn newer_release(&self) -> Option<&Row> {
        match self.known(Channel::Release) {
            Known::Has(Some(r)) if r.version[2] == 0 && r.version > our_version() => Some(r),
            _ => None,
        }
    }

    pub fn busy(&self) -> bool {
        self.busy
    }

    /// Download the channel's build (any build other than this one, older too: changing channel is a tap).
    pub fn get(&mut self, c: Channel) {
        if self.busy {
            return;
        }
        let Some(row) = self.offer(c).cloned() else { return };
        self.busy = true;
        self.getting = Some(c);
        self.status = format!("Downloading {} {}", c.label(), row.name.split('+').next().unwrap_or(&row.name));
        self.progress = Some((0, row.size));
        let ext = if row.url.ends_with(".apk") { "apk" } else { "bin" };
        let dest = self.dir.join(format!("mahere-update.{ext}"));
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let p = tx.clone();
            let progress = move |done: u64, total: u64| {
                let _ = p.send(Event::Progress(done, total));
            };
            let _ = tx.send(match download(&row, &dest, &progress) {
                Ok(()) => Event::Downloaded(dest),
                Err(e) => Event::Failed(e),
            });
        });
    }

    /// Take in what the threads found; true when anything shown changed. Runs the automatic check when it is due.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        if self.auto && !self.busy && self.unmetered && self.started.elapsed() >= FIRST_CHECK && self.last_auto.is_none_or(|t| t.elapsed() >= EVERY) {
            self.last_auto = Some(std::time::Instant::now());
            self.check();
            changed = true;
        }
        while let Ok(e) = self.rx.try_recv() {
            changed = true;
            match e {
                Event::Checked(c, Ok(row)) => {
                    *self.known_mut(c) = Known::Has(row);
                    // Automatic: a newer release is taken as soon as it is seen; nothing else ever is.
                    if c == Channel::Release && self.auto && self.unmetered && self.last_auto.is_some() && self.newer_release().is_some() {
                        self.get(c);
                    }
                }
                Event::Checked(c, Err(e)) => *self.known_mut(c) = Known::Failed(e),
                Event::Progress(done, total) => self.progress = Some((done, total)),
                Event::Downloaded(path) => {
                    self.busy = false;
                    self.progress = None;
                    self.status = "Downloaded \u{2713}  installing\u{2026}".into();
                    self.ready = Some(path);
                }
                Event::Failed(e) => {
                    self.busy = false;
                    self.progress = None;
                    self.status = format!("Update failed, nothing changed: {e}");
                }
            }
        }
        changed
    }

    /// A downloaded, verified build for the host to install, once.
    pub fn take_ready(&mut self) -> Option<PathBuf> {
        self.ready.take()
    }
}

#[cfg(all(test, feature = "publish"))]
mod binary_tests {
    use super::*;

    #[test]
    fn a_signed_binary_verifies_and_a_changed_one_does_not() {
        let Ok(k) = std::fs::read("/mnt/Harbor/Code/keys/photon-signing-key") else { return };
        let key: [u8; 32] = k.try_into().unwrap();
        let dir = std::env::temp_dir().join(format!("mahere-sign-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("bin");
        std::fs::write(&f, b"a program's bytes").unwrap();
        assert!(verify_file(&f).is_err(), "an unsigned file passed");
        sign_binary(&f, &key).unwrap();
        assert!(verify_file(&f).is_ok());
        let mut b = std::fs::read(&f).unwrap();
        b[0] ^= 1;
        std::fs::write(&f, &b).unwrap();
        assert!(verify_file(&f).is_err(), "a changed file passed");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
