//! The fleet (Nick 2026-10-22; ported 2026-11-23 from the vault branch that never reached main): the person's preferences and themes follow them across their devices as Photon's language does, entries in the fleet-shared state (fstate) on FGTW, sealed under the fleet key, merged key by key, newest write wins. Mahere's entries are `mahere.*` keys beside Photon's; a push pulls first and merges, so it never disturbs the roster or Photon's own settings.
//! Each preference is an entry of its own, natively typed as fstate v7 wants (a preference changed on one device and another changed on a second both survive the merge); a theme is its VSF document under a key named by the hash of its name (fgtw's keys are ASCII, a theme's name is whatever the person typed).
//! Mahere joins as the same fleet member Photon is on this machine: the device key comes from the machine fingerprint, and the fleet key is recovered from the fan-out with the person's session, never minted here.
//! What never goes: the camera (where the person last looked), the cell cache, the UI scale, sensors, recording, the tool: what only makes sense on one device.

use fgtw::client::{FgtwResponse, FgtwTransport, FleetSealer};
use fgtw::fstate::{FleetState, fstate_from_bytes, fstate_to_bytes, merge_fstate};
pub use fgtw::fstate::SettingEntry;
use mahere_engine::theme::Theme;
use vsf::{Tensor, VsfType};

use crate::{Settings, Store, name, vault_key};

const FGTW_URL: &str = "https://fgtw.org";

/// Every key Mahere writes into the fleet state begins so.
pub const PREFIX: &str = "mahere.";
const LAYERS: &str = "mahere.layers";
/// The theme in use, by name: the lists of themes differ from device to device, so never by its place in one.
const THEME: &str = "mahere.theme";
const COMPRESSED: &str = "mahere.compressed";
const RULER: &str = "mahere.ruler";
const PEN: &str = "mahere.pen";
const TRAVEL: &str = "mahere.travel";
/// The held contour interval in metres; a tombstone when the interval fits the view.
const CONTOUR_HOLD: &str = "mahere.contourhold";
const AUTO_UPDATE: &str = "mahere.autoupdate";
/// A theme's document, the hash of its name after this.
const THEME_DOC: &str = "mahere.themes.";

struct Http;

impl FgtwTransport for Http {
    fn post(&self, body: Vec<u8>) -> Result<FgtwResponse, String> {
        let r = ureq::post(FGTW_URL).set("Content-Type", "application/octet-stream").timeout(std::time::Duration::from_secs(15)).send_bytes(&body);
        let resp = match r {
            Ok(r) => r,
            Err(ureq::Error::Status(_, r)) => r,
            Err(e) => return Err(format!("reach FGTW: {e}")),
        };
        let status = resp.status();
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut resp.into_reader(), &mut out).map_err(|e| format!("FGTW body: {e}"))?;
        Ok(FgtwResponse { status, body: out })
    }
}

/// The fleet state's AEAD: kete's, as Photon seals it, so both apps open the same slot.
struct Sealer;

impl FleetSealer for Sealer {
    fn seal(&self, plaintext: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, String> {
        kete::encrypt_bytes(plaintext, key)
    }
    fn open(&self, sealed: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, String> {
        kete::decrypt_bytes(sealed, key)
    }
}

/// This device's seat in the person's fleet.
pub struct Fleet {
    handle_proof: [u8; 32],
    /// Every scheme this device declared to its fleet (Ed25519, Falcon, SLH-DSA), derived from the machine as Photon derives it: a fleet whose members all declared post-quantum keys refuses an envelope signed with less.
    signer: fgtw::pq::SigningBundle,
    key: [u8; 32],
}

impl Fleet {
    /// Join the fleet Photon has put this device in: Ok(None) when nobody is attested, or this device holds no fleet key (not a member, locked, awaiting its sponsor). Network: one or two requests to FGTW.
    pub fn join() -> Result<Option<Fleet>, String> {
        let Some(s) = tohu::session() else { return Ok(None) };
        let fp = tohu::device::machine_fingerprint().map_err(|e| format!("machine fingerprint: {e}"))?;
        let device = fgtw::keys::derive_device_keypair(&fp);
        let Some(key) = fgtw::client::recover_fleet_key(&Http, &s.handle_proof, &device, &s.identity_seed)? else { return Ok(None) };
        let signer = fgtw::pq::SigningBundle::derive(&fp);
        Ok(Some(Fleet { handle_proof: s.handle_proof, signer, key }))
    }

    /// The slot's document, opened: None when nothing is published yet.
    fn pull_raw(&self) -> Result<Option<Vec<u8>>, String> {
        let mut section = vsf::VsfSection::new("fstate_get");
        section.add_field("hp", VsfType::hP(self.handle_proof.to_vec()));
        let req = vsf::VsfBuilder::new().creation_time_oscillations(vsf::eagle_time_oscillations()).add_section_direct(section).build().map_err(|e| format!("fstate_get build: {e}"))?;
        let resp = Http.post(req)?;
        if fgtw::client::is_error(&resp.body, "not_found") {
            return Ok(None);
        }
        if let Some((reason, detail)) = fgtw::client::error_frame(&resp.body) {
            return Err(format!("fgtw fstate_get {reason}: {detail}"));
        }
        if !(200..300).contains(&resp.status) {
            return Err(format!("FGTW transport {}", resp.status));
        }
        let (header, end) = vsf::verification::read_verified(&resp.body, None).map_err(|e| format!("fstate_get response: {e}"))?;
        let stored = header.primary_section(&resp.body, end).map_err(|e| format!("fstate_get section: {e}"))?;
        let sealed = match stored.get_field("bl").and_then(|f| f.values.first()) {
            Some(VsfType::ge(b)) => b.clone(),
            _ => return Ok(None),
        };
        Sealer.open(&sealed, &self.key).map(Some)
    }

    /// The fleet's Mahere entries (tombstones included: a delete must reach every device).
    pub fn pull(&self) -> Result<Vec<SettingEntry>, String> {
        let Some(doc) = self.pull_raw()? else { return Ok(Vec::new()) };
        Ok(fstate_from_bytes(&doc)?.global_settings.into_iter().filter(|e| e.key.starts_with(PREFIX)).collect())
    }

    /// Publish entries: pull the slot, merge ours in key by key, push the result. A failed pull pushes nothing: the merge base would be unknown, and pushing over it could erase the roster (Photon's rule, kept).
    /// Nor does a slot written in a NEWER format than this build knows: fgtw reads a roster it does not understand as empty, so merging over it and pushing would wipe the fleet's roster. Mahere waits for an update instead.
    /// The write replaces the slot without naming the body it merged over (the published fgtw has no compare-and-set): a write of Photon's landing between this pull and this push is lost, a window of one round trip.
    pub fn push(&self, entries: Vec<SettingEntry>) -> Result<(), String> {
        if entries.iter().any(|e| !e.key.starts_with(PREFIX)) {
            return Err("only mahere.* keys are Mahere's to write".into());
        }
        let slot = match self.pull_raw() {
            Ok(Some(doc)) => {
                if newer_than_ours(&doc) {
                    return Err("the fleet state is a newer format than this Mahere knows; not pushing over it, Mahere needs an update".into());
                }
                fstate_from_bytes(&doc)?
            }
            Ok(None) => FleetState::default(),
            Err(e) => return Err(format!("push skipped, pull failed: {e}")),
        };
        let (roster, others, maps) = (slot.roster.len(), slot.global_settings.iter().filter(|e| !e.key.starts_with(PREFIX)).count(), slot.device_settings.len());
        let merged = merge_with(slot, entries);
        // What rides through untouched, said every time: the roster, everyone else's settings, the device maps.
        let after = (merged.roster.len(), merged.global_settings.iter().filter(|e| !e.key.starts_with(PREFIX)).count(), merged.device_settings.len());
        if after != (roster, others, maps) {
            return Err(format!("merge changed what is not Mahere's ({roster}, {others}, {maps} -> {after:?}); not pushing"));
        }
        eprintln!("fleet: pushing over {roster} roster entries, {others} other settings, {maps} device maps, unchanged");
        fgtw::client::push_fstate(&Http, &Sealer, &self.handle_proof, &self.signer, &self.key, &merged)
    }
}

/// The highest format version any section of a fleet-state document carries.
fn max_version(doc: &[u8]) -> Option<usize> {
    let (header, end) = vsf::VsfHeader::decode(doc).ok()?;
    let sections = header.sections(doc, end).ok()?;
    sections
        .iter()
        .flat_map(|s| s.fields.iter())
        .filter(|f| f.name == "version")
        .filter_map(|f| match f.values.first() {
            Some(VsfType::z(n)) => Some(*n),
            _ => None,
        })
        .max()
}

/// Whether a slot document carries a section newer than this build's fgtw writes.
fn newer_than_ours(doc: &[u8]) -> bool {
    let ours = max_version(&fstate_to_bytes(&FleetState::default())).unwrap_or(0);
    max_version(doc).is_some_and(|v| v > ours)
}

/// The slot with our entries merged in: everything else in it (the roster, Photon's settings, every device map) exactly as it was.
pub fn merge_with(slot: FleetState, entries: Vec<SettingEntry>) -> FleetState {
    merge_fstate(slot, FleetState { roster: Vec::new(), global_settings: entries, device_settings: Vec::new() })
}

/// Now, as the fleet's logical clock counts it.
pub fn now() -> i64 {
    vsf::eagle_time_oscillations()
}

fn entry(key: &str, value: Option<VsfType>, updated: i64) -> SettingEntry {
    SettingEntry { key: key.to_string(), tombstone: value.is_none(), value: value.unwrap_or(VsfType::u0(false)), updated }
}

/// An unsigned value at whatever width it was written.
fn unsigned(v: &VsfType) -> Option<u64> {
    Some(match v {
        VsfType::u(n, _) => *n as u64,
        VsfType::u3(n) => *n as u64,
        VsfType::u4(n) => *n as u64,
        VsfType::u5(n) => *n as u64,
        VsfType::u6(n) => *n,
        _ => return None,
    })
}

fn boolean(v: &VsfType) -> Option<bool> {
    match v {
        VsfType::u0(b) => Some(*b),
        _ => unsigned(v).map(|n| n != 0),
    }
}

/// When this device last wrote or took the fleet entry `key` (0 = never): a fleet entry is applied only when it is newer. Kept with the person's data.
pub fn stamp(store: &Store, key: &str) -> i64 {
    let Some(b) = store.read(&stamp_key(key)).ok().flatten() else { return 0 };
    let mut ptr = 0usize;
    match vsf::parse(&b, &mut ptr) {
        Ok(VsfType::i6(t)) => t,
        _ => 0,
    }
}

pub fn set_stamp(store: &Store, key: &str, t: i64) {
    let _ = store.write(&stamp_key(key), &VsfType::i6(t).flatten());
}

fn stamp_key(key: &str) -> String {
    vault_key(&[name("fleet"), name("stamp"), VsfType::x(key.to_string())])
}

/// The preferences that follow the person: what they see and draw with, not what one device's screen, sensors or disk want.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Prefs {
    pub layer_bits: u64,
    pub theme: String,
    pub compressed: bool,
    pub ruler: bool,
    pub pen_rgba: Option<[u8; 4]>,
    pub travel: u8,
    pub contour_hold: Option<f32>,
    pub auto_update: bool,
}

impl Prefs {
    /// A device's settings as preferences, the theme by its name in `themes`.
    pub fn of(s: &Settings, themes: &[Theme]) -> Prefs {
        Prefs { layer_bits: s.layer_bits, theme: themes.get(s.theme as usize).map(|t| t.name.clone()).unwrap_or_default(), compressed: s.compressed, ruler: s.ruler, pen_rgba: s.pen_rgba, travel: s.travel, contour_hold: s.contour_hold, auto_update: s.auto_update }
    }

    /// These preferences over a device's settings, which keep their own budget, scale, sensors and tool; a theme this device does not have leaves its own.
    pub fn onto(&self, s: &mut Settings, themes: &[Theme]) {
        s.layer_bits = self.layer_bits;
        if let Some(i) = themes.iter().position(|t| t.name == self.theme) {
            s.theme = i as u64;
        }
        s.compressed = self.compressed;
        s.ruler = self.ruler;
        if self.pen_rgba.is_some() {
            s.pen_rgba = self.pen_rgba;
        }
        s.travel = self.travel;
        s.contour_hold = self.contour_hold;
        s.auto_update = self.auto_update;
    }

    /// An entry for each preference that differs from `before` (every one when there is no before), stamped `t`.
    pub fn entries(&self, before: Option<&Prefs>, t: i64) -> Vec<SettingEntry> {
        let all: [(&str, Option<VsfType>, bool); 8] = [
            (LAYERS, Some(VsfType::u6(self.layer_bits)), before.is_some_and(|b| b.layer_bits == self.layer_bits)),
            (THEME, Some(VsfType::x(self.theme.clone())), before.is_some_and(|b| b.theme == self.theme)),
            (COMPRESSED, Some(VsfType::u0(self.compressed)), before.is_some_and(|b| b.compressed == self.compressed)),
            (RULER, Some(VsfType::u0(self.ruler)), before.is_some_and(|b| b.ruler == self.ruler)),
            // No pen colour yet is not a preference: nothing is sent, and a device keeps its own.
            (PEN, self.pen_rgba.map(|c| VsfType::t_u3(Tensor::new(vec![4], c.to_vec()))), before.is_some_and(|b| b.pen_rgba == self.pen_rgba) || self.pen_rgba.is_none()),
            (TRAVEL, Some(VsfType::u3(self.travel)), before.is_some_and(|b| b.travel == self.travel)),
            (CONTOUR_HOLD, self.contour_hold.map(VsfType::f5), before.is_some_and(|b| b.contour_hold == self.contour_hold)),
            (AUTO_UPDATE, Some(VsfType::u0(self.auto_update)), before.is_some_and(|b| b.auto_update == self.auto_update)),
        ];
        all.into_iter().filter(|(_, _, same)| !same).map(|(k, v, _)| entry(k, v, t)).collect()
    }

    /// A fleet entry taken in, when it is one of the preferences and reads as one: true when it changed them.
    pub fn take(&mut self, e: &SettingEntry) -> bool {
        let before = self.clone();
        let v = &e.value;
        match e.key.as_str() {
            CONTOUR_HOLD if e.tombstone => self.contour_hold = None,
            _ if e.tombstone => {}
            LAYERS => self.layer_bits = unsigned(v).unwrap_or(self.layer_bits),
            THEME => {
                if let VsfType::x(n) = v {
                    self.theme = n.clone();
                }
            }
            COMPRESSED => self.compressed = boolean(v).unwrap_or(self.compressed),
            RULER => self.ruler = boolean(v).unwrap_or(self.ruler),
            PEN => {
                if let VsfType::t_u3(t) = v {
                    if let Ok(c) = <[u8; 4]>::try_from(t.data.as_slice()) {
                        self.pen_rgba = Some(c);
                    }
                }
            }
            // Travel picks one of four (none, car, bike, foot): a value from another device outside them is refused, as any field read off the network that does not fit.
            TRAVEL => {
                if let Some(n) = unsigned(v).filter(|&n| n < 4) {
                    self.travel = n as u8;
                }
            }
            CONTOUR_HOLD => {
                if let VsfType::f5(m) = v {
                    self.contour_hold = Some(*m);
                }
            }
            AUTO_UPDATE => self.auto_update = boolean(v).unwrap_or(self.auto_update),
            _ => {}
        }
        *self != before
    }
}

/// Whether a fleet entry is one of the preferences.
pub fn is_pref(key: &str) -> bool {
    [LAYERS, THEME, COMPRESSED, RULER, PEN, TRAVEL, CONTOUR_HOLD, AUTO_UPDATE].contains(&key)
}

/// A theme's key in the fleet: the hash of its name, spelled as fgtw's keys must be.
pub fn theme_key(theme_name: &str) -> String {
    format!("{THEME_DOC}{}", mahere_tiles::base64url(blake3::hash(theme_name.as_bytes()).as_bytes()))
}

/// A theme for the fleet, or its tombstone when deleted.
pub fn theme_entry(theme_name: &str, theme: Option<&Theme>, updated: i64) -> Option<SettingEntry> {
    let doc = match theme {
        Some(t) => Some(VsfType::v(b'r', crate::themes::encode(t)?)),
        None => None,
    };
    Some(entry(&theme_key(theme_name), doc, updated))
}

/// What a theme entry from the fleet says: Some(Some(theme)) to keep, Some(None) a deletion, None when the entry is not a theme's.
pub fn theme_of(e: &SettingEntry) -> Option<Option<Theme>> {
    if !e.key.starts_with(THEME_DOC) {
        return None;
    }
    if e.tombstone {
        return Some(None);
    }
    match &e.value {
        VsfType::v(_, b) => crate::themes::decode(b).map(Some),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefs_travel_one_by_one_and_leave_the_device_alone() {
        let p = Prefs { layer_bits: 0b1011, theme: "Pō".into(), compressed: false, ruler: true, pen_rgba: Some([1, 2, 3, 0]), travel: 2, contour_hold: Some(20.), auto_update: false };
        let mut q = Prefs::default();
        for e in p.entries(None, 7) {
            assert!(is_pref(&e.key) && e.key.is_ascii());
            q.take(&e);
        }
        assert_eq!(q, p, "every preference round-trips as its own entry");
        let mut changed = p.clone();
        changed.ruler = false;
        changed.contour_hold = None;
        let sent = changed.entries(Some(&p), 8);
        assert_eq!(sent.len(), 2, "only what changed goes");
        assert!(sent.iter().any(|e| e.key == CONTOUR_HOLD && e.tombstone), "a hold let go travels as a tombstone");
        for e in &sent {
            q.take(e);
        }
        assert_eq!(q, changed);
        let bad = entry(TRAVEL, Some(VsfType::u3(9)), 9);
        assert!(!q.take(&bad), "a travel outside the four is refused");
        let mut s = Settings { cache_budget: 7 << 30, ui_ru: 1.5, real_sun: true, tool: 2, ..Default::default() };
        p.onto(&mut s, &[]);
        assert_eq!((s.cache_budget, s.ui_ru, s.real_sun, s.tool), (7 << 30, 1.5, true, 2), "the device's own settings stay");
    }

    /// A document from a newer fgtw is refused before any merge; one in our own format is not.
    #[test]
    fn a_newer_format_is_never_pushed_over() {
        let ours = fstate_to_bytes(&FleetState::default());
        assert!(!newer_than_ours(&ours));
        let v = max_version(&ours).unwrap();
        // A section a future fgtw would write: our roster schema at the next version.
        let mut sec = vsf::VsfSection::new("roster");
        sec.add_field("version", VsfType::z(v + 1));
        let future = vsf::VsfBuilder::new().creation_time_oscillations(vsf::eagle_time_oscillations()).add_section_direct(sec).build().unwrap();
        assert!(newer_than_ours(&future));
    }

    /// Our merge leaves everything that is not ours exactly as it was, and newest wins key by key.
    #[test]
    fn a_push_touches_only_mahere_keys() {
        let photon = SettingEntry { key: "display.lang".into(), value: VsfType::x("es".into()), updated: 50, tombstone: false };
        let old = entry(RULER, Some(VsfType::u0(false)), 10);
        let slot = FleetState { roster: Vec::new(), global_settings: vec![photon.clone(), old], device_settings: Vec::new() };
        let merged = merge_with(slot.clone(), vec![entry(RULER, Some(VsfType::u0(true)), 20), entry(&theme_key("Dusk"), None, 20)]);
        assert!(merged.global_settings.contains(&photon), "Photon's language untouched");
        let ruler = merged.global_settings.iter().find(|e| e.key == RULER).unwrap();
        assert_eq!(ruler.value, VsfType::u0(true));
        assert!(merged.global_settings.iter().any(|e| e.key == theme_key("Dusk") && e.tombstone), "a delete travels as a tombstone");
        let stale = merge_with(merged.clone(), vec![entry(RULER, Some(VsfType::u0(false)), 5)]);
        assert_eq!(stale.global_settings.iter().find(|e| e.key == RULER).unwrap().value, VsfType::u0(true), "an older write loses");
        assert_eq!(merged.roster, slot.roster);
    }
}
