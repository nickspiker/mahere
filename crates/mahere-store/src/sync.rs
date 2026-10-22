//! The fleet (Nick 2026-10-22): the person's preferences and themes follow them across their devices exactly as Photon's language does — entries in the fleet-shared state (fstate) on FGTW, sealed under the fleet key, merged key by key, newest write wins. Mahere's entries are `mahere.*` keys beside Photon's; a push pulls first and merges, so it never disturbs the roster or Photon's own settings.
//! Mahere joins as the same fleet member Photon is on this machine: the device key comes from the machine fingerprint, and the fleet key is recovered from the fan-out with the person's session — never minted here.
//! What never goes: the camera (where the person last looked), the cell cache, anything that only makes sense on one device.

use fgtw::client::{FgtwResponse, FgtwTransport, FleetSealer};
use fgtw::fstate::{merge_fstate, FleetState};
pub use fgtw::fstate::SettingEntry;
use vsf::VsfType;

const FGTW_URL: &str = "https://fgtw.org";

/// Every key Mahere writes into the fleet state begins so.
pub const PREFIX: &str = "mahere.";
/// The synced preferences (see [`Prefs`]).
pub const PREFS_KEY: &str = "mahere.prefs";
/// A theme, by name after this.
pub const THEME_PREFIX: &str = "mahere.theme.";

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
    device: fgtw::keys::Keypair,
    key: [u8; 32],
}

impl Fleet {
    /// Join the fleet Photon has put this device in: Ok(None) when nobody is attested, or this device holds no fleet key (not a member, locked, awaiting its sponsor). Network: one or two requests to FGTW.
    pub fn join() -> Result<Option<Fleet>, String> {
        let Some(s) = tohu::session() else { return Ok(None) };
        let fp = tohu::device::machine_fingerprint().map_err(|e| format!("machine fingerprint: {e}"))?;
        let device = fgtw::keys::derive_device_keypair(&fp);
        Ok(fgtw::client::recover_fleet_key(&Http, &s.handle_proof, &device, &s.identity_seed)?.map(|key| Fleet { handle_proof: s.handle_proof, device, key }))
    }

    /// The fleet's Mahere entries (tombstones included — a delete must reach every device).
    pub fn pull(&self) -> Result<Vec<SettingEntry>, String> {
        Ok(fgtw::client::pull_fstate(&Http, &Sealer, &self.handle_proof, &self.key)?
            .map(|s| s.global_settings.into_iter().filter(|e| e.key.starts_with(PREFIX)).collect())
            .unwrap_or_default())
    }

    /// Publish entries: pull the slot, merge ours in key by key, push the result. A failed pull pushes nothing — the merge base would be unknown, and pushing over it could erase the roster (Photon's rule, kept).
    pub fn push(&self, entries: Vec<SettingEntry>) -> Result<(), String> {
        if entries.iter().any(|e| !e.key.starts_with(PREFIX)) {
            return Err("only mahere.* keys are Mahere's to write".into());
        }
        let slot = match fgtw::client::pull_fstate(&Http, &Sealer, &self.handle_proof, &self.key) {
            Ok(Some(s)) => s,
            Ok(None) => FleetState::default(),
            Err(e) => return Err(format!("push skipped, pull failed: {e}")),
        };
        let merged = merge_with(slot, entries);
        fgtw::client::push_fstate(&Http, &Sealer, &self.handle_proof, &self.device, &self.key, &merged)
    }
}

/// The slot with our entries merged in: everything else in it — the roster, Photon's settings, every device map — exactly as it was.
pub fn merge_with(slot: FleetState, entries: Vec<SettingEntry>) -> FleetState {
    merge_fstate(slot, FleetState { roster: Vec::new(), global_settings: entries, device_settings: Vec::new() })
}

/// Now, as the fleet's logical clock counts it.
pub fn now() -> i64 {
    vsf::eagle_time_oscillations()
}

/// A setting entry carrying `bytes` (a vault document) under `key`, stamped `updated`; None = a tombstone.
pub fn entry(key: &str, bytes: Option<&[u8]>, updated: i64) -> SettingEntry {
    SettingEntry { key: key.to_string(), value: VsfType::v(b'r', bytes.unwrap_or_default().to_vec()), updated, tombstone: bytes.is_none() }
}

/// The bytes an entry carries, unless it is a tombstone.
pub fn bytes_of(e: &SettingEntry) -> Option<Vec<u8>> {
    if e.tombstone {
        return None;
    }
    match &e.value {
        VsfType::v(_, b) => Some(b.clone()),
        _ => None,
    }
}

/// When this device last wrote or took the fleet entry `key` (0 = never): a fleet entry is applied only when it is newer.
pub fn stamp(store: &crate::Store, key: &str) -> i64 {
    store.read(&format!("syncstamp|{key}")).ok().flatten().and_then(|b| <[u8; 8]>::try_from(b.as_slice()).ok()).map_or(0, i64::from_le_bytes)
}

pub fn set_stamp(store: &crate::Store, key: &str, t: i64) {
    let _ = store.write(&format!("syncstamp|{key}"), &t.to_le_bytes());
}

/// The preferences that follow the person: what they see and draw with, not what one device's screen, sensors or disk want.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Prefs {
    pub layer_bits: u64,
    pub theme: u64,
    pub compressed: bool,
    pub ruler: bool,
    pub pen_rgba: Option<[u8; 4]>,
    pub travel: u8,
    pub contour_hold: Option<f32>,
    pub auto_update: bool,
}

impl Prefs {
    pub fn of(s: &crate::Settings) -> Prefs {
        Prefs { layer_bits: s.layer_bits, theme: s.theme, compressed: s.compressed, ruler: s.ruler, pen_rgba: s.pen_rgba, travel: s.travel, contour_hold: s.contour_hold, auto_update: s.auto_update }
    }

    /// These preferences over a device's settings, which keep their own budget, scale, sensors and tool.
    pub fn onto(&self, s: &mut crate::Settings) {
        s.layer_bits = self.layer_bits;
        s.theme = self.theme;
        s.compressed = self.compressed;
        s.ruler = self.ruler;
        if self.pen_rgba.is_some() {
            s.pen_rgba = self.pen_rgba;
        }
        s.travel = self.travel;
        s.contour_hold = self.contour_hold;
        s.auto_update = self.auto_update;
    }

    pub fn encode(&self) -> Vec<u8> {
        let flags = self.compressed as u64 | (self.ruler as u64) << 1 | (!self.auto_update as u64) << 2;
        let pen = self.pen_rgba.map_or(0, |c| u32::from_be_bytes(c) as u64 + 1);
        let hold = self.contour_hold.map_or(0, |m| (m * 1000.0).round().max(1.0) as u64);
        VsfType::t_u6(vsf::Tensor::new(vec![6], vec![self.layer_bits, self.theme, flags, pen, self.travel as u64, hold])).flatten()
    }

    pub fn decode(bytes: &[u8]) -> Option<Prefs> {
        let mut ptr = 0usize;
        let VsfType::t_u6(t) = vsf::parse(bytes, &mut ptr).ok()? else { return None };
        let d = t.data;
        if d.len() < 6 {
            return None;
        }
        Some(Prefs {
            layer_bits: d[0],
            theme: d[1],
            compressed: d[2] & 1 != 0,
            ruler: d[2] & 2 != 0,
            auto_update: d[2] & 4 == 0,
            pen_rgba: (d[3] != 0).then(|| ((d[3] - 1) as u32).to_be_bytes()),
            travel: d[4].min(3) as u8,
            contour_hold: (d[5] != 0).then(|| d[5] as f32 / 1000.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefs_round_trip_and_leave_the_device_alone() {
        let p = Prefs { layer_bits: 0b1011, theme: 3, compressed: false, ruler: true, pen_rgba: Some([1, 2, 3, 0]), travel: 2, contour_hold: Some(20.0), auto_update: false };
        assert_eq!(Prefs::decode(&p.encode()), Some(p));
        let mut s = crate::Settings { cache_budget: 7 << 30, ui_ru: 1.5, real_sun: true, tool: 2, ..Default::default() };
        p.onto(&mut s);
        assert_eq!((s.cache_budget, s.ui_ru, s.real_sun, s.tool), (7 << 30, 1.5, true, 2), "the device's own settings stay");
        assert_eq!(Prefs::of(&s), p);
    }

    /// Our merge leaves everything that is not ours exactly as it was, and newest wins key by key.
    #[test]
    fn a_push_touches_only_mahere_keys() {
        let photon = SettingEntry { key: "display.lang".into(), value: VsfType::x("es".into()), updated: 50, tombstone: false };
        let old = entry(PREFS_KEY, Some(b"old"), 10);
        let slot = FleetState { roster: Vec::new(), global_settings: vec![photon.clone(), old], device_settings: Vec::new() };
        let merged = merge_with(slot.clone(), vec![entry(PREFS_KEY, Some(b"new"), 20), entry("mahere.theme.Dusk", None, 20)]);
        assert!(merged.global_settings.contains(&photon), "Photon's language untouched");
        let prefs = merged.global_settings.iter().find(|e| e.key == PREFS_KEY).unwrap();
        assert_eq!(bytes_of(prefs).as_deref(), Some(&b"new"[..]));
        assert!(merged.global_settings.iter().any(|e| e.key == "mahere.theme.Dusk" && e.tombstone), "a delete travels as a tombstone");
        let stale = merge_with(merged.clone(), vec![entry(PREFS_KEY, Some(b"stale"), 5)]);
        assert_eq!(bytes_of(stale.global_settings.iter().find(|e| e.key == PREFS_KEY).unwrap()).as_deref(), Some(&b"new"[..]), "an older write loses");
        assert_eq!(merged.roster, slot.roster);
    }
}
