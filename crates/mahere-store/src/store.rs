//! Where Mahere keeps everything (Nick 2026-10-22): one vault, its device secret the machine's own (tohu), and two kinds of entry in it.
//! - The person's data — session, settings, themes, tracks, ink and selections, areas — in the IDENTITY scope, once Photon has attested who is using this device: keyed hash(thing | device | person), sealed under keys only that person's session opens, and synced with that person's other devices. Before then it is held in RAM and never written: the session is not being saved, and the app says so.
//! - Public data — map cells, font packs — in the DEVICE scope, sealed like everything else (what a device holds says where its person has been), cells as the vault's cache class: evicted by the vault, never grown for, never synced.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use kete::{FlatStorage, StorageError};

/// The vault and the person's data as it stands: written through once attested, held in RAM before.
pub struct Store {
    vault: Arc<FlatStorage>,
    /// Some until attestation: the person's entries this session, by key (None = deleted).
    held: Mutex<Option<HashMap<String, Option<Vec<u8>>>>>,
    /// A vault from before (its secret, its key file), whose person data comes in once someone is attested to own it.
    older: Mutex<Option<([u8; 32], std::path::PathBuf)>>,
    /// The person's data goes to the DEVICE scope, saved from first launch as before attestation existed — Android until Photon's session reaches it (Photon's capsule lives in Photon's own storage there; the shared capsule broadcast is not wired yet).
    device_scope: bool,
}

impl Store {
    /// The store over `vault`. Attested already when the vault holds an identity (a session Photon left this process, or one set earlier).
    pub fn new(vault: Arc<FlatStorage>) -> Arc<Store> {
        let held = if vault.identity().is_some() { None } else { Some(HashMap::new()) };
        Arc::new(Store { vault, held: Mutex::new(held), older: Mutex::new(None), device_scope: false })
    }

    /// A store that saves the person's data in the device scope from the start (see `device_scope`).
    pub fn device(vault: Arc<FlatStorage>) -> Arc<Store> {
        Arc::new(Store { vault, held: Mutex::new(None), older: Mutex::new(None), device_scope: true })
    }

    pub(crate) fn set_older(&self, older: Option<([u8; 32], std::path::PathBuf)>) {
        *self.older.lock().unwrap() = older;
    }

    pub(crate) fn take_older(&self) -> Option<([u8; 32], std::path::PathBuf)> {
        self.older.lock().unwrap().take()
    }

    /// Whether the person's data is being saved: Photon has attested who this is.
    pub fn attested(&self) -> bool {
        self.held.lock().unwrap().is_none()
    }

    /// Photon has attested: bind the vault to the person's vault seed and write what this session held — but only where the person has nothing saved: what they saved wins, this session fills the gaps (a session that began unattested started from defaults, which must not overwrite their settings). Returns how many entries were written. A vault already bound to someone else refuses (one person per device).
    pub fn attest(&self, vault_seed: [u8; 32]) -> Result<usize, StorageError> {
        self.vault.set_identity(vault_seed)?;
        let mut held = self.held.lock().unwrap();
        let Some(entries) = held.take() else { return Ok(0) };
        let mut n = 0;
        let mut failed = None;
        for (key, value) in &entries {
            let Some(v) = value else { continue };
            match self.vault.read(key) {
                Ok(Some(_)) => continue,
                Ok(None) => {}
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
            match self.vault.write(key, v) {
                Ok(()) => n += 1,
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = failed {
            // Put it all back: nothing is lost, and the next attest tries again.
            *held = Some(entries);
            return Err(e);
        }
        Ok(n)
    }

    /// The person's entry under `key`.
    pub fn read(&self, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        if let Some(held) = self.held.lock().unwrap().as_ref() {
            return Ok(held.get(key).cloned().flatten());
        }
        if self.device_scope {
            return self.vault.read_device(key);
        }
        self.vault.read(key)
    }

    pub fn write(&self, key: &str, data: &[u8]) -> Result<(), StorageError> {
        if let Some(held) = self.held.lock().unwrap().as_mut() {
            held.insert(key.to_string(), Some(data.to_vec()));
            return Ok(());
        }
        if self.device_scope {
            return self.vault.write_device(key, data);
        }
        self.vault.write(key, data)
    }

    pub fn delete(&self, key: &str) -> Result<(), StorageError> {
        if let Some(held) = self.held.lock().unwrap().as_mut() {
            held.insert(key.to_string(), None);
            return Ok(());
        }
        if self.device_scope {
            return self.vault.delete_device(key);
        }
        self.vault.delete(key)
    }

    /// The vault itself, for public data (device scope) and its stats.
    pub fn vault(&self) -> &Arc<FlatStorage> {
        &self.vault
    }
}
