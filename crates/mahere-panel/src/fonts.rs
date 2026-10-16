//! Font packs for the panel and the map's labels (Nick 2026-10-16: every name native, the binary small): the faces fluor does not carry, fetched the first time text needs one ([`fluor::packs`]).
//!
//! Each renderer (the panel's on the drawing thread, the letterer's on its worker) tells [`FontPacks::want`] what its text needed; each pack is fetched once, on a thread of its own, from the host's [`PackSource`] (the vault, then the bucket); each renderer loads what arrived on its own thread with [`load_into`], which checks the author's signature.
//! Until a pack arrives its text draws as it always has; when it does, the panel paints again and the labels lay out again in the script's own face.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fluor::packs::Pack;
use fluor::text::TextRenderer;

/// Where pack documents come from: the host's (the vault's copy, else the bucket's, kept in the vault).
pub trait PackSource: Send + Sync {
    fn get(&self, id: &str) -> Result<Vec<u8>, String>;
    /// A copy a renderer refused (its signature did not check): dropped, so the next fetch goes to the source.
    fn forget(&self, id: &str);
}

/// A failed fetch is tried again after this, not every frame (an offline phone).
const RETRY: Duration = Duration::from_secs(30);

#[derive(Default)]
struct State {
    /// Every document fetched, in arrival order: each renderer keeps how far through it has loaded.
    docs: Vec<(&'static str, Arc<Vec<u8>>)>,
    /// Packs being fetched, or fetched already.
    asked: HashSet<&'static str>,
    /// Packs whose fetch failed, and when.
    failed: HashMap<&'static str, Instant>,
}

/// The packs fetched for this app, shared by every renderer.
pub struct FontPacks {
    source: Arc<dyn PackSource>,
    author: [u8; 32],
    state: Mutex<State>,
}

impl FontPacks {
    /// Packs from `source`, accepted only when signed by `author`.
    pub fn new(source: Arc<dyn PackSource>, author: [u8; 32]) -> Arc<FontPacks> {
        Arc::new(FontPacks { source, author, state: Mutex::new(State::default()) })
    }

    /// Packs a renderer's text needed ([`TextRenderer::wanted_packs`]), each fetched once; a failed one again after [`RETRY`].
    pub fn want(self: &Arc<Self>, packs: Vec<&'static Pack>) {
        let mut st = self.state.lock().unwrap();
        // Failures old enough to try again.
        let now = Instant::now();
        let again: Vec<&'static str> = st.failed.iter().filter(|(_, t)| now.duration_since(**t) >= RETRY).map(|(id, _)| *id).collect();
        for id in &again {
            st.failed.remove(id);
            st.asked.remove(id);
        }
        let ids = packs.iter().map(|p| p.id).chain(again);
        for id in ids {
            if !st.asked.insert(id) {
                continue;
            }
            let me = self.clone();
            std::thread::Builder::new()
                .name(format!("font pack {id}"))
                .spawn(move || {
                    let got = me.source.get(id);
                    let mut st = me.state.lock().unwrap();
                    match got {
                        Ok(doc) => st.docs.push((id, Arc::new(doc))),
                        Err(e) => {
                            eprintln!("font pack {id}: {e}");
                            st.failed.insert(id, Instant::now());
                        }
                    }
                })
                .ok();
        }
    }

    /// Whether a fetch is under way (a host idle otherwise keeps a frame coming until it lands).
    pub fn busy(&self) -> bool {
        let st = self.state.lock().unwrap();
        st.asked.iter().any(|id| !st.failed.contains_key(id) && !st.docs.iter().any(|(d, _)| d == id))
    }

    /// How many documents have arrived: a renderer behind this has more to load.
    pub fn arrived(&self) -> usize {
        self.state.lock().unwrap().docs.len()
    }

    fn since(&self, from: usize) -> Vec<(&'static str, Arc<Vec<u8>>)> {
        self.state.lock().unwrap().docs.get(from..).map(<[_]>::to_vec).unwrap_or_default()
    }

    /// A document a renderer refused: forgotten at the source and fetched again after [`RETRY`]. The list of arrivals only grows, so every renderer's place in it holds.
    fn refused(&self, id: &'static str) {
        self.source.forget(id);
        self.state.lock().unwrap().failed.entry(id).or_insert_with(Instant::now);
    }
}

/// What arrived since `cursor` loaded into a renderer, each checked against the author's key; true when a face was added (whatever measured text before measures again).
pub fn load_into(text: &mut TextRenderer, fonts: &FontPacks, cursor: &mut usize) -> bool {
    let docs = fonts.since(*cursor);
    *cursor += docs.len();
    let mut added = false;
    for (id, doc) in docs {
        match text.load_pack(&doc, fonts.author) {
            Ok(_) => added = true,
            Err(e) => {
                eprintln!("font pack {id} refused: {e}");
                fonts.refused(id);
                text.offer_again(id);
            }
        }
    }
    added
}

/// The regional face for Han in a view's names: Japanese where any name has kana, Korean where any has Hangul and none kana, else Simplified Chinese.
pub fn han_for<'a>(names: impl Iterator<Item = &'a str>) -> fluor::packs::Han {
    use fluor::packs::{Han, Need, need};
    let mut hangul = false;
    for n in names {
        for c in n.chars() {
            match need(c) {
                Some(Need::Kana) => return Han::Japanese,
                Some(Need::Hangul) => hangul = true,
                _ => {}
            }
        }
    }
    if hangul { Han::Korean } else { Han::Simplified }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluor::packs::Han;

    /// A view's names choose its Han face: any kana Japanese, Hangul without kana Korean, else Simplified Chinese.
    #[test]
    fn a_views_names_choose_its_han() {
        assert_eq!(han_for(["北京市", "朝阳区"].into_iter()), Han::Simplified);
        assert_eq!(han_for(["東京都", "さいたま市"].into_iter()), Han::Japanese);
        assert_eq!(han_for(["서울특별시", "中區"].into_iter()), Han::Korean);
        assert_eq!(han_for(["Seattle"].into_iter()), Han::Simplified);
    }
}
