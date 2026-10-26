//! Search (Nick 2026-10-25): the names inside the screen's bounding box, from the cells kept on the device — every depth of the pyramid the device holds under the box, nothing fetched for it. A name counts where one of its texels lies on the screen; matches sort by how well the text matches, then the kind's tier (towns before streets, as the lettering goes), then rank, then size.

use std::sync::Arc;

use mahere_tiles::{CellKey, Named, NameCell, TRI};

use crate::labels::NamesCache;
use crate::residency::{CellStore, Fetch};
use crate::{Camera, MapCore};

/// Results kept, best first.
pub const MAX_FOUND: usize = 60;

/// A place the search found.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub name: String,
    pub reference: String,
    pub kind: u8,
    pub rank: u32,
    /// Where to show it: its texel nearest the screen's centre, latitude and longitude.
    pub lat: f64,
    pub lon: f64,
    /// Its extent on the screen: texels inside the box at the finest depth it was found at, times that depth's texel edge, metres.
    pub size_m: f64,
    /// How the text matched: 0 the whole name, 1 its start, 2 the start of a word in it, 3 anywhere in it (the reference counts as a name).
    pub fit: u8,
}

/// A search's inputs, for another thread: the view, the query, the cells the device keeps and where to read them.
pub struct SearchJob {
    pub cam: Camera,
    pub w: usize,
    pub h: usize,
    pub query: String,
    pub cached: Vec<CellKey>,
    store: Arc<dyn CellStore>,
}

impl MapCore {
    /// A search of the view for `query` over `cached` (the base cells the device keeps, any depth), for another thread to run ([`run`]).
    pub fn search_job(&self, query: &str, w: usize, h: usize, cached: Vec<CellKey>) -> SearchJob {
        SearchJob { cam: self.cam, w, h, query: query.to_string(), cached, store: self.res.store() }
    }
}

/// How `query` (lowercase, trimmed) fits `text`: see [`Found::fit`]; None when it does not appear.
pub fn fit(query: &str, text: &str) -> Option<u8> {
    if query.is_empty() {
        return None;
    }
    let t = text.to_lowercase();
    if t == query {
        return Some(0);
    }
    if t.starts_with(query) {
        return Some(1);
    }
    let at = t.find(query)?;
    let word_start = t[..at].chars().next_back().map_or(true, |c| !c.is_alphanumeric());
    Some(if word_start { 2 } else { 3 })
}

/// Whether the cell's footprint meets the screen: its corners' screen box overlaps the screen, or it holds the screen's centre (a coarse cell larger than the view).
fn meets(cam: &Camera, w: usize, h: usize, key: CellKey) -> bool {
    let (clat, clon) = cam.screen_to_geo(w as f64 / 2.0, h as f64 / 2.0, w, h);
    if CellKey::containing(mahere_coord::Coord::from_lat_lon(clat, clon), key.depth) == key {
        return true;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for (lat, lon) in key.corners() {
        let (x, y) = cam.geo_to_screen(lat, lon, w, h);
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    x1 >= 0.0 && y1 >= 0.0 && x0 <= w as f64 && y0 <= h as f64
}

/// A cell's names from the device alone (never the network), through the worker's cache.
fn names_of(store: &dyn CellStore, cache: &mut NamesCache, key: CellKey) -> Option<Arc<NameCell>> {
    if let Some(n) = cache.get(&key) {
        return n.clone();
    }
    let n = match store.get_local(key) {
        Fetch::Bytes(b) => mahere_tiles::decode_names(&b).map(Arc::new),
        _ => None,
    };
    cache.insert(key, n.clone());
    n
}

/// Run a search: every cached cell under the screen read for names that fit the query, each kept where a texel of it is on the screen, merged across cells and depths by kind and name, best first. `tier` orders the kinds (the lettering's own; None leaves a kind out).
pub fn run(job: &SearchJob, cache: &mut NamesCache, tier: &dyn Fn(&Named) -> Option<u8>) -> Vec<Found> {
    let q = job.query.trim().to_lowercase();
    if q.is_empty() {
        return Vec::new();
    }
    let (w, h) = (job.w as f64, job.h as f64);
    let (cx, cy) = (w / 2.0, h / 2.0);
    // Merged by (kind, name, reference): the best fit, the position and size at the finest depth seen (a coarse texel is hundreds of metres wide: a trailhead placed from a depth-5 cell stood a kilometre from where the map letters it), nearest the centre among that depth's.
    let mut found: Vec<(Found, u8, f64)> = Vec::new();
    for &key in &job.cached {
        if !meets(&job.cam, job.w, job.h, key) {
            continue;
        }
        let Some(names) = names_of(&*job.store, cache, key) else { continue };
        // The entries that fit, by their 1-based index.
        let fits: Vec<(usize, u8)> = names
            .entries
            .iter()
            .enumerate()
            .filter(|(_, n)| tier(n).is_some())
            .filter_map(|(i, n)| {
                let f = [fit(&q, &n.name), fit(&q, &n.reference)].into_iter().flatten().min()?;
                Some((i + 1, f))
            })
            .collect();
        if fits.is_empty() {
            continue;
        }
        // Each fitting entry's texels on the screen: how many, and the one nearest the centre.
        let mut seen: Vec<(usize, u32, f64, (f64, f64))> = fits.iter().map(|&(i, _)| (i, 0, f64::MAX, (0.0, 0.0))).collect();
        for plane in [&names.line, &names.area] {
            if plane.len() < TRI {
                continue;
            }
            for (t, &v) in plane.iter().enumerate() {
                if v == 0 {
                    continue;
                }
                let Some(s) = seen.iter_mut().find(|s| s.0 == v as usize) else { continue };
                let (lat, lon) = key.texel_lat_lon(t);
                let (x, y) = job.cam.geo_to_screen(lat, lon, job.w, job.h);
                if x < 0.0 || y < 0.0 || x > w || y > h {
                    continue;
                }
                s.1 += 1;
                let d = (x - cx).powi(2) + (y - cy).powi(2);
                if d < s.2 {
                    s.2 = d;
                    s.3 = (lat, lon);
                }
            }
        }
        for ((i, count, d, (lat, lon)), &(_, f)) in seen.into_iter().zip(&fits) {
            if count == 0 {
                continue;
            }
            let n = &names.entries[i - 1];
            let size_m = count as f64 * key.texel_m();
            match found.iter_mut().find(|e| e.0.kind == n.kind && e.0.name == n.name && e.0.reference == n.reference) {
                Some(e) => {
                    e.0.fit = e.0.fit.min(f);
                    e.0.rank = e.0.rank.max(n.rank);
                    if key.depth > e.1 {
                        (e.0.lat, e.0.lon, e.2, e.0.size_m, e.1) = (lat, lon, d, size_m, key.depth);
                    } else if key.depth == e.1 && d < e.2 {
                        (e.0.lat, e.0.lon, e.2) = (lat, lon, d);
                    }
                }
                None => found.push((Found { name: n.name.clone(), reference: n.reference.clone(), kind: n.kind, rank: n.rank, lat, lon, size_m, fit: f }, key.depth, d)),
            }
        }
    }
    let order = |f: &Found| {
        let n = Named { kind: f.kind, rank: f.rank, name: String::new(), reference: String::new() };
        (f.fit, tier(&n).unwrap_or(u8::MAX))
    };
    found.sort_by(|a, b| order(&a.0).cmp(&order(&b.0)).then(b.0.rank.cmp(&a.0.rank)).then(b.0.size_m.total_cmp(&a.0.size_m)));
    found.truncate(MAX_FOUND);
    found.into_iter().map(|e| e.0).collect()
}

#[cfg(test)]
mod tests {
    use super::fit;

    #[test]
    fn fits_rank_whole_start_word_anywhere() {
        assert_eq!(fit("harper", "Harper"), Some(0));
        assert_eq!(fit("harp", "Harper"), Some(1));
        assert_eq!(fit("colby", "South Colby"), Some(2));
        assert_eq!(fit("olby", "South Colby"), Some(3));
        assert_eq!(fit("seattle", "Harper"), None);
        assert_eq!(fit("", "Harper"), None);
    }
}
