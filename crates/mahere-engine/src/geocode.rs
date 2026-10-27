//! Typed addresses to places (Nick 2026-10-27: "type in my bro's address and hit navigate"): "647 Talon Trail, 59912", "647 Talon Ln Columbia Falls MT 59912", or "647 Talon Trail" alone near where the map is. The number and the ZIP are read off the ends, the rest is the street (and whatever town and state follow it); the ZIP's address document says where the number lies.

use mahere_tiles::addr::{Found, ZipDoc, ZipIndex, INDEX_PATH, house_number, zip_path};

/// Where address documents come from: the device's copy, else the bucket, kept on the device.
pub trait DocStore: Send + Sync {
    fn doc(&self, path: &str) -> Option<Vec<u8>>;
}

/// A typed address taken apart: the house number, the words after it, the ZIP if one ends it.
#[derive(Clone, Debug, PartialEq)]
pub struct Typed {
    pub number: u32,
    pub street: String,
    pub zip: Option<u32>,
}

/// An address's parts, or None when the text does not start with a house number followed by words.
pub fn parse(text: &str) -> Option<Typed> {
    let t = text.trim();
    let mut words: Vec<&str> = t.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()).collect();
    // A house number is digits, perhaps a letter after them ("12B"); a decimal is a coordinate, not a house.
    let first = words.first()?;
    if first.contains('.') || first.starts_with('-') {
        return None;
    }
    let number = house_number(first)?;
    words.remove(0);
    // A ZIP, or ZIP+4, ends it; a two-letter state before the ZIP is left in the street words, where the matcher ignores what follows the street.
    let zip = words.last().and_then(|w| {
        let five: String = w.chars().take(5).collect();
        (five.len() == 5 && five.chars().all(|c| c.is_ascii_digit()) && (w.len() == 5 || w.as_bytes().get(5) == Some(&b'-'))).then(|| five.parse().ok()).flatten()
    });
    if zip.is_some() {
        words.pop();
    }
    if words.is_empty() || !words[0].chars().any(|c| c.is_alphabetic()) {
        return None;
    }
    Some(Typed { number, street: words.join(" "), zip })
}

/// How far from the map's middle the ZIPs searched reach when none was typed.
pub const NEAR_M: f64 = 40_000.0;

/// The places an address could be, best first: in its ZIP, or in the ZIPs within [`NEAR_M`] of `near` when it names none.
pub fn geocode(text: &str, near: (f64, f64), docs: &dyn DocStore) -> Vec<Found> {
    let Some(t) = parse(text) else { return Vec::new() };
    let zips: Vec<u32> = match t.zip {
        Some(z) => vec![z],
        None => docs.doc(INDEX_PATH).and_then(|b| ZipIndex::decode(&b)).map(|i| i.near(near.0, near.1, NEAR_M)).unwrap_or_default().into_iter().take(16).collect(),
    };
    let mut out: Vec<Found> = Vec::new();
    for z in zips {
        let Some(doc) = docs.doc(&zip_path(z)).and_then(|b| ZipDoc::decode(&b)) else { continue };
        out.extend(doc.find(t.number, &t.street));
    }
    out.sort_by_key(|f| f.score);
    out.truncate(8);
    out
}

#[cfg(test)]
mod tests {
    use super::{parse, Typed};

    #[test]
    fn addresses_come_apart() {
        assert_eq!(parse("647 Talon Trail, 59912"), Some(Typed { number: 647, street: "Talon Trail".into(), zip: Some(59912) }));
        assert_eq!(parse("647 Talon Ln Columbia Falls MT 59912-1234"), Some(Typed { number: 647, street: "Talon Ln Columbia Falls MT".into(), zip: Some(59912) }));
        assert_eq!(parse("647 Talon Trail"), Some(Typed { number: 647, street: "Talon Trail".into(), zip: None }));
        assert_eq!(parse("12B 9th St"), Some(Typed { number: 12, street: "9th St".into(), zip: None }));
        assert_eq!(parse("Snow Lake"), None);
        assert_eq!(parse("47.4455, -121.4235"), None, "degrees are not an address");
        assert_eq!(parse("98366"), None);
    }
}
