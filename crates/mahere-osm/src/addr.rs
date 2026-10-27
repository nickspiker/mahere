//! Addresses (Nick 2026-10-27: "type in my bro's address and hit navigate"): street names in one normal form, the Census Bureau's TIGER/Line address ranges (every US street's house numbers by block and side, with the ZIP on each side), and OSM's address points.

use osmpbf::{Element, ElementReader};

/// A street name in its normal form: capitals, no punctuation, directions and street types in their USPS abbreviations ("North 9th Street" and "N 9th St" are one name).
pub fn normal(name: &str) -> String {
    let words: Vec<String> = words(name);
    let n = words.len();
    words.into_iter().enumerate().map(|(i, w)| abbreviate(&w, i, n)).collect::<Vec<_>>().join(" ")
}

/// A street name's base: its normal form without the directions at either end or the street type at its end ("N 9th St" is "9TH", "Talon Trail" and "Talon Ln" are both "TALON").
pub fn base(name: &str) -> String {
    let n = normal(name);
    let mut w: Vec<&str> = n.split(' ').filter(|w| !w.is_empty()).collect();
    if w.len() > 1 && DIRECTIONS.iter().any(|(_, a)| a == w.first().unwrap()) {
        w.remove(0);
    }
    if w.len() > 1 && DIRECTIONS.iter().any(|(_, a)| a == w.last().unwrap()) {
        w.pop();
    }
    if w.len() > 1 && TYPES.iter().any(|(_, a)| a == w.last().unwrap()) {
        w.pop();
    }
    w.join(" ")
}

fn words(name: &str) -> Vec<String> {
    name.to_uppercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

const DIRECTIONS: [(&str, &str); 8] = [("NORTH", "N"), ("SOUTH", "S"), ("EAST", "E"), ("WEST", "W"), ("NORTHEAST", "NE"), ("NORTHWEST", "NW"), ("SOUTHEAST", "SE"), ("SOUTHWEST", "SW")];

/// Street types and their USPS abbreviations; a long form, a common short form, or the abbreviation itself.
const TYPES: [(&str, &str); 40] = [
    ("STREET", "ST"), ("STR", "ST"), ("AVENUE", "AVE"), ("AV", "AVE"), ("ROAD", "RD"), ("DRIVE", "DR"), ("LANE", "LN"), ("COURT", "CT"), ("PLACE", "PL"), ("BOULEVARD", "BLVD"),
    ("TRAIL", "TRL"), ("TERRACE", "TER"), ("CIRCLE", "CIR"), ("WAY", "WAY"), ("HIGHWAY", "HWY"), ("PARKWAY", "PKWY"), ("LOOP", "LOOP"), ("POINT", "PT"), ("RIDGE", "RDG"), ("SQUARE", "SQ"),
    ("CROSSING", "XING"), ("EXPRESSWAY", "EXPY"), ("FREEWAY", "FWY"), ("ALLEY", "ALY"), ("COVE", "CV"), ("HEIGHTS", "HTS"), ("HILL", "HL"), ("RUN", "RUN"), ("PATH", "PATH"), ("PIKE", "PIKE"),
    ("ROW", "ROW"), ("VIEW", "VW"), ("VISTA", "VIS"), ("WALK", "WALK"), ("ESTATES", "ESTS"), ("MEADOWS", "MDWS"), ("GLEN", "GLN"), ("GROVE", "GRV"), ("HOLLOW", "HOLW"), ("BEND", "BND"),
];

fn abbreviate(w: &str, i: usize, n: usize) -> String {
    // A direction at either end, a type at the end: "North Bend Way" keeps North as a name only in the middle.
    if i == 0 || i + 1 == n {
        if let Some((_, a)) = DIRECTIONS.iter().find(|(l, a)| *l == w || *a == w) {
            return a.to_string();
        }
    }
    if i + 1 == n && n > 1 {
        if let Some((_, a)) = TYPES.iter().find(|(l, a)| *l == w || *a == w) {
            return a.to_string();
        }
    }
    w.to_string()
}

/// A house number's leading digits ("647", "647A", "647 1/2"); None for a number that does not start with one.
pub fn house_number(s: &str) -> Option<u32> {
    let digits: String = s.trim().chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// One side of a block's numbers: the first and last, and which numbers lie on it (1 odd, 2 even, 3 both).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Side {
    pub from: u32,
    pub to: u32,
    pub parity: u8,
    pub zip: u32,
}

/// A TIGER address range edge: the street's name, each side's numbers and ZIP, and the edge's line from its first point to its last (the ranges run in that direction).
#[derive(Clone, Debug)]
pub struct Edge {
    pub street: String,
    pub left: Option<Side>,
    pub right: Option<Side>,
    pub pts: Vec<(f64, f64)>,
}

/// Every address range edge in a TIGER/Line ADDRFEAT shapefile.
pub fn read_addrfeat(shp: &str) -> Result<Vec<Edge>, String> {
    use shapefile::dbase::FieldValue;
    let mut reader = shapefile::Reader::from_path(shp).map_err(|e| format!("{shp}: {e}"))?;
    let text = |rec: &shapefile::dbase::Record, name: &str| -> String {
        match rec.get(name) {
            Some(FieldValue::Character(Some(s))) => s.trim().to_string(),
            _ => String::new(),
        }
    };
    let mut out = Vec::new();
    for item in reader.iter_shapes_and_records() {
        let (shape, rec) = item.map_err(|e| format!("{shp}: {e}"))?;
        let pts: Vec<(f64, f64)> = match shape {
            shapefile::Shape::Polyline(l) => l.parts().iter().flatten().map(|p| (p.y, p.x)).collect(),
            shapefile::Shape::PolylineZ(l) => l.parts().iter().flatten().map(|p| (p.y, p.x)).collect(),
            shapefile::Shape::PolylineM(l) => l.parts().iter().flatten().map(|p| (p.y, p.x)).collect(),
            _ => continue,
        };
        let street = text(&rec, "FULLNAME");
        if street.is_empty() || pts.len() < 2 {
            continue;
        }
        let side = |from: &str, to: &str, parity: &str, zip: &str| -> Option<Side> {
            // A range whose numbers carry letters or hyphens (Queens' "12-34") is left out: it does not interpolate.
            let (f, t) = (text(&rec, from), text(&rec, to));
            if f.is_empty() || t.is_empty() || !f.chars().all(|c| c.is_ascii_digit()) || !t.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            let parity = match text(&rec, parity).as_str() {
                "O" => 1,
                "E" => 2,
                _ => 3,
            };
            Some(Side { from: f.parse().ok()?, to: t.parse().ok()?, parity, zip: text(&rec, zip).parse().unwrap_or(0) })
        };
        let left = side("LFROMHN", "LTOHN", "PARITYL", "ZIPL");
        let right = side("RFROMHN", "RTOHN", "PARITYR", "ZIPR");
        if left.is_none() && right.is_none() {
            continue;
        }
        out.push(Edge { street, left, right, pts });
    }
    Ok(out)
}

/// An OSM address: its house number as written, its street, its ZIP when tagged, where it is (a node, or a building's first corner).
#[derive(Clone, Debug)]
pub struct AddrPoint {
    pub number: String,
    pub street: String,
    pub zip: u32,
    pub lat: f64,
    pub lon: f64,
}

/// Every address in an OSM extract with a number and a street.
pub fn load_addresses(path: &str) -> Result<Vec<AddrPoint>, osmpbf::Error> {
    // An address's tags: number, street, ZIP (the first five digits of a postcode).
    fn tags_of<'a>(tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<(String, String, u32)> {
        let (mut num, mut street, mut zip) = (String::new(), String::new(), 0u32);
        for (k, v) in tags {
            match k {
                "addr:housenumber" => num = v.to_string(),
                "addr:street" => street = v.to_string(),
                "addr:postcode" => zip = v.chars().take(5).collect::<String>().parse().unwrap_or(0),
                _ => {}
            }
        }
        (!num.is_empty() && !street.is_empty()).then_some((num, street, zip))
    }
    enum Rec {
        Node(AddrPoint),
        Way(i64, (String, String, u32)),
    }
    let recs: Vec<Rec> = ElementReader::from_path(path)?.par_map_reduce(
        |e| match e {
            Element::Node(n) => tags_of(n.tags()).map(|(number, street, zip)| Rec::Node(AddrPoint { number, street, zip, lat: n.lat(), lon: n.lon() })).into_iter().collect(),
            Element::DenseNode(n) => tags_of(n.tags()).map(|(number, street, zip)| Rec::Node(AddrPoint { number, street, zip, lat: n.lat(), lon: n.lon() })).into_iter().collect(),
            Element::Way(w) => match (tags_of(w.tags()), w.refs().next()) {
                (Some(t), Some(first)) => vec![Rec::Way(first, t)],
                _ => Vec::new(),
            },
            _ => Vec::new(),
        },
        Vec::new,
        |mut a: Vec<Rec>, mut b| {
            a.append(&mut b);
            a
        },
    )?;
    let mut out = Vec::new();
    let mut ways: Vec<(i64, (String, String, u32))> = Vec::new();
    for r in recs {
        match r {
            Rec::Node(p) => out.push(p),
            Rec::Way(id, t) => ways.push((id, t)),
        }
    }
    let mut needed: Vec<i64> = ways.iter().map(|(id, _)| *id).collect();
    needed.sort_unstable();
    needed.dedup();
    let mut located: Vec<(i64, f64, f64)> = ElementReader::from_path(path)?.par_map_reduce(
        |e| {
            let (id, lat, lon) = match e {
                Element::Node(n) => (n.id(), n.lat(), n.lon()),
                Element::DenseNode(n) => (n.id(), n.lat(), n.lon()),
                _ => return Vec::new(),
            };
            if needed.binary_search(&id).is_ok() { vec![(id, lat, lon)] } else { Vec::new() }
        },
        Vec::new,
        |mut a, mut b| {
            a.append(&mut b);
            a
        },
    )?;
    located.sort_unstable_by_key(|l| l.0);
    for (id, (number, street, zip)) in ways {
        if let Ok(i) = located.binary_search_by_key(&id, |l| l.0) {
            out.push(AddrPoint { number, street, zip, lat: located[i].1, lon: located[i].2 });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{base, house_number, normal};

    #[test]
    fn street_names_meet_in_one_form() {
        assert_eq!(normal("North 9th Street"), "N 9TH ST");
        assert_eq!(normal("N. 9th St."), "N 9TH ST");
        assert_eq!(normal("Talon Trail"), "TALON TRL");
        assert_eq!(base("Talon Trail"), "TALON");
        assert_eq!(base("Talon Ln"), "TALON");
        assert_eq!(base("Southeast Sedgwick Road"), "SEDGWICK");
        assert_eq!(normal("North Bend Way"), "N BEND WAY");
        assert_eq!(base("Way"), "WAY", "a street called only by its type keeps it");
        assert_eq!(house_number("647A"), Some(647));
        assert_eq!(house_number("A12"), None);
    }
}
