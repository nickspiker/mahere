//! Names: what a texel can say it is. The baker interns every named line, area and point into one table while it reads the extract, and texels point into it (Nick 2026-10-13: tap a trail and it says which one; labels come from the same table).
use std::collections::HashMap;

/// A kind at or above this is an area: `AREA` + its [`crate::AreaClass`].
pub const AREA: u8 = 32;
/// A kind at or above this is a point: `POINT` + its [`PointKind`].
pub const POINT: u8 = 64;

/// One named thing: its kind (a line's class id 1..=17, an area's `AREA` + class, a point's `POINT` + kind), how much it matters against others of its sort (a point's population or height, an area's size, a river's catchment), its name and its ref ("US 93", "1551").
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Named {
    pub kind: u8,
    pub rank: u32,
    pub name: String,
    pub reference: String,
}

impl Named {
    pub fn is_point(&self) -> bool {
        self.kind >= POINT
    }

    pub fn is_area(&self) -> bool {
        (AREA..POINT).contains(&self.kind)
    }

    /// What a reader calls it: the name, the ref where there is no name, both where both say something.
    pub fn label(&self) -> String {
        match (self.name.is_empty(), self.reference.is_empty()) {
            (false, true) => self.name.clone(),
            (true, false) => self.reference.clone(),
            (false, false) if !self.name.contains(&self.reference) => format!("{} ({})", self.name, self.reference),
            (false, false) => self.name.clone(),
            (true, true) => String::new(),
        }
    }
}

/// The named places a map marks with a point. Ordered as their names in [`POINT_NAMES`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PointKind {
    City,
    Town,
    Village,
    Hamlet,
    Peak,
    Volcano,
    Saddle,
    Viewpoint,
    Trailhead,
    Waterfall,
    Spring,
    CampSite,
    Hut,
}

/// What each [`PointKind`] is called, by its index.
pub const POINT_NAMES: [&str; 13] = ["City", "Town", "Village", "Hamlet", "Peak", "Volcano", "Pass", "Viewpoint", "Trailhead", "Waterfall", "Spring", "Campsite", "Hut"];

/// A height from an OSM `ele` tag, metres: "4392", "4392 m", "14,411 ft".
pub fn parse_ele(v: &str) -> Option<f32> {
    let v = v.trim().replace(',', "");
    let (num, feet) = match v.strip_suffix("ft").or_else(|| v.strip_suffix('\'')) {
        Some(n) => (n.trim().to_string(), true),
        None => (v.trim_end_matches('m').trim().to_string(), false),
    };
    let e: f32 = num.parse().ok()?;
    Some(if feet { e * 0.3048 } else { e })
}

impl PointKind {
    /// A node's point kind and rank from its tags. Rank: a place's population (its kind's typical one where untagged); a summit's height in metres times 25, so Rainier's 4,392 m stands with a city of 110,000; a pass's height times 2; small marks a little.
    pub fn from_tags<'a>(tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<(PointKind, u32)> {
        use PointKind::*;
        let (mut kind, mut population, mut ele) = (None, None, None);
        for (k, v) in tags {
            match (k, v) {
                ("place", "city") => kind = Some(City),
                ("place", "town") => kind = Some(Town),
                ("place", "village") => kind = Some(Village),
                ("place", "hamlet") => kind = Some(Hamlet),
                ("natural", "peak") => kind = Some(Peak),
                ("natural", "volcano") => kind = Some(Volcano),
                ("natural", "saddle") | ("mountain_pass", "yes") => kind = kind.or(Some(Saddle)),
                ("tourism", "viewpoint") => kind = kind.or(Some(Viewpoint)),
                ("highway", "trailhead") => kind = kind.or(Some(Trailhead)),
                ("waterway", "waterfall") | ("natural", "waterfall") => kind = kind.or(Some(Waterfall)),
                ("natural", "spring") => kind = kind.or(Some(Spring)),
                ("tourism", "camp_site") => kind = kind.or(Some(CampSite)),
                ("tourism", "alpine_hut") | ("tourism", "wilderness_hut") => kind = kind.or(Some(Hut)),
                ("population", p) => population = p.replace([',', ' '], "").parse::<u32>().ok(),
                ("ele", e) => ele = parse_ele(e),
                _ => {}
            }
        }
        let kind = kind?;
        let height = |times: f32| ele.map_or(0, |e| (e.max(0.0) * times) as u32);
        let rank = match kind {
            City => population.unwrap_or(100_000),
            Town => population.unwrap_or(10_000),
            Village => population.unwrap_or(1_000),
            Hamlet => population.unwrap_or(100),
            Peak | Volcano => height(25.0),
            Saddle => height(2.0),
            Waterfall => 50,
            _ => 10,
        };
        Some((kind, rank))
    }
}

/// A named point: its entry in the table and where it is, codec-quantized like every other position.
#[derive(Clone, Debug)]
pub struct Point {
    pub name: u32,
    pub lat: f32,
    pub lon: f32,
}

/// The baker's table of named things. Entry 0 is "nothing named"; the same kind, name and ref intern to one entry wherever they appear (every piece of 3rd Avenue is one entry), keeping the highest rank seen.
#[derive(Clone, Debug)]
pub struct NameTable {
    pub list: Vec<Named>,
    index: HashMap<(u8, String, String), u32>,
}

impl Default for NameTable {
    fn default() -> Self {
        NameTable { list: vec![Named::default()], index: HashMap::new() }
    }
}

impl NameTable {
    /// The entry for a named thing; 0 when it has neither a name nor a ref.
    pub fn intern(&mut self, kind: u8, rank: u32, name: &str, reference: &str) -> u32 {
        let (name, reference) = (name.trim(), reference.trim());
        if name.is_empty() && reference.is_empty() {
            return 0;
        }
        let key = (kind, name.to_string(), reference.to_string());
        if let Some(&i) = self.index.get(&key) {
            let e = &mut self.list[i as usize];
            e.rank = e.rank.max(rank);
            return i;
        }
        let i = self.list.len() as u32;
        self.list.push(Named { kind, rank, name: key.1.clone(), reference: key.2.clone() });
        self.index.insert(key, i);
        i
    }

    /// Raise an entry's rank (a river's catchment is known only after the network pass).
    pub fn raise(&mut self, i: u32, rank: u32) {
        if i != 0 {
            let e = &mut self.list[i as usize];
            e.rank = e.rank.max(rank);
        }
    }
}

/// A ring's area in square metres, roughly: the shoelace over an equirectangular frame at the ring's latitude, enough to tell a pond from a lake.
pub fn ring_area_m2(ring: &[(f32, f32)]) -> f64 {
    let Some(&(lat0, _)) = ring.first() else { return 0.0 };
    let kx = 111_320.0 * (lat0 as f64).to_radians().cos();
    let mut s = 0.0;
    for w in ring.windows(2) {
        let (y0, x0) = (w[0].0 as f64 * 111_320.0, w[0].1 as f64 * kx);
        let (y1, x1) = (w[1].0 as f64 * 111_320.0, w[1].1 as f64 * kx);
        s += x0 * y1 - x1 * y0;
    }
    (s / 2.0).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heights_read_in_metres_or_feet() {
        assert_eq!(parse_ele("4392"), Some(4392.0));
        assert_eq!(parse_ele("4392 m"), Some(4392.0));
        assert!((parse_ele("14,411 ft").unwrap() - 4392.5).abs() < 0.5);
        assert_eq!(parse_ele("high"), None);
    }

    #[test]
    fn a_name_interns_once_and_keeps_its_highest_rank() {
        let mut t = NameTable::default();
        let a = t.intern(6, 0, "3rd Avenue", "");
        let b = t.intern(6, 5, "3rd Avenue", "");
        assert_eq!(a, b);
        assert_eq!(t.list[a as usize].rank, 5);
        assert_eq!(t.intern(6, 0, "", ""), 0);
        assert_ne!(t.intern(7, 0, "3rd Avenue", ""), a, "another kind is another thing");
    }

    #[test]
    fn rainier_outranks_a_cascade_bump_and_stands_with_a_city() {
        let rank = |tags: &[(&str, &str)]| PointKind::from_tags(tags.iter().copied()).unwrap().1;
        let rainier = rank(&[("natural", "volcano"), ("ele", "4392")]);
        let bump = rank(&[("natural", "peak"), ("ele", "1800")]);
        assert!(rainier > bump);
        assert!(rainier > rank(&[("place", "town"), ("population", "9000")]));
        assert!(rainier < rank(&[("place", "city"), ("population", "750000")]));
    }

    #[test]
    fn a_label_says_the_name_and_the_ref_once() {
        let n = |name: &str, r: &str| Named { kind: 3, rank: 0, name: name.into(), reference: r.into() }.label();
        assert_eq!(n("", "US 93"), "US 93");
        assert_eq!(n("Icicle Creek Trail", "1551"), "Icicle Creek Trail (1551)");
        assert_eq!(n("US Highway 93", "93"), "US Highway 93");
    }
}
