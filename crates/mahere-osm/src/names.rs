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

    /// What a reader calls it: the name, the ref where there is no name, both where both say something; a point by its name alone (a summit's ref is a survey number); an exit by its number, then its name ("165A Seneca St").
    pub fn label(&self) -> String {
        if self.kind == POINT + PointKind::Exit as u8 {
            return [self.reference.as_str(), self.name.as_str()].iter().filter(|s| !s.is_empty()).copied().collect::<Vec<_>>().join(" ");
        }
        if self.is_point() && !self.name.is_empty() {
            return self.name.clone();
        }
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
    // Added 2026-10-16 (Nick: exit numbers, hot springs, fire lookouts "and other goodies"), after the first thirteen so a cell's kinds keep their meaning; an older reader letters these as plain small marks.
    /// A freeway exit: its number the ref, its name if it has one.
    Exit,
    Island,
    Islet,
    Bay,
    Strait,
    Sea,
    Cape,
    /// A district of a city: suburb, quarter or neighbourhood.
    Neighbourhood,
    /// A named place that is not a settlement: a meadow, a flat, a basin.
    Locality,
    DrinkingWater,
    Toilets,
    Shelter,
    RangerStation,
    /// A fire lookout or other observation tower.
    Lookout,
    HotSpring,
    Cave,
    Gate,
    Ford,
}

/// What each [`PointKind`] is called, by its index.
pub const POINT_NAMES: [&str; 31] = [
    "City", "Town", "Village", "Hamlet", "Peak", "Volcano", "Pass", "Viewpoint", "Trailhead", "Waterfall", "Spring", "Campsite", "Hut", "Exit", "Island", "Islet", "Bay", "Strait", "Sea", "Cape", "Neighbourhood", "Locality", "Drinking water", "Toilets", "Shelter", "Ranger station", "Lookout", "Hot spring", "Cave", "Gate", "Ford",
];

impl PointKind {
    /// The kinds marked on the map whether or not they have a name (a tap, a gate): an unnamed one carries its kind's word for the tap card, and the map draws its mark alone.
    pub fn marks_unnamed(self) -> bool {
        use PointKind::*;
        matches!(self, Exit | DrinkingWater | Toilets | Shelter | RangerStation | Lookout | HotSpring | Cave | Gate | Ford)
    }

    /// The kind an island's, a bay's or a cape's outline names (a closed way or a multipolygon), labelled at a point inside it.
    pub fn of_outline<'a>(tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<PointKind> {
        let mut kind = None;
        for (k, v) in tags {
            kind = kind.or(match (k, v) {
                ("place", "island") => Some(PointKind::Island),
                ("place", "islet") => Some(PointKind::Islet),
                ("natural", "cape") | ("natural", "peninsula") => Some(PointKind::Cape),
                ("place", "locality") => Some(PointKind::Locality),
                _ => None,
            });
        }
        kind
    }
}

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
        let (mut kind, mut population, mut ele, mut tower) = (None, None, None, (false, false));
        for (k, v) in tags {
            match (k, v) {
                ("highway", "motorway_junction") => kind = kind.or(Some(Exit)),
                ("place", "island") => kind = kind.or(Some(Island)),
                ("place", "islet") => kind = kind.or(Some(Islet)),
                ("natural", "bay") => kind = kind.or(Some(Bay)),
                ("natural", "strait") => kind = kind.or(Some(Strait)),
                ("place", "sea") | ("place", "ocean") => kind = kind.or(Some(Sea)),
                ("natural", "cape") | ("natural", "peninsula") => kind = kind.or(Some(Cape)),
                ("place", "suburb") | ("place", "quarter") | ("place", "neighbourhood") => kind = kind.or(Some(Neighbourhood)),
                ("place", "locality") => kind = kind.or(Some(Locality)),
                ("amenity", "drinking_water") | ("man_made", "drinking_fountain") => kind = kind.or(Some(DrinkingWater)),
                ("amenity", "toilets") => kind = kind.or(Some(Toilets)),
                ("amenity", "shelter") => kind = kind.or(Some(Shelter)),
                ("amenity", "ranger_station") => kind = kind.or(Some(RangerStation)),
                ("man_made", "tower") => tower.0 = true,
                ("tower:type", "observation") | ("tower:type", "fire_lookout") => tower.1 = true,
                ("man_made", "fire_lookout") => kind = kind.or(Some(Lookout)),
                ("natural", "hot_spring") | ("leisure", "hot_spring") => kind = kind.or(Some(HotSpring)),
                ("natural", "cave_entrance") => kind = kind.or(Some(Cave)),
                ("barrier", "gate") | ("barrier", "lift_gate") => kind = kind.or(Some(Gate)),
                ("ford", "yes") | ("ford", "stepping_stones") => kind = kind.or(Some(Ford)),
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
        // A tower is a lookout only when it is for looking out from.
        if tower == (true, true) {
            kind = kind.or(Some(Lookout));
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
            Sea => 10_000_000,
            Bay | Strait => 100_000,
            Island => 50_000,
            Neighbourhood => population.unwrap_or(5_000),
            Lookout => height(1.0).max(20),
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

/// A point well inside an outline, for its name: across the largest ring's centroid latitude, the middle of the widest run inside the outline (even-odd over every ring, so a lake in an island is outside). A crescent's centroid can sit in the water; this does not. None for an outline with no area.
pub fn interior_point(rings: &[Vec<(f32, f32)>]) -> Option<(f32, f32)> {
    let largest = rings.iter().max_by(|a, b| ring_area_m2(a).total_cmp(&ring_area_m2(b)))?;
    if largest.len() < 4 {
        return None;
    }
    // The ring's area-weighted centroid latitude (planar on degrees, fine at a label's scale).
    let (mut a2, mut cy) = (0.0f64, 0.0f64);
    for w in largest.windows(2) {
        let (y0, x0, y1, x1) = (w[0].0 as f64, w[0].1 as f64, w[1].0 as f64, w[1].1 as f64);
        let cross = x0 * y1 - x1 * y0;
        a2 += cross;
        cy += (y0 + y1) * cross;
    }
    let lat = if a2.abs() > 1e-18 { cy / (3.0 * a2) } else { largest.iter().map(|p| p.0 as f64).sum::<f64>() / largest.len() as f64 };
    // Where the line of that latitude crosses each ring's edges, in order: inside between alternate crossings.
    let mut xs: Vec<f64> = Vec::new();
    for ring in rings {
        for w in ring.windows(2) {
            let (y0, x0, y1, x1) = (w[0].0 as f64, w[0].1 as f64, w[1].0 as f64, w[1].1 as f64);
            if (y0 <= lat) != (y1 <= lat) {
                xs.push(x0 + (lat - y0) / (y1 - y0) * (x1 - x0));
            }
        }
    }
    xs.sort_by(f64::total_cmp);
    let run = xs.chunks_exact(2).max_by(|a, b| (a[1] - a[0]).total_cmp(&(b[1] - b[0])))?;
    Some((lat as f32, ((run[0] + run[1]) / 2.0) as f32))
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
        let peak = Named { kind: POINT + PointKind::Volcano as u8, rank: 0, name: "Mauna Kea".into(), reference: "332030".into() };
        assert_eq!(peak.label(), "Mauna Kea");
        let exit = |name: &str, r: &str| Named { kind: POINT + PointKind::Exit as u8, rank: 0, name: name.into(), reference: r.into() }.label();
        assert_eq!(exit("Seneca Street", "165A"), "165A Seneca Street");
        assert_eq!(exit("", "165A"), "165A");
    }

    /// A C-shaped island's label lands on the island, not in the bay its centroid falls in.
    #[test]
    fn an_outlines_point_is_inside_it() {
        // A C: a square 0..10 with the middle of its right side cut away (x 3..10, y 3..7).
        let c: Vec<(f32, f32)> = [(0.0, 0.0), (0.0, 10.0), (3.0, 10.0), (3.0, 3.0), (7.0, 3.0), (7.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)].iter().map(|&(y, x)| (y, x)).collect();
        let (lat, lon) = interior_point(&[c]).unwrap();
        assert!((3.0..7.0).contains(&lat) && lon < 3.0, "inside the C's back, not its open bay: ({lat}, {lon})");
        let square: Vec<(f32, f32)> = vec![(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0), (0.0, 0.0)];
        let (lat, lon) = interior_point(&[square]).unwrap();
        assert!((lat - 2.0).abs() < 0.01 && (lon - 2.0).abs() < 0.01);
    }

    #[test]
    fn the_new_kinds_read_from_their_tags() {
        let kind = |tags: &[(&str, &str)]| PointKind::from_tags(tags.iter().copied()).map(|k| k.0);
        assert_eq!(kind(&[("highway", "motorway_junction"), ("ref", "165A")]), Some(PointKind::Exit));
        assert_eq!(kind(&[("man_made", "tower"), ("tower:type", "observation")]), Some(PointKind::Lookout));
        assert_eq!(kind(&[("man_made", "tower"), ("tower:type", "communication")]), None, "a radio mast is no lookout");
        assert_eq!(kind(&[("natural", "hot_spring")]), Some(PointKind::HotSpring));
        assert_eq!(kind(&[("place", "neighbourhood")]), Some(PointKind::Neighbourhood));
        assert_eq!(POINT_NAMES.len(), PointKind::Ford as usize + 1);
        assert!(PointKind::Gate.marks_unnamed() && !PointKind::Island.marks_unnamed());
    }
}
