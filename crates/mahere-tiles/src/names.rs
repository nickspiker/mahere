//! Names in cells (Nick 2026-10-13: tap a trail and it says which one; labels read the same planes). At bake time a texel holds an entry of the bake's whole name table; a cell on disk holds its own small table and, per texel, an index into it.
//!
//! Two planes: `line` for lines and points, a point taking its one texel over any line there, and `area` for the land cover or water a texel lies in. Up the pyramid a name follows what is drawn: a parent's line texel names the child whose line won its class, a point outranks a line and a higher-ranked point a lower one (so Rainier keeps its texel when the Cascades' lesser summits lose theirs), and an area texel names the area most of its children lie in.
use std::collections::HashMap;

use rayon::prelude::*;

pub use mahere_osm::names::{AREA, POINT, POINT_NAMES};

use super::{child_cell_texels, disk_to_mem, mem_to_disk, raw_u16, tri_idx, CellKey, ClassCell, Named, TEX, TRI};

/// Bake-time name planes: entries of the bake's name table per texel, 0 where nothing is named. A plane stays empty until something is written to it.
#[derive(Clone, Default)]
pub struct IdCell {
    pub line: Vec<u32>,
    pub area: Vec<u32>,
}

impl IdCell {
    pub fn set_line(&mut self, i: usize, id: u32) {
        if self.line.is_empty() {
            self.line = vec![0; TRI];
        }
        self.line[i] = id;
    }

    pub fn set_area(&mut self, i: usize, id: u32) {
        if self.area.is_empty() {
            self.area = vec![0; TRI];
        }
        self.area[i] = id;
    }

    pub fn line_at(&self, i: usize) -> u32 {
        self.line.get(i).copied().unwrap_or(0)
    }

    pub fn area_at(&self, i: usize) -> u32 {
        self.area.get(i).copied().unwrap_or(0)
    }
}

/// What a cell's texels are called: its own table of named things, and per texel which entry, counting from 1 (0 = nothing named). An empty plane is one nothing in the cell is named on.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct NameCell {
    pub line: Vec<u16>,
    pub area: Vec<u16>,
    pub entries: Vec<Named>,
}

/// The most entries one cell's table holds: past it (a cell would need more distinct winners than two bytes count, which no real cell comes near) the lowest-ranked fall back to unnamed.
pub const MAX_ENTRIES: usize = u16::MAX as usize;

impl NameCell {
    fn entry(&self, plane: &[u16], i: usize) -> Option<&Named> {
        let n = *plane.get(i)?;
        (n != 0).then(|| &self.entries[n as usize - 1])
    }

    /// The line or point named at a texel.
    pub fn line_at(&self, i: usize) -> Option<&Named> {
        self.entry(&self.line, i)
    }

    /// The area a texel lies in, by name.
    pub fn area_at(&self, i: usize) -> Option<&Named> {
        self.entry(&self.area, i)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A cell's own table from bake-time planes: the entries its texels point at, in first-seen order.
    pub fn from_ids(ids: &IdCell, names: &[Named]) -> NameCell {
        let mut count: HashMap<u32, u32> = HashMap::new();
        for &id in ids.line.iter().chain(&ids.area) {
            if id != 0 {
                *count.entry(id).or_default() += 1;
            }
        }
        let mut kept: Vec<u32> = count.keys().copied().collect();
        kept.sort_unstable();
        if kept.len() > MAX_ENTRIES {
            kept.sort_by_key(|&id| std::cmp::Reverse((names[id as usize].rank, count[&id])));
            kept.truncate(MAX_ENTRIES);
            kept.sort_unstable();
        }
        let local: HashMap<u32, u16> = kept.iter().enumerate().map(|(n, &id)| (id, n as u16 + 1)).collect();
        let plane = |p: &[u32]| -> Vec<u16> {
            if p.iter().any(|id| local.contains_key(id)) {
                p.iter().map(|id| local.get(id).copied().unwrap_or(0)).collect()
            } else {
                Vec::new()
            }
        };
        NameCell { line: plane(&ids.line), area: plane(&ids.area), entries: kept.iter().map(|&id| names[id as usize].clone()).collect() }
    }

    /// Overlay a new bake's names on a cell's old ones: per texel the new where `take_new` says so, the old elsewhere; one table for both, without the entries no texel uses any more.
    pub fn merge(new: Option<NameCell>, old: Option<NameCell>, take_new: &dyn Fn(usize, u16) -> bool) -> Option<NameCell> {
        let (new, old) = match (new, old) {
            (n, None) => return n,
            (None, Some(o)) => (NameCell::default(), o),
            (Some(n), Some(o)) => (n, o),
        };
        let mut entries = new.entries.clone();
        let mut at: HashMap<Named, u16> = entries.iter().enumerate().map(|(i, e)| (e.clone(), i as u16 + 1)).collect();
        let remap: Vec<u16> = std::iter::once(0)
            .chain(old.entries.iter().map(|e| {
                if let Some(&i) = at.get(e) {
                    return i;
                }
                if entries.len() >= MAX_ENTRIES {
                    return 0;
                }
                entries.push(e.clone());
                at.insert(e.clone(), entries.len() as u16);
                entries.len() as u16
            }))
            .collect();
        let plane = |n: &[u16], o: &[u16]| -> Vec<u16> {
            if n.is_empty() && o.is_empty() {
                return Vec::new();
            }
            (0..TRI)
                .map(|i| {
                    let nv = n.get(i).copied().unwrap_or(0);
                    if take_new(i, nv) { nv } else { remap[o.get(i).copied().unwrap_or(0) as usize] }
                })
                .collect()
        };
        let merged = NameCell { line: plane(&new.line, &old.line), area: plane(&new.area, &old.area), entries };
        Some(merged.compact())
    }

    /// The same names without the entries no texel points at, and without planes that name nothing.
    pub fn compact(self) -> NameCell {
        let mut used = vec![false; self.entries.len() + 1];
        for &n in self.line.iter().chain(&self.area) {
            used[n as usize] = true;
        }
        let mut remap = vec![0u16; self.entries.len() + 1];
        let mut entries = Vec::new();
        for (i, e) in self.entries.into_iter().enumerate() {
            if used[i + 1] {
                entries.push(e);
                remap[i + 1] = entries.len() as u16;
            }
        }
        let plane = |p: Vec<u16>| -> Vec<u16> {
            if p.iter().any(|&n| remap[n as usize] != 0) {
                p.into_iter().map(|n| remap[n as usize]).collect()
            } else {
                Vec::new()
            }
        };
        NameCell { line: plane(self.line), area: plane(self.area), entries }
    }

    /// The cell's `name` section: the table as parallel lists (kind, rank, names and refs one per line) and the two planes, a byte per texel while the table has under 256 entries.
    pub fn section(&self) -> Option<vsf::VsfSection> {
        use vsf::{types::Tensor, VsfType};
        if self.entries.is_empty() {
            return None;
        }
        let clean = |s: &str| s.replace(['\n', '\r'], " ");
        let mut fields = vec![
            ("kind".to_string(), VsfType::t_u3(Tensor::new(vec![self.entries.len()], self.entries.iter().map(|e| e.kind).collect()))),
            ("rank".to_string(), VsfType::t_u5(Tensor::new(vec![self.entries.len()], self.entries.iter().map(|e| e.rank).collect()))),
            ("text".to_string(), VsfType::x(self.entries.iter().map(|e| clean(&e.name)).collect::<Vec<_>>().join("\n"))),
            ("ref".to_string(), VsfType::x(self.entries.iter().map(|e| clean(&e.reference)).collect::<Vec<_>>().join("\n"))),
        ];
        let narrow = self.entries.len() < 256;
        for (name, p) in [("line", &self.line), ("area", &self.area)] {
            if p.len() == TRI {
                let disk = mem_to_disk(p);
                let v = if narrow { VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], disk.iter().map(|&n| n as u8).collect())) } else { VsfType::t_u4(Tensor::new(vec![2, TEX * TEX], disk)) };
                fields.push((name.to_string(), v));
            }
        }
        let mut s = vsf::VsfSection::new("name");
        for (name, v) in fields {
            s.add_field(name, v);
        }
        Some(s)
    }

    /// A `name` section's fields back into a cell's names.
    pub fn decode(fields: &HashMap<String, Vec<vsf::VsfType>>) -> Option<NameCell> {
        use vsf::VsfType;
        let first = |name: &str| fields.get(name).and_then(|v| v.first());
        let kinds: Vec<u8> = match first("kind")? {
            VsfType::t_u3(t) => t.data.clone(),
            _ => return None,
        };
        let ranks: Vec<u32> = match first("rank")? {
            VsfType::t_u5(t) => t.data.clone(),
            VsfType::t_u4(t) => t.data.iter().map(|&r| r as u32).collect(),
            VsfType::t_u3(t) => t.data.iter().map(|&r| r as u32).collect(),
            _ => return None,
        };
        let text = |name: &str| -> Vec<String> {
            match first(name) {
                Some(VsfType::x(s) | VsfType::a(s)) => s.split('\n').map(str::to_string).collect(),
                _ => Vec::new(),
            }
        };
        let (names, refs) = (text("text"), text("ref"));
        let n = kinds.len();
        if ranks.len() != n || names.len() != n || refs.len() != n {
            return None;
        }
        let entries: Vec<Named> = (0..n).map(|i| Named { kind: kinds[i], rank: ranks[i], name: names[i].clone(), reference: refs[i].clone() }).collect();
        let plane = |name: &str| -> Vec<u16> {
            first(name).and_then(raw_u16).filter(|d| d.len() == TRI).map(|d| disk_to_mem(&d)).unwrap_or_default()
        };
        let cell = NameCell { line: plane("line"), area: plane("area"), entries };
        cell.line.iter().chain(&cell.area).all(|&i| (i as usize) <= n).then_some(cell)
    }
}

/// Which of two areas a texel is named for: water over land, then the class the land plane draws (the higher), then the smaller area, which is the more particular (a named park inside a named forest).
pub fn area_beats(a: &Named, b: &Named) -> bool {
    let water = mahere_osm::names::AREA + mahere_osm::AreaClass::Water as u8;
    let key = |n: &Named| (n.kind == water, n.kind, std::cmp::Reverse(n.rank));
    key(a) > key(b)
}

/// Build every level's names from the base level up: a parent texel's line names the child whose line won the parent's class (the most coverage among them), unless a child holds a point, when the highest-ranked point wins; its area names the area most of its children lie in. `line` must hold the already-built line pyramid.
pub fn pyramid_ids(mut ids: HashMap<CellKey, IdCell>, line: &HashMap<CellKey, ClassCell>, names: &[Named], base_depth: u8, min_depth: u8) -> HashMap<CellKey, IdCell> {
    let mut depth = base_depth;
    while depth > min_depth {
        let mut parents: Vec<CellKey> = ids.keys().filter(|k| k.depth == depth).map(|k| k.parent()).collect();
        parents.sort();
        parents.dedup();
        let built: Vec<(CellKey, IdCell)> = parents
            .par_iter()
            .filter_map(|&p| {
                let kids: [Option<&IdCell>; 4] = std::array::from_fn(|q| ids.get(&p.child(q as u64)));
                let kid_lines: [Option<&ClassCell>; 4] = std::array::from_fn(|q| line.get(&p.child(q as u64)));
                let parent_line = line.get(&p);
                let any_line = kids.iter().flatten().any(|k| !k.line.is_empty());
                let any_area = kids.iter().flatten().any(|k| !k.area.is_empty());
                let mut cell = IdCell::default();
                for ty in 0..TEX {
                    for tx in 0..TEX {
                        for half in 0..2 {
                            let pi = tri_idx(tx, ty, half);
                            let kid_texels = child_cell_texels(tx, ty, half);
                            if any_line {
                                let class = parent_line.map_or(0, |c| c.class[pi]);
                                let (mut point, mut drawn, mut drawn_cov) = (0u32, 0u32, 0u8);
                                for (q, i) in kid_texels {
                                    let Some(k) = kids[q as usize] else { continue };
                                    let id = k.line_at(i);
                                    if id == 0 {
                                        continue;
                                    }
                                    let n = &names[id as usize];
                                    if n.is_point() {
                                        if point == 0 || n.rank > names[point as usize].rank {
                                            point = id;
                                        }
                                    } else if let Some(l) = kid_lines[q as usize] {
                                        if class != 0 && l.class[i] == class && (drawn == 0 || l.cov[i] > drawn_cov) {
                                            drawn = id;
                                            drawn_cov = l.cov[i];
                                        }
                                    }
                                }
                                let id = if point != 0 { point } else { drawn };
                                if id != 0 {
                                    cell.set_line(pi, id);
                                }
                            }
                            if any_area {
                                let mut votes: [(u32, u8); 4] = [(0, 0); 4];
                                for (q, i) in kid_texels {
                                    let Some(k) = kids[q as usize] else { continue };
                                    let id = k.area_at(i);
                                    if id == 0 {
                                        continue;
                                    }
                                    if let Some(v) = votes.iter_mut().find(|v| v.0 == id || v.0 == 0) {
                                        v.0 = id;
                                        v.1 += 1;
                                    }
                                }
                                let mut best: (u32, u8) = (0, 0);
                                for &(id, n) in votes.iter().filter(|v| v.0 != 0) {
                                    if n > best.1 || (n == best.1 && area_beats(&names[id as usize], &names[best.0 as usize])) {
                                        best = (id, n);
                                    }
                                }
                                if best.0 != 0 {
                                    cell.set_area(pi, best.0);
                                }
                            }
                        }
                    }
                }
                (!cell.line.is_empty() || !cell.area.is_empty()).then_some((p, cell))
            })
            .collect();
        ids.extend(built);
        depth -= 1;
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(kind: u8, rank: u32, name: &str) -> Named {
        Named { kind, rank, name: name.into(), reference: String::new() }
    }

    #[test]
    fn a_cell_keeps_only_the_names_its_texels_use_and_round_trips() {
        let names = vec![Named::default(), named(6, 0, "Pike Street"), named(6, 0, "Pine Street"), named(44, 9, "Lake Union")];
        let mut ids = IdCell::default();
        ids.set_line(5, 1);
        ids.set_line(6, 1);
        ids.set_area(7, 3);
        let cell = NameCell::from_ids(&ids, &names);
        assert_eq!(cell.entries.len(), 2, "Pine Street is in the table but on no texel");
        assert_eq!(cell.line_at(5).unwrap().name, "Pike Street");
        assert_eq!(cell.area_at(7).unwrap().name, "Lake Union");
        assert!(cell.line_at(7).is_none());
        let s = cell.section().unwrap();
        let fields: HashMap<String, Vec<vsf::VsfType>> = s.fields.into_iter().map(|f| (f.name, f.values)).collect();
        assert_eq!(NameCell::decode(&fields), Some(cell));
    }

    #[test]
    fn a_merge_takes_the_new_inside_and_keeps_the_old_outside() {
        let mut old = NameCell { line: vec![0; TRI], area: Vec::new(), entries: vec![named(9, 0, "Old Trail")] };
        old.line[1] = 1;
        old.line[2] = 1;
        let mut new = NameCell { line: vec![0; TRI], area: Vec::new(), entries: vec![named(9, 0, "New Trail")] };
        new.line[2] = 1;
        let m = NameCell::merge(Some(new), Some(old), &|i, _| i == 2).unwrap();
        assert_eq!(m.line_at(1).unwrap().name, "Old Trail");
        assert_eq!(m.line_at(2).unwrap().name, "New Trail");
        let gone = NameCell::merge(Some(NameCell::default()), Some(m), &|i, _| i == 1 || i == 2);
        assert!(gone.unwrap().is_empty(), "names no texel uses are dropped");
    }

    #[test]
    fn up_the_pyramid_a_point_beats_a_line_and_the_higher_point_wins() {
        let names = vec![Named::default(), named(2, 0, "Highway"), named(64 + 4, 110_000, "Mount Rainier"), named(64 + 4, 45_000, "Lesser Peak")];
        let p = CellKey { depth: 9, prefix: 5 << 18 };
        let kids: Vec<CellKey> = (0..4).map(|q| p.child(q)).collect();
        // The parent texel 0's four children, and a line drawn through all of them.
        let texels = child_cell_texels(0, 0, 0);
        let mut ids: HashMap<CellKey, IdCell> = HashMap::new();
        let mut lines: HashMap<CellKey, ClassCell> = HashMap::new();
        for (q, i) in texels {
            let k = kids[q as usize];
            ids.entry(k).or_default().set_line(i, 1);
            let l = lines.entry(k).or_insert_with(ClassCell::new_line);
            l.class[i] = 2;
            l.cov[i] = 200;
        }
        let mut parent_line = ClassCell::new_line();
        parent_line.class[tri_idx(0, 0, 0)] = 2;
        lines.insert(p, parent_line);
        let built = pyramid_ids(ids.clone(), &lines, &names, 10, 9);
        assert_eq!(built[&p].line_at(tri_idx(0, 0, 0)), 1, "a line alone names its texel");
        let (q0, i0) = texels[0];
        let (q1, i1) = texels[1];
        ids.get_mut(&kids[q0 as usize]).unwrap().set_line(i0, 3);
        ids.get_mut(&kids[q1 as usize]).unwrap().set_line(i1, 2);
        let built = pyramid_ids(ids, &lines, &names, 10, 9);
        assert_eq!(built[&p].line_at(tri_idx(0, 0, 0)), 2, "Rainier wins over the lesser peak and the highway");
    }
}
