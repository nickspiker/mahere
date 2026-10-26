//! Routing in cells (Nick 2026-10-13: "flood fill our way to a fairly reasonable navigation platform", overpasses as "a special dual/triple case list"). The ways' centrelines are stamped a trixel wide at the base depth, apart from the drawn lines (whose widths overlap at every crossing). A cell holds a table of the ways in it (by OSM id, so a route knows a way across cells), the plane naming one way per trixel, the stack of further ways sharing a trixel without meeting there (a bridge over a road, a road over a tunnel), the trixels where ways truly meet (a node they share; a route changes ways only there), and each one-way trixel's direction of travel.
//! The maze's graph (Nick 2026-10-25: "nobody's figured out this noded maze fill?"): on each way, the trixels where a route can do anything but go on — a junction, the cell's edge, a dead end — are its NODES, clustered where they touch and named by the cluster's lowest trixel; the runs of the way's trixels between nodes are its CORRIDORS, each with its two nodes, its length and which way it may be travelled. A search hops node to node along corridors instead of trixel to trixel, which is what a vector router gets from its edges; the raster is still the only thing stored. Built by [`RouteCell::build_graph`] from the stamped planes, at the base depth and at every coarser one the pyramid makes.
use std::collections::HashMap;

use mahere_coord::Coord;
use mahere_osm::Road;

use super::{disk_to_mem, mem_to_disk, raw_u16, texel_of, tri_at, tri_idx, CellKey, Named, TEX, TRI};

/// One way as a route sees it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Way {
    pub osm: i64,
    /// The line class id (1 = motorway .. 9 = path).
    pub class: u8,
    /// Who may use it ([`mahere_osm::use_bits`]).
    pub uses: u8,
    /// The posted limit, km/h; 0 untagged.
    pub speed_kmh: u8,
    pub oneway: i8,
    pub layer: i8,
    /// Difficulty low, roughness high ([`mahere_osm::Travel::grade`], 2026-10-16); 0 in a cell made before.
    pub grade: u8,
    /// What turn-by-turn calls it.
    pub name: String,
}

/// A cell's routing: its ways and where each runs. Way numbers count from 1 (0 = no way).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouteCell {
    /// A way per trixel, memory order; empty when no way runs through the cell.
    pub way: Vec<u16>,
    pub ways: Vec<Way>,
    /// Further ways on a trixel the plane already gives another: (trixel, way).
    pub stack: Vec<(u32, u16)>,
    /// Trixels where ways meet.
    pub joins: Vec<u32>,
    /// A one-way trixel's direction of travel, per way: (trixel, way, angle in 256ths of a turn, in the cell's grid).
    pub dir: Vec<(u32, u16, u8)>,
    /// The graph: every node trixel with its way and its cluster's name (the cluster's lowest trixel), sorted by (trixel, way).
    pub nodes: Vec<(u32, u16, u32)>,
    /// Each corridor: its two nodes (cluster names), its way, its length in 256ths of a texel edge, and its direction (0 either way, 1 from a to b only, 2 from b to a only).
    pub corridors: Vec<Corridor>,
    /// Per trixel, the corridor (1-based; 0 none: a node, or no way) of the plane's way; and of each stacked way, parallel to `stack`.
    pub cor: Vec<u16>,
    pub stackcor: Vec<u16>,
}

/// A run of one way's trixels between two nodes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Corridor {
    pub a: u32,
    pub b: u32,
    pub way: u16,
    pub len: u32,
    pub dir: u8,
}

/// The two halves of the squares around a trixel: its eight-ish neighbours in the cell, by memory index.
fn around(i: usize) -> impl Iterator<Item = usize> {
    let (tx, ty) = ((i >> 1) & (TEX - 1), i >> (1 + super::TEX_BITS));
    (-1i64..=1).flat_map(move |dy| (-1i64..=1).flat_map(move |dx| (0..2usize).map(move |h| (dx, dy, h)))).filter_map(move |(dx, dy, h)| {
        let (x, y) = (tx as i64 + dx, ty as i64 + dy);
        if x < 0 || y < 0 || x >= TEX as i64 || y >= TEX as i64 {
            return None;
        }
        let j = tri_idx(x as usize, y as usize, h);
        (j != i).then_some(j)
    })
}

/// The node trixels clustered where they touch, each given its cluster's name (the lowest trixel in it); None for a trixel that is not a node. Indexed as `trixels` is.
fn cluster_names(trixels: &[u32], is_node: &[bool]) -> Vec<Option<u32>> {
    let mut name: Vec<Option<u32>> = vec![None; trixels.len()];
    for k in 0..trixels.len() {
        if !is_node[k] || name[k].is_some() {
            continue;
        }
        let mut members = vec![k];
        name[k] = Some(0);
        let mut q = 0;
        while q < members.len() {
            let i = trixels[members[q]] as usize;
            q += 1;
            for j in around(i) {
                if let Ok(m) = trixels.binary_search(&(j as u32)) {
                    if is_node[m] && name[m].is_none() {
                        // Claimed as found, so a cluster never takes a trixel twice.
                        name[m] = Some(0);
                        members.push(m);
                    }
                }
            }
        }
        let low = members.iter().map(|&m| trixels[m]).min().unwrap();
        for &m in &members {
            name[m] = Some(low);
        }
    }
    name
}

/// Whether a trixel's square touches the cell's edge.
fn on_border(i: usize) -> bool {
    let (tx, ty) = ((i >> 1) & (TEX - 1), i >> (1 + super::TEX_BITS));
    tx == 0 || ty == 0 || tx == TEX - 1 || ty == TEX - 1
}

/// The centroid of a trixel in texel units.
fn centre(i: usize) -> (f64, f64) {
    let (half, tx, ty) = (i & 1, (i >> 1) & (TEX - 1), i >> (1 + super::TEX_BITS));
    super::tri_centroid(tx, ty, half)
}

/// Texel edges between two trixels' centroids (the grid's axes are 60° apart).
fn grid_len(a: usize, b: usize) -> f64 {
    let (p, q) = (centre(a), centre(b));
    let (du, dv) = (q.0 - p.0, q.1 - p.1);
    (du * du + dv * dv + du * dv).max(0.0).sqrt()
}

impl RouteCell {
    /// The ways on a trixel: the plane's and the stack's.
    pub fn ways_at(&self, i: usize) -> impl Iterator<Item = u16> + '_ {
        let first = self.way.get(i).copied().filter(|&w| w != 0);
        let i = i as u32;
        let lo = self.stack.partition_point(|&(t, _)| t < i);
        first.into_iter().chain(self.stack[lo..].iter().take_while(move |&&(t, _)| t == i).map(|&(_, w)| w))
    }

    pub fn is_join(&self, i: usize) -> bool {
        self.joins.binary_search(&(i as u32)).is_ok()
    }

    /// A one-way trixel's direction of travel for a way, radians in the cell's grid (u along x, v along y).
    pub fn direction(&self, i: usize, way: u16) -> Option<f32> {
        let i = i as u32;
        let lo = self.dir.partition_point(|&(t, _, _)| t < i);
        self.dir[lo..].iter().take_while(|&&(t, _, _)| t == i).find(|&&(_, w, _)| w == way).map(|&(_, _, a)| a as f32 / 256.0 * std::f32::consts::TAU)
    }

    fn add(&mut self, i: usize, w: u16) {
        if self.way.is_empty() {
            self.way = vec![0; TRI];
        }
        if self.way[i] == 0 {
            self.way[i] = w;
        } else if self.way[i] != w && !self.stack.contains(&(i as u32, w)) {
            self.stack.push((i as u32, w));
        }
    }

    /// The corridor of a trixel's way, 1-based; 0 for a node or no way.
    pub fn corridor_at(&self, i: usize, way: u16) -> u16 {
        if self.way.get(i).copied() == Some(way) {
            return self.cor.get(i).copied().unwrap_or(0);
        }
        let lo = self.stack.partition_point(|&(t, _)| t < i as u32);
        self.stack[lo..].iter().zip(&self.stackcor[lo..]).take_while(|((t, _), _)| *t == i as u32).find(|((_, w), _)| *w == way).map_or(0, |(_, &c)| c)
    }

    /// The node cluster a node trixel of a way belongs to, by its name; None where the trixel is not a node of that way.
    pub fn node_of(&self, i: usize, way: u16) -> Option<u32> {
        self.nodes.binary_search_by_key(&(i as u32, way), |n| (n.0, n.1)).ok().map(|k| self.nodes[k].2)
    }

    /// The trixels of a node cluster on a way.
    pub fn node_trixels(&self, name: u32, way: u16) -> Vec<u32> {
        self.nodes.iter().filter(|n| n.2 == name && n.1 == way).map(|n| n.0).collect()
    }

    /// Whether the cell's graph is built (a cell baked before the graph has none, and routes the slow way).
    pub fn has_graph(&self) -> bool {
        !self.corridors.is_empty() || !self.nodes.is_empty()
    }

    /// The graph from the stamped planes: on each way, its nodes (the junctions, the trixels on the cell's edge, the dead ends), clustered where they touch, and the corridors between them with their lengths and directions. Built after the stamping, and again after a pyramid merge.
    pub fn build_graph(&mut self) {
        self.nodes.clear();
        self.corridors.clear();
        self.cor = if self.way.is_empty() { Vec::new() } else { vec![0; TRI] };
        self.stackcor = vec![0; self.stack.len()];
        if self.way.is_empty() {
            return;
        }
        let n_ways = self.ways.len() as u16;
        // Each way's trixels, from the plane and the stack.
        let mut on_way: Vec<Vec<u32>> = vec![Vec::new(); n_ways as usize + 1];
        for (i, &w) in self.way.iter().enumerate() {
            if w != 0 {
                on_way[w as usize].push(i as u32);
            }
        }
        for &(t, w) in &self.stack {
            on_way[w as usize].push(t);
        }
        for w in 1..=n_ways {
            let mut trixels = std::mem::take(&mut on_way[w as usize]);
            trixels.sort_unstable();
            trixels.dedup();
            self.graph_of_way(w, &trixels);
        }
        self.nodes.sort_unstable();
    }

    /// One way's nodes and corridors. The way's trixels are visited in a set the size of the cell; components of the non-node trixels are its corridors.
    fn graph_of_way(&mut self, w: u16, trixels: &[u32]) {
        // Nodes first: junctions and the edge.
        let mut is_node: Vec<bool> = trixels.iter().map(|&t| self.is_join(t as usize) || on_border(t as usize)).collect();
        // A way that shares only one trixel with the cell is a node by itself.
        if trixels.len() == 1 {
            is_node[0] = true;
        }
        // The non-node components, each a corridor (or a hub touching several node clusters).
        let mut seen = vec![false; trixels.len()];
        let mut comps: Vec<Vec<usize>> = Vec::new();
        for k in 0..trixels.len() {
            if is_node[k] || seen[k] {
                continue;
            }
            let mut comp = vec![k];
            seen[k] = true;
            let mut q = 0;
            while q < comp.len() {
                let i = trixels[comp[q]] as usize;
                q += 1;
                for j in around(i) {
                    if let Ok(m) = trixels.binary_search(&(j as u32)) {
                        if !is_node[m] && !seen[m] {
                            seen[m] = true;
                            comp.push(m);
                        }
                    }
                }
            }
            comps.push(comp);
        }
        // Dead ends and isolated runs get nodes at their far ends, so every corridor has two: a component touching one cluster of nodes (or none) is such a run.
        let first_names = cluster_names(trixels, &is_node);
        for comp in &comps {
            let mut touching: Vec<u32> = comp.iter().flat_map(|&k| around(trixels[k] as usize)).filter_map(|j| trixels.binary_search(&(j as u32)).ok()).filter_map(|m| first_names[m]).collect();
            touching.sort_unstable();
            touching.dedup();
            let far_from = |start: &[usize], comp: &[usize]| -> usize {
                // Breadth first across the component from the given trixels: the last reached is the far end.
                let mut in_comp = vec![false; trixels.len()];
                for &k in comp {
                    in_comp[k] = true;
                }
                let mut reached = vec![false; trixels.len()];
                let mut queue: std::collections::VecDeque<usize> = start.iter().copied().collect();
                for &k in start {
                    reached[k] = true;
                }
                let mut last = start[0];
                while let Some(k) = queue.pop_front() {
                    last = k;
                    for j in around(trixels[k] as usize) {
                        if let Ok(m) = trixels.binary_search(&(j as u32)) {
                            if in_comp[m] && !reached[m] {
                                reached[m] = true;
                                queue.push_back(m);
                            }
                        }
                    }
                }
                last
            };
            match touching.len() {
                0 => {
                    let p = far_from(&[comp[0]], comp);
                    let q = far_from(&[p], comp);
                    is_node[p] = true;
                    is_node[q] = true;
                }
                1 => {
                    let near: Vec<usize> = comp.iter().copied().filter(|&k| around(trixels[k] as usize).any(|j| trixels.binary_search(&(j as u32)).is_ok_and(|m| is_node[m]))).collect();
                    let p = far_from(&near, comp);
                    is_node[p] = true;
                }
                _ => {}
            }
        }
        // Clusters and corridors, again with any trixel left out promoted to a node, until every trixel is a node or in a corridor (a shape the rules above missed: a single trixel between two clusters, a loop).
        loop {
        let nodes_before = self.nodes.len();
        let corridors_before = self.corridors.len();
        // Clusters of touching nodes, named by the lowest trixel.
        let name = cluster_names(trixels, &is_node);
        for k in 0..trixels.len() {
            if let Some(n) = name[k] {
                self.nodes.push((trixels[k], w, n));
            }
        }
        // Corridors: within each component (now cut by the nodes added), the shortest path between every pair of touching clusters.
        let mut comps: Vec<Vec<usize>> = Vec::new();
        let mut seen = vec![false; trixels.len()];
        for k in 0..trixels.len() {
            if is_node[k] || seen[k] {
                continue;
            }
            let mut comp = vec![k];
            seen[k] = true;
            let mut q = 0;
            while q < comp.len() {
                let i = trixels[comp[q]] as usize;
                q += 1;
                for j in around(i) {
                    if let Ok(m) = trixels.binary_search(&(j as u32)) {
                        if !is_node[m] && !seen[m] {
                            seen[m] = true;
                            comp.push(m);
                        }
                    }
                }
            }
            comps.push(comp);
        }
        for comp in comps {
            let mut in_comp = vec![false; trixels.len()];
            for &k in &comp {
                in_comp[k] = true;
            }
            // The clusters this component touches, each with the component trixels next to it.
            let mut clusters: Vec<(u32, Vec<usize>)> = Vec::new();
            for &k in &comp {
                for j in around(trixels[k] as usize) {
                    if let Ok(m) = trixels.binary_search(&(j as u32)) {
                        if let Some(n) = name[m].filter(|_| is_node[m]) {
                            match clusters.iter_mut().find(|c| c.0 == n) {
                                Some(c) => {
                                    if c.1.last() != Some(&k) {
                                        c.1.push(k);
                                    }
                                }
                                None => clusters.push((n, vec![k])),
                            }
                        }
                    }
                }
            }
            clusters.sort_by_key(|c| c.0);
            let cor_id = (self.corridors.len() + 1) as u16;
            // A component touching one cluster at both ends is a loop: one corridor from the cluster to itself.
            let pairs: Vec<(usize, usize)> = if clusters.len() == 1 { vec![(0, 0)] } else { (0..clusters.len()).flat_map(|x| (x + 1..clusters.len()).map(move |y| (x, y))).collect() };
            let mut made = 0u16;
            for (x, y) in pairs {
                // Dijkstra from cluster x's touching trixels to cluster y's, within the component, in texel edges.
                let mut dist: Vec<f64> = vec![f64::MAX; trixels.len()];
                let mut heap = std::collections::BinaryHeap::new();
                for &k in &clusters[x].1 {
                    dist[k] = 0.0;
                    heap.push(std::cmp::Reverse((0u64, k)));
                }
                let mut goal = vec![false; trixels.len()];
                for &k in &clusters[y].1 {
                    goal[k] = true;
                }
                let mut best: Option<f64> = None;
                while let Some(std::cmp::Reverse((dk, k))) = heap.pop() {
                    let d = dk as f64 / 256.0;
                    if d > dist[k] + 1e-9 {
                        continue;
                    }
                    if goal[k] && (x != y || d > 0.0) {
                        best = Some(d);
                        break;
                    }
                    for j in around(trixels[k] as usize) {
                        if let Ok(m) = trixels.binary_search(&(j as u32)) {
                            if in_comp[m] {
                                let nd = d + grid_len(trixels[k] as usize, trixels[m] as usize);
                                if nd < dist[m] {
                                    dist[m] = nd;
                                    heap.push(std::cmp::Reverse(((nd * 256.0) as u64, m)));
                                }
                            }
                        }
                    }
                }
                let Some(inner) = best else { continue };
                // The two steps from the clusters onto the component, a texel edge each or so.
                let len = ((inner + 2.0) * 256.0) as u32;
                let (a, b) = (clusters[x].0, clusters[y].0);
                // The direction, from the one-way angles on the component against the run from a to b.
                let dir = {
                    let (pa, pb) = (centre(a as usize), centre(b as usize));
                    let run = (pb.0 - pa.0, pb.1 - pa.1);
                    let mut dot = 0.0f64;
                    let mut any = false;
                    for &k in &comp {
                        if let Some(angle) = self.direction(trixels[k] as usize, w) {
                            any = true;
                            dot += run.0 * angle.cos() as f64 + run.1 * angle.sin() as f64;
                        }
                    }
                    if !any || a == b { 0 } else if dot >= 0.0 { 1 } else { 2 }
                };
                self.corridors.push(Corridor { a, b, way: w, len, dir });
                made += 1;
            }
            if made == 0 {
                continue;
            }
            // The component's trixels carry its first corridor (a hub's others share the trixels).
            for &k in &comp {
                let i = trixels[k] as usize;
                if self.way[i] == w {
                    self.cor[i] = cor_id;
                } else if let Ok(lo) = self.stack.binary_search(&(i as u32, w)) {
                    self.stackcor[lo] = cor_id;
                }
            }
        }
        let orphans: Vec<usize> = (0..trixels.len()).filter(|&k| !is_node[k] && self.corridor_at(trixels[k] as usize, w) == 0).collect();
        if orphans.is_empty() {
            break;
        }
        for k in orphans {
            is_node[k] = true;
        }
        self.nodes.truncate(nodes_before);
        self.corridors.truncate(corridors_before);
        for &t in trixels {
            let i = t as usize;
            if self.way[i] == w {
                self.cor[i] = 0;
            } else if let Ok(lo) = self.stack.binary_search(&(i as u32, w)) {
                self.stackcor[lo] = 0;
            }
        }
        }
    }

    /// The cell's `route` section.
    pub fn section(&self) -> Option<vsf::VsfSection> {
        use vsf::{types::Tensor, VsfType};
        if self.ways.is_empty() {
            return None;
        }
        let n = self.ways.len();
        let mut s = vsf::VsfSection::new("route");
        let narrow = n < 256;
        let disk = mem_to_disk(&self.way);
        s.add_field("way", if narrow { VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], disk.iter().map(|&w| w as u8).collect())) } else { VsfType::t_u4(Tensor::new(vec![2, TEX * TEX], disk)) });
        s.add_field("osm", VsfType::t_u6(Tensor::new(vec![n], self.ways.iter().map(|w| w.osm as u64).collect())));
        s.add_field("class", VsfType::t_u3(Tensor::new(vec![n], self.ways.iter().map(|w| w.class).collect())));
        s.add_field("uses", VsfType::t_u3(Tensor::new(vec![n], self.ways.iter().map(|w| w.uses).collect())));
        s.add_field("speed", VsfType::t_u3(Tensor::new(vec![n], self.ways.iter().map(|w| w.speed_kmh).collect())));
        s.add_field("oneway", VsfType::t_u3(Tensor::new(vec![n], self.ways.iter().map(|w| (w.oneway + 1) as u8).collect())));
        s.add_field("layer", VsfType::t_u3(Tensor::new(vec![n], self.ways.iter().map(|w| (w.layer as i16 + 128) as u8).collect())));
        s.add_field("grade", VsfType::t_u3(Tensor::new(vec![n], self.ways.iter().map(|w| w.grade).collect())));
        s.add_field("name", VsfType::x(self.ways.iter().map(|w| w.name.replace(['\n', '\r'], " ")).collect::<Vec<_>>().join("\n")));
        // The sparse lists, each in two or three parallel tensors; an empty list is left out.
        if !self.stack.is_empty() {
            s.add_field("stack", VsfType::t_u5(Tensor::new(vec![self.stack.len()], self.stack.iter().map(|s| s.0).collect())));
            s.add_field("stackway", VsfType::t_u4(Tensor::new(vec![self.stack.len()], self.stack.iter().map(|s| s.1).collect())));
        }
        if !self.joins.is_empty() {
            s.add_field("join", VsfType::t_u5(Tensor::new(vec![self.joins.len()], self.joins.clone())));
        }
        if !self.dir.is_empty() {
            s.add_field("dir", VsfType::t_u5(Tensor::new(vec![self.dir.len()], self.dir.iter().map(|d| d.0).collect())));
            s.add_field("dirway", VsfType::t_u4(Tensor::new(vec![self.dir.len()], self.dir.iter().map(|d| d.1).collect())));
            s.add_field("dirangle", VsfType::t_u3(Tensor::new(vec![self.dir.len()], self.dir.iter().map(|d| d.2).collect())));
        }
        // The graph: nodes (trixel, way, cluster), corridors (a, b, way, length, direction), and the corridor per trixel and per stacked way.
        if !self.nodes.is_empty() {
            s.add_field("node", VsfType::t_u5(Tensor::new(vec![self.nodes.len()], self.nodes.iter().map(|n| n.0).collect())));
            s.add_field("nodeway", VsfType::t_u4(Tensor::new(vec![self.nodes.len()], self.nodes.iter().map(|n| n.1).collect())));
            s.add_field("nodename", VsfType::t_u5(Tensor::new(vec![self.nodes.len()], self.nodes.iter().map(|n| n.2).collect())));
        }
        if !self.corridors.is_empty() {
            let m = self.corridors.len();
            s.add_field("cora", VsfType::t_u5(Tensor::new(vec![m], self.corridors.iter().map(|c| c.a).collect())));
            s.add_field("corb", VsfType::t_u5(Tensor::new(vec![m], self.corridors.iter().map(|c| c.b).collect())));
            s.add_field("corway", VsfType::t_u4(Tensor::new(vec![m], self.corridors.iter().map(|c| c.way).collect())));
            s.add_field("corlen", VsfType::t_u5(Tensor::new(vec![m], self.corridors.iter().map(|c| c.len).collect())));
            s.add_field("cordir", VsfType::t_u3(Tensor::new(vec![m], self.corridors.iter().map(|c| c.dir).collect())));
            let disk = mem_to_disk(&self.cor);
            s.add_field("cor", if m < 256 { VsfType::t_u3(Tensor::new(vec![2, TEX * TEX], disk.iter().map(|&c| c as u8).collect())) } else { VsfType::t_u4(Tensor::new(vec![2, TEX * TEX], disk)) });
            if !self.stackcor.is_empty() {
                s.add_field("stackcor", VsfType::t_u4(Tensor::new(vec![self.stackcor.len()], self.stackcor.clone())));
            }
        }
        Some(s)
    }

    /// A `route` section's fields back into a cell's routing.
    pub fn decode(fields: &HashMap<String, Vec<vsf::VsfType>>) -> Option<RouteCell> {
        use vsf::VsfType;
        let first = |name: &str| fields.get(name).and_then(|v| v.first());
        let u8s = |name: &str| -> Vec<u8> {
            match first(name) {
                Some(VsfType::t_u3(t)) => t.data.clone(),
                _ => Vec::new(),
            }
        };
        let u16s = |name: &str| -> Vec<u16> { first(name).and_then(raw_u16).unwrap_or_default() };
        let u32s = |name: &str| -> Vec<u32> {
            match first(name) {
                Some(VsfType::t_u5(t)) => t.data.clone(),
                _ => Vec::new(),
            }
        };
        let osm: Vec<i64> = match first("osm")? {
            VsfType::t_u6(t) => t.data.iter().map(|&v| v as i64).collect(),
            _ => return None,
        };
        let n = osm.len();
        let (class, uses, speed, oneway, layer) = (u8s("class"), u8s("uses"), u8s("speed"), u8s("oneway"), u8s("layer"));
        let names: Vec<String> = match first("name") {
            Some(VsfType::x(s) | VsfType::a(s)) => s.split('\n').map(str::to_string).collect(),
            _ => vec![String::new(); n],
        };
        if [class.len(), uses.len(), speed.len(), oneway.len(), layer.len(), names.len()].iter().any(|&l| l != n) {
            return None;
        }
        // A cell made before grades read every way as graded 0.
        let grade = u8s("grade");
        let ways = (0..n).map(|i| Way { osm: osm[i], class: class[i], uses: uses[i], speed_kmh: speed[i], oneway: oneway[i] as i8 - 1, layer: (layer[i] as i16 - 128) as i8, grade: grade.get(i).copied().unwrap_or(0), name: names[i].clone() }).collect();
        let way = first("way").and_then(raw_u16).filter(|d| d.len() == TRI).map(|d| disk_to_mem(&d)).unwrap_or_default();
        let stack: Vec<(u32, u16)> = u32s("stack").into_iter().zip(u16s("stackway")).collect();
        let dir = u32s("dir").into_iter().zip(u16s("dirway")).zip(u8s("dirangle")).map(|((t, w), a)| (t, w, a)).collect();
        let nodes = u32s("node").into_iter().zip(u16s("nodeway")).zip(u32s("nodename")).map(|((t, w), n)| (t, w, n)).collect();
        let corridors: Vec<Corridor> = u32s("cora").into_iter().zip(u32s("corb")).zip(u16s("corway")).zip(u32s("corlen")).zip(u8s("cordir")).map(|((((a, b), way), len), dir)| Corridor { a, b, way, len, dir }).collect();
        let cor = first("cor").and_then(raw_u16).filter(|d| d.len() == TRI).map(|d| disk_to_mem(&d)).unwrap_or_default();
        let mut stackcor = u16s("stackcor");
        if stackcor.len() != stack.len() {
            stackcor = vec![0; stack.len()];
        }
        Some(RouteCell { way, ways, stack, joins: u32s("join"), dir, nodes, corridors, cor, stackcor })
    }
}

/// Every routable way's centreline stamped a trixel wide at `depth`, with the stacks, the joins (the junctions' trixels) and the one-way directions. Only base-depth cells: routing reads the finest level.
pub fn bake_routes(roads: &[Road], names: &[Named], junctions: &[(f32, f32)], depth: u8) -> HashMap<CellKey, RouteCell> {
    let mut cells: HashMap<CellKey, RouteCell> = HashMap::new();
    // Each cell's way numbers by OSM id.
    let mut numbers: HashMap<CellKey, HashMap<i64, u16>> = HashMap::new();
    // Each cell's one-way trixels already given a direction.
    let mut directed: HashMap<CellKey, std::collections::HashSet<(u32, u16)>> = HashMap::new();
    let locate = |lat: f32, lon: f32| -> (u8, f64, f64) { texel_of(Coord::from_lat_lon(lat as f64, lon as f64), depth) };
    let at = |d: u8, gx: f64, gy: f64| -> (CellKey, usize) {
        let (tx, ty, half) = tri_at(gx, gy);
        (CellKey::from_grid(d, depth, (tx / TEX) as u64, (ty / TEX) as u64), tri_idx(tx % TEX, ty % TEX, half))
    };
    for road in roads.iter().filter(|r| r.class.routable()) {
        let way = Way {
            osm: road.travel.way,
            class: road.class_id(),
            uses: road.uses,
            speed_kmh: road.travel.speed_kmh,
            oneway: road.travel.oneway,
            layer: road.travel.layer,
            grade: road.travel.grade(),
            name: names.get(road.name as usize).map(|n| n.label()).unwrap_or_default(),
        };
        let mut prev: Option<(u8, f64, f64)> = None;
        for &(lat, lon) in &road.pts {
            let (d, gx, gy) = locate(lat, lon);
            if let Some((pd, px, py)) = prev.filter(|p| p.0 == d) {
                let (dx, dy) = (gx - px, gy - py);
                let len = dx.hypot(dy);
                // A quarter trixel a step, so successive trixels always touch.
                let steps = (len * 4.0).ceil().max(1.0) as usize;
                let travel = if road.travel.oneway < 0 { (-dy).atan2(-dx) } else { dy.atan2(dx) };
                let angle = (travel.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU * 256.0).round() as u32 as u8;
                for k in 0..=steps {
                    let t = k as f64 / steps as f64;
                    let (key, i) = at(pd, px + dx * t, py + dy * t);
                    let cell = cells.entry(key).or_default();
                    let nums = numbers.entry(key).or_default();
                    let w = *nums.entry(way.osm).or_insert_with(|| {
                        cell.ways.push(way.clone());
                        cell.ways.len() as u16
                    });
                    cell.add(i, w);
                    if road.travel.oneway != 0 && directed.entry(key).or_default().insert((i as u32, w)) {
                        cell.dir.push((i as u32, w, angle));
                    }
                }
            }
            prev = Some((d, gx, gy));
        }
    }
    for &(lat, lon) in junctions {
        let (d, gx, gy) = locate(lat, lon);
        let (key, i) = at(d, gx, gy);
        if let Some(cell) = cells.get_mut(&key) {
            cell.joins.push(i as u32);
        }
    }
    for cell in cells.values_mut() {
        cell.stack.sort_unstable();
        cell.stack.dedup();
        cell.joins.sort_unstable();
        cell.joins.dedup();
        cell.dir.sort_unstable();
        cell.build_graph();
    }
    cells
}

/// The coarsest depth a way of a class is kept to in the route pyramid (the long middle of a trip stays on the major roads, as the lettering keeps them): motorways, trunks, primaries and ferries to depth 8, secondaries to 9, tertiaries to 10, tracks and paths to 11 (a level deeper than the streets of their rank, for the long walks), residential streets to 12, service roads to 13. Anything else routes only at the base depth.
pub fn class_floor(class: u8) -> u8 {
    const FERRY: u8 = mahere_osm::RoadClass::Ferry as u8 + 1;
    match class {
        1..=3 | FERRY => 8,
        4 => 9,
        5 => 10,
        8 | 9 => 11,
        6 => 12,
        7 => 13,
        _ => 14,
    }
}

/// The parent trixel of a child at the next finer depth, in the parent's grid: the inverse of [`super::tri_children`].
pub fn tri_parent(cx: usize, cy: usize, ch: usize) -> (usize, usize, usize) {
    let half = match (cx & 1, cy & 1) {
        (1, 1) => 1,
        (0, 0) => 0,
        _ => ch,
    };
    (cx >> 1, cy >> 1, half)
}

/// The route pyramid: each parent cell's routing from its four children, the way plane the union of theirs (the stack holding what a trixel gets beyond one), the junctions and one-way directions carried up, the ways kept by [`class_floor`] at each depth, and the graph built again at each. The maze stays a maze: a way's trixels are a connected run at every depth, so a route follows it there as at the base (Nick 2026-10-25: "linear preservation rather than straight averaging").
pub fn pyramid_routes(mut cells: HashMap<CellKey, RouteCell>, base_depth: u8, min_depth: u8) -> HashMap<CellKey, RouteCell> {
    use rayon::prelude::*;
    let mut depth = base_depth;
    while depth > min_depth {
        let mut parents: Vec<CellKey> = cells.keys().filter(|k| k.depth == depth).map(|k| k.parent()).collect();
        parents.sort();
        parents.dedup();
        let pd = depth - 1;
        let built: Vec<(CellKey, RouteCell)> = parents
            .par_iter()
            .filter_map(|&p| {
                let kids: [Option<&RouteCell>; 4] = std::array::from_fn(|q| cells.get(&p.child(q as u64)));
                let mut cell = RouteCell::default();
                // The ways kept at this depth, by OSM id, numbered afresh; each child's numbers mapped.
                let mut nums: Vec<HashMap<u16, u16>> = vec![HashMap::new(); 4];
                for (q, kid) in kids.iter().enumerate() {
                    let Some(kid) = kid else { continue };
                    for (n, w) in kid.ways.iter().enumerate() {
                        if class_floor(w.class) > pd {
                            continue;
                        }
                        let num = match cell.ways.iter().position(|x| x.osm == w.osm) {
                            Some(i) => i as u16 + 1,
                            None => {
                                cell.ways.push(w.clone());
                                cell.ways.len() as u16
                            }
                        };
                        nums[q].insert(n as u16 + 1, num);
                    }
                }
                if cell.ways.is_empty() {
                    return None;
                }
                let mut joins = std::collections::HashSet::new();
                let mut dirs: HashMap<(u32, u16), u8> = HashMap::new();
                for ty in 0..TEX {
                    for tx in 0..TEX {
                        for half in 0..2 {
                            let i = tri_idx(tx, ty, half);
                            for (q, ci) in super::child_cell_texels(tx, ty, half) {
                                let Some(kid) = kids[q as usize] else { continue };
                                for w in kid.ways_at(ci) {
                                    let Some(&num) = nums[q as usize].get(&w) else { continue };
                                    cell.add(i, num);
                                    if kid.is_join(ci) {
                                        joins.insert(i as u32);
                                    }
                                    if let Some(a) = kid.dir.iter().find(|d| d.0 == ci as u32 && d.1 == w).map(|d| d.2) {
                                        dirs.entry((i as u32, num)).or_insert(a);
                                    }
                                }
                            }
                        }
                    }
                }
                // A junction only where two kept ways still meet.
                cell.joins = joins.into_iter().filter(|&j| cell.ways_at(j as usize).count() >= 2).collect();
                cell.joins.sort_unstable();
                cell.dir = dirs.into_iter().map(|((t, w), a)| (t, w, a)).collect();
                cell.dir.sort_unstable();
                cell.stack.sort_unstable();
                cell.stack.dedup();
                cell.build_graph();
                Some((p, cell))
            })
            .collect();
        cells.extend(built);
        depth -= 1;
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahere_osm::{RoadClass, Travel};

    fn road(class: RoadClass, osm: i64, pts: Vec<(f32, f32)>, oneway: i8, layer: i8) -> Road {
        Road { class, pts, weight: 0.5, uses: 0, name: 0, travel: Travel { way: osm, speed_kmh: 0, oneway, layer, ..Default::default() } }
    }

    #[test]
    fn an_overpass_stacks_and_a_junction_joins() {
        // A bridge west to east and a road south to north crossing under it, and a third road meeting the bridge at a node they share.
        let (lat, lon) = (47.60f32, -122.33f32);
        let bridge = road(RoadClass::Primary, 1, vec![(lat, lon - 0.002), (lat, lon + 0.002)], 1, 1);
        let under = road(RoadClass::Residential, 2, vec![(lat - 0.002, lon), (lat + 0.002, lon)], 0, 0);
        let side = road(RoadClass::Residential, 3, vec![(lat - 0.002, lon + 0.001), (lat, lon + 0.001)], 0, 0);
        let cells = bake_routes(&[bridge, under, side], &[Named::default()], &[(lat, lon + 0.001)], 14);
        let stacks: usize = cells.values().map(|c| c.stack.len()).sum();
        let joins: usize = cells.values().map(|c| c.joins.len()).sum();
        assert!(stacks >= 2, "the crossings stack: {stacks}");
        assert_eq!(joins, 1, "one junction, where the side road meets the bridge");
        let (key, c) = cells.iter().find(|(_, c)| !c.joins.is_empty()).unwrap();
        let j = c.joins[0] as usize;
        let at_join: Vec<i64> = c.ways_at(j).map(|w| c.ways[w as usize - 1].osm).collect();
        assert!(at_join.contains(&1) && at_join.contains(&3), "{key:?}: {at_join:?}");
        // The bridge is one-way east: its trixels carry the direction from its west end to its east end in the grid.
        let (_, wx, wy) = texel_of(Coord::from_lat_lon(lat as f64, (lon - 0.002) as f64), 14);
        let (_, ex, ey) = texel_of(Coord::from_lat_lon(lat as f64, (lon + 0.002) as f64), 14);
        let want = ((ey - wy).atan2(ex - wx).rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU * 256.0).round() as i32;
        let east = c.dir.iter().find(|d| c.ways[d.1 as usize - 1].osm == 1).unwrap();
        assert!((east.2 as i32 - want).rem_euclid(256).min((want - east.2 as i32).rem_euclid(256)) <= 1, "{} against {want}", east.2);
        // And the section round-trips, graph included.
        let s = c.section().unwrap();
        let fields: HashMap<String, Vec<vsf::VsfType>> = s.fields.into_iter().map(|f| (f.name, f.values)).collect();
        assert_eq!(RouteCell::decode(&fields).as_ref(), Some(c));
        for c in cells.values() {
            graph_holds(c);
        }
        // The side road: a dead end at its south end, the junction at its north: one corridor between two nodes. The bridge and the road under it each cross the cell (or end in it) with no junction between: one corridor each, unless they leave the cell.
        let side = c.ways.iter().position(|w| w.osm == 3).unwrap() as u16 + 1;
        let side_cors: Vec<&Corridor> = c.corridors.iter().filter(|k| k.way == side).collect();
        assert_eq!(side_cors.len(), 1, "{side_cors:?}");
        assert_ne!(side_cors[0].a, side_cors[0].b);
        assert!(side_cors[0].len > 50 * 256, "about 220 m of side road, {} texel edges", side_cors[0].len / 256);
        // The one-way bridge's corridors run east: from the node nearer the west end.
        let bridge = c.ways.iter().position(|w| w.osm == 1).unwrap() as u16 + 1;
        for k in c.corridors.iter().filter(|k| k.way == bridge) {
            assert_ne!(k.dir, 0, "the bridge is one-way");
            let (pa, pb) = (key.texel_lat_lon(k.a as usize), key.texel_lat_lon(k.b as usize));
            let east = pb.1 >= pa.1;
            assert_eq!(k.dir == 1, east, "{k:?}: a at {pa:?}, b at {pb:?}");
        }
    }

    /// What every built graph must satisfy: each corridor's ends are node names of its way; every trixel of a way is a node of it or in one of its corridors; a node's name is a node trixel of the same way; lengths are at least a texel edge.
    fn graph_holds(c: &RouteCell) {
        for k in &c.corridors {
            for n in [k.a, k.b] {
                assert!(c.nodes.iter().any(|x| x.0 == n && x.1 == k.way && x.2 == n), "corridor {k:?}: {n} is not a node name on its way");
            }
            assert!(k.len >= 256, "{k:?}");
        }
        for (t, w, name) in &c.nodes {
            assert!(c.ways_at(*t as usize).any(|x| x == *w), "node {t} on way {w} has no such way there");
            assert!(c.nodes.iter().any(|x| x.0 == *name && x.1 == *w), "node {t}'s name {name} is not a node");
        }
        for i in 0..TRI {
            for w in c.ways_at(i) {
                let cor = c.corridor_at(i, w);
                let node = c.node_of(i, w).is_some();
                assert!(node || cor != 0, "trixel {i} of way {w} is neither node nor corridor");
                if cor != 0 {
                    assert_eq!(c.corridors[cor as usize - 1].way, w);
                }
            }
        }
    }

    /// The pyramid keeps a way connected a depth up, drops the classes below their floor, and carries junctions and directions; the parent of a child trixel is the one whose children include it.
    #[test]
    fn the_pyramid_keeps_the_maze() {
        for tx in 0..6 {
            for ty in 0..6 {
                for h in 0..2 {
                    for (cx, cy, ch) in super::super::tri_children(tx, ty, h) {
                        assert_eq!(tri_parent(cx, cy, ch), (tx, ty, h), "child {:?}", (cx, cy, ch));
                    }
                }
            }
        }
        let (lat, lon) = (47.60f32, -122.33f32);
        let main = road(RoadClass::Primary, 1, vec![(lat, lon - 0.01), (lat, lon + 0.01)], 1, 0);
        let lane = road(RoadClass::Service, 2, vec![(lat - 0.003, lon), (lat, lon)], 0, 0);
        let cells = bake_routes(&[main, lane], &[Named::default()], &[(lat, lon)], 14);
        let all = pyramid_routes(cells, 14, 11);
        for d in [13u8, 12, 11] {
            let at: Vec<&RouteCell> = all.iter().filter(|(k, _)| k.depth == d).map(|(_, c)| c).collect();
            assert!(!at.is_empty(), "depth {d}");
            let has_lane = at.iter().any(|c| c.ways.iter().any(|w| w.osm == 2));
            assert_eq!(has_lane, d >= class_floor(RoadClass::Service as u8 + 1), "depth {d}: service road kept {has_lane}");
            for c in &at {
                graph_holds(c);
                // The primary is one way: its corridors say so.
                let main = c.ways.iter().position(|w| w.osm == 1).map(|p| p as u16 + 1);
                if let Some(m) = main {
                    assert!(c.corridors.iter().filter(|k| k.way == m).all(|k| k.dir != 0 || k.a == k.b), "depth {d}: a corridor of the one-way primary lost its direction");
                }
            }
            // The primary's trixels at this depth: one connected run per cell.
            for c in &at {
                let Some(m) = c.ways.iter().position(|w| w.osm == 1).map(|p| p as u16 + 1) else { continue };
                let on: Vec<usize> = (0..TRI).filter(|&i| c.ways_at(i).any(|w| w == m)).collect();
                let mut seen = vec![false; on.len()];
                seen[0] = true;
                let mut q = vec![0usize];
                while let Some(k) = q.pop() {
                    for j in around(on[k]) {
                        if let Ok(x) = on.binary_search(&j) {
                            if !seen[x] {
                                seen[x] = true;
                                q.push(x);
                            }
                        }
                    }
                }
                assert!(seen.iter().all(|&s| s), "depth {d}: the primary broke into pieces in a cell");
            }
        }
        // The junction survives to depth 13 (both ways kept) and is gone at 12 with the service road.
        let joins = |d: u8| all.iter().filter(|(k, _)| k.depth == d).map(|(_, c)| c.joins.len()).sum::<usize>();
        assert!(joins(13) >= 1);
        assert_eq!(joins(12), 0);
    }

    /// A road ending inside a cell: a border cluster, a dead-end node at its far end, and one corridor between them — not every trixel a node.
    #[test]
    fn a_dead_end_is_one_corridor() {
        let (lat, lon) = (47.60f32, -122.33f32);
        let r = road(RoadClass::Residential, 7, vec![(lat, lon - 0.003), (lat, lon + 0.003)], 1, 0);
        let cells = bake_routes(&[r], &[Named::default()], &[], 14);
        for (k, c) in &cells {
            graph_holds(c);
            let on = (0..TRI).filter(|&i| c.ways_at(i).next().is_some()).count();
            let clusters: std::collections::HashSet<u32> = c.nodes.iter().map(|n| n.2).collect();
            eprintln!("{}: {on} trixels, {} nodes in {} clusters, {} corridors", k.name(), c.nodes.len(), clusters.len(), c.corridors.len());
            assert!(c.nodes.len() < on, "{}: every trixel a node", k.name());
            assert!(!c.corridors.is_empty(), "{}: no corridor", k.name());
        }
    }

    /// A road across two cells: a node on each side of the edge, a corridor from the cell's edge to the far end in each.
    #[test]
    fn a_road_across_an_edge_has_border_nodes() {
        // Long enough east-west to cross a depth-14 cell (about 430 m).
        let (lat, lon) = (47.60f32, -122.33f32);
        let r = road(RoadClass::Secondary, 7, vec![(lat, lon - 0.01), (lat, lon + 0.01)], 0, 0);
        let cells = bake_routes(&[r], &[Named::default()], &[], 14);
        assert!(cells.len() >= 3, "{} cells", cells.len());
        let mut borders = 0;
        let mut corridors = 0;
        for c in cells.values() {
            graph_holds(c);
            borders += c.nodes.iter().filter(|n| on_border(n.0 as usize)).count().min(2);
            corridors += c.corridors.len();
        }
        // The cells the road crosses whole each have a border cluster at each side and a corridor between; a cell it only clips at a corner may be nodes alone.
        assert!(borders >= 4, "border nodes: {borders}");
        assert!(corridors >= 2, "corridors: {corridors}");
    }
}
