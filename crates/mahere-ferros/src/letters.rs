//! The map's lettering on the GPU (2026-11-13, the owner on the 8a: "nor does it load labels"): the labeller's layout (mahere-panel labels.rs, the CPU path's own: the same names, places, priorities and collisions) drawn by the map kernel in the same frame as the map (`kernel::letters`), under the panel and the marks, the halo under the letters and the map under both, front to back.
//!
//! Each frame the lettering's letters are quads at their anchors' places on the screen now (`Labeller::glyphs`): a point's name level, a line's or an area's turned with the map and half a turn more where it would read upside down.
//! The kernel finds a pixel's letters by its tile (2^`TILE_SHIFT` pixels a side): each tile a word, where its list of letters starts and how many; each letter a record of `GLYPH_WORDS` words; then it samples each letter's glyph from the atlas by its quad's turn and adds it under what it has, the letters first and their halo under them.
//!
//! The atlas is the labeller's glyphs at the screen's own scale: the labeller rasterises each glyph at `ATLAS_DENSITY` (3) atlas texels a pixel, and the CPU path averages each pixel's 3 x 3 texels as it draws; here each glyph is averaged once, as it first appears, into a texel a pixel, two bytes a texel (the letter's coverage and its halo's), two texels a word.
//! A glyph at rest is placed on whole pixels, so every pixel's sample is a texel's centre and the letter is the CPU's average exactly (crisp); a turned one (a line's letters by the rest of their angle's bin, any turning label while the map turns) is sampled between texels.
//! The atlas lives in each frame region (the region the next frame is planned into is never the one the GPU is reading), written there again only when it changed; it starts again when the labeller's does, and when it is full.

use crate::arena::Layout;
use crate::kernel::{self, Out};
use mahere_panel::labels::{Atlas, GlyphQuad};

/// The labeller's glyphs as this side reads them: its atlas's side and texels (two bytes each, the letter's coverage and its halo's), and the generation they are of (`Atlas::generation`) and their version (`Atlas::version`).
#[derive(Clone, Copy)]
pub struct Source<'a> {
    pub side: usize,
    pub rg: &'a [u8],
    pub generation: u64,
    pub version: u64,
}

impl<'a> From<&'a Atlas> for Source<'a> {
    fn from(a: &'a Atlas) -> Source<'a> {
        Source { side: a.side, rg: &a.rg, generation: a.generation, version: a.version }
    }
}

/// A tile is 2^4 pixels a side: on the 8a's 1080 x 2400 at the device's text size (37 px), 5402 of 10200 tiles hold any letter and the most in one is 8 (a Seattle view, 2026-11-13).
pub const TILE_SHIFT: u32 = 4;
/// The most letters a tile's list may hold (the kernel's loop runs at most this many times); more, and the rest are left out of that tile.
pub const TILE_MOST: u32 = 1 << 5;
/// A letter's record, words (`kernel::letters` reads them by these offsets): its centre on the screen (f32 x, y), its glyph's centre in the atlas (f32 texels x, y), the cosine and sine of its turn (an f16 pair), its reach from the centre along its own axes (an f16 pair, pixels), its colour in the kernel's output (an f16 pair, red and green; then blue and its opacity).
pub const GLYPH_SHIFT: u32 = 3;
pub const GLYPH_WORDS: usize = 1 << GLYPH_SHIFT;
/// The atlas's row: 2^9 texels, two to a word.
pub const ATLAS_STRIDE: usize = 1 << 9;
pub const ATLAS_ROW: usize = ATLAS_STRIDE * 2;

/// The lettering's GPU side: the atlas as the kernel reads it (the host's copy, written into each frame region when it changed), where each of the labeller's glyphs is in it, and the frame's tiles.
pub struct Letters {
    /// Two texels a word, `ATLAS_STRIDE` texels a row, as many rows as a region holds (`Layout::atlas_rows`).
    atlas: Vec<u32>,
    rows: usize,
    /// The shelf being filled: its left edge free from, its top, its height; and the rows below it free from.
    shelf: [usize; 3],
    /// Each of the labeller's glyphs (its texels' corner in the labeller's atlas, x | y << 16) and its place here (x, y, width, height in texels, its borders in), by key.
    places: Vec<(u32, [u16; 4])>,
    /// The labeller's atlas generation the places are for.
    generation: Option<u64>,
    /// The labeller's atlas version whose glyphs this atlas started again for and still could not hold.
    crowded: Option<u64>,
    /// Bumped whenever the atlas changed, and by frame region the version written there.
    version: u64,
    written: [u64; 2],
    /// The frame's tile counts, reused.
    counts: Vec<u32>,
    pub stats: Stats,
}

/// What the lettering cost, since the last report.
#[derive(Default, Clone, Copy)]
pub struct Stats {
    /// Frames lettered, the letters drawn in them, those left out (no room in the atlas, the region or a full tile), tile list entries.
    pub frames: u64,
    pub letters: u64,
    pub left_out: u64,
    pub entries: u64,
    /// Glyphs averaged into the atlas, the times it started again, its rows in use now; the host's µs a frame (the quads, the atlas, the tiles), and the atlas's copies into a region with their µs.
    pub glyphs: u64,
    pub resets: u64,
    pub rows_used: usize,
    pub us: u64,
    pub copies: u64,
    pub copy_us: u64,
}

/// Where the lettering is in a frame region, as the kernel takes it.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub consts: kernel::Letters,
    pub letters: usize,
}

impl Letters {
    pub fn new(lay: &Layout) -> Letters {
        Letters { atlas: vec![0; lay.atlas_rows * ATLAS_STRIDE / 2], rows: lay.atlas_rows, shelf: [1, 1, 0], places: Vec::new(), generation: None, crowded: None, version: 1, written: [0; 2], counts: Vec::new(), stats: Stats::default() }
    }

    /// The atlas empty: every glyph averaged into it again as a frame next asks for it.
    fn reset(&mut self) {
        self.atlas.iter_mut().for_each(|w| *w = 0);
        // The first row and column stay empty: a sample at a glyph's very edge may round a texel before its border, which is then still the atlas's own.
        self.shelf = [1, 1, 0];
        self.places.clear();
        self.version += 1;
        self.stats.resets += 1;
    }

    /// A glyph's place in the atlas, averaged into it if it is not there yet: `r` its texels in the labeller's atlas (x, y, width, height); None when the atlas has no room.
    fn place(&mut self, src: Source, r: [usize; 4]) -> Option<[u16; 4]> {
        let key = (r[0] | r[1] << 16) as u32;
        let at = match self.places.binary_search_by_key(&key, |p| p.0) {
            Ok(i) => return Some(self.places[i].1),
            Err(i) => i,
        };
        // A texel a pixel, with a border: one empty texel before it and two after, so a sample anywhere within the glyph's reach (`reach`) reads only its own texels and empty ones.
        let (cw, ch) = (r[2].div_ceil(3), r[3].div_ceil(3));
        let (w, h) = (cw + 3, ch + 3);
        let [mut x, mut y, mut sh] = self.shelf;
        if x + w > ATLAS_STRIDE {
            (x, y, sh) = (1, y + sh, 0);
        }
        // WHY: the glyph's size is the labeller's (its text size, the user's pinch), and a glyph placed past the atlas's last row would be written over the next region's tables (or past the block): no room, and the caller starts the atlas again or leaves the letter out.
        if y + h > self.rows || w > ATLAS_STRIDE - 1 {
            return None;
        }
        self.shelf = [x + w, y, sh.max(h)];
        self.stats.rows_used = self.stats.rows_used.max(y + sh.max(h));
        // Each texel the mean of the labeller's 3 x 3 under it, as the CPU path's draw averages a pixel's (labels.rs `draw_cpu`), the glyph's edge where a 3 x 3 runs past it counting nothing.
        let side = src.side;
        for j in 0..ch {
            for i in 0..cw {
                let (mut c, mut hl) = (0u32, 0u32);
                for b in 0..3 {
                    let sy = 3 * j + b;
                    if sy >= r[3] {
                        break;
                    }
                    for a in 0..3 {
                        let sx = 3 * i + a;
                        if sx >= r[2] {
                            break;
                        }
                        let o = ((r[1] + sy) * side + r[0] + sx) * 2;
                        c += src.rg[o] as u32;
                        hl += src.rg[o + 1] as u32;
                    }
                }
                let texel = (c + 4) / 9 | ((hl + 4) / 9) << 8;
                let (tx, ty) = (x + 1 + i, y + 1 + j);
                let wd = &mut self.atlas[ty * (ATLAS_STRIDE / 2) + (tx >> 1)];
                let s = (tx & 1) << 4;
                *wd = (*wd & !(0xFFFF << s)) | texel << s;
            }
        }
        let p = [x as u16, y as u16, w as u16, h as u16];
        self.places.insert(at, (key, p));
        self.version += 1;
        self.stats.glyphs += 1;
        Some(p)
    }

    /// The frame's lettering into frame region `region`: the quads (`Labeller::glyphs`, the labeller's `src` atlas), each glyph's place in the atlas, the atlas copied into the region when it changed there, and the tiles, records and lists. None when nothing is lettered: the kernel then has no lettering in it.
    pub fn frame(&mut self, src: Source, quads: &[GlyphQuad], lay: &Layout, region: usize, arena: &mut [u32]) -> Option<Frame> {
        let t = std::time::Instant::now();
        if quads.is_empty() {
            return None;
        }
        if self.generation != Some(src.generation) {
            self.generation = Some(src.generation);
            self.reset();
        }
        let side = src.side as f32;
        let rect = |g: &GlyphQuad| [(g[8] * side).round() as usize, (g[9] * side).round() as usize, ((g[10] - g[8]) * side).round() as usize, ((g[11] - g[9]) * side).round() as usize];
        // Each quad's place; a full atlas starts again with this frame's glyphs alone, and what still has no room is left out. Once only for a lettering whose glyphs do not fit even then (`crowded`): every frame would start it again.
        let mut places: Vec<Option<[u16; 4]>> = quads.iter().map(|g| self.place(src, rect(g))).collect();
        if places.iter().any(Option::is_none) && self.crowded != Some(src.version) {
            self.crowded = Some(src.version);
            self.reset();
            places = quads.iter().map(|g| self.place(src, rect(g))).collect();
        }
        let (w, h) = (lay.w, lay.h);
        let (cols, rows) = (w.div_ceil(1 << TILE_SHIFT), h.div_ceil(1 << TILE_SHIFT));
        let tiles = cols * rows;
        // The records after the tiles, on a record's boundary; the lists after the records. All of it within the region's lettering (`kernel::LETTERS` up to the overlay).
        let base = (lay.regions[region] + kernel::LETTERS) / 4;
        let room = (kernel::OVERLAY - kernel::LETTERS) / 4;
        let glyphs_at = tiles.next_multiple_of(GLYPH_WORDS);
        let out = lay.out;
        // A colour as the kernel adds it: the theme's (VSF RGB at gamma 2, 0..1) as the surface's working value (gamma 2, white at 2^14), or as fluor's g2t code (256 · tanh of it).
        let value = |c: f32| match out {
            Out::Surface => c * (1 << 14) as f32,
            Out::Argb8 => 256. * c.tanh(),
        };
        let mut records: Vec<[u32; GLYPH_WORDS]> = Vec::with_capacity(quads.len());
        let mut boxes: Vec<[usize; 4]> = Vec::with_capacity(quads.len());
        let mut halo = [0f32; 4];
        self.counts.clear();
        self.counts.resize(tiles, 0);
        let mut entries = 0usize;
        let mut left_out = 0u64;
        for (g, p) in quads.iter().zip(&places) {
            let Some(p) = p else {
                left_out += 1;
                continue;
            };
            let r = rect(g);
            // The glyph's half extents in pixels (a texel a pixel), and its centre in the atlas past the border.
            let (hx, hy) = (r[2] as f32 / 6., r[3] as f32 / 6.);
            let (ax, ay) = (p[0] as f32 + 1. + hx, p[1] as f32 + 1. + hy);
            let (mut cx, mut cy, mut turn) = (g[0], g[1], g[4]);
            // At rest (no turn left over from its bin) the glyph's texels on whole pixels: every pixel's sample is a texel's centre, the CPU's own average there.
            if turn.abs() < 1e-3 {
                cx = (cx - hx).round() + hx;
                cy = (cy - hy).round() + hy;
                turn = 0.;
            }
            let (reach_x, reach_y) = (hx + 0.5, hy + 0.5);
            // The tiles its turned quad can reach: the box about it turned.
            let (s, c) = turn.sin_cos();
            let (ex, ey) = (c.abs() * reach_x + s.abs() * reach_y, s.abs() * reach_x + c.abs() * reach_y);
            let (x0, x1, y0, y1) = (cx - ex, cx + ex, cy - ey, cy + ey);
            // WHY: the labeller hands over the letters of every label whose anchor is within a screen of this one (`Lettering::glyphs`), and a letter wholly off the screen has no tile.
            if x1 < 0. || y1 < 0. || x0 >= w as f32 || y0 >= h as f32 {
                continue;
            }
            // WHY the min: a letter across the screen's right or bottom edge reaches past the last tile, which has no tile after it (one across the left or top edge casts to 0: a float's cast to usize stops at 0).
            let tile = |v: f32, n: usize| ((v as usize) >> TILE_SHIFT).min(n - 1);
            let b = [tile(x0, cols), tile(x1, cols), tile(y0, rows), tile(y1, rows)];
            let n = (b[1] - b[0] + 1) * (b[3] - b[2] + 1);
            // WHY: the records and lists end where the overlay starts (`kernel::OVERLAY`); a lettering with more letters than fit (tens of thousands, at a tiny text size) leaves the last ones (the least important, the layout's order) out rather than writing over it.
            if glyphs_at + (records.len() + 1) * GLYPH_WORDS + entries + n > room {
                left_out += 1;
                continue;
            }
            entries += n;
            for ty in b[2]..=b[3] {
                for tx in b[0]..=b[1] {
                    self.counts[ty * cols + tx] += 1;
                }
            }
            if g[19] > 0. {
                halo = [value(g[16]), value(g[17]), value(g[18]), g[19]];
            }
            records.push([cx.to_bits(), cy.to_bits(), ax.to_bits(), ay.to_bits(), h2(c, s), h2(reach_x, reach_y), h2(value(g[12]), value(g[13])), h2(value(g[14]), g[15])]);
            boxes.push(b);
        }
        if records.is_empty() {
            return None;
        }
        let list_at = glyphs_at + records.len() * GLYPH_WORDS;
        let lettering = &mut arena[base..base + room];
        // Each tile's list where the one before it ends: its word the list's start << 8 | its count, at most `TILE_MOST`.
        let mut start = 0usize;
        for (k, n) in self.counts.iter_mut().enumerate() {
            // WHY the min: the kernel's loop runs `TILE_MOST` times at most and the count has a byte; past it a tile's letters are left out there.
            let shown = (*n).min(TILE_MOST);
            left_out += (*n - shown) as u64;
            lettering[k] = (start as u32) << 8 | shown;
            // From here the count is where the tile's next entry goes.
            *n = start as u32;
            start += shown as usize;
        }
        for (j, (rec, b)) in records.iter().zip(&boxes).enumerate() {
            lettering[glyphs_at + j * GLYPH_WORDS..glyphs_at + (j + 1) * GLYPH_WORDS].copy_from_slice(rec);
            for ty in b[2]..=b[3] {
                for tx in b[0]..=b[1] {
                    let k = ty * cols + tx;
                    // A tile past its most takes no more: its word holds the most, and the entries after it are its neighbour's.
                    let next = self.counts[k] as usize;
                    if next < (lettering[k] >> 8) as usize + (lettering[k] & 255) as usize {
                        lettering[list_at + next] = j as u32;
                        self.counts[k] += 1;
                    }
                }
            }
        }
        // The atlas into this region when it changed since it was last written there.
        if self.written[region] != self.version {
            let tc = std::time::Instant::now();
            let at = (lay.regions[region] + lay.atlas_at) / 4;
            let used = self.stats.rows_used * ATLAS_STRIDE / 2;
            arena[at..at + used].copy_from_slice(&self.atlas[..used]);
            self.written[region] = self.version;
            self.stats.copies += 1;
            self.stats.copy_us += tc.elapsed().as_micros() as u64;
        }
        let s = &mut self.stats;
        s.frames += 1;
        s.letters += records.len() as u64;
        s.left_out += left_out;
        s.entries += start as u64;
        s.us += t.elapsed().as_micros() as u64;
        let consts = kernel::Letters { tiles_at: base as u32, cols: cols as u32, glyphs_at: (base + glyphs_at) as u32, list_at: (base + list_at) as u32, atlas_at: ((lay.regions[region] + lay.atlas_at) / 4) as u32, halo };
        Some(Frame { consts, letters: records.len() })
    }

    /// The atlas's bytes on the host, and the most a region holds.
    pub fn atlas_bytes(&self) -> (usize, usize) {
        (self.stats.rows_used * ATLAS_ROW, self.rows * ATLAS_ROW)
    }

    /// The stats since the last call, the atlas's use kept.
    pub fn take_stats(&mut self) -> Stats {
        let rows = self.stats.rows_used;
        let s = std::mem::take(&mut self.stats);
        self.stats.rows_used = rows;
        s
    }
}

/// Two f16 in a word, `lo` in its low half: a v2f16 the kernel reads as it is.
fn h2(lo: f32, hi: f32) -> u32 {
    valhall::f32_to_f16(lo) as u32 | (valhall::f32_to_f16(hi) as u32) << 16
}
