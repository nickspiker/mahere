//! A file on a web server read as if it were local, by byte ranges, for the sources too big to fetch whole (Canada's HRDEM: one lidar project a cloud-optimised GeoTIFF of up to 68 GB, of which a piece needs a window): only what a decoder asks for crosses the network, in blocks kept while they are used, and a span known in advance (the tiles under a window) fetched in parallel first.

use std::io::{Read, Seek, SeekFrom};

/// The unit a read fetches and keeps.
const BLOCK: u64 = 1 << 20;
/// Blocks kept: what a window's tiles and the file's tile tables need at once, and a bound on memory (256 MB).
const KEEP: usize = 1 << 8;

pub struct RangeReader {
    agent: ureq::Agent,
    url: String,
    len: u64,
    pos: u64,
    /// (block index, its bytes), the most recently used last.
    blocks: Vec<(u64, Vec<u8>)>,
}

impl RangeReader {
    pub fn open(url: &str) -> Result<RangeReader, String> {
        let agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(120)).build();
        let resp = agent.head(url).call().map_err(|e| format!("{url}: {e}"))?;
        let len: u64 = resp.header("Content-Length").and_then(|v| v.parse().ok()).ok_or_else(|| format!("{url}: no length"))?;
        Ok(RangeReader { agent, url: url.to_string(), len, pos: 0, blocks: Vec::new() })
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    /// Bytes `start..end` of the file, retried a few times (a long bake should not die on one dropped connection).
    fn fetch(&self, start: u64, end: u64) -> std::io::Result<Vec<u8>> {
        let mut last = String::new();
        for _ in 0..5 {
            match self.agent.get(&self.url).set("Range", &format!("bytes={start}-{}", end - 1)).call() {
                Ok(resp) => {
                    let mut buf = Vec::with_capacity((end - start) as usize);
                    match resp.into_reader().read_to_end(&mut buf) {
                        Ok(_) if buf.len() as u64 == end - start => return Ok(buf),
                        Ok(_) => last = format!("{} bytes where {} were asked", buf.len(), end - start),
                        Err(e) => last = e.to_string(),
                    }
                }
                Err(e) => last = e.to_string(),
            }
        }
        Err(std::io::Error::other(format!("{}: bytes {start}..{end}: {last}", self.url)))
    }

    fn keep(&mut self, index: u64, bytes: Vec<u8>) {
        if let Some(i) = self.blocks.iter().position(|b| b.0 == index) {
            self.blocks.remove(i);
        }
        if self.blocks.len() >= KEEP {
            self.blocks.remove(0);
        }
        self.blocks.push((index, bytes));
    }

    fn block(&mut self, index: u64) -> std::io::Result<&[u8]> {
        if let Some(i) = self.blocks.iter().position(|b| b.0 == index) {
            let b = self.blocks.remove(i);
            self.blocks.push(b);
        } else {
            let start = index * BLOCK;
            let bytes = self.fetch(start, (start + BLOCK).min(self.len))?;
            self.keep(index, bytes);
        }
        Ok(&self.blocks.last().unwrap().1)
    }

    /// The blocks covering these byte spans, fetched in parallel (eight at a time) and kept, so the decoder's reads of them never wait on the network one at a time.
    pub fn prefetch(&mut self, spans: &[(u64, u64)]) -> std::io::Result<()> {
        let mut want: Vec<u64> = spans.iter().flat_map(|&(s, e)| s / BLOCK..e.div_ceil(BLOCK)).filter(|i| !self.blocks.iter().any(|b| b.0 == *i)).collect();
        want.sort_unstable();
        want.dedup();
        want.truncate(KEEP);
        for group in want.chunks(8) {
            let me = &*self;
            let got: Vec<std::io::Result<Vec<u8>>> = std::thread::scope(|s| {
                let handles: Vec<_> = group.iter().map(|&i| s.spawn(move || me.fetch(i * BLOCK, ((i + 1) * BLOCK).min(me.len)))).collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for (&i, bytes) in group.iter().zip(got) {
                self.keep(i, bytes?);
            }
        }
        Ok(())
    }
}

impl Read for RangeReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let (index, at) = (self.pos / BLOCK, (self.pos % BLOCK) as usize);
        let block = self.block(index)?;
        let n = (block.len() - at).min(buf.len());
        buf[..n].copy_from_slice(&block[at..at + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for RangeReader {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let pos = match to {
            SeekFrom::Start(p) => p as i128,
            SeekFrom::End(d) => self.len as i128 + d as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
        };
        if pos < 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek before the start"));
        }
        self.pos = pos as u64;
        Ok(self.pos)
    }
}

/// A [`RangeReader`] that clones cheaply, every clone the same file, position and kept blocks: one handle goes to a decoder that keeps its reader, another stays with the caller to [`RangeReader::prefetch`] what the decoder will read next.
#[derive(Clone)]
pub struct SharedRange(std::sync::Arc<std::sync::Mutex<RangeReader>>);

impl SharedRange {
    pub fn open(url: &str) -> Result<SharedRange, String> {
        Ok(SharedRange(std::sync::Arc::new(std::sync::Mutex::new(RangeReader::open(url)?))))
    }

    pub fn len(&self) -> u64 {
        self.0.lock().unwrap().len()
    }

    pub fn prefetch(&self, spans: &[(u64, u64)]) -> std::io::Result<()> {
        self.0.lock().unwrap().prefetch(spans)
    }
}

impl Read for SharedRange {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().read(buf)
    }
}

impl Seek for SharedRange {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        self.0.lock().unwrap().seek(to)
    }
}
