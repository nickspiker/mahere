//! `cog_info <url>`: a remote GeoTIFF's layout as its decoder sees it, read by byte ranges (dimensions, tiles, sample type, compression, predictor, georeferencing keys), without fetching the file.
use tiff::tags::Tag;

fn main() {
    let url = std::env::args().nth(1).expect("url");
    let r = mahere_tiles::remote::RangeReader::open(&url).expect("open");
    println!("{} bytes", r.len());
    let mut dec = tiff::decoder::Decoder::new(r).expect("tiff").with_limits(tiff::decoder::Limits::unlimited());
    println!("dimensions {:?}, chunk {:?}, colour {:?}", dec.dimensions().unwrap(), dec.chunk_dimensions(), dec.colortype().unwrap());
    for (name, tag) in [("compression", Tag::Compression), ("predictor", Tag::Predictor), ("sample format", Tag::SampleFormat), ("bits", Tag::BitsPerSample)] {
        println!("{name}: {:?}", dec.get_tag_u32_vec(tag).ok());
    }
    println!("scale {:?}", dec.get_tag_f64_vec(Tag::ModelPixelScaleTag).ok());
    println!("tie {:?}", dec.get_tag_f64_vec(Tag::ModelTiepointTag).ok());
    println!("geokeys {:?}", dec.get_tag_u32_vec(Tag::GeoKeyDirectoryTag).ok());
    println!("nodata {:?}", dec.get_tag_ascii_string(Tag::Unknown(42113)).ok());
    let t = std::time::Instant::now();
    let chunk = dec.read_chunk(0);
    println!("chunk 0: {:?} in {:.0} ms", chunk.as_ref().map(|_| "ok").map_err(|e| e.to_string()), t.elapsed().as_secs_f64() * 1e3);
}
