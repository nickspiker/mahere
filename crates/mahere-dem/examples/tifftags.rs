// tifftags <tif>: the layout tags a raw reader needs — compression, predictor, sample size, tiling.
use tiff::decoder::Decoder;
use tiff::tags::Tag;

fn main() {
    let path = std::env::args().nth(1).expect("path");
    let mut d = Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap())).unwrap();
    for (name, tag) in [("Compression", Tag::Compression), ("Predictor", Tag::Predictor), ("BitsPerSample", Tag::BitsPerSample), ("SampleFormat", Tag::SampleFormat), ("TileWidth", Tag::TileWidth), ("TileLength", Tag::TileLength), ("ImageWidth", Tag::ImageWidth), ("RowsPerStrip", Tag::RowsPerStrip), ("PlanarConfiguration", Tag::PlanarConfiguration)] {
        println!("{name}: {:?}", d.get_tag_u32_vec(tag).ok());
    }
    println!("TileOffsets: {:?}", d.get_tag_u64_vec(Tag::TileOffsets).map(|v| v.len()).ok());
}
