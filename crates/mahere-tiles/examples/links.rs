//! `links <base.vsf> <layer doc>...`: a published base's layer links and each layer document's provenance hash, to see whether they match.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let base = std::fs::read(&args[0]).expect("base");
    for l in [mahere_tiles::publish::Layer::Dem, mahere_tiles::publish::Layer::Img, mahere_tiles::publish::Layer::Nir] {
        println!("base names {}: {:?}", l.dir(), mahere_tiles::publish::link_name(&base, l));
    }
    for f in &args[1..] {
        let d = std::fs::read(f).expect("doc");
        let hp = match mahere_tiles::publish::provenance(&d) { Some(vsf::VsfType::hp(h)) => h.iter().take(16).map(|b| format!("{b:02x}")).collect::<String>(), other => format!("{other:?}") };
        println!("{f}: {hp}");
    }
}
