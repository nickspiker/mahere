//! `field_check <cell>`: every field of a baked cell coded and decoded back, compared value for value.
fn main() {
    let z = std::fs::read(std::env::args().nth(1).unwrap()).unwrap();
    let data = zstd::decode_all(&z[..]).unwrap();
    let (h, end) = vsf::VsfHeader::decode(&data).unwrap();
    for s in h.sections(&data, end).unwrap() {
        for f in s.fields {
            let Some(c) = mahere_tiles::ac::code_field(&f.values, None) else {
                println!("{}.{} kept", s.name, f.name);
                continue;
            };
            let ok = if let Some(u) = mahere_tiles::ac::uncode_field(&c) {
                format!("{:?}", u) == format!("{:?}", f.values[0])
            } else {
                mahere_tiles::ac::coefs(&c, None) == mahere_tiles::pyr::from_vsf(&f.values, mahere_tiles::pyr::NCOEF)
            };
            println!("{}.{} {} {:?}", s.name, f.name, if ok { "ok" } else { "DIFFERS" }, f.values.iter().map(|v| format!("{:.40}", format!("{v:?}"))).collect::<Vec<_>>());
        }
    }
}
