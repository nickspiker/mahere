//! `codec_probe <cell files...>`: the cells re-encoded with every section coding itself through the range coder ([`mahere_tiles::ac`]: pyramid planes, per-trixel planes, short tables) and no zstd around the file, against the files as stored; every recoded value is decoded back and checked.
use vsf::VsfType;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (mut stored, mut plain, mut recoded) = (0usize, 0usize, 0usize);
    let mut per: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    let n = a.len() - 1;
    for f in &a[1..] {
        let z = std::fs::read(f).unwrap();
        stored += z.len();
        let data = zstd::decode_all(&z[..]).unwrap();
        plain += data.len();
        let (header, end) = vsf::VsfHeader::decode(&data).unwrap();
        let sections = header.sections(&data, end).unwrap();
        let mut b = vsf::VsfBuilder::new();
        for s in sections {
            let mut out = vsf::VsfSection::new(s.name.clone());
            // The first band of a section is the others' reference (PROBE_NOREF to code each alone).
            let mut reference: Option<Vec<i32>> = None;
            for field in s.fields {
                let v = field.values;
                let new: Vec<VsfType> = if let Some(coef) = mahere_tiles::pyr::from_vsf(&v, mahere_tiles::pyr::NCOEF) {
                    let r = reference.as_deref().filter(|_| std::env::var_os("PROBE_NOREF").is_none());
                    let e = mahere_tiles::ac::encode_coefs_with(&coef, r);
                    assert_eq!(mahere_tiles::ac::decode_coefs_with(&e, r).unwrap(), coef);
                    if reference.is_none() {
                        reference = Some(coef);
                    }
                    vec![VsfType::t_u3(vsf::types::Tensor::new(vec![e.len()], e))]
                } else if let Some(p) = plane(&v) {
                    let e = mahere_tiles::ac::encode_plane(&p);
                    assert_eq!(mahere_tiles::ac::decode_plane(&e).unwrap(), p);
                    vec![VsfType::t_u3(vsf::types::Tensor::new(vec![e.len()], e))]
                } else if let Some(t) = series(&v) {
                    let e = mahere_tiles::ac::encode_series(&t);
                    assert_eq!(mahere_tiles::ac::decode_series(&e, t.len()).unwrap(), t);
                    vec![VsfType::t_u3(vsf::types::Tensor::new(vec![e.len()], e))]
                } else {
                    v
                };
                if std::env::var_os("PROBE_FIELDS").is_some() {
                    let kind = if matches!(new.as_slice(), [VsfType::t_u3(_)]) { "coded" } else { "kept" };
                    let bytes: usize = new.iter().map(|x| match x { VsfType::p(t) => t.data.len(), VsfType::t_u3(t) => t.data.len(), VsfType::t_u4(t) => t.data.len() * 2, _ => 0 }).sum();
                    eprintln!("{}.{} {kind} {bytes} B", s.name, field.name);
                }
                out.add_field_multi(field.name, new);
            }
            b = b.add_section_direct(out);
        }
        let doc = b.build().unwrap();
        recoded += doc.len();
        let (h, _) = vsf::VsfHeader::decode(&doc).unwrap();
        for e in &h.fields {
            per.entry(e.name.clone()).or_default().0 += e.size_bytes;
        }
        let (h, _) = vsf::VsfHeader::decode(&data).unwrap();
        for e in &h.fields {
            per.entry(e.name.clone()).or_default().1 += e.size_bytes;
        }
    }
    let kb = |b: usize| b as f64 / n.max(1) as f64 / 1024.0;
    println!("{n} cells: stored (zstd) {:.1} KB, plain VSF {:.1} KB, recoded without zstd {:.1} KB ({:+.0}%)", kb(stored), kb(plain), kb(recoded), (recoded as f64 / stored as f64 - 1.0) * 100.0);
    for (name, (new, old)) in &per {
        println!("  {name:6} {:7.1} KB recoded   (raw {:7.1} KB)", kb(*new), kb(*old));
    }
}

/// A one-value field holding one number per trixel, or one bit per trixel (a mask).
fn plane(v: &[VsfType]) -> Option<Vec<u32>> {
    if v.len() != 1 {
        return None;
    }
    let tri = mahere_tiles::TRI;
    match &v[0] {
        VsfType::t_u3(t) if t.data.len() == tri => Some(t.data.iter().map(|&x| x as u32).collect()),
        VsfType::t_u4(t) if t.data.len() == tri => Some(t.data.iter().map(|&x| x as u32).collect()),
        VsfType::p(t) if t.bit_depth == 1 && t.shape.iter().product::<usize>() == tri => {
            Some((0..tri).map(|i| ((t.data[i >> 3] >> (7 - (i & 7))) & 1) as u32).collect())
        }
        _ => None,
    }
}

/// A one-value field of 16-bit numbers longer than a few: a table worth coding (the terrain's apron).
fn series(v: &[VsfType]) -> Option<Vec<u32>> {
    match v {
        [VsfType::t_u4(t)] if t.data.len() >= 64 => Some(t.data.iter().map(|&x| x as u32).collect()),
        _ => None,
    }
}
