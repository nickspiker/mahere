//! Visual proof: composite dem shading + the line pyramid for a small area
//! at two depths, from the baked cell directory.
use mahere_coord::Coord;
use mahere_tiles::{CellKey, TEX, read_cell_fields, tensor_f32, tensor_u8};
use std::path::Path;

const LUT: [[u8; 3]; 13] = [
    [0, 0, 0],
    [245, 150, 60],
    [238, 175, 62],
    [240, 208, 84],
    [212, 212, 168],
    [182, 192, 182],
    [142, 147, 158],
    [112, 117, 128],
    [152, 120, 88],
    [80, 230, 120],
    [125, 122, 128],
    [148, 136, 160],
    [84, 150, 210],
];

fn save(path: &str, px: &[u32], w: usize, h: usize) {
    let f = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    let mut wr = enc.write_header().unwrap();
    let mut buf = Vec::with_capacity(w * h * 3);
    for &p in px {
        buf.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8]);
    }
    wr.write_image_data(&buf).unwrap();
}

fn main() {
    let out = Path::new("data/cells");
    // Cold Springs / South Climb corner of Mt Adams.
    let center = Coord::from_lat_lon(46.137, -121.556);
    for depth in [10u8, 12, 13] {
        let key = CellKey::containing(center, depth);
        let p = out.join(key.path("line"));
        let Ok(fields) = read_cell_fields(&p) else {
            eprintln!("no line cell at depth {depth}");
            continue;
        };
        let class = tensor_u8(&fields["class"]).unwrap();
        let cov = tensor_u8(&fields["cov"]).unwrap();
        // Optional dem underlay at its own depth if present.
        let demkey = CellKey::containing(center, 12);
        let shade: Option<(Vec<f32>, Vec<f32>, Vec<f32>)> =
            read_cell_fields(&out.join(demkey.path("dem"))).ok().map(|f| {
                (
                    tensor_f32(&f["elev"]).unwrap(),
                    tensor_f32(&f["ge"]).unwrap(),
                    tensor_f32(&f["gn"]).unwrap(),
                )
            });
        let mut px = vec![0x14161Au32; TEX * TEX];
        if depth == 12 {
            if let Some((_e, ge, gn)) = &shade {
                for i in 0..TEX * TEX {
                    let inv = 1.0 / (1.0 + ge[i] * ge[i] + gn[i] * gn[i]).sqrt();
                    let ndl = ((-ge[i] * 0.5 - gn[i] * 0.5) * inv + inv * 0.707).max(0.0);
                    let b = (40.0 + 180.0 * ndl) as u32;
                    px[i] = (b << 16) | (b << 8) | b;
                }
            }
        }
        for i in 0..TEX * TEX {
            if cov[i] > 0 {
                let c = LUT[class[i].min(12) as usize];
                let a = cov[i] as u32;
                let bg = px[i];
                let mix = |b: u32, f: u8| ((b * (255 - a) + f as u32 * a) / 255) & 255;
                px[i] = (mix((bg >> 16) & 255, c[0]) << 16)
                    | (mix((bg >> 8) & 255, c[1]) << 8)
                    | mix(bg & 255, c[2]);
            }
        }
        let path = format!("/tmp/claude-1000/cell-d{depth}.png");
        save(&path, &px, TEX, TEX);
        eprintln!("wrote {path} ({:?})", key);
    }
}
