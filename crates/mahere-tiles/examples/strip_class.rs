// strip_class <cell dir> <class id> <cell names on stdin>: clear every line texel of that class in the named cells, the other sections kept byte for byte. For taking a class out before a bake puts it back without what no longer belongs (the maritime boundaries).
use std::io::BufRead;

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("cell dir"));
    let class: u8 = std::env::args().nth(2).expect("class id").parse().expect("class id");
    let (mut cells, mut texels) = (0usize, 0usize);
    for name in std::io::stdin().lock().lines().map_while(Result::ok) {
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        let mut n = 0;
        let done = mahere_tiles::edit_vectors(&dir.join(name), |p| {
            if let Some(line) = &mut p.line {
                for i in 0..line.class.len() {
                    if line.class[i] == class && line.cov[i] != 0 {
                        line.class[i] = 0;
                        line.cov[i] = 0;
                        if line.mag.len() == line.class.len() {
                            line.mag[i] = 0;
                            line.uses[i] = 0;
                        }
                        n += 1;
                    }
                }
            }
        })
        .expect("edit");
        if done && n > 0 {
            cells += 1;
            texels += n;
        }
    }
    eprintln!("{texels} texels of class {class} cleared in {cells} cells");
}
