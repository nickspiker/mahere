// texel_sizes: the ground edge of a triangle texel at each depth — nominal (the icosahedron edge on the sphere over 256 · 2^depth), and as measured on the sphere at a face's centre and next to one of its corners, where the projection stretches it least and most.
use mahere_coord::uv_to_lat_lon;

fn dist_m(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (la1, lo1, la2, lo2) = (a.0.to_radians(), a.1.to_radians(), b.0.to_radians(), b.1.to_radians());
    let h = ((la2 - la1) / 2.0).sin().powi(2) + la1.cos() * la2.cos() * ((lo2 - lo1) / 2.0).sin().powi(2);
    2.0 * 6_371_000.0 * h.sqrt().asin()
}

fn main() {
    println!("depth  nominal edge    at face centre   near a corner    triangle area (nominal)");
    for depth in 0u32..=20 {
        let n = 256.0 * f64::powi(2.0, depth as i32);
        let step = 1.0 / n;
        // Mean of the three edge directions of a texel (u, v and the diagonal) at a UV point.
        let edge = |u: f64, v: f64| -> f64 {
            let p = uv_to_lat_lon(0, u, v);
            (dist_m(p, uv_to_lat_lon(0, u + step, v)) + dist_m(p, uv_to_lat_lon(0, u, v + step)) + dist_m(uv_to_lat_lon(0, u + step, v), uv_to_lat_lon(0, u, v + step))) / 3.0
        };
        let nominal = 7_054_000.0 / n;
        let centre = edge(1.0 / 3.0, 1.0 / 3.0);
        let corner = edge(1e-3, 1e-3);
        let area = 3f64.sqrt() / 4.0 * nominal * nominal;
        let fmt = |m: f64| if m >= 1000.0 { format!("{:8.2} km", m / 1000.0) } else if m >= 1.0 { format!("{:8.2} m ", m) } else { format!("{:8.1} cm", m * 100.0) };
        let fa = if area >= 1e6 { format!("{:10.2} km²", area / 1e6) } else { format!("{:10.2} m² ", area) };
        println!("{depth:>5}  {}     {}      {}      {}", fmt(nominal), fmt(centre), fmt(corner), fa);
    }
}
