//! `worldcode <seven words>` prints the place as latitude and longitude; `worldcode <lat> <lon>` prints its seven words (the Dymaxion coordinate of VSF).
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() == 2 {
        if let (Ok(lat), Ok(lon)) = (a[0].parse::<f64>(), a[1].parse::<f64>()) {
            println!("{}", vsf::WorldCoord::from_lat_lon(lat, lon).to_words());
            return;
        }
    }
    match vsf::WorldCoord::from_words(&a.join(" ")) {
        Some(c) => {
            let (lat, lon) = c.to_lat_lon();
            println!("{lat:.6} {lon:.6}");
        }
        None => eprintln!("not a world code: {}", a.join(" ")),
    }
}
