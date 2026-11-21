//! Land ownership (PAD-US 4.1) and motor vehicle use (USFS MVUM) for one area, as the shapefiles mahere-overlay reads: own.shp, wild.shp and mvum.shp, in latitude and longitude, clipped to the area's box (mahere_osm::padus).
//!   mahere-landuse <lat0,lon0,lat1,lon1> <out dir> <PADUS4_1_State<XX>.gdb>... --mvum <roads.shp> <trails.shp>
//! The box is what `area <file.vsf> bbox` prints.
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: mahere-landuse <lat0,lon0,lat1,lon1> <out dir> <PAD-US .gdb>... --mvum <roads.shp> <trails.shp>";
    let Some(m) = args.iter().position(|a| a == "--mvum") else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    if m < 2 || args.len() != m + 3 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let b: Vec<f64> = args[0].split(',').map(|v| v.parse().expect("the box: four numbers, lat0,lon0,lat1,lon1")).collect();
    let [lat0, lon0, lat1, lon1] = b[..] else { panic!("the box: four numbers, lat0,lon0,lat1,lon1") };
    let gdbs: Vec<&Path> = args[2..m].iter().map(Path::new).collect();
    let mvum: Vec<&Path> = args[m + 1..].iter().map(Path::new).collect();
    if let Err(e) = mahere_tiles::osm::padus::prep((lat0, lon0, lat1, lon1), Path::new(&args[1]), &gdbs, &mvum) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
