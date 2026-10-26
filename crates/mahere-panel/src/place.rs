//! A place typed any way people write one (Nick 2026-10-26: "N23.6, W48.14", which some maps cannot read): decimal degrees, degrees and decimal minutes, or degrees, minutes and seconds; the hemisphere as a letter before or after each number, touching it or not, or as a sign; the symbols ° ′ ″ or ' " or the letters d m s or nothing between the parts; a comma, a slash or a space between the two.

/// A group of numbers read as one coordinate: its parts (degrees, minutes, seconds, each with whether it was written with a minus) and the hemisphere letter found for it.
#[derive(Default, Debug)]
struct Group {
    parts: Vec<(f64, bool)>,
    hemisphere: Option<char>,
}

impl Group {
    fn value(&self) -> Option<f64> {
        if self.parts.is_empty() || self.parts.len() > 3 || self.parts[1..].iter().any(|&(p, neg)| neg || !(0.0..60.0).contains(&p)) {
            return None;
        }
        let v = self.parts.iter().enumerate().map(|(k, (p, _))| p / 60f64.powi(k as i32)).sum::<f64>();
        let south_or_west = matches!(self.hemisphere, Some('s') | Some('w'));
        Some(if self.parts[0].1 != south_or_west { -v } else { v })
    }
}

/// Latitude and longitude from the text, or None when it does not read as a place.
pub fn parse(text: &str) -> Option<(f64, f64)> {
    let t: String = text
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| match c {
            '\u{b0}' => 'd',
            '\u{2032}' | '\'' | '\u{2019}' => 'm',
            '\u{2033}' | '"' | '\u{201d}' => 's',
            _ => c,
        })
        .collect();
    let t = t.replace("deg", "d").replace("min", "m").replace("sec", "s");
    // Tokens: a number (its sign kept), a letter run, or a separator between the two coordinates.
    #[derive(Debug, PartialEq)]
    enum Tok {
        Num(f64, bool),
        Letters(String, bool),
        Sep,
    }
    let mut toks: Vec<Tok> = Vec::new();
    let chars: Vec<char> = t.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let glued = i > 0 && !chars[i - 1].is_whitespace();
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())) || ((c == '-' || c == '+') && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit() || *d == '.')) {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let s: String = chars[start..i].iter().collect();
            let neg = s.starts_with('-');
            toks.push(Tok::Num(s.trim_start_matches(['-', '+']).parse().ok()?, neg));
        } else if c.is_alphabetic() {
            let start = i;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
            toks.push(Tok::Letters(chars[start..i].iter().collect(), glued));
        } else if c == ',' || c == '/' || c == ';' {
            toks.push(Tok::Sep);
            i += 1;
        } else if c.is_whitespace() || c == '-' || c == '+' {
            i += 1;
        } else {
            return None;
        }
    }
    // The groups: a hemisphere letter closes the numbers before it or opens the next coordinate.
    let mut groups: Vec<Group> = Vec::new();
    let mut open: Option<Group> = None;
    let close = |open: &mut Option<Group>, groups: &mut Vec<Group>| {
        if let Some(g) = open.take() {
            groups.push(g);
        }
    };
    for tok in toks {
        match tok {
            Tok::Sep => close(&mut open, &mut groups),
            Tok::Num(v, neg) => {
                let g = open.get_or_insert_with(Group::default);
                if g.parts.len() == 3 {
                    // A fourth number starts the next coordinate.
                    close(&mut open, &mut groups);
                    open = Some(Group::default());
                }
                open.as_mut().unwrap().parts.push((v, neg));
            }
            Tok::Letters(run, glued) => {
                // A hemisphere word whole; else letter by letter, so "43.8sn" is seconds then north.
                let letters: Vec<(String, bool)> = if is_hemisphere(&run) { vec![(run, glued)] } else { run.chars().enumerate().map(|(k, c)| (c.to_string(), glued || k > 0)).collect() };
                for (l, glued) in letters {
                    let hemi = is_hemisphere(&l);
                    // d or m after a number mark it; s glued to a third number is seconds, not south.
                    let marker = |g: &Group| !g.parts.is_empty() && (l == "d" || l == "m" || (l == "s" && glued && g.parts.len() == 3));
                    match open.as_mut() {
                        Some(g) if marker(g) => {}
                        Some(g) if !g.parts.is_empty() && hemi && g.hemisphere.is_none() => {
                            g.hemisphere = Some(l.chars().next().unwrap());
                            close(&mut open, &mut groups);
                        }
                        _ if hemi => {
                            close(&mut open, &mut groups);
                            open = Some(Group { hemisphere: Some(l.chars().next().unwrap()), ..Default::default() });
                        }
                        _ if l == "d" || l == "m" || l == "s" => {}
                        _ => return None,
                    }
                }
            }
        }
    }
    close(&mut open, &mut groups);
    // Numbers alone with no letters or separator between: shared evenly.
    if groups.len() == 1 && groups[0].hemisphere.is_none() {
        let g = groups.remove(0);
        if g.parts.len() % 2 != 0 || g.parts.is_empty() {
            return None;
        }
        let half = g.parts.len() / 2;
        let (a, b) = g.parts.split_at(half);
        groups.push(Group { parts: a.to_vec(), hemisphere: None });
        groups.push(Group { parts: b.to_vec(), hemisphere: None });
    }
    if groups.len() != 2 {
        return None;
    }
    let (a, b) = (&groups[0], &groups[1]);
    // Which is the latitude: the one lettered n or s, else the first.
    let swap = matches!(a.hemisphere, Some('e') | Some('w')) || matches!(b.hemisphere, Some('n') | Some('s'));
    let (lat, lon) = if swap { (b.value()?, a.value()?) } else { (a.value()?, b.value()?) };
    (lat.abs() <= 90.0 && lon.abs() <= 180.0).then_some((lat, lon))
}

fn is_hemisphere(l: &str) -> bool {
    matches!(l, "n" | "s" | "e" | "w" | "north" | "south" | "east" | "west")
}

#[cfg(test)]
mod tests {
    use super::parse;

    fn near(got: Option<(f64, f64)>, lat: f64, lon: f64, text: &str) {
        let (a, b) = got.unwrap_or_else(|| panic!("{text:?} did not read as a place"));
        assert!((a - lat).abs() < 1e-4 && (b - lon).abs() < 1e-4, "{text:?} read as {a}, {b}, not {lat}, {lon}");
    }

    #[test]
    fn every_way_of_writing_a_place() {
        let (lat, lon) = (47.4455, -121.4235);
        for text in [
            "47.4455, -121.4235",
            "47.4455 -121.4235",
            "47.4455/-121.4235",
            "N47.4455 W121.4235",
            "N 47.4455, W 121.4235",
            "47.4455N 121.4235W",
            "47.4455 N, 121.4235 W",
            "W121.4235 N47.4455",
            "121.4235 W 47.4455 N",
            "47\u{b0}26\u{2032}43.8\u{2033} N  121\u{b0}25\u{2032}24.6\u{2033} W",
            "47\u{b0}26'43.8\"N 121\u{b0}25'24.6\"W",
            "N47\u{b0}26'43.8\" W121\u{b0}25'24.6\"",
            "47d26m43.8s n 121d25m24.6s w",
            "47 26 43.8 N 121 25 24.6 W",
            "47 26 43.8, -121 25 24.6",
            "47 26 43.8 -121 25 24.6",
            "47\u{b0}26.73' N 121\u{b0}25.41' W",
            "47 26.73 N 121 25.41 W",
            "-47.4455 S 121.4235 W",
            "47.4455\u{b0} N, 121.4235\u{b0} W",
            "  47.4455 ,  -121.4235  ",
        ] {
            near(parse(text), lat, lon, text);
        }
    }

    #[test]
    fn the_one_google_cannot_read() {
        near(parse("N23.6, W48.14"), 23.6, -48.14, "N23.6, W48.14");
        near(parse("s23.6 e48.14"), -23.6, 48.14, "s23.6 e48.14");
    }

    #[test]
    fn what_is_not_a_place() {
        for text in ["snow lake", "47.4455", "91 0", "0 181", "47 61 0 N 121 0 0 W", "", "12 34 56 78 90"] {
            assert!(parse(text).is_none(), "{text:?} read as a place");
        }
    }
}
