//! De-wrap hard-wrapped comment prose: one thought per line (the house rule in every one of Nick's repos; see scripts/comment-gate.sh for the gate).
//!
//! `rustc -O tools/dewrap.rs -o /tmp/dewrap && /tmp/dewrap <files...>` rewrites each file in place, joining a comment line onto the previous one when the previous line ends mid-thought (no terminal punctuation) and the next line is the same comment marker at the same indentation continuing with prose — never a bullet, a heading, a fence, a banner or a blank comment line. Markdown paragraphs get the same treatment outside code fences, lists and tables. Std-only: this tree ships no Python.

use std::fs;

fn main() {
    let mut changed = 0;
    for path in std::env::args().skip(1) {
        let Ok(src) = fs::read_to_string(&path) else { continue };
        let out = if path.ends_with(".md") { dewrap_markdown(&src) } else { dewrap_comments(&src, markers_for(&path)) };
        if out != src {
            fs::write(&path, out).expect("write");
            changed += 1;
            eprintln!("dewrapped {path}");
        }
    }
    eprintln!("{changed} files changed");
}

/// Comment markers for a source language, longest first so `//!` and `///` win over `//`.
fn markers_for(path: &str) -> &'static [&'static str] {
    if path.ends_with(".rs") {
        &["//!", "///", "//"]
    } else if path.ends_with(".kt") || path.ends_with(".gradle") || path.ends_with(".java") {
        &["//"]
    } else {
        &["#"]
    }
}

/// Split a line into (indent, marker, body) if it is a comment line.
fn split<'a>(line: &'a str, markers: &[&'static str]) -> Option<(&'a str, &'static str, &'a str)> {
    let trimmed = line.trim_start();
    let indent = &line[..line.len() - trimmed.len()];
    for m in markers {
        if let Some(rest) = trimmed.strip_prefix(m) {
            if *m == "#" && rest.starts_with('!') {
                return None; // shebang
            }
            return Some((indent, m, rest.strip_prefix(' ').unwrap_or(rest)));
        }
    }
    None
}

/// Does this body end mid-thought — i.e. was the line broken for width, not meaning?
fn ends_open(body: &str) -> bool {
    let b = body.trim_end();
    if b.is_empty() {
        return false;
    }
    let last = b.chars().last().unwrap();
    !matches!(last, '.' | ':' | '!' | '?' | '`' | '|') && !b.ends_with("```")
}

/// Does this body start something that must stay its own line?
fn starts_discrete(body: &str) -> bool {
    let b = body.trim_start();
    b.is_empty()
        || b.starts_with("- ")
        || b.starts_with("* ")
        || b.starts_with("• ")
        || b.starts_with("```")
        || b.starts_with('|')
        || b.starts_with('#')
        || b.starts_with('=')
        || b.starts_with("---")
        || b.starts_with("**Why")
        || b.starts_with("**How")
        || body.starts_with("  ") // indented: code or a nested item
        || b.chars().next().is_some_and(|c| c.is_ascii_digit()) && b.contains(". ") && b.find(". ").unwrap() < 3
}

fn dewrap_comments(src: &str, markers: &[&'static str]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut in_fence = false;
    for line in src.lines() {
        let cur = split(line, markers);
        if let Some((_, _, body)) = cur {
            if body.trim_start().starts_with("```") {
                in_fence = !in_fence;
                out.push(line.to_string());
                continue;
            }
        }
        if !in_fence {
            if let (Some((indent, marker, body)), Some(prev)) = (cur, out.last()) {
                if let Some((pindent, pmarker, pbody)) = split(prev, markers) {
                    if pindent == indent && pmarker == marker && ends_open(pbody) && !starts_discrete(body) {
                        let joined = format!("{} {}", prev.trim_end(), body.trim_start());
                        *out.last_mut().unwrap() = joined;
                        continue;
                    }
                }
            }
        }
        out.push(line.to_string());
    }
    let mut s = out.join("\n");
    if src.ends_with('\n') {
        s.push('\n');
    }
    s
}

/// Markdown: join wrapped paragraph lines and wrapped list items; leave code fences, tables, headings and blank lines alone.
fn dewrap_markdown(src: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut in_fence = false;
    for line in src.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            out.push(line.to_string());
            continue;
        }
        if !in_fence {
            if let Some(prev) = out.last() {
                let p = prev.trim_end();
                let joinable_prev = !p.is_empty()
                    && !p.starts_with('#')
                    && !p.starts_with('|')
                    && !p.starts_with("```")
                    && !p.ends_with("  ")
                    && !p.starts_with("---")
                    && !p.starts_with("<");
                let b = line.trim_start();
                let joinable_cur = !b.is_empty()
                    && !b.starts_with('#')
                    && !b.starts_with('|')
                    && !b.starts_with("- ")
                    && !b.starts_with("* ")
                    && !b.starts_with("---")
                    && !b.starts_with('<')
                    && !line.starts_with("    ")
                    && !(b.chars().next().is_some_and(|c| c.is_ascii_digit()) && b.contains(". ") && b.find(". ").unwrap() < 3);
                if joinable_prev && joinable_cur {
                    *out.last_mut().unwrap() = format!("{} {}", p, b);
                    continue;
                }
            }
        }
        out.push(line.to_string());
    }
    let mut s = out.join("\n");
    if src.ends_with('\n') {
        s.push('\n');
    }
    s
}
