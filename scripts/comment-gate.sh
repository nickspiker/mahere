#!/bin/sh
# Wrapped-comment ratchet (photon's, pointed at this workspace): hard-wrapped comment prose is banned — Rust (// /// //!), Kotlin (//), shell/TOML/gradle (#). One line per thought; a sentence never continues onto the next comment line.
# A wrap = a comment line longer than 60 columns ending mid-clause (a word character) whose next line is the same comment marker continuing in lowercase. Long lines are CORRECT; continuations are the defect.
# Baseline is ZERO. Do not add a baseline mechanism — fix the comment instead (tools/dewrap.rs joins a whole tree if it ever comes to that).
# Runs standalone (`scripts/comment-gate.sh`) and from the cargo test crates/mahere-tiles/tests/comment_gate.rs.
cd "$(dirname "$0")/.." || exit 2

rs_off=$(find crates tools -name "*.rs" -not -path "*/target/*" | sort | while read -r f; do
    awk -v F="$f" '
        FNR == 1 { pm = ""; prev = "" }
        {
            m = ""
            if ($0 ~ /^[[:space:]]*\/\//) {
                t = $0; sub(/^[[:space:]]*/, "", t)
                m = (substr(t, 1, 3) == "///") ? "///" : (substr(t, 1, 3) == "//!") ? "//!" : "//"
            }
            if (pm != "" && m == pm) {
                body = $0; sub(/^[[:space:]]*\/+!?\/* ?/, "", body)
                if (length(prev) > 60 && prev ~ /[A-Za-z]$/ && body ~ /^[a-z]/) print F ":" FNR - 1
            }
            pm = m; prev = $0
        }
    ' "$f"
done)

kt_off=$(find android -type f \( -name "*.kt" -o -name "*.gradle" \) -not -path "*/build/*" -not -path "*/.gradle/*" 2>/dev/null | sort | while read -r f; do
    awk -v F="$f" '
        FNR == 1 { pm = ""; prev = "" }
        {
            m = ($0 ~ /^[[:space:]]*\/\//) ? "//" : ""
            if (pm == "//" && m == "//") {
                body = $0; sub(/^[[:space:]]*\/+ ?/, "", body)
                if (length(prev) > 60 && prev ~ /[A-Za-z]$/ && body ~ /^[a-z]/) print F ":" FNR - 1
            }
            pm = m; prev = $0
        }
    ' "$f"
done)

sh_off=$(find scripts android -maxdepth 2 \( -name "*.sh" -o -name "*.toml" \) 2>/dev/null | sort -u | while read -r f; do
    awk -v F="$f" '
        FNR == 1 { pm = ""; prev = ""; term = "" }
        {
            if (term != "") { tl = $0; sub(/^[[:space:]]*/, "", tl); if (tl == term) term = ""; pm = ""; prev = $0; next }
            if (match($0, /<<-?["'"'"']?[A-Za-z_][A-Za-z0-9_]*/)) { h = substr($0, RSTART, RLENGTH); sub(/^<<-?["'"'"']?/, "", h); term = h }
            m = ""
            if ($0 ~ /^[[:space:]]*#/ && $0 !~ /^[[:space:]]*#!/) m = "#"
            if (pm == "#" && m == "#") {
                body = $0; sub(/^[[:space:]]*#+ ?/, "", body)
                if (length(prev) > 60 && prev ~ /[A-Za-z]$/ && body ~ /^[a-z]/) print F ":" FNR - 1
            }
            pm = m; prev = $0
        }
    ' "$f"
done)

offenders=$(printf '%s\n%s\n%s\n' "$rs_off" "$kt_off" "$sh_off" | grep -v '^$')
if [ -n "$offenders" ]; then
    echo "COMMENT GATE: hard-wrapped comment prose (one line per thought, never wrapped):" >&2
    echo "$offenders" >&2
    echo "COMMENT GATE: blocked — join each sentence onto one line." >&2
    exit 1
fi
exit 0
