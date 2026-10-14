#!/bin/sh
# Promote a build already on the dev channel to release: every platform's row of that build number from the dev manifest into the release manifest, the other platforms' release rows kept. Usage: promote-release.sh <build number>
. "$(dirname "$0")/lib-publish.sh"
[ -n "$1" ] || { echo "usage: promote-release.sh <build number>" >&2; exit 2; }
tmp=$(mktemp -d)
curl -sf -H "Cache-Control: no-cache" "$BASE/manifest-dev.vsf?v=$(date +%s%N)" -o "$tmp/dev.vsf"
curl -sf -H "Cache-Control: no-cache" "$BASE/manifest-release.vsf?v=$(date +%s%N)" -o "$tmp/release.vsf" || true
$MANIFEST promote "$tmp/dev.vsf" "$1" "$tmp/new.vsf" --merge "$tmp/release.vsf"
put "$tmp/new.vsf" manifest-release.vsf application/octet-stream
rm -rf "$tmp"
