#!/bin/sh
# The one-line installers (Nick 2026-10-17: Photon's, one paste and done) uploaded beside the release builds: install-release.sh for Linux and macOS, install-release.ps1 for Windows with the SHA-256 of the Windows build it installs written in, and the icons they fetch.
# publish-installers.sh [windows exe]: release.sh passes the build it just published; alone it hashes the published mahere-windows-x86_64.exe.
. "$(dirname "$0")/lib-publish.sh"
EXE="$1"
if [ -z "$EXE" ]; then
    EXE=$(mktemp)
    curl -sSfL -H "Cache-Control: no-cache" "$BASE/mahere-windows-x86_64.exe?v=$(date +%s%N)" -o "$EXE"
fi
HASH=$(sha256sum "$EXE" | cut -d' ' -f1 | tr a-f A-F)
PS1=$(mktemp)
sed "s/^\$expectedHash = \"[0-9A-F]*\"/\$expectedHash = \"$HASH\"/" installers/install-release.ps1 > "$PS1"
grep -q "$HASH" "$PS1" || { echo "the Windows hash did not go into install-release.ps1" >&2; exit 1; }
put installers/install-release.sh install-release.sh "text/plain; charset=utf-8" no-cache
put "$PS1" install-release.ps1 "text/plain; charset=utf-8" no-cache
put installers/icon-1024.png icon-1024.png image/png no-cache
put installers/icon-256.png icon-256.png image/png no-cache
put installers/mahere.ico mahere.ico image/x-icon no-cache
rm -f "$PS1"
echo "installers published (Windows build $HASH)"
