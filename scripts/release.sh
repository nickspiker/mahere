#!/bin/sh
# A release (Nick 2026-10-15: Photon's numbering): the version moved to the next minor's x.y.0 (or the one given, release.sh 0.0 for 0.0.0), built fresh for Android, Linux, macOS (Apple Silicon) and Windows, each published to the release channel and as the project page's mahere.apk, mahere-linux-x86_64, mahere-macos-arm64 and mahere-windows-x86_64.exe, with the one-line installers that fetch them (publish-installers.sh); then the tree opens the next dev line at x.y.1. Phones and desktops on any build older take it on their own.
. "$(dirname "$0")/lib-publish.sh"
stamp
NEXT="${1:-$MAJOR.$((MINOR + 1))}.0"
set_version "$NEXT" "Release $NEXT"
stamp
build_android
publish_row release "$APK" Android arm64 "android/mahere-$(version).apk"
put "$APK" mahere.apk application/vnd.android.package-archive no-cache
build_linux
publish_row release "$BIN" Linux x86_64 "linux/mahere-$(version)-x86_64"
put "$BIN" mahere-linux-x86_64 application/octet-stream no-cache
build_macos
publish_row release "$BIN" macOS arm64 "macos/mahere-$(version)-arm64"
put "$BIN" mahere-macos-arm64 application/octet-stream no-cache
build_windows
publish_row release "$BIN" Windows x86_64 "windows/mahere-$(version)-x86_64.exe"
put "$BIN" mahere-windows-x86_64.exe application/vnd.microsoft.portable-executable no-cache
"$(dirname "$0")/publish-installers.sh" "$BIN"
echo "released mahere $MAHERE_VERSION_NAME for Android, Linux, macOS and Windows"
set_version "$MAJOR.$MINOR.1" "Version $MAJOR.$MINOR.1, the dev line after release $MAJOR.$MINOR.0"
