#!/bin/sh
# A release (Nick 2026-10-15: Photon's numbering): the version moved to the next minor's x.y.0 (or the one given, release.sh 0.216), built fresh for Android and Linux, each published to the release channel and as the project page's mahere.apk and mahere-linux-x86_64; then the tree opens the next dev line at x.y.1. Phones and desktops on any build older take it on their own.
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
echo "released mahere $MAHERE_VERSION_NAME for Android and Linux"
set_version "$MAJOR.$MINOR.1" "Version $MAJOR.$MINOR.1, the dev line after release $MAJOR.$MINOR.0"
