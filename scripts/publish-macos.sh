#!/bin/sh
# A dev build for macOS (Apple Silicon): the signed binary under its version and the dev manifest's macOS row; the tree then moves to the next patch. A release is release.sh's (Nick 2026-10-25: a dev build for the Mac).
. "$(dirname "$0")/lib-publish.sh"
begin_dev
build_macos
publish_dev_row "$BIN" macOS arm64 "macos/mahere-$(version)-arm64"
put "$BIN" mahere-dev-macos-arm64 application/octet-stream no-cache
echo "published mahere $MAHERE_VERSION_NAME for macOS arm64, dev channel"
end_dev
