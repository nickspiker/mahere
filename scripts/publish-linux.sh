#!/bin/sh
# A dev build for Linux x86_64: the binary under its version and the dev manifest's Linux row; the patch moves on at the next publish after a change. A release is release.sh's.
. "$(dirname "$0")/lib-publish.sh"
begin_dev
build_linux
publish_dev_row "$BIN" Linux x86_64 "linux/mahere-$(version)-x86_64"
put "$BIN" mahere-dev-linux-x86_64 application/octet-stream no-cache
echo "published mahere $MAHERE_VERSION_NAME for Linux x86_64, dev channel"
end_dev
