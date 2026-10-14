#!/bin/sh
# Build the Linux x86_64 desktop app and publish it to the dev channel: the binary under its build number and the dev manifest's Linux row.
. "$(dirname "$0")/lib-publish.sh"
echo "mahere linux $MAHERE_VERSION_NAME"
cargo build --release -q -p mahere-app
publish_dev_row target/release/mahere-app Linux x86_64 "linux/mahere-$MAHERE_VERSION_CODE-x86_64"
echo "published mahere $MAHERE_VERSION_NAME for Linux x86_64, dev channel"
