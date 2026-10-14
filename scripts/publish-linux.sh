#!/bin/sh
# Build the Linux x86_64 desktop app and publish it to the dev channel: the binary under its build number and the dev manifest's Linux row.
. "$(dirname "$0")/lib-publish.sh"
echo "mahere linux $MAHERE_VERSION_NAME"
cargo build --release -q -p mahere-app
# Signed as Photon's binaries are: the author's signature appended, checked by the program as it starts and by the updater before it swaps it in.
mkdir -p target/publish
cp target/release/mahere-app target/publish/mahere-linux-x86_64
$MANIFEST sign target/publish/mahere-linux-x86_64
publish_dev_row target/publish/mahere-linux-x86_64 Linux x86_64 "linux/mahere-$MAHERE_VERSION_CODE-x86_64"
put target/publish/mahere-linux-x86_64 mahere-linux-x86_64 application/octet-stream no-cache
echo "published mahere $MAHERE_VERSION_NAME for Linux x86_64, dev channel, and https://brobdingnagian.holdmyoscilloscope.com/mahere/mahere-linux-x86_64"
