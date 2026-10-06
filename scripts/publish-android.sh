#!/bin/sh
# Build the release-signed APK and put it on the bucket beside the cells, where the project page links it: https://brobdingnagian.holdmyoscilloscope.com/mahere/mahere.apk
# Signing key: keys/mahere-release.jks + keys/mahere-release.pass (same key every time, so a published build updates over the previous one). Version: commit count as versionCode, "0.<count>+<hash>" as versionName.
set -e
cd "$(dirname "$0")/.."
KEYS=/mnt/Harbor/Code/keys
[ -r "$KEYS/mahere-release.jks" ] || { echo "no $KEYS/mahere-release.jks" >&2; exit 1; }
export MAHERE_KEYSTORE="$KEYS/mahere-release.jks"
export MAHERE_KEYSTORE_PASS="$(tr -d ' \n' < "$KEYS/mahere-release.pass")"
export MAHERE_VERSION_CODE="$(git rev-list --count HEAD)"
export MAHERE_VERSION_NAME="0.$MAHERE_VERSION_CODE+$(git rev-parse --short HEAD)"
echo "mahere android $MAHERE_VERSION_NAME"
(cd android && ./gradlew -q assembleRelease)
APK=android/app/build/outputs/apk/release/app-release.apk
ls -la "$APK"
# Headless wrangler auth from the keys dir, as photon does it.
. /mnt/Harbor/Code/photon/scripts/lib/wrangler-auth.sh
export WRANGLER_SEND_METRICS=false
timeout 600 wrangler r2 object put "holdmyoscilloscope/mahere/mahere.apk" --file "$APK" --content-type application/vnd.android.package-archive --remote
echo "published https://brobdingnagian.holdmyoscilloscope.com/mahere/mahere.apk ($MAHERE_VERSION_NAME)"
