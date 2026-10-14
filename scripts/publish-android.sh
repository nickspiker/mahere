#!/bin/sh
# Build the release-signed APK and publish it to the dev channel: the APK under its build number, the dev manifest's Android row, and the plain mahere.apk the project page links. Promote a build to release with promote-release.sh.
# Signing key: keys/mahere-release.jks + keys/mahere-release.pass (the same key every time, so a published build installs over the previous one). Version: the commit count as versionCode, "0.<count>+<hash>" as versionName, stamped into the Rust too.
. "$(dirname "$0")/lib-publish.sh"
KEYS=/mnt/Harbor/Code/keys
[ -r "$KEYS/mahere-release.jks" ] || { echo "no $KEYS/mahere-release.jks" >&2; exit 1; }
export MAHERE_KEYSTORE="$KEYS/mahere-release.jks"
export MAHERE_KEYSTORE_PASS="$(tr -d ' \n' < "$KEYS/mahere-release.pass")"
echo "mahere android $MAHERE_VERSION_NAME"
(cd android && ./gradlew -q assembleRelease)
APK=android/app/build/outputs/apk/release/app-release.apk
ls -la "$APK"
publish_dev_row "$APK" Android arm64 "android/mahere-$MAHERE_VERSION_CODE.apk"
put "$APK" mahere.apk application/vnd.android.package-archive
echo "published https://brobdingnagian.holdmyoscilloscope.com/mahere/mahere.apk ($MAHERE_VERSION_NAME), dev channel"
