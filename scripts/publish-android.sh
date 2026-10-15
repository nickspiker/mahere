#!/bin/sh
# A dev build for Android: the APK under its version, the dev manifest's Android row, and mahere-dev.apk; the tree then moves to the next patch. A release is release.sh's.
. "$(dirname "$0")/lib-publish.sh"
begin_dev
build_android
publish_dev_row "$APK" Android arm64 "android/mahere-$(version).apk"
put "$APK" mahere-dev.apk application/vnd.android.package-archive no-cache
echo "published mahere $MAHERE_VERSION_NAME for Android, dev channel"
end_dev
