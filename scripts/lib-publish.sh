# Sourced by the publish scripts: the version, the manifest tool, uploads to the bucket, and a manifest row merged in and uploaded last (Nick 2026-10-13: updates after Photon's).
# Versions are Photon's (Nick 2026-10-15): 0.MINOR.PATCH in the workspace's Cargo.toml, a release x.y.0 built fresh by release.sh, a dev build x.y.patch (patch 1 and up); the tree must be clean so a build is the commit its manifest names.
set -e
cd "$(dirname "$0")/.."
if ! git diff --quiet HEAD -- crates android scripts Cargo.toml Cargo.lock; then
    echo "uncommitted changes: commit first, so the build is the commit its manifest names" >&2
    exit 1
fi

# The workspace's version, from Cargo.toml's [workspace.package].
version() {
    sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"/\1/p' Cargo.toml | head -1
}

# set_version <x.y.z> <message>: the workspace's version changed, Cargo.lock with it, committed.
set_version() {
    sed -i '/^\[workspace.package\]/,/^\[/s/^version = ".*"/version = "'"$1"'"/' Cargo.toml
    cargo metadata -q --format-version 1 > /dev/null
    git add Cargo.toml Cargo.lock
    git commit -q -m "$2

Built with Claude Opus 5.5"
}

# stamp: the environment a build carries, from the version and the commit: Android's version code (minor times a thousand plus patch) and the name a person reads.
stamp() {
    V=$(version)
    MAJOR=$(echo "$V" | cut -d. -f1)
    MINOR=$(echo "$V" | cut -d. -f2)
    PATCH=$(echo "$V" | cut -d. -f3)
    export MAHERE_VERSION_CODE=$((MAJOR * 1000000 + MINOR * 1000 + PATCH))
    export MAHERE_VERSION_NAME="$V+$(git rev-parse --short HEAD)"
}

# begin_dev: a dev build never wears a release's patch 0 (right after a release the tree is moved to patch 1 first).
begin_dev() {
    stamp
    if [ "$PATCH" = 0 ]; then
        set_version "$MAJOR.$MINOR.1" "Version $MAJOR.$MINOR.1, the dev line after release $MAJOR.$MINOR.0"
        stamp
    fi
}

# end_dev: the dev build published, the tree moved to the next patch, so a local build never wears a published version.
end_dev() {
    set_version "$MAJOR.$MINOR.$((PATCH + 1))" "Version $MAJOR.$MINOR.$((PATCH + 1)), after dev $MAJOR.$MINOR.$PATCH was published"
}

BASE=https://brobdingnagian.holdmyoscilloscope.com/mahere
BUCKET=holdmyoscilloscope/mahere
. /mnt/Harbor/Code/photon/scripts/lib/wrangler-auth.sh
export WRANGLER_SEND_METRICS=false
cargo build --release -q -p mahere-update --features publish --bin mahere-manifest
MANIFEST=target/release/mahere-manifest

# put <local file> <bucket path> <content type> [cache control]: a fixed name (mahere.apk, the manifests) goes up no-cache, or Cloudflare's edge keeps handing out the old file for hours (it caches .apk by default: build 178 was served after 183 was up, 2026-10-13); a build under its own number never changes and may be cached.
put() {
    # rclone, which retries: wrangler's single fetch failed outright with the uplink full of cell uploads (build 191, 2026-10-13).
    rclone copyto "$1" "r2:$BUCKET/$2" --retries 10 --low-level-retries 20 --s3-no-check-bucket --header-upload "Content-Type: $3" --header-upload "Cache-Control: ${4:-public, max-age=31536000, immutable}"
}

# publish_row <channel> <file> <platform> <arch> <bucket path>: the build uploaded under its own name, then the channel's manifest fetched, this platform's row replaced, signed and uploaded last.
publish_row() {
    set -- "$1" "$2" "$3" "$4" "$5" $($MANIFEST hash "$2")
    put "$2" "$5" application/octet-stream
    tmp=$(mktemp -d)
    curl -sf -H "Cache-Control: no-cache" "$BASE/manifest-$1.vsf?v=$(date +%s%N)" -o "$tmp/old.vsf" || true
    $MANIFEST write "$1" "$tmp/new.vsf" --merge "$tmp/old.vsf" --row "$3" "$4" "$MAHERE_VERSION_CODE" "$MAHERE_VERSION_NAME" "$BASE/$5" "$6" "$7"
    put "$tmp/new.vsf" "manifest-$1.vsf" application/octet-stream no-cache
    rm -rf "$tmp"
}

publish_dev_row() {
    publish_row dev "$@"
}

# build_android: the release-signed APK of the stamped version (keys/mahere-release.jks + .pass, the same key every time, so a published build installs over the previous one); APK names it.
build_android() {
    KEYS=/mnt/Harbor/Code/keys
    [ -r "$KEYS/mahere-release.jks" ] || { echo "no $KEYS/mahere-release.jks" >&2; exit 1; }
    export MAHERE_KEYSTORE="$KEYS/mahere-release.jks"
    export MAHERE_KEYSTORE_PASS="$(tr -d ' \n' < "$KEYS/mahere-release.pass")"
    echo "mahere android $MAHERE_VERSION_NAME"
    (cd android && ./gradlew -q assembleRelease)
    APK=android/app/build/outputs/apk/release/app-release.apk
}

# build_linux: the Linux x86_64 desktop app of the stamped version, signed as Photon's binaries are (checked as it starts and by the updater before it swaps it in); BIN names it.
build_linux() {
    echo "mahere linux $MAHERE_VERSION_NAME"
    cargo build --release -q -p mahere-app
    mkdir -p target/publish
    BIN=target/publish/mahere-linux-x86_64
    cp target/release/mahere-app "$BIN"
    $MANIFEST sign "$BIN"
}
