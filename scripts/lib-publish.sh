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

# stamp: the environment a build carries, from the version and the commit: Android's version code and the name a person reads.
# The code is the commit count, as Photon's (Nick 2026-10-22): Android's private "is this newer" number, rising by one a commit, read for nothing else; no positional version in it to overflow or collide. The first numbering's codes ran to 225012 and 0.0.0 went out once at 300000, so a phone holding either uninstalls once.
stamp() {
    V=$(version)
    MAJOR=$(echo "$V" | cut -d. -f1)
    MINOR=$(echo "$V" | cut -d. -f2)
    PATCH=$(echo "$V" | cut -d. -f3)
    export MAHERE_VERSION_CODE=$(git rev-list --count HEAD)
    # The name people read is the version alone, digits only (Nick 2026-10-26: letters in a version); the build number is the manifest's own field.
    export MAHERE_VERSION_NAME="$V"
}

# begin_dev: the version a dev build wears. A dev build never wears a release's patch 0 (right after a release the tree is moved to patch 1 first). One version per tree, whatever the platform: the published commit is tagged dev-<version>, and the patch moves on only when the code has changed since (Nick 2026-10-26: each platform's publish moved the patch, so one commit wore three numbers).
begin_dev() {
    stamp
    if [ "$PATCH" = 0 ]; then
        set_version "$MAJOR.$MINOR.1" "Version $MAJOR.$MINOR.1, the dev line after release $MAJOR.$MINOR.0"
        stamp
    fi
    if git rev-parse -q --verify "refs/tags/dev-$V" > /dev/null; then
        if [ "$(git rev-parse "dev-$V")" != "$(git rev-parse HEAD)" ]; then
            set_version "$MAJOR.$MINOR.$((PATCH + 1))" "Version $MAJOR.$MINOR.$((PATCH + 1)), the code changed since dev $V was published"
            stamp
            git tag "dev-$V"
        fi
    else
        git tag "dev-$V"
    fi
}

# end_dev: nothing since the tag (begin_dev moves the patch at the next publish after a change); kept so the platform scripts read the same.
end_dev() {
    :
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

# build_macos: the Apple Silicon desktop app, cross-built with osxcross (as Photon's is), given the stable Apple code signature (Photon's self-signed certificate, identifier nz.mahere.app) and then the author's Ed25519 signature appended, in that order or the Apple signature would not cover the binary macOS loads; BIN names it.
build_macos() {
    echo "mahere macos arm64 $MAHERE_VERSION_NAME"
    OSX=/mnt/Harbor/Code/osxcross/target/bin
    SDK=$(ls -d /mnt/Harbor/Code/osxcross/target/SDK/MacOSX*.sdk | sort -V | tail -1)
    CC_aarch64_apple_darwin=$OSX/aarch64-apple-darwin-clang-wrapper CXX_aarch64_apple_darwin=$OSX/aarch64-apple-darwin-clang-wrapper CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER=$OSX/aarch64-apple-darwin-clang-wrapper COREAUDIO_SDK_PATH=$SDK OSXCROSS_TRIPLE=aarch64-apple-darwin CMAKE_TOOLCHAIN_FILE_aarch64_apple_darwin=/mnt/Harbor/Code/photon/scripts/lib/osxcross-cmake.toolchain \
        cargo build --release -q -p mahere-app --target aarch64-apple-darwin
    mkdir -p target/publish
    BIN=target/publish/mahere-macos-arm64
    cp target/aarch64-apple-darwin/release/mahere-app "$BIN"
    CERT=/mnt/Harbor/Code/keys/photon-macos-codesign
    if command -v rcodesign >/dev/null && [ -f "$CERT.crt" ]; then
        rcodesign sign --pem-file "$CERT.crt" --pem-file "$CERT.key" --binary-identifier nz.mahere.app "$BIN" >/dev/null
    else
        echo "WARN: rcodesign or the certificate missing: an ad-hoc Apple signature only" >&2
    fi
    $MANIFEST sign "$BIN"
}

# build_windows: the x86_64 Windows desktop app, cross-built with mingw-w64, the author's Ed25519 signature appended (a PE file carries trailing bytes as an overlay); BIN names it.
build_windows() {
    echo "mahere windows x86_64 $MAHERE_VERSION_NAME"
    cargo build --release -q -p mahere-app --target x86_64-pc-windows-gnu
    mkdir -p target/publish
    BIN=target/publish/mahere-windows-x86_64.exe
    cp target/x86_64-pc-windows-gnu/release/mahere-app.exe "$BIN"
    $MANIFEST sign "$BIN"
}
