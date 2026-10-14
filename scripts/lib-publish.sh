# Sourced by the publish scripts: the version every build of this commit carries, the manifest tool, uploads to the bucket, and a dev manifest row merged in and uploaded last (Nick 2026-10-13: updates after Photon's).
# A build is stamped with the commit count (MAHERE_VERSION_CODE, the build number every platform shares) and "0.<count>+<hash>"; the tree must be clean so the build is the commit the manifest names.
set -e
cd "$(dirname "$0")/.."
if ! git diff --quiet HEAD -- crates android scripts Cargo.toml Cargo.lock; then
    echo "uncommitted changes: commit first, so the build is the commit its manifest names" >&2
    exit 1
fi
export MAHERE_VERSION_CODE="$(git rev-list --count HEAD)"
export MAHERE_VERSION_NAME="0.$MAHERE_VERSION_CODE+$(git rev-parse --short HEAD)"
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

# publish_dev_row <file> <platform> <arch> <bucket path>: the build uploaded under its own name, then the dev manifest fetched, this platform's row replaced, signed and uploaded last.
publish_dev_row() {
    set -- "$1" "$2" "$3" "$4" $($MANIFEST hash "$1")
    put "$1" "$4" application/octet-stream
    tmp=$(mktemp -d)
    curl -sf -H "Cache-Control: no-cache" "$BASE/manifest-dev.vsf?v=$(date +%s%N)" -o "$tmp/old.vsf" || true
    $MANIFEST write dev "$tmp/new.vsf" --merge "$tmp/old.vsf" --row "$2" "$3" "$MAHERE_VERSION_CODE" "$MAHERE_VERSION_NAME" "$BASE/$4" "$5" "$6"
    put "$tmp/new.vsf" manifest-dev.vsf application/octet-stream no-cache
    rm -rf "$tmp"
}
