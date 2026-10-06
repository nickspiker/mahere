#!/bin/sh
# Publish the baked cell directory to the bucket. The directory layout IS the object layout: {layer}/{cell}.vsf.zst (the cell named by its flattened VSF value, base64url), served publicly at https://brobdingnagian.holdmyoscilloscope.com/mahere/cells/... which is what every client fetches from (mahere_engine::residency::DEFAULT_CELLS_URL).
# rclone's `r2` remote holds the bucket's S3 credentials (keys dir).
set -e
cd "$(dirname "$0")/.."
# Cache-Control is short on purpose: the custom domain fronts the bucket with Cloudflare's edge cache (4 h by default), and a cell overwritten under the same key would otherwise serve stale for hours. --ignore-times re-uploads everything so the header lands on every object, not just changed ones.
rclone sync data/cells r2:holdmyoscilloscope/mahere/cells \
    --header-upload "Cache-Control: max-age=300" --ignore-times \
    --transfers 32 --checkers 32 --fast-list --stats 30s --stats-one-line
rclone size r2:holdmyoscilloscope/mahere/cells
