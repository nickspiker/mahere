#!/bin/sh
# Publish the baked cell directory to the bucket. The directory layout IS
# the object layout: {layer}/{dd}/{prefix}.vsf.zst, served publicly at
# https://brobdingnagian.holdmyoscilloscope.com/mahere/cells/... which is
# what every client fetches from (mahere_engine::residency::DEFAULT_CELLS_URL).
# rclone's `r2` remote holds the bucket's S3 credentials (keys dir).
set -e
cd "$(dirname "$0")/.."
rclone sync data/cells r2:holdmyoscilloscope/mahere/cells \
    --transfers 32 --checkers 32 --fast-list --stats 30s --stats-one-line
rclone size r2:holdmyoscilloscope/mahere/cells
