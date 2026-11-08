#!/usr/bin/env sh
# Regenerate requirements.txt (hash-locked, Linux x86_64, Python 3.12) for the Modal image.
set -eu
cd "$(dirname "$0")/.."
UV_NO_CONFIG=1 uv pip compile tools/requirements.in --generate-hashes --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 --exclude-newer "${CUTOFF:-2026-11-01T00:00:00Z}" --no-header -o requirements.txt
