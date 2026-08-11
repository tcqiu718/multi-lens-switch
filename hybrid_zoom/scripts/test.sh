#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python hybrid_zoom/test.py --config hybrid_zoom/config.yaml "$@"
