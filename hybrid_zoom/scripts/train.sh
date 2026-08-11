#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python hybrid_zoom/train.py --config hybrid_zoom/config.yaml "$@"
