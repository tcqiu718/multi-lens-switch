#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MAX_JOBS="${MAX_JOBS:-8}"

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install --no-build-isolation -v "$SCRIPT_DIR/submodules/diff-gaussian-rasterization-confidence"
python -m pip install --no-build-isolation -v "$SCRIPT_DIR/submodules/simple-knn"

python - <<'PY'
import diff_gaussian_rasterization
import simple_knn._C
print("ZoomGS CUDA extensions imported successfully.")
PY
