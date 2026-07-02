#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

: "${MAX_JOBS:=8}"
export MAX_JOBS

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install --no-build-isolation -v "$SCRIPT_DIR/submodules/diff-gaussian-rasterization-confidence"
python -m pip install --no-build-isolation -v "$SCRIPT_DIR/submodules/simple-knn"

python - <<'PY'
import diff_gaussian_rasterization
import simple_knn._C
print("ZoomGS CUDA extensions imported successfully.")
PY
