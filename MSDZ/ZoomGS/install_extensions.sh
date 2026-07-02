#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

: "${MAX_JOBS:=8}"
export MAX_JOBS

python - <<'PY'
import os
import re
import shutil
import subprocess
import sys

try:
    import torch
except ImportError as exc:
    raise SystemExit("ERROR: PyTorch is not installed in the active Python environment.") from exc

torch_cuda = torch.version.cuda
nvcc = shutil.which("nvcc")

print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA: {torch_cuda}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or '(not set)'}")

if torch_cuda is None:
    raise SystemExit("ERROR: The active PyTorch install is CPU-only; ZoomGS extensions require a CUDA build of PyTorch.")

if nvcc is None:
    raise SystemExit(
        "ERROR: nvcc was not found. Install/load a CUDA Toolkit that matches PyTorch, "
        "then make sure CUDA_HOME and PATH point to it."
    )

try:
    nvcc_output = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT, text=True)
except subprocess.CalledProcessError as exc:
    raise SystemExit(f"ERROR: Failed to execute nvcc at {nvcc}:\n{exc.output}") from exc

match = re.search(r"release\s+(\d+\.\d+)", nvcc_output)
nvcc_cuda = match.group(1) if match else None

print(f"nvcc: {nvcc}")
print(f"nvcc CUDA: {nvcc_cuda or 'unknown'}")

if nvcc_cuda is not None:
    torch_major = torch_cuda.split(".")[0]
    nvcc_major = nvcc_cuda.split(".")[0]
    if torch_major != nvcc_major:
        raise SystemExit(
            "ERROR: CUDA Toolkit and PyTorch CUDA versions are incompatible: "
            f"nvcc reports CUDA {nvcc_cuda}, but PyTorch was built with CUDA {torch_cuda}. "
            "Load/install a CUDA 12.x Toolkit, preferably CUDA 12.4 for cu124 PyTorch, "
            "and update CUDA_HOME/PATH before building."
        )
    if nvcc_cuda != torch_cuda:
        print(
            "WARNING: nvcc CUDA and PyTorch CUDA minor versions differ. "
            "This often works when the major version matches, but CUDA 12.4 is recommended for cu124 PyTorch.",
            file=sys.stderr,
        )
PY

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install --no-build-isolation -v "$SCRIPT_DIR/submodules/diff-gaussian-rasterization-confidence"
python -m pip install --no-build-isolation -v "$SCRIPT_DIR/submodules/simple-knn"

python - <<'PY'
import diff_gaussian_rasterization
import simple_knn._C
print("ZoomGS CUDA extensions imported successfully.")
PY
