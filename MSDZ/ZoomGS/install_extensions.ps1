$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $env:MAX_JOBS) {
    $env:MAX_JOBS = "8"
}

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install --no-build-isolation -v "$ScriptDir\submodules\diff-gaussian-rasterization-confidence"
python -m pip install --no-build-isolation -v "$ScriptDir\submodules\simple-knn"

@'
import diff_gaussian_rasterization
import simple_knn._C
print("ZoomGS CUDA extensions imported successfully.")
'@ | python -
