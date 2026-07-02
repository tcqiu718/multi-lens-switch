# MSDZ Usage Guide

This repository contains two runnable parts:

- `ZoomGS`: train and render dual-camera smooth zoom sequences.
- `FI`: train and test frame interpolation models on DCSZ data.

The code has been adjusted for a Python 3.10 + PyTorch 2.5.x CUDA 12.x environment. This README focuses on installation, data layout, training, testing, and rendering.

## 1. Environment

Recommended base environment:

```bash
conda create -n msdz python=3.10 -y
conda activate msdz
```

Install PyTorch and torchvision with the CUDA wheel that matches your server. For CUDA 12.4:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining Python packages:

```bash
pip install lpips timm pyiqa cupy-cuda12x kornia torchtyping PyMCubes \
    opencv-python pillow tqdm scikit-image matplotlib plyfile ninja setuptools wheel
```

ZoomGS still requires CUDA extension compilation:

```bash
cd ZoomGS
bash install_extensions.sh
```

On Windows PowerShell:

```powershell
cd ZoomGS
.\install_extensions.ps1
```

The extension installer builds:

- `diff_gaussian_rasterization`
- `simple_knn`

## 2. Data and Checkpoints

Run commands from the `MSDZ` directory unless otherwise noted.

Expected data layout:

```text
MSDZ/
  dataset/
    ZoomGS_dataset/
      01/
      ...
    DCSZ_dataset/
      DCSZ_syn/
      DCSZ_real/
  FI/
    pretrained_dirs/
      EDSC/
      IFRNet/
      RIFE/
      AMT/
      UPRNet/
      EMAVFI/
    ckpt/
```

Download links from the original project:

- ZoomGS dataset: `https://pan.baidu.com/s/1lKcAs12vDzHODBKBPa3fEw?pwd=tarf`
- FI dataset: `https://pan.baidu.com/s/1rIaAc2Huprl796qguiB8AQ`, extraction code: `w4zf`
- FI pretrained models: `https://pan.baidu.com/s/1_bfNrij8HwtwlON32TiCWg?pwd=x66g`, extraction code: `x66g`
- FI fine-tuned checkpoints: `https://pan.baidu.com/s/1QeuSrRo4E5dIEMNGiJRLiw`, extraction code: `hya8`

Put FI pretrained models into:

```text
MSDZ/FI/pretrained_dirs/
```

Put FI fine-tuned checkpoints into:

```text
MSDZ/FI/ckpt/
```

## 3. ZoomGS Usage

Enter the ZoomGS directory:

```bash
cd ZoomGS
```

Train the base UW Gaussian model:

```bash
CUDA_VISIBLE_DEVICES=0 python zoomgs_train.py \
    -s ../dataset/ZoomGS_dataset/01 \
    -m ./ckpt/zoomgs/01 \
    --iterations 30000 \
    --eval \
    --stage uw_pretrain \
    --data_device cuda:0
```

Jointly train the UW-to-wide camera transition model:

```bash
CUDA_VISIBLE_DEVICES=0 python zoomgs_train.py \
    -s ../dataset/ZoomGS_dataset/01 \
    -m ./ckpt/zoomgs/01 \
    --iterations 30000 \
    --eval \
    --stage uw2wide \
    --data_device cuda:0
```

Test the trained ZoomGS model:

```bash
CUDA_VISIBLE_DEVICES=0 python zoomgs_test.py \
    -s ../dataset/ZoomGS_dataset/01 \
    -m ./ckpt/zoomgs/01 \
    --iteration 30000 \
    --target cx \
    --data_device cuda:0
```

Render smooth zoom sequences:

```bash
CUDA_VISIBLE_DEVICES=0 python zoomgs_render.py \
    -s ../dataset/ZoomGS_dataset/01 \
    -m ./ckpt/zoomgs/01 \
    --iteration 30000 \
    --target cx \
    --data_device cuda:0
```

Generated ZoomGS outputs are saved under the selected model directory, for example:

```text
MSDZ/ZoomGS/ckpt/zoomgs/01/
  point_cloud/
  train/
  zoom_sequences/
```

You can also run the provided script after editing the scene id and GPU id:

```bash
bash zoomgs_train.sh
```

## 4. FI Model Usage

Enter the FI directory:

```bash
cd FI
```

Supported model names:

```text
EDSC, IFRNet, RIFE, AMT, UPRNet, EMAVFI
```

Train one FI model on the synthetic DCSZ dataset with PyTorch 2.x:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=29502 \
    train.py \
    --model RIFE \
    --log_dir ./ckpt/RIFE_finetuned \
    --dataset_dir ../dataset/DCSZ_dataset/DCSZ_syn \
    --epoch 100 \
    --world_size 2
```

For single-GPU training, use:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_port=29502 \
    train.py \
    --model RIFE \
    --log_dir ./ckpt/RIFE_finetuned \
    --dataset_dir ../dataset/DCSZ_dataset/DCSZ_syn \
    --epoch 100 \
    --world_size 1
```

Test on synthetic data:

```bash
CUDA_VISIBLE_DEVICES=0 python test_syn.py \
    --model RIFE \
    --log_dir ./ckpt/RIFE_finetuned \
    --dataset_dir ../dataset/DCSZ_dataset/DCSZ_syn \
    --save_dir ./syn_results/
```

Test on real-world data:

```bash
CUDA_VISIBLE_DEVICES=0 python test_real.py \
    --model RIFE \
    --log_dir ./ckpt/RIFE_finetuned \
    --dataset_dir ../dataset/DCSZ_dataset/DCSZ_real \
    --save_dir ./real_results/
```

To switch FI models, change both `--model` and `--log_dir`, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python test_real.py \
    --model UPRNet \
    --log_dir ./ckpt/UPRNet_finetuned \
    --dataset_dir ../dataset/DCSZ_dataset/DCSZ_real
```

## 5. Notes for the Updated Environment

- ZoomGS requires CUDA and compiled extensions. It cannot run as pure Python or CPU-only code.
- FI models `EDSC` and `UPRNet` use CuPy kernels. With newer CuPy versions, this repository uses `FI/model/cupy_compat.py` to replace the removed `cupy.cuda.compile_with_cache` path.
- The old `python -m torch.distributed.launch` command is deprecated in PyTorch 2.x. Use `torchrun` for FI training.
- `--data_device cuda:0` can be changed to another visible CUDA device when running ZoomGS.

## 6. Common Commands

Check PyTorch and CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

Check CuPy:

```bash
python - <<'PY'
import cupy
print(cupy.__version__)
print(cupy.cuda.runtime.runtimeGetVersion())
PY
```

Check ZoomGS extensions:

```bash
python - <<'PY'
import diff_gaussian_rasterization
import simple_knn._C
print("ZoomGS extensions OK")
PY
```
