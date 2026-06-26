# ZoomGS PyTorch 2.5 Notes

This refactor keeps the original ZoomGS pipeline while making the Python side
compatible with a PyTorch 2.5 / CUDA 12.x environment.

## Expected Environment

- Python 3.10
- PyTorch 2.5.x CUDA build, for example cu124
- torchvision matching the PyTorch build
- CUDA Toolkit with `nvcc` visible when building the extensions
- `plyfile`, `opencv-python`, `matplotlib`, `tqdm`, `ninja`, `setuptools`, `wheel`

## Build Extensions

From this directory:

```bash
bash install_extensions.sh
```

On Windows PowerShell:

```powershell
.\install_extensions.ps1
```

The script installs:

- `diff_gaussian_rasterization`
- `simple_knn`

These are still required for ZoomGS training and rendering.

## Device Selection

The scripts keep CUDA as the default device. You can select a visible GPU with
either `CUDA_VISIBLE_DEVICES` or the existing `--data_device` argument:

```bash
CUDA_VISIBLE_DEVICES=1 python zoomgs_train.py -s ../dataset/ZoomGS_dataset/01 -m ./ckpt/zoomgs/01 --stage uw_pretrain
```

or:

```bash
python zoomgs_train.py -s ../dataset/ZoomGS_dataset/01 -m ./ckpt/zoomgs/01 --stage uw_pretrain --data_device cuda:0
```

ZoomGS still requires CUDA because the rasterizer and KNN extensions are CUDA
extensions.
