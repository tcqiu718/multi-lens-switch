# MSDZ 使用指南

本项目包含两个可运行部分：

- `ZoomGS`：用于训练并渲染双摄平滑变焦序列。
- `FI`：用于在 DCSZ 数据上训练和测试帧插值模型。

当前代码已适配 Python 3.10 + PyTorch 2.5.x + CUDA 12.x 环境。本 README 只保留安装、数据组织、训练、测试和渲染相关的使用说明。

## 1. 环境配置

推荐创建基础环境：

```bash
conda create -n msdz python=3.10 -y
conda activate msdz
```

根据服务器 CUDA 版本安装对应的 PyTorch 和 torchvision。以 CUDA 12.4 为例：

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

安装其余 Python 依赖：

```bash
pip install lpips timm pyiqa cupy-cuda12x kornia torchtyping PyMCubes \
    opencv-python pillow tqdm scikit-image matplotlib plyfile ninja setuptools wheel
```

ZoomGS 仍然需要编译 CUDA 扩展：

```bash
cd ZoomGS
bash install_extensions.sh
```

Windows PowerShell 下使用：

```powershell
cd ZoomGS
.\install_extensions.ps1
```

扩展安装脚本会编译：

- `diff_gaussian_rasterization`
- `simple_knn`

## 2. 数据与权重

除特别说明外，以下命令默认从 `MSDZ` 目录执行。

推荐的数据目录结构：

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

原项目提供的下载链接：

- ZoomGS 数据集：`https://pan.baidu.com/s/1lKcAs12vDzHODBKBPa3fEw?pwd=tarf`
- FI 数据集：`https://pan.baidu.com/s/1rIaAc2Huprl796qguiB8AQ`，提取码：`w4zf`
- FI 预训练模型：`https://pan.baidu.com/s/1_bfNrij8HwtwlON32TiCWg?pwd=x66g`，提取码：`x66g`
- FI 微调权重：`https://pan.baidu.com/s/1QeuSrRo4E5dIEMNGiJRLiw`，提取码：`hya8`

FI 预训练模型放置到：

```text
MSDZ/FI/pretrained_dirs/
```

FI 微调权重放置到：

```text
MSDZ/FI/ckpt/
```

## 3. ZoomGS 用法

进入 ZoomGS 目录：

```bash
cd ZoomGS
```

训练基础 UW Gaussian 模型：

```bash
CUDA_VISIBLE_DEVICES=0 python zoomgs_train.py \
    -s ../dataset/ZoomGS_dataset/01 \
    -m ./ckpt/zoomgs/01 \
    --iterations 30000 \
    --eval \
    --stage uw_pretrain \
    --data_device cuda:0
```

联合训练 UW 到广角相机的过渡模型：

```bash
CUDA_VISIBLE_DEVICES=0 python zoomgs_train.py \
    -s ../dataset/ZoomGS_dataset/01 \
    -m ./ckpt/zoomgs/01 \
    --iterations 30000 \
    --eval \
    --stage uw2wide \
    --data_device cuda:0
```

测试训练后的 ZoomGS 模型：

```bash
CUDA_VISIBLE_DEVICES=0 python zoomgs_test.py \
    -s ../dataset/ZoomGS_dataset/01 \
    -m ./ckpt/zoomgs/01 \
    --iteration 30000 \
    --target cx \
    --data_device cuda:0
```

渲染平滑变焦序列：

```bash
CUDA_VISIBLE_DEVICES=0 python zoomgs_render.py \
    -s ../dataset/ZoomGS_dataset/01 \
    -m ./ckpt/zoomgs/01 \
    --iteration 30000 \
    --target cx \
    --data_device cuda:0
```

生成的 ZoomGS 输出会保存到所选模型目录下，例如：

```text
MSDZ/ZoomGS/ckpt/zoomgs/01/
  point_cloud/
  train/
  zoom_sequences/
```

也可以先修改脚本中的场景编号和 GPU 编号，然后运行项目自带脚本：

```bash
bash zoomgs_train.sh
```

## 4. FI 模型用法

进入 FI 目录：

```bash
cd FI
```

支持的模型名称：

```text
EDSC, IFRNet, RIFE, AMT, UPRNet, EMAVFI
```

使用 PyTorch 2.x 在合成 DCSZ 数据集上训练一个 FI 模型：

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

单卡训练可以使用：

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

在合成数据上测试：

```bash
CUDA_VISIBLE_DEVICES=0 python test_syn.py \
    --model RIFE \
    --log_dir ./ckpt/RIFE_finetuned \
    --dataset_dir ../dataset/DCSZ_dataset/DCSZ_syn \
    --save_dir ./syn_results/
```

在真实数据上测试：

```bash
CUDA_VISIBLE_DEVICES=0 python test_real.py \
    --model RIFE \
    --log_dir ./ckpt/RIFE_finetuned \
    --dataset_dir ../dataset/DCSZ_dataset/DCSZ_real \
    --save_dir ./real_results/
```

切换 FI 模型时，需要同时修改 `--model` 和 `--log_dir`，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python test_real.py \
    --model UPRNet \
    --log_dir ./ckpt/UPRNet_finetuned \
    --dataset_dir ../dataset/DCSZ_dataset/DCSZ_real
```

## 5. 新环境注意事项

- ZoomGS 需要 CUDA 和已编译的扩展，不能作为纯 Python 或仅 CPU 代码运行。
- FI 中的 `EDSC` 和 `UPRNet` 会使用 CuPy 内核。对于新版 CuPy，本仓库使用 `FI/model/cupy_compat.py` 替代已移除的 `cupy.cuda.compile_with_cache` 调用路径。
- PyTorch 2.x 中旧的 `python -m torch.distributed.launch` 已废弃，FI 训练建议使用 `torchrun`。
- 运行 ZoomGS 时，可以把 `--data_device cuda:0` 改成其他可见 CUDA 设备。

## 6. 常用检查命令

检查 PyTorch 和 CUDA：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

检查 CuPy：

```bash
python - <<'PY'
import cupy
print(cupy.__version__)
print(cupy.cuda.runtime.runtimeGetVersion())
PY
```

检查 ZoomGS 扩展：

```bash
python - <<'PY'
import diff_gaussian_rasterization
import simple_knn._C
print("ZoomGS extensions OK")
PY
```
