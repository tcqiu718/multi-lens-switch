# Hybrid Zoom Camera Fusion（PyTorch 简化复现）

这是论文 [Efficient Hybrid Zoom using Camera Fusion on Mobile Phones](https://arxiv.org/abs/2401.01461)
的第一版研究实现。项目优先完成一个可在 PC/CUDA 上运行、便于观察中间结果和继续扩展的闭环：

```text
Wide + Tele -> RAFT 双向光流 -> Warp Tele -> Occlusion / Rejection
            -> 5-level Fusion UNet -> Adaptive Blending -> RGB output
```

这不是作者官方代码，也不声称复现论文的移动端速度或全部质量。当前实现直接使用 torchvision RAFT，
只融合亮度，并在不可靠区域退回 Wide。所有图像张量在项目内部均为 **RGB、float、[0, 1]**。

## 1. 当前功能

- torchvision 官方 RAFT large/small 封装，可使用预训练权重、冻结或微调；CUDA 与 CPU fallback。
- 可在较低分辨率估计 flow，再用正确的位移尺度恢复到融合分辨率。
- 基于 `grid_sample` 的 batched backward warp，同时返回有效区域。
- forward-backward consistency 的 hard/soft occlusion mask。
- 局部去均值、对比度归一化 patch 的 L1/L2 alignment rejection mask。
- 支持 3/4 通道输入、任意合理尺寸的轻量 5-level Fusion UNet。
- 动态组合 occlusion、rejection 以及未来 defocus/flow uncertainty mask。
- VGG19 perceptual、真正的 Contextual Loss、Gaussian brightness loss。
- AMP、AdamW、cosine scheduler、resume、best/last checkpoint、TensorBoard 与验证。
- 无 GT 测试、带 GT 的 PSNR/SSIM、标准光流 color wheel 与完整中间结果保存。
- 无 Fusion checkpoint 时，residual head 零初始化为 Wide identity，仍可安全检查 flow/masks。

## 2. 安装

建议 Python 3.9+。先按显卡驱动安装匹配的 PyTorch/CUDA 版本，再安装其余依赖：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

# 按 https://pytorch.org/get-started/locally/ 选择适合本机的命令
pip install torch torchvision
pip install -r hybrid_zoom/requirements.txt
```

`flow.weights: default` 第一次使用时会由 torchvision 下载官方 RAFT 权重。
离线结构测试可设为 `none`，但随机 RAFT 的 flow **没有配准意义**：

```bash
python hybrid_zoom/demo.py \
  --wide wide.png --tele tele.png --output demo_result \
  --flow-variant small --flow-weights none --height 128 --width 128
```

训练默认还会加载官方预训练 VGG19。完全离线时可加
`--set loss.vgg_weights=null`；这只适合代码 smoke test，不适合正式训练。

## 3. 光流和 Warp 坐标约定

Wide 始终是目标坐标系。项目定义：

```text
flow_w2t: [B, 2, H, W]
flow_w2t[:, 0] = dx
flow_w2t[:, 1] = dy

p_tele = p_wide + flow_w2t(p_wide)
warped_tele = warp(tele, flow_w2t)
```

这是 backward sampling flow：输出 Wide 像素 `(x, y)` 从 Tele 的
`(x + dx, y + dy)` 取样。比如常量 `dx=+20` 会让可见的 Tele 内容在
`warped_tele` 中向左移动 20 像素。它不是把 Wide 像素向右 forward-splat 20 像素。

`warp` 将像素坐标转换到 `[-1, 1]` 后调用 `torch.nn.functional.grid_sample`。
默认 `align_corners=True`，网格转换和 `grid_sample` 始终使用同一个设置。
`return_mask=True` 时还返回 `[B,1,H,W]` valid mask。

光流 resize 不能只插值：

```text
Hf x Wf -> H x W
dx *= W / Wf
dy *= H / Hf
```

项目统一通过 `modules.resize_flow()` 完成这一步。

快速验证以上坐标约定：

```bash
python -m unittest discover -s hybrid_zoom/tests -v
```

其中包含人工正 `dx`、`align_corners=True/False` identity、flow resize、颜色往返、
mask、非 16 倍数 UNet、完整 mock-flow pipeline、Dataset/loss/metrics/checkpoint 测试。

## 4. 关键 Tensor Shape

| 名称 | Shape | 含义 |
|---|---:|---|
| `wide`, `tele` | `[B,3,H,W]` | 预处理后的 RGB `[0,1]` |
| `flow_w2t` | `[B,2,H,W]` | Wide 像素到 Tele 采样位置的 `(dx,dy)` |
| `flow_t2w` | `[B,2,H,W]` | Tele 到 Wide 的反向 flow |
| `warped_tele` | `[B,3,H,W]` | Wide 坐标系中的 Tele |
| `fusion_y` | `[B,1,H,W]` | UNet 预测的融合亮度 |
| `occlusion_mask` | `[B,1,H,W]` | 0 可靠，1 遮挡/不可靠 |
| `rejection_mask` | `[B,1,H,W]` | 0 对齐，1 应拒绝 Tele |
| `blend_mask` | `[B,1,H,W]` | 1 使用 fusion，0 退回 Wide |
| `output` | `[B,3,H,W]` | Wide 色度 + 最终亮度的 RGB 输出 |

除 `blend_mask` 是“融合使用量”外，所有 reliability mask 均统一为：
**0 = reliable，1 = unreliable**。

完整 forward 返回：

```python
{
    "output": final_rgb,
    "wide": wide,
    "tele": tele,
    "flow_w2t": flow_w2t,
    "flow_t2w": flow_t2w,
    "warped_tele": warped_tele,
    "occlusion_mask": m_occ,
    "rejection_mask": m_reject,
    "blend_mask": m_blend,
    "fusion_y": y_fusion,
    # 额外诊断项
    "fusion_rgb": fusion_rgb,
    "warp_valid_mask": valid,
    "final_y": y_final,
}
```

## 5. 数据集格式

有 GT 的训练/验证：

```text
dataset/
├── train/
│   ├── wide/
│   │   ├── 000001.png
│   │   └── 000002.png
│   ├── tele/
│   │   ├── 000001.png
│   │   └── 000002.png
│   └── gt/
│       ├── 000001.png
│       └── 000002.png
└── val/
    ├── wide/
    ├── tele/
    └── gt/
```

无 GT 的真实双摄测试：

```text
dataset/
└── test/
    ├── wide/
    │   ├── 000101.png
    │   └── 000102.png
    └── tele/
        ├── 000101.png
        └── 000102.png
```

每个 split 内启用的目录必须拥有**完全相同的文件名（含扩展名）**；Dataset 会严格检查，
避免静默错配。读取结果统一为 RGB `float32 [0,1]`。训练增强会同步应用归一化 crop 和水平翻转。

若要让 Wide 在 resize 前做中心 FOV crop，请显式配置原生像素尺寸，例如
`image.wide_crop_size: [3024, 4032]`（或同时设置 `crop_height/crop_width`）。
裸写 `center_crop: true` 会报错，而不会因推断尺寸不同在 demo 与 Dataset 中产生含糊行为。
train/test 在 Dataset 内完成一次 crop/resize 并以 `preprocessed=True` 调模型；demo 则由模型处理原图。
有 GT 时默认按同一归一化 FOV 同步裁 GT；若数据集里的 GT 已经是裁好的目标 FOV，设置
`image.crop_gt_with_wide: false`。

## 6. 运行命令

以下命令均从包含 `hybrid_zoom/` 的仓库根目录执行。

### 训练

```bash
python hybrid_zoom/train.py \
  --config hybrid_zoom/config.yaml \
  --data-root ./dataset \
  --output-dir ./runs/hybrid_zoom
```

恢复训练：

```bash
python hybrid_zoom/train.py \
  --config hybrid_zoom/config.yaml \
  --data-root ./dataset \
  --resume ./runs/hybrid_zoom/last.pth
```

本项目生成的 checkpoint 包含 Fusion UNet 与 RAFT 的完整状态。resume/test/demo 会先检测这一点，
用 `weights=None` 构造 RAFT 后再严格恢复 checkpoint，因此离线恢复不会先触发重复下载。

YAML 可通过重复的 `--set key=value` 临时覆盖：

```bash
python hybrid_zoom/train.py --config hybrid_zoom/config.yaml --data-root ./dataset \
  --set training.batch_size=1 \
  --set flow.variant=small \
  --set fusion.base_channels=16
```

默认冻结 RAFT，只优化有梯度的 Fusion UNet 参数。若设 `flow.freeze=false`，RAFT 也进入优化器；
这会显著增加显存与计算量。

### 整个测试集

无 GT：

```bash
python hybrid_zoom/test.py \
  --config hybrid_zoom/config.yaml \
  --data-root ./dataset \
  --split test \
  --checkpoint ./runs/hybrid_zoom/best.pth \
  --output ./results
```

有 GT（例如 val）：

```bash
python hybrid_zoom/test.py \
  --config hybrid_zoom/config.yaml \
  --data-root ./dataset \
  --split val \
  --checkpoint ./runs/hybrid_zoom/best.pth \
  --output ./results_val
```

无 checkpoint 也可运行，用来检查 RAFT、warp 和 masks；由于 Fusion UNet 是安全 identity 初始化，
这种情况下不代表已获得 detail enhancement。

测试输出：

```text
results/
├── final/
├── warped_tele/
├── fusion/
├── flow/                 # 标准 optical-flow color wheel
├── masks/
│   ├── occlusion/
│   ├── rejection/
│   └── blend/
├── metrics.txt           # 有 GT 时包含逐图和平均 PSNR/SSIM
└── run_config.json
```

### 单张图 Demo

```bash
python hybrid_zoom/demo.py \
  --config hybrid_zoom/config.yaml \
  --wide ./wide.png \
  --tele ./tele.png \
  --checkpoint ./runs/hybrid_zoom/best.pth \
  --output ./demo_result
```

保存 `wide.png`、`tele.png`、`warped_tele.png`、`fusion.png`、`final.png`、
`flow.png`、`occlusion_mask.png`、`rejection_mask.png`、`blend_mask.png`。

## 7. Loss

当前训练目标是：

```text
L_total = 1.0 * L_vgg + 0.05 * L_contextual + 1.0 * L_brightness
```

- `L_vgg`：冻结的 torchvision VGG19 多层 feature L1。
- `L_contextual`：feature 中心化、L2 normalize、pairwise cosine distance、relative distance、
  contextual soft matching 和 `-log`；不是用普通 L1 冒充。为限制显存，默认将空间采样限制为 1024。
- `L_brightness`：预测与 GT 的 luminance 经 `sigma=10` Gaussian blur 后做 L1。

## 8. 扩展 Flow Uncertainty / Defocus

`AdaptiveBlending` 不把两个现有 mask 写死，而是动态计算：

```python
masks = {
    "occlusion": m_occ,
    "rejection": m_reject,
    "flow_uncertainty": None,
    "defocus": None,
}
m_blend = clamp(1 - sum(non_none_masks), 0, 1)
```

也可以立即把外部估计结果传给模型：

```python
outputs = model(
    wide,
    tele,
    flow_uncertainty_mask=m_flow,
    defocus_mask=m_defocus,
)
```

因此未来添加 `FlowUncertaintyMask`、`DefocusMask` 或 SEA-RAFT 时不需要修改 blending 公式。

## 9. 与原论文的主要差异

1. **对齐模型**：论文使用裁剪/FAST 全局平移/传感器颜色匹配，再用裁剪后的轻量 PWC-Net；
   本版提供可选中心裁剪/resize，直接使用 torchvision RAFT，没有复制 RAFT 源码。
2. **RAFT 在论文中的角色**：论文主要用 RAFT 生成 pseudo ground-truth 来微调 PWC-Net；本版将 RAFT
   直接作为推理 flow estimator。
3. **缺少两个 mask**：当前不内部估计 `M_flow` 和 `M_defocus`，只提供动态接口；完整论文使用四类 mask。
4. **Rejection 简化**：论文还按焦距比模拟 Tele/Wide 的 optical resolution，并采用其特定指数公式；
   本版按需求实现局部去均值/RMS 归一化 patch 的 L1/L2 confidence。
5. **Fusion UNet**：这里是清晰的通用 5-level 轻量 UNet，不保证逐层等同论文 supplementary architecture。
6. **训练数据**：没有复现双手机 rig、额外 Tele GT、训练期双重 warp 与 availability loss mask；
   Dataset 接收用户准备的严格配对 `wide/tele/gt`。
7. **Brightness loss**：按本项目验收要求计算 prediction 与 GT 的低频差；论文原式约束
   `Y_fusion` 与 `Y_source`，两者并不完全相同。
8. **空间回填**：本版输出位于配置的 Wide 分辨率，没有实现论文的原生 12MP FOV crop/uncrop 相机管线。
9. **部署**：未实现 PWC-Net 移动版、TensorFlow/TFLite、NNAPI、量化、12MP tiling 或论文的
   500ms/300MB 移动端指标。

## 10. 项目结构

```text
hybrid_zoom/
├── README.md
├── requirements.txt
├── config.yaml
├── config_utils.py
├── train.py
├── test.py
├── demo.py
├── datasets/
│   ├── __init__.py
│   └── hybrid_zoom_dataset.py
├── models/
│   ├── __init__.py
│   ├── flow_estimator.py
│   ├── fusion_unet.py
│   └── hybrid_zoom_model.py
├── modules/
│   ├── __init__.py
│   ├── warp.py
│   ├── preprocessing.py
│   ├── occlusion_mask.py
│   ├── rejection_mask.py
│   └── adaptive_blending.py
├── losses/
│   ├── __init__.py
│   ├── perceptual_loss.py
│   ├── contextual_loss.py
│   ├── brightness_loss.py
│   └── total_loss.py
├── utils/
│   ├── __init__.py
│   ├── image_utils.py
│   ├── checkpoint.py
│   ├── logger.py
│   ├── metrics.py
│   └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_support.py
└── scripts/
    ├── train.sh
    └── test.sh
```

## 11. 已知限制

- Wide/Tele 必须是近同步且大致 FOV 对应的输入；本版没有相机标定、畸变校正或 homography。
- RAFT 预训练域与真实多摄手机域可能不匹配，后续应使用真实数据微调或接入 SEA-RAFT。
- 无训练 checkpoint 时输出刻意接近 Wide；这是一种防伪影的安全默认值，不是超分效果。
- PSNR/SSIM 只衡量与 GT 的像素接近程度，不一定与真实 Tele 纹理迁移的视觉质量一致。
- LPIPS 仅预留为后续评价项，当前没有将它伪装成其他指标。
