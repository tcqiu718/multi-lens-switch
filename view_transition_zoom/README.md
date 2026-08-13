# View Transition Zoom

一个可实际运行的 PyTorch/OpenCV 研究代码库，用于：

1. 依据论文 **View Transition based Dual Camera Image Fusion** 重建 `paper_repro`；
2. 将空间 View Transition 扩展为 Wide -> Tele 的时序连续变焦 `continuous_zoom`。

> 项目状态：研究复现，不是作者官方实现。截至 2026-08-12，论文的
> [arXiv 页面](https://arxiv.org/abs/2312.11184)未给出代码仓库，公开代码索引仍为
> “Request Code”。论文使用的光流骨干有独立的
> [FlowFormer 官方仓库](https://github.com/drinkingcoder/FlowFormer-Official)。

## 1. 核心几何约定

项目内图像统一为 RGB、`float32`、`[0,1]`：

| 张量 | shape | 坐标含义 |
|---|---:|---|
| `I_W`, `I_T` | `[B,3,H,W]` | Wide / Tele RGB |
| `F_T2W` | `[B,2,H,W]` | 定义在 W 输出坐标；通道 0=`dx`，1=`dy` |
| `M_*` | `[B,1,H,W]` | mask / distance / weight |
| `F_hat` | `[B,2,H,W]` | 受约束的 mixed-view target flow |
| `delta_F` | `[B,2,H,W]` | `F_hat - F_T2W`，W -> O 的 forward displacement |
| `F_T2O` | `[B,2,H,W]` | 定义在 O 输出坐标，用于 backward warp Tele |

对 W 中像素 `p=(x,y)`：

```text
p_tele = p + F_T2W(p)
WarpBackward(I_T, F_T2W) -> Tele aligned to W
```

因此，恒定 `dx=+20` 时：

- `backward_warp(image, flow)` 的可见内容向左移动 20 px；
- `forward_warp(image, flow)` 将源内容向右 splat 20 px。

这两个方向由独立单元测试固定。`forward_warp` 是真正的 nearest/bilinear
forward splatting，会累积 many-to-one 权重并显式返回 hole/valid mask。

## 2. Paper Repro 流程

```text
Wide + Tele
  -> FlowFormer / RAFT: F_T2W
  -> target flow: F_M, foreground/background means, F_target
  -> flow-aware distance M_dis
  -> F_hat = clip(F_target, F_original +/- rho*M_dis)
  -> delta_F = F_hat - F_original
  -> forward-splat F_hat by delta_F, fill holes: F_T2O
  -> backward-warp Tele by F_T2O: I_T^O
  -> valid-weighted multi-offset forward-splat Wide: I_W^O
  -> occlusion + regional histogram tone matching
  -> Laplacian pyramid blend in overlap
  -> pyramid blend with Wide non-overlap
```

遮挡掩码语义在全项目固定为：`M_occ=1` 使用 transformed Wide，`M_occ=0`
使用 transformed Tele。`rho=0` 会精确退化到原始 T->W flow。

## 3. Continuous Zoom 扩展

三个参数严格分离：

```text
zoom ratio z -> FOV(z)          # crop / intrinsics
             -> alpha(z)        # viewpoint: W -> paper mixed view
rho          -> local constraint # 几何安全边界，不等于 alpha
beta(z)      -> Tele endpoint    # mixed view -> pure Tele，平滑启动
```

连续模式使用：

```text
delta_paper = F_hat_paper - F_original
delta_z     = alpha(z) * delta_paper
F_hat_z     = F_original + delta_z
```

支持 `linear/log` 倍率归一化，`linear/smoothstep/smootherstep/cosine/custom`
曲线。`tele_endpoint` 在 `terminal_start` 后用第二个 smoothstep `beta` 渐进收敛；
不存在阈值硬切镜。在中心裁剪模式下，`z=z_T, beta=1` 的输出严格收敛到 Tele。

视频模式平滑 paper delta、occlusion/overlap mask 和 local-affine tone 参数；可选
current-Wide -> previous-Wide 光流传播，并在场景切换时清空历史状态。

## 4. 安装

建议 Python 3.8+，PyTorch 与 torchvision 必须版本匹配：

```powershell
cd E:\pycharmproject\multi-lens\view_transition_zoom
python -m pip install -r requirements.txt
```

### Torchvision RAFT

RAFT 已作为可直接选择的光流后端，不需要额外克隆仓库。推荐的质量配置是：

```yaml
flow:
  model: torchvision_raft
  fallback: null
  raft_variant: large
  raft_weights: default
  raft_input_size: null
  raft_progress: true
```

也可用 `raft`、`raft_large` 或 `raft_small` 作为 `model` 简写。`large` 通常精度更好，
`small` 更快且显存占用更低。`raft_weights` 接受 `default`、`none` 或 torchvision
公开的权重枚举名，例如 `C_T_SKHT_V2`；`none` 是随机初始化，仅适合接口测试。

`raft_input_size: [H, W]` 可在显存不足时降低估计分辨率，输出会恢复到原尺寸并同步缩放
dx/dy；设为 `null` 使用输入原分辨率。适配器自动完成 `[0,1] -> [-1,1]`、最小
`128x128` 和 8 倍数 padding，并取 RAFT 迭代列表中的最后一个 flow。首次使用
`default` 权重时，torchvision 会从 PyTorch 权重服务器下载模型。

命令行可直接切换，无需修改 YAML：

```powershell
python demo_paper.py --wide wide.png --tele tele.png `
  --set flow.model=torchvision_raft --set flow.fallback=null `
  --set flow.raft_variant=large --set flow.raft_weights=default
```

### FlowFormer

FlowFormer 不在本项目中伪造或重写。按其官方仓库安装：

```powershell
git clone https://github.com/drinkingcoder/FlowFormer-Official third_party/FlowFormer-Official
python -m pip install -r requirements-flowformer.txt
```

根据官方 README 下载 checkpoint，例如放在：

```text
third_party/FlowFormer-Official/checkpoints/things.pth
```

并检查 `config_*.yaml`：

```yaml
flow:
  model: flowformer
  fallback: raft
  flowformer_repo: third_party/FlowFormer-Official
  flowformer_checkpoint: checkpoints/things.pth
  raft_variant: large
  raft_weights: default
```

官方仓库依赖较旧，常需要独立环境，并依赖 `yacs loguru einops timm==0.4.12`。
加载失败时会明确告警并尝试 torchvision RAFT；fallback 使用同一组 `raft_*` 参数。
离线调试支持 `flow.model=precomputed` 或 `farneback`，二者不代表论文精度。

## 5. 运行

论文单图复现：

```powershell
python demo_paper.py `
  --wide wide.png `
  --tele tele.png `
  --config config_paper.yaml `
  --output outputs/paper
```

使用已计算光流：

```powershell
python demo_paper.py --wide wide.png --tele tele.png `
  --flow flow_t2w.npy --config config_paper.yaml --output outputs/paper
```

静态图像对生成连续变焦：

```powershell
python demo_zoom.py `
  --wide wide.png --tele tele.png `
  --zoom-start 1.0 --zoom-end 3.0 --frames 60 --fps 30 `
  --config config_zoom.yaml --output outputs/zoom
```

同步双视频：

```powershell
python process_video.py `
  --wide-video wide.mp4 --tele-video tele.mp4 `
  --zoom-start 1.0 --zoom-end 3.0 `
  --config config_zoom.yaml --output zoom_result.mp4
```

同步图像目录也可用 `--wide-dir` / `--tele-dir`。自定义倍率 CSV/TXT 用
`--zoom-schedule`，每行可为 `zoom` 或 `frame,zoom`。任意配置可从命令行覆盖：

```powershell
python demo_zoom.py ... --set view_transition.ratio=0.02 `
  --set zoom.schedule=smoothstep --set temporal.enabled=false
```

## 6. 合成测试与测试套件

```powershell
python -m unittest discover -s tests -v
python synthetic_test.py --output outputs/synthetic_test
```

合成场景为棋盘背景、矩形前景，背景 disparity=10 px、前景=30 px。脚本输出 direct
T->W 与 mixed-view 几何、光流、遮挡图和 `metrics.json`。当前测试还检查变换后估计
遮挡比例不高于 direct baseline，但这只是合成参考指标，不等价于真实遮挡 ground truth。

## 7. 输出

`demo_paper.py` 保存：

```text
input/{wide,tele}.png
flow/{original,mean,foreground,background,target,transformed,delta}.{png,npy}
masks/{foreground,distance,motion_boundary,hole,occlusion,overlap}.png
transformed/{tele_O,wide_O,tele_tone}.png
results/{overlap_result,full_result}.png
```

`demo_zoom.py` 保存 `zoom_result.mp4`、三联 `comparison.mp4`、5 类 baseline 抽帧、
`zoom_schedule.csv`、`zoom_parameters.csv` 和 `metrics.csv`。指标包含 adjacent-frame
difference、temporal warping error、flow consistency、mask temporal difference、亮度变化、
`mean_flow`、`mean_delta`、occlusion ratio 与 `tele_usage_ratio`。
`d_alpha/dd_alpha` 按实际 FPS 记录为每秒一阶/二阶变化率，crop box 也逐帧写入 CSV。

## 8. PAPER_AMBIGUITY 与当前近似

论文无官方源码，以下细节不能被宣称为作者精确实现：

| 未唯一确定的细节 | 当前实现 | 需要优先微调的参数 |
|---|---|---|
| 前景/背景比较规则 | 默认 `|F| > |box(F)|`；另有 component ablation | `foreground_mode` |
| 大核边缘 padding | replicate-padded integral box mean | `kernel_size: 100/300/600` |
| non-connected distance 算法 | flow gradient barrier + 区域内截断 distance transform | threshold/quantile、`max_extra_distance` |
| 空洞背景选择 | O 坐标最近有效 background flow，再 nearest fallback | `fill_mode` |
| paper occlusion 扫描/矩形 rasterization | 按相机 left/right、upper/lower 邻接边界和 flow jump 画矩形 | `rectangle_scale`、相机方位 |
| RHE 重叠块权重 | 抬高 Hann 窗加权，避免 block seam | block=200、stride=30、bins |
| FlowFormer checkpoint | 官方没有唯一手机双摄 checkpoint | things/sintel/kitti 或域内微调 |
| overlap mask | 未提供标定 mask 时假设全帧 overlap | 应由真实 FOV 标定提供 |
| mixed -> pure Tele endpoint | 在 Tele 有效 FOV 内以 beta 多带渐入 | `terminal_start: 0.75~0.9` |
| 视频时序传播 | EMA；可选 Wide current->previous flow guidance | EMA、scene-cut threshold |

最敏感参数通常依次是：光流模型/权重、输入标定与 overlap、`rho`、flow-gradient 阈值、
相机相对方位、occlusion rectangle、tone 模式，然后才是 pyramid 层数和 feather width。

## 9. 建议消融

无需改源码即可实验：

- `view_transition.ratio`: `0, 0.005, 0.01, 0.02, 0.05`
- `target_flow.kernel_size`: `100, 300, 600`
- `boundary.mode`: `simple, flow_aware`
- `zoom.schedule`: `linear, smoothstep, smootherstep`
- `temporal.enabled`: `true, false`
- `tone.mode`: `paper_rhe, local_affine_temporal, none`
- `occlusion.mode`: `paper, fb_consistency`
- `zoom.endpoint_mode`: `paper_mixed, tele_endpoint`
- `zoom.fov_mode`: `center_crop, intrinsics`

建议先在合成场景确认方向，再在静态真实 pair 调 `rho/boundary/occlusion`，最后才开启 tone
与 temporal。几何问题不应通过更软的 blending 掩盖。

## 10. 当前限制

- 输入应先完成去畸变、时间同步和粗标定；代码不自动估计镜头畸变或 rolling shutter。
- `center_crop` 是无标定基线。真正手机变焦应提供 W/T intrinsics 与真实 overlap mask。
- FlowFormer 官方依赖老旧；本项目仅提供适配器，未重新发布其代码或权重。
- paper occlusion 与 flow-aware distance 是可解释近似，需用真实遮挡 ground truth 校准。
- 视频模式每帧仍需一对 W/T 光流；开启 flow-guided temporal 会额外增加一条时域 flow。
- endpoint 中 Tele 在低于原生倍率时只覆盖中心有效区，外部保持 mixed/Wide，不生成不存在的 Tele 内容。
- 当前仅实现解析几何与经典融合；预留 learned transition、learned mask、frame interpolation、
  novel-view renderer 接口，但没有用网络替代论文困难步骤。

## 11. 目录

```text
view_transition_zoom/
  demo_paper.py       demo_zoom.py       process_video.py
  paper_pipeline.py   zoom_runner.py     baselines.py
  synthetic_test.py   config_paper.yaml  config_zoom.yaml
  models/             view_transition/   fusion/
  zoom/               utils/             tests/
```
