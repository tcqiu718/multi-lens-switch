# View Transition Fusion Demo

This folder contains an unofficial demo inspired by **View Transition based Dual
Camera Image Fusion**. The paper has no released source code, so this demo keeps
the uncertain parts as parameters and saves intermediate images for tuning.

## Run a smoke test

Dependencies: `numpy`, `scipy`, `scikit-image`, and `pillow`. These are enough
for the default TV-L1 demo backend; FlowFormer is only needed if you want to
provide a precomputed flow field.

```bash
cd E:/pycharmproject/multi-lens
python view_transition_fusion_demo/view_transition_fusion.py \
  --self-test \
  --out-dir outputs/view_transition_fusion_self_test
```

The output directory will contain:

- `fused.png`: final fusion result.
- `tele_warped.png`: tele image warped to the wide-camera view.
- `tele_tone_matched.png`: warped tele image after local tone matching.
- `tele_weight.png`: soft mask controlling where tele detail is trusted.
- `raw_flow_mag.png`: optical-flow magnitude.
- `transition_flat_weight.png`: where the view-transition smoothing is active.
- `transition_delta.png`: how much the transition step changed the raw flow.
- `photometric_residual.png`: wide/warped-tele mismatch map.
- `params.json`: exact parameters used in this run.

## Run on real dual-camera images

```bash
cd E:/pycharmproject/multi-lens
python view_transition_fusion_demo/view_transition_fusion.py \
  --wide path/to/wide.png \
  --tele path/to/tele.png \
  --out-dir outputs/view_transition_fusion_real \
  --resize-long-edge 1280
```

The script resizes `tele` to the `wide` image shape before optical flow. For a
closer reproduction, replace the default TV-L1 optical flow with FlowFormer and
save the flow as `HxWx2` `.npy`:

```bash
python view_transition_fusion_demo/view_transition_fusion.py \
  --wide path/to/wide.png \
  --tele path/to/tele.png \
  --flow-backend precomputed \
  --flow-npy path/to/flow.npy \
  --flow-format yx \
  --out-dir outputs/view_transition_fusion_flowformer
```

`--flow-format yx` means channel 0 is vertical displacement and channel 1 is
horizontal displacement. Use `--flow-format xy` if your flow is saved as
OpenCV/RAFT-style `(dx, dy)`.

## Main uncertain details

- The official paper does not define the complete code-level behavior of the T
  map and W map transformations. This demo approximates the view-transition
  step by smoothing optical flow in connected flat regions while preserving
  strong image/flow edges.
- The threshold for "non-connected points" is not fully specified. This demo
  uses image-gradient and flow-gradient quantiles.
- The paper describes forward warping and empty-flow handling; this demo uses
  backward sampling plus validity and residual masks. It is more stable for a
  first demo, but not identical.
- The paper's regional histogram equalization is approximated here by local
  weighted mean/std tone matching. It is easier to inspect and tune, but may
  differ from the paper on strong color casts.
- Occlusion handling is heuristic. The demo combines out-of-bounds validity,
  photometric residuals, and Gaussian softening instead of a learned or exact
  forward/backward occlusion test.

## Parameters worth tuning first

- `--transition-ratio`: small view-transition update toward the regularized
  flow. The paper mentions `0.01`; larger values such as `0.05` to `0.25` may
  be visibly stronger on small images.
- `--transition-box`: large local smoothing window. The paper mentions `600`,
  but for resized images this should scale with resolution.
- `--flow-smooth-strength`: how strongly flat connected regions use smoothed
  flow.
- `--edge-quantile` and `--flow-jump-quantile`: control which pixels are treated
  as structure/flow discontinuities.
- `--edge-keep-distance`: protects a band around structure edges from flow
  smoothing.
- `--occlusion-soft-zone`: mask feather width. The paper mentions `15 px`; use
  a larger value for high-resolution images or visible seams.
- `--residual-sigma`: lower values reject misaligned tele pixels more
  aggressively.
- `--tone-block`: local tone matching window. The paper mentions `200x200`
  blocks; scale it when using `--resize-long-edge`.
- `--tone-strength`: blend between original warped tele color and locally
  matched tele color.
- `--pyramid-levels`: multi-band blending depth. Increase for large images and
  gradual seams; decrease if details look washed out.
