# UniDepth V2 接入与下一步实验

更新：2026-10-29。本轮将官方 UniDepth V2 ViT-L/14 接入 NightTrace 的冻结八条件强度扫描。允许它与 DAV2 具有相同或不同的失效原因，不预设其危险组合或过渡区。

## 接口做了什么

- `models/inference.py` 按 model.name 显式选择模型，未知名称报错。
- UniDepth 使用官方 `infer(rgb*255, camera=None, normalize=True)`：输入仍为浮点，不转 uint8，不丢失暗部浮点信号；使用官方归一化、padding、resize 和还原。
- 第一轮为 RGB-only，不额外提供相机内参。`resolution_level: 9` 固定，独立记录；模型原生处理分辨率与 DAV2 不必相同，但看到的是相同数据集的同一退化 RGB 网格。
- UniDepth 输出原生 metric z-depth，直接送入 GT 评价。其倒数仅用于既有逆深度可视化和相对 clean 的漂移。
- 主指标仍为相同 GT 像素对上的远近排序错误，以及相对各模型 clean 的增量。当前 `gt_abs_rel`、`gt_a1` 仍采用中值对齐，便于相对几何比较，不能标为未经对齐的 metric-depth 性能。
- 官方权重严格加载，模型冻结；目录里配置、权重、实际导入源码及依赖版本均记录。推理不调用 HF 下载 API。
- `diagnose_severity.py` 支持新模型；旧训练、test.py、DAV2 特征交换入口仍是其原接口，不因新增配置自动获得 UniDepth 训练/机制探针支持。

## 1. 更新代码

```bash
cd /080226/Nighttime_Depth_Estimation/NightTrace
git status --short
git fetch origin
git switch feat/severity-boundary-20261015
git pull --ff-only
```

原实验文件保留用于比较。扫描实现哈希本轮发生变化，已经完成的 DAV2 目录不重跑；若仍需恢复旧的未完成 DAV2 实验，应使用其原始代码版本，避免合同不一致。新模型使用新的输出目录。

## 2. 权重：手动下载两个文件

官方页面：https://huggingface.co/lpiccinelli/unidepth-v2-vitl14/tree/main

只需要：

1. `config.json`（约 3.85 KB）：https://huggingface.co/lpiccinelli/unidepth-v2-vitl14/resolve/main/config.json?download=true
2. `model.safetensors`（约 1.42 GB）：https://huggingface.co/lpiccinelli/unidepth-v2-vitl14/resolve/main/model.safetensors?download=true

放在：

```text
NightTrace/models/checkpoints/unidepth_v2_vitl14/
  config.json
  model.safetensors
```

两者取自同一官方模型快照。也支持同页面的 `pytorch_model.bin` 代替 safetensors，无需两个权重文件都下载。不要用 v1、v2old 或其他 backbone 的配置。官方 ViT-L/14 配置的 pixel_encoder.pretrained 应为 null，避免构造时额外下载 DINOv2；本入口会拒绝其他值。

## 3. 独立 Python 3.10 环境与官方源码

现有 nighttime_ds 为 Python 3.8；官方 UniDepth 要求 Python≥3.10。使用单独环境，保留原 DAV2 环境。

```bash
conda create -n nighttrace_uni python=3.10 -y
conda activate nighttrace_uni
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-unidepth.txt
python tools/prepare_unidepth.py
```

最后一条只下载官方源码，导入本仓库 `models/unidepth/` 并保留许可证，不下载任何权重。源码固定在官方 commit `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`，收据为 `UNIDEPTH_SOURCE_MANIFEST.json`。

源码也可以离线准备：下载 https://github.com/lpiccinelli-eth/UniDepth/archive/8d8cfe4c7ee15297099983607febf0d4f32eb3d6.zip ，然后执行：

```bash
python tools/prepare_unidepth.py --zip /path/to/UniDepth-8d8cfe4c7ee15297099983607febf0d4f32eb3d6.zip
```

本轮只推理，不要求编译训练用 CUDA patch 提取扩展，也不要求安装 xformers；官方相应路径有回退实现。若提示无法运行 original UniDepth V1，不能仅凭该提示判定 V2 不可用；以随后的 smoke 是否成功为准。

额外依赖文件是固定源码下的推理依赖集合，未直接安装包含训练/展示全部依赖的上游 requirements。正式模型在服务器的依赖兼容性由下一步真实 smoke 验证。跨环境的 torch/numpy 等版本会写入元数据，比较时同时核对参数与实际输入统计。

## 4. 本轮先执行：一张图真实权重检查

已有 `config/nuscenes_unidepthv2.yaml` 与 `config/robotcar_unidepthv2.yaml`，数据设置沿用仓库对应配置。若服务器此前修改过 data.root、GT 空间或 Flare7K++ 路径，请将这些数据设置同步到新配置，保留其 UniDepth model 块。

GPU02 沿用已确认的 `CUDA_VISIBLE_DEVICES=1`：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/smoke_inference.py \
  --config config/nuscenes_unidepthv2.yaml --split day \
  --out runs/uni_ns_smoke
```

输出 `smoke.json`、未经显示归一化的浮点 `depth.npy` 和 `inverse_depth.npy`。这一步只检查原生输出、形状、有限性及本地权重/源码身份，不估计失效边界。把 smoke.json 和实际报错（如有）提供后，再确认完整实验。

需要检查八条件端到端时，只用两张图做一次软件连通检查（最多 58 次前向），不把两张图当研究样本结论：

```bash
CUDA_VISIBLE_DEVICES=1 python diagnose_severity.py \
  --config config/nuscenes_unidepthv2.yaml --source day-test \
  --profile config/severity.yaml --limit 2 --levels 1 2 3 4 \
  --repeats 1 --seed 42 --save-vis --num-vis 2 \
  --defer-summary --out runs/uni_ns_check2

python tools/summarize_sweep.py --runs runs/uni_ns_check2 \
  --no-fit --bootstrap 1000 --out runs/uni_ns_check2_analysis
```

如果显存不足，可在新配置中降低 resolution_level 并重新记录设置、使用新目录。不要在原目录中改变分辨率后续写。

## 5. 真实接口通过后：完整正式扫描

正式实验使用完整官方 day-test，保持七退化条件、0.25 剂量步长、两次重复和 seed=42；不只跑 DAV2 最敏感的 LN/LNG。GT 候选对数、近似等深过滤和主阈值沿用当前默认 8192 / 0.1 / 0.10。

### nuScenes

```bash
CUDA_VISIBLE_DEVICES=1 python diagnose_severity.py \
  --config config/nuscenes_unidepthv2.yaml --source day-test \
  --all-split --limit 0 --profile config/severity.yaml \
  --level-step 0.25 --repeats 2 --seed 42 \
  --pairs 8192 --min-pairs 128 --pair-gap 0.1 --failure-delta 0.10 \
  --defer-summary --out runs/uni_ns_full_gt

python tools/summarize_sweep.py --runs runs/uni_ns_full_gt \
  --no-fit --bootstrap 1000 --out runs/uni_ns_analysis
```

500 张时，112,500 次前向、113,000 行，与 DAV2 nuScenes 的设计一致。

### RobotCar（nuScenes 通过后再执行）

```bash
CUDA_VISIBLE_DEVICES=1 python diagnose_severity.py \
  --config config/robotcar_unidepthv2.yaml --source day-test \
  --all-split --limit 0 --profile config/severity.yaml \
  --level-step 0.25 --repeats 2 --seed 42 \
  --pairs 8192 --min-pairs 128 --pair-gap 0.1 --failure-delta 0.10 \
  --defer-summary --out runs/uni_rc_full_gt

python tools/summarize_sweep.py --runs runs/uni_rc_full_gt \
  --no-fit --bootstrap 1000 --out runs/uni_rc_analysis
```

159 张时，35,775 次前向、35,934 行。恢复任一未完成新扫描时保持同一环境、代码、权重及参数，在原命令末尾加 --resume。

已有 nuScenes sample token 不能自动恢复真实 scene ID，统计分组仍如实保留 unknown_recording；没有真实分组时不解释场景 bootstrap 置信区间。当前不为此重跑 DAV2。

## 交付与验证范围

代码测试使用 CPU、小张量和假 UniDepth 类，验证浮点输入与官方 infer 调用参数、本地严格权重加载、冻结、源码身份、原生深度不被 DAV2 截断、GT 排序方向及现有恢复/分片/指标回归。它们不替代真实 checkpoint 或 GPU 验证。

本地未下载 1.42GB 权重、未在本地完成真实 UniDepth 推理，尚无 UniDepth 失效结论。下一步只需完成环境/源码/权重准备，并执行第 4 节真实 smoke，再继续正式扫描。

官方来源：

- https://github.com/lpiccinelli-eth/UniDepth
- https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/models/unidepthv2/unidepthv2.py
- https://huggingface.co/lpiccinelli/unidepth-v2-vitl14/tree/main
