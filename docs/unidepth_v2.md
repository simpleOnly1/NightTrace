# UniDepth V2 接入与下一步实验

更新：2026-11-04。UniDepth V2 ViT-L/14 接入冻结八条件强度扫描，并为现有 Python 3.8 环境提供项目兼容层。允许它与 DAV2 具有相同或不同的失效原因，不预设其危险组合或过渡区。

当前进度：用户提供的环境检查和单张真实权重结果已通过；但正式扫描在首张 clean 推理的 cuBLASLt 算法选择处失败。当前先执行第 4b 节的 CUDA 短检查，再继续第 5 节。保留现有 Python 3.8.12 环境；尚无 UniDepth GT 精度或失效边界结论。

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

## 3. 继续使用现有 nighttime_ds 环境

保持现有 Python 3.8、Torch、Torchvision、NumPy、OpenCV 和 SciPy。上游声明 Python≥3.10；本项目将固定源码中的新类型标注转换为 Python 3.8 可用的 typing 标注，并取消推理不需要的可视化模块自动导入。模型计算、预处理和权重名称不改。这是 NightTrace 的兼容方案，不代表上游官方支持 Python 3.8。

```bash
conda activate nighttime_ds
python tools/prepare_unidepth.py
python tools/check_unidepth_env.py --install --out runs/uni_env_check.json
```

`prepare_unidepth.py` 下载并兼容处理官方源码，导入 `models/unidepth/` 并保留许可证，不下载权重。源码固定在官方 commit `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`；`UNIDEPTH_SOURCE_MANIFEST.json` 记录原始与处理后的文件哈希、兼容版本及修改清单。已经准备过未经修改的同版官方源码，也可以直接重跑，不需要 --force；未知本地修改仍报错。

`check_unidepth_env.py --install` 先检查现有 Torch/Torchvision 接口，再通过 `pip --no-deps --target` 将补充依赖安装到项目内 `.deps/unidepth-py38/`。仅加载 UniDepth 时启用该目录；不写入 conda 环境的 site-packages，也不安装或升级上述核心数值包。之后真实导入 UniDepthV2，并在 JSON 中记录检查前后的核心包版本。不要执行旧文档中创建 Python 3.10 环境或全局安装 requirements 的命令。

现有记录为 Torch 2.4.1+cu121 / NumPy 1.23.5，可保留。检查要求 Torch 的 scaled_dot_product_attention/autocast 和 Torchvision v2 normalize 接口可用；若检查失败，把 `runs/uni_env_check.json` 和终端错误发回，不要先升级环境。

源码也可以离线准备：下载 https://github.com/lpiccinelli-eth/UniDepth/archive/8d8cfe4c7ee15297099983607febf0d4f32eb3d6.zip ，然后执行：

```bash
python tools/prepare_unidepth.py --zip /path/to/UniDepth-8d8cfe4c7ee15297099983607febf0d4f32eb3d6.zip
```

本轮只推理，不要求编译训练用 CUDA patch 提取扩展，也不要求安装 xformers；官方相应路径有回退实现。若提示无法运行 original UniDepth V1，不能仅凭该提示判定 V2 不可用；以随后的 smoke 是否成功为准。

额外依赖固定为 Python 3.8 可用版本，未安装上游全部训练/展示依赖。无需 wandb、trimesh 或 xformers。正式模型在服务器上的可用性由下一步真实 smoke 验证。Torch/NumPy 等版本和 Python 版本写入元数据。

## 4. 本轮先执行：一张图真实权重检查

已有 `config/nuscenes_unidepthv2.yaml` 与 `config/robotcar_unidepthv2.yaml`，数据设置沿用仓库对应配置。若服务器此前修改过 data.root、GT 空间或 Flare7K++ 路径，请将这些数据设置同步到新配置，保留其 UniDepth model 块。

GPU02 沿用已确认的 `CUDA_VISIBLE_DEVICES=1`：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/smoke_inference.py \
  --config config/nuscenes_unidepthv2.yaml --split day \
  --out runs/uni_ns_smoke
```

输出 `smoke.json`、未经显示归一化的浮点 `depth.npy` 和 `inverse_depth.npy`。这一步只检查原生输出、形状、有限性及本地权重/源码身份，不估计失效边界。本步骤已于 2026-11-04 根据用户提供的结果验收。

以下两图检查供首次排查 GT/Flare 路径时使用，不是正式样本规模。本次 CUDA 错误优先使用下方第 4b 节命令；这里保留通用路径排查命令（最多 58 次前向）：

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

## 4b. cuBLASLt 算法选择错误：保持环境，先做短检查

本次完整 traceback 在首张 clean 的 DINOv2 qkv 线性层报告 `CUBLAS_STATUS_NOT_SUPPORTED` / `cublasLtMatmulAlgoGetHeuristic`，不是退化图达到失效强度。单张 smoke 通过仍不足以排除 CUDA 执行路径问题；尚不能仅凭日志确定具体的驱动、库或资源原因。

PyTorch 2.4.1 源码提供 `DISABLE_ADDMM_CUDA_LT=1`，可禁用 addmm 的 cuBLASLt 路径，作为本次错误的优先绕行检查。该变量必须在新 Python 进程启动前设置；不更新 Python、Torch 或 Torchvision。官方出处：[PyTorch v2.4.1 Blas.cpp](https://github.com/pytorch/pytorch/blob/v2.4.1/aten/src/ATen/native/cuda/Blas.cpp#L163-L186)。不自动切换精度，不静默重试失败样本。

更新项目代码后，先做一张基图、clean＋七种 S4 退化的 8 次前向检查：

```bash
DISABLE_ADDMM_CUDA_LT=1 CUDA_VISIBLE_DEVICES=1 python diagnose_severity.py \
  --config config/nuscenes_unidepthv2.yaml --source day-test \
  --profile config/severity.yaml --limit 1 --levels 4 \
  --conditions L N G LN LG NG LNG --repeats 1 --seed 42 \
  --pairs 8192 --min-pairs 128 --pair-gap 0.1 --failure-delta 0.10 \
  --save-vis --num-vis 1 --defer-summary --out runs/uni_ns_cuda_check1
```

看到 `[1/1]` 和 `Checkpoints complete` 后执行下节正式命令，保留同样的进程开关。如果仍报错，提供完整终端输出和 `nvidia-smi` 输出，继续查运行时；不要升级整个环境或改变退化强度。

适配器将 CUDA build、可见 GPU 和相关 BLAS 环境变量写入模型身份/恢复合同。原失败目录保留，新全量实验使用 `runs/uni_ns_full_gt_cublas`，不向旧合同追加；今后恢复也保持同样的开关。单图成功和此次绕行的 GPU 成功是两项不同的验证，后者仍需服务器确认。

## 5. CUDA 短检查通过后：完整正式扫描

正式实验使用完整官方 day-test，保持七退化条件、0.25 剂量步长、两次重复和 seed=42；不只跑 DAV2 最敏感的 LN/LNG。GT 候选对数、近似等深过滤和主阈值沿用当前默认 8192 / 0.1 / 0.10。

### nuScenes

```bash
DISABLE_ADDMM_CUDA_LT=1 CUDA_VISIBLE_DEVICES=1 python diagnose_severity.py \
  --config config/nuscenes_unidepthv2.yaml --source day-test \
  --all-split --limit 0 --profile config/severity.yaml \
  --level-step 0.25 --repeats 2 --seed 42 \
  --pairs 8192 --min-pairs 128 --pair-gap 0.1 --failure-delta 0.10 \
  --save-vis --num-vis 2 --defer-summary --out runs/uni_ns_full_gt_cublas

python tools/summarize_sweep.py --runs runs/uni_ns_full_gt_cublas \
  --no-fit --bootstrap 1000 --out runs/uni_ns_analysis
```

500 张时，112,500 次前向、113,000 行，与 DAV2 nuScenes 的设计一致。两张仅指保存可视化的数量，全部 500 张仍参与定量分析。完成后提供 `runs/uni_ns_analysis/` 整个目录以及 `runs/uni_ns_full_gt_cublas/` 中的 `run.yaml` 和 `contract.json`，用于核对与 DAV2 的样本、合成参数、输入审计及 GT 协议。

### RobotCar（nuScenes 通过后再执行）

```bash
DISABLE_ADDMM_CUDA_LT=1 CUDA_VISIBLE_DEVICES=1 python diagnose_severity.py \
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

兼容层已使用真实 Python 3.8.10 完成固定官方 ZIP 的导入、全部 Python 源码编译和纯源码测试；数值语句的 AST 对比确认未修改计算。另在本地 Python 3.10 / Torch 2.0.1 / Torchvision 0.15.2 下，成功真实导入兼容后的 UniDepthV2，核心包版本检查前后一致，并验证 TorchScript 类型标注可用。2026-11-04 已进一步验收用户服务器 Python 3.8.12 下的真实权重单张 CUDA 推理。

CPU、小张量及假 UniDepth 类测试还覆盖浮点输入、官方 infer 参数、严格权重加载、冻结、源码身份、原生深度语义和现有扫描/恢复/指标回归。

真实大权重由用户在服务器加载，本地不保存该权重。当前通过的是单张接口检查，尚无 UniDepth 失效结论。下一步先完成第 4b 节 CUDA 检查，再执行第 5 节 nuScenes 全量 GT 扫描，核对实测曲线后再推进 RobotCar。

官方来源：

- https://github.com/lpiccinelli-eth/UniDepth
- https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/models/unidepthv2/unidepthv2.py
- https://huggingface.co/lpiccinelli/unidepth-v2-vitl14/tree/main
