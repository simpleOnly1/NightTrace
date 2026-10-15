# NightTrace

**2026-10-15：强度扫描与失效边界。** 推荐使用 `diagnose_severity.py`：先用
`--render-only --save-vis` 审查 S0–S4 输入，再用 `--source day-test` 评估真实 GT。
详见 [运行指南与科学解释边界](docs/severity_boundary.md)。旧 `diagnose.py` 保留
原有参数分布与 DPT swaps，便于复现历史结果。

**Nighttime degradation propagation in monocular depth foundation encoders**

这是我们自己的单仓库：**ACDepth 的目录组织 + STEPS 的数据/评估协议 + 原版 DAV2-L + NightTrace 退化机理研究**。
不包含 `steps_original`、`rnw_original`、`md4all_original`、`third_party` 或任何并列运行仓库。

## 0. 我们正在做什么？

当前阶段不预设夜间网络或新的 loss，而是先固定一个可信的 DAV2 相对深度基线和可复现的评估协议，随后研究曝光衰减、噪声、白色光学眩光对 ViT-L F1/F2/F3/F4 的传播影响。否则评估或预处理的改变可能会被误当成方法改进。后续机制成立后只修改 `models/`、`losses/` 和 `analysis/`。

## 1. 项目内目录与职责

```text
NightTrace/
├── config/{robotcar,nuscenes}.yaml
├── data/{RobotCar-Night,nuScenes-Night}/       # 你已有数据，可软链接或外部绝对路径
├── datasets/{common,robotcar,nuscenes}.py     # STEPS 数据格式；严格测试样本 ID 排序
├── split_files/                              # 一次性导入 STEPS 官方划分文件
├── models/
│   ├── depth_anything_v2/                    # 官方 DAV2 源码在这里
│   ├── checkpoints/depth_anything_v2_vitl.pth
│   ├── depth_net.py                          # 相对逆深度/深度的唯一接口
│   └── pose.py
├── losses/photometric.py                     # 白天 self-supervised 几何约束
├── evaluation/{metrics,protocol,eval_robotcar,eval_nuscenes}.py
├── analysis/{degradation,relations,spread}.py
├── visualization/depth.py
├── scripts/{prepare,train_rc_day,train_ns_day,test_robotcar,test_nuscenes,...}.sh
├── tools/{prepare_sources,set_data_roots,check_setup,check_eval_data,check_parity,smoke_model}.py
├── train.py  trainer.py  test.py  evaluate.py
├── test_robotcar_disp.py  test_nuscenes_disp.py  show_eval_result.py
└── diagnose.py  diagnose_spread.py  tests/
```

`evaluation/metrics.py` 直接采用 STEPS 的 7 项指标与参数顺序；`evaluation/protocol.py` 保留 median scaling 与最近邻 resize 到 GT 的逻辑；两个 `test_*_disp.py` 保留 STEPS 风格的分数据集测试入口；`show_eval_result.py` 直接呈现对应指标表。原 STEPS 使用 MMCV + 旧 Lightning 的模型启动逻辑不会被塞进新项目，因为那部分已由 `trainer.py` 和 DAV2 接口承担。

## 2. 从零开始：服务器准备

假设你在 `/080226/Nighttime_Depth_Estimation/` 下工作：

```bash
cd /080226/Nighttime_Depth_Estimation
unzip NightTrace.zip
cd NightTrace

# 复用当前能跑 PyTorch 的环境；先确认 torch 与 CUDA，别无必要重装
python -V
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m pip install -r requirements.txt
```

**为什么需要一次性 source import？** 压缩包包含 NightTrace 的全部自有运行代码，但这里的离线构建环境不能直接获取 GitHub 官方 ZIP，且大型 DAV2-L checkpoint 不应重新分发。只需获取一次官方 DAV2 的 Python 包以及 STEPS 官方 split 文件，然后它们都直接成为 NightTrace 内部文件；运行时不需要 Git、不需要远程下载，也不存在第二个仓库。

服务器可连网：

```bash
bash scripts/prepare.sh
```

服务器断网：先自行下载两个官方 source ZIP，再上传；它们是**输入安装包**，不是未来的运行依赖：

- https://github.com/ucaszyp/STEPS/archive/refs/heads/master.zip
- https://github.com/DepthAnything/Depth-Anything-V2/archive/refs/heads/main.zip

```bash
bash scripts/prepare.sh \
  --steps-zip /path/to/STEPS-master.zip \
  --dav2-zip /path/to/Depth-Anything-V2-main.zip
```

如果服务器已有解压好的官方源码文件夹，也能一次性拷入本工程的标准位置：

```bash
python tools/prepare_sources.py \
  --steps-dir /path/to/STEPS-master \
  --dav2-dir /path/to/Depth-Anything-V2-main
```

这些目录只是导入时的源文件路径，**导入后训练无需依赖它们**。

**不会创建上游目录副本。** STEPS split 落在 `split_files/`，DAV2 Python 落在 `models/depth_anything_v2/`。已有 `depth_anything_v2_vitl.pth` 放在：

```text
models/checkpoints/depth_anything_v2_vitl.pth
```

这是**相对深度** DAV2-L 权重，不是 metric-depth 专用权重。文件不存在时，运行会明确报错。

## 3. 设置数据路径（为什么不重新复制数据？）

数据完全采用你提供的布局。`data/RobotCar-Night/<package>/rgb,depth,intrinsic.npy,train_split.txt,test_split.txt`；`data/nuScenes-Night/sequences/<clip>/file_list.txt,intrinsic.npy,images`，以及 `splits/` 和 `test/color,test/gt`。

数据如果不在仓库 `data/` 下，无需重复复制几十 GB，只需：

```bash
python tools/set_data_roots.py \
  --robotcar /ABS/PATH/data/RobotCar-Night \
  --nuscenes /ABS/PATH/data/nuScenes-Night
```

训练和测试 **一律使用已有 split 文件**，绝不随机重分。`split_files/` 用来备份/参考 STEPS 官方划分文件；运行时使用你数据目录已安装的 split。

```bash
python tools/check_setup.py --config config/robotcar.yaml --stage day
python tools/check_setup.py --config config/robotcar.yaml --stage test
python tools/check_setup.py --config config/nuscenes.yaml --stage test
python tools/check_eval_data.py --config config/robotcar.yaml --limit 3
python tools/check_eval_data.py --config config/nuscenes.yaml --limit 3
```

**目的：** 在正式耗费 GPU 前检查图像 ID、RGB、GT 路径以及 split 顺序。**GT 不参与训练。**

## 4. DAV2 与 STEPS 为什么不能直接接 `disp_to_depth()`？

STEPS 旧模型输出 sigmoid disparity `s∈(0,1)`，通过固定 min/max disparity 范围转为深度：

\[
\hat D_{STEPS}=\frac1{1/d_{max}+(1/d_{min}-1/d_{max})s}.
\]

DAV2-L 原版 DPT 给出 `q≥0`，较大响应通常表示近处，属于**相对逆深度表征**，并不是 STEPS sigmoid `s`。NightTrace 严格保留原始 `q`，单独构建用于 STEPS evaluator 的正相对深度：

\[
\boxed{\hat D_{rel}=1/\max(q,\epsilon)},\qquad \epsilon=10^{-3}.
\]

`predictions.npy` 存 `D_rel`，`raw_inverse_depth.npy` 存原始 q。**不采用逐图 min-max 归一化、不会在测试时用 GT 调整预测**。评估时才对每幅图做 STEPS median scaling：

\[
\hat D^{align}=\frac{\mathrm{median}(D^{GT}_{valid})}{\mathrm{median}(\hat D_{rel,valid})}\hat D_{rel}.
\]

`q→1/q` 是相对深度接口假设，不是“已经恢复米制深度”，必须先通过 `tools/smoke_model.py` 检查 q 的零值、近远次序和实际分布。若尝试 scale-shift alignment，它只能用于单独报告的诊断，不与 STEPS median 主结果混合。

## 5. 首先跑 DAV2 原生基线（为什么？）

**不要一上来训练。** 先确认原版权重能 forward、STEPS 式数据与尺度协议可用，这样后续改 loss/network 才能清楚定位增益来源。

```bash
python tools/smoke_model.py --config config/robotcar.yaml

# STEPS 同名入口，night 表示使用相应测试划分
python test_robotcar_disp.py night \
  --config config/robotcar.yaml \
  --out evaluation/rc_result --vis
python evaluate.py \
  --config config/robotcar.yaml --pred-dir evaluation/rc_result
python show_eval_result.py --file evaluation/rc_result/metrics.json

python test_nuscenes_disp.py night \
  --config config/nuscenes.yaml \
  --out evaluation/ns_result --vis
python evaluate.py \
  --config config/nuscenes.yaml --pred-dir evaluation/ns_result
python show_eval_result.py --file evaluation/ns_result/metrics.json
```

`predictions.npy` 的顺序严格是按数据中的测试 split 排序；测试入口不碰 GT，仅在 evaluator 中调用相应 `.npy`。RobotCar 使用 STEPS 的 **1152×640 中心裁剪**再 resize 到测试尺寸；nuScenes 遵循 split 图像列表顺序。评估阶段先依据预测索引的坐标元数据对齐 GT/RGB 视场，再将 prediction 最近邻插值到裁剪后的 GT shape，逐图算 median scale 和 7 项指标。nuScenes 默认 `(1e-5,60m)`，RobotCar 默认 `(0.1,40m)`。

**GT 坐标检查：** `data.gt_space: raw` 要求 GT 与原始 RGB 同网格，然后使用相同中心裁剪；已核实预裁剪 GT 才可设 `cropped`。缺少坐标元数据的旧预测需重新生成，尺寸不符直接报错。形状检查不能替代真实投影来源的核验，详见 [最小前期验证](docs/minimal_validation.md)。

## 6. 核对评估与 STEPS 数值实现（为什么？）

`evaluation/metrics.py` 是 STEPS 七项指标公式的原生端口。你可以使用同一个官方 STEPS zip 做**独立数学一致性对照**，原源码只在内存解析，**不会生成任何另一个项目目录**：

```bash
python tests/test_parity_local.py
python tools/check_parity.py --steps-zip /path/to/STEPS-master.zip
```

前者验证基础指标与 median scaling，后者从官方 `evaluation/eval_nuscenes.py` 提取纯函数，在 20 组随机预测上比较 7 项指标。它只证明**数值公式一致**；实际 RGB/GT 顺序、RobotCar crop、GT mask 仍须在真实数据上验证。

## 7. 白天自监督基线与 NightTrace 退化诊断

下列白天训练仅为可选基线，当前冻结模型诊断不需要它。白天训练提供 DAV2-L + DPT + PoseNet 的自监督基线，loss 使用 daytime view synthesis（最小重投影、SSIM+L1、automask、smoothness）。训练只使用白天 RGB 视频序列与 K，没有 GT depth/pose。`train.py` 与 `trainer.py` 采用 ACDepth 风格职责分离。

```bash
# 一次小型 GPU 检查，不会跑完整训练
python train.py --stage day --config config/robotcar.yaml --smoke

# 完整白天自监督训练（可根据显存修改 batch_size）
bash scripts/train_rc_day.sh
# 或
bash scripts/train_ns_day.sh
```

现在真正优先运行的科学实验是 **冻结 DAV2 的 F1–F4 退化响应**：

```bash
bash scripts/diagnose_robotcar.sh
python diagnose_spread.py --config config/robotcar.yaml \
    --sample-list runs/diagnose_robotcar/samples.txt --out runs/diagnose_spread
```

主诊断默认跨完整白天 split 分组筛选 512 张，2 个随机实例，运行完整 L/N/G 八组配对实验；传播脚本使用独立的紧支撑局部探针。结果仅描述关系/输出一致性和 DPT oracle 恢复潜力。详细命令、统计定义、GT 坐标修复及真实夜间验证边界见 [最小前期验证](docs/minimal_validation.md)。

**暂时没有实现夜间损失。** 当前 `python train.py --stage night ...` 会明确提示尚未确定，不会假装完成真实夜间训练。这是当前研究阶段有意保留的科学边界。

## 8. 本地无大模型测试

```bash
python tests/test_diagnosis_protocol.py
python tests/test_core.py
python tests/test_dataset.py
python tests/test_geometry.py
python tests/test_model_adapter.py
python tests/test_eval_contract.py
python tests/test_source_import.py
python tests/test_parity_local.py
```

`test_model_adapter.py` 用小型假 DAV2 验证接口/梯度，**不代表已经用 335M DAV2-L 权重完成 GPU forward**。这一步要在你服务器运行 `tools/smoke_model.py` 完成。

## 9. 引用与边界

使用 STEPS 的数学评估/组织规范并适配成项目内 native 模块，不再保留老版训练框架的 MMCV/Lightning 依赖。官方 DAV2 包与 STEPS split 由一次性工具直接导入适当的最终目录，参见 [ATTRIBUTION.md](ATTRIBUTION.md)。正式发表指标前完成真实数据集上的样本、裁剪、评估等价性审计，参见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。

