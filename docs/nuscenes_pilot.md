# 当前原则与下一步：DAV2 / nuScenes 小批实验

更新：2026-10-27。RobotCar＋DAV2 已有先导结果；nuScenes 结果尚待服务器执行。本轮只新增执行工具，不声称已经完成新模型适配或 GPU 实验。

## 研究原则

分别测量 DAV2、UniDepth V2、Marigold V2 在 RobotCar 和 nuScenes 上的退化响应与边界位置，然后分析各自的失效原因。**不预设不同模型具有共同原因，也不把找出共性作为研究成功条件。** 相同原因可以支持共享的优化思路；不同原因也可以提供有价值的模型比较和设计启发。多个模型在同一条件下变差，并不能单独证明失效机制相同。

顺序保持为：失效响应 → 模型各自的机制分析 → 针对性优化及真实夜间验证 → 选择一个下游方法检验迁移。优化收益需要与同数据、同预算的普通适配比较。训练仍遵循 RNW 划分：train 无 GT，test GT 用于评价。

先导结果用于选取研究方向，保留非单调和未越阈情况；其他模型首先保留八条件粗扫描，不预设 LN/LNG 一定也是它们最危险的组合。

## 本轮唯一执行目标

保持 DAV2 不变，跑通 nuScenes 的 RGB/GT 加载、八条件输入、GT 指标及汇总。小批结果通过后再安排完整 nuScenes day-test；随后逐个接入其他模型。

新增入口仅支持 `data.name: nuscenes` 和 `model.name: dav2_vitl`，避免只改配置名却仍运行 DAV2。它调用既有数据检查、推理和汇总程序，不改变原有合成参数、模型权重或指标实现。

## 1. 更新服务器代码

```bash
cd /080226/Nighttime_Depth_Estimation/NightTrace
git status --short
git fetch origin
git switch feat/severity-boundary-20261015
git pull --ff-only
```

如果 Git 因本地配置修改而停止，保留这些配置再处理，不用 reset --hard。确认 config/nuscenes.yaml 中的数据根目录、GT 空间、权重和 Flare7K++ 路径仍对应服务器现有文件。

## 2. 数据与 GT 检查：不运行模型

```bash
python tools/run_nuscenes_pilot.py --stage audit
```

输出在 `runs/ns_dav2_pilot/gt_audit/`。此步检查完整 day-test 的加载、GT 形状/视野、有效范围，并保存八张 overlay。实际 checked/available 由服务器数据决定，不假装已经检查了固定 500 张。

查看 `summary.json` 和 `*_aligned_sparse_GT.png`。自动通过说明尺寸和有效 GT 检查通过，投影/时间对齐仍需结合 overlay 与数据来源检查。

## 3. 八张图、八条件：运行模型并自动汇总

```bash
python tools/run_nuscenes_pilot.py --stage pilot
```

默认选择最多八张基图；clean 自动加入，七退化为 L/N/G/LN/LG/NG/LNG，正强度为 S1/S2/S3/S4，一次重复，seed=42，8192 候选 GT 对。使用既有 config/severity.yaml；若选满八张，为 **232 次前向、232 行**。

工具先核对完整 nuScenes audit 存在，再调用 diagnose_severity.py，成功后调用 summarize_sweep.py（--no-fit）。推理或汇总失败时返回非零状态，后续步骤停止。

输出：

- `runs/ns_dav2_pilot/sweep/`：run.yaml、样本清单、逐图检查点、两张基图的输入/深度面板。
- `runs/ns_dav2_pilot/analysis/`：逐图指标、实测响应曲线、参数和越阈记录。
- `runs/ns_dav2_pilot/*_status.json`：对应阶段的命令与执行状态。

执行状态 completed 只表示程序运行成功，不是科学结论已验证。八张图用于入口验证与初步观察，不能据此宣称跨数据集规律成立。

## 查看命令、恢复和重新分析

```bash
# 只显示命令和预算，不运行、不创建输出
python tools/run_nuscenes_pilot.py --stage pilot --dry-run

# 推理中断：保持参数、数据和代码一致，校验检查点并恢复
python tools/run_nuscenes_pilot.py --stage pilot --resume

# 只重新汇总到一个新的分析目录，不运行模型
python tools/run_nuscenes_pilot.py --stage summarize \
  --analysis-out runs/ns_dav2_pilot/analysis_review2
```

已有非空分析目录会被拒绝写入；若上一轮汇总失败留下部分文件，恢复时也指定新的 `--analysis-out`。自定义输出父目录时，各阶段保持同一个 `--out`；可用 `--config` 指向服务器单独保存的 nuScenes 配置。

## 运行后提供什么，以及下一步是什么

提供 `gt_audit/summary.json`、少量 GT overlay、`sweep/run.yaml` 和 `analysis/`。先核对样本覆盖、有效 GT 对数、八条件是否齐全和主要指标是否有限；确定入口正常后再运行完整 nuScenes day-test。

当前模型接口仍是 DAV2 专用。UniDepth V2、Marigold V2 的输出含义和预处理将在各自接入时核验；不能仅修改 model.name 就宣称完成了新模型实验。后续比较统一 GT 排序及相对各模型 clean 的错误增量，并同时报告 clean 绝对错误。各数据集分别报告，不要求所有模型具有相同失效原因或相同临界 S。

## 本轮软件验证范围

标准库测试检查阶段隔离、八条件与剂量配置、错误模型/数据集拒绝、完整 audit 前置条件、子进程失败停止和执行日志。没有在本地运行真实 nuScenes、DAV2 权重或 GPU 推理；服务器结果仍待提供。
