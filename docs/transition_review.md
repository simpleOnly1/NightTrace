# 下一小步：复核 LN / LNG 过渡区（阶段 2.1）

本轮只回答：**过渡和回落在不同退化实现下是否持续，哪些样本值得进入机制实验？**
先核对已有数据，再做约 12 张基图的小规模复测。不训练，不开展全训练集扫描，
不把本轮称为独立验证或机制发现。长期主线仍是失效规律 → 机制证据 → 针对性方法。

## 已完成的离线复核

2026-10-20 对用户提供的 RobotCar 阶段 2 `per_image.csv` 进行检查：
159 张图，3 个时间组（31/65/63），2 个重复，35,934 行。无重复键、缺失配对网格，
GT 排序增量与配对 clean 一致。输入 SHA256 为
`08d70a17553918fb3238c432fbd9ba8f051382aec703f45e75ed602e238583de`。
这些是 CSV 内部检查，不替代 GT 投影、权重和原实验合同审计。

在错误增量 .10（10 个百分点）阈值下：

- LNG：重复 0 首次越阈在 (3.00, 3.25]，重复 1 在 (2.75, 3.00]。
- LN：两个重复的组均值都在 (3.00, 3.25] 首次越阈并回落。
- LN：移除第一个时间组后，剩余组均值的首次越阈变为 (3.75, 4.00]。

因此目前可以报告过渡现象，不能锁定一个不依赖场景/退化实现的精确临界值。
默认同时报告 .05/.10/.15 的事后敏感性，不挑选最漂亮的阈值。
每次 repeat 同时改变眩光模板、几何位置及噪声；这不是纯噪声重复实验。

## 1. 更新研究分支并生成清单（CPU，无模型）

在服务器仓库根目录执行。工作区有修改时先保留，不使用 reset --hard。

```bash
git status --short
git fetch origin
git switch feat/severity-boundary-20261015
git pull --ff-only

python tools/audit_transition.py \
  --csv runs/rc_stage2_analysis/per_image.csv \
  --select-per-group 4 --selection-seed 42 --min-gap-seconds 1 \
  --out runs/rc_stage2_transition_review
```

产物：`report.md`（摘要）、`boundary_sensitivity.csv`（逐组、留一组、逐重复）、
`per_image_boundaries.csv`、`selected_cases.csv`、`selection_coverage.csv`、`samples.txt`。

清单固定使用 .10：优先未越阈对照，再选 LN 回落、重复不一致、LNG 持续越阈。
类别允许重叠，每张图只选一次；同类用固定哈希顺序，能解析时间戳时保持同序列
至少 1 秒间隔，不强行补齐。当前文件选出 **11 张**（3/4/4），并保留 2 张 LNG
未越阈对照。选择依赖已观察到的错误，**不用于估计总体失败率或证明泛化**。

## 2. 运行 11 张图的小复测（需要 GPU 和真实 DAV2-L）

沿用原实验 `runs/rc_stage2_full_gt/run.yaml` 中的配置、素材库、权重、GT 范围、
profile、seed 42 及 GT pair 策略。若服务器原实验不同，应先对齐，不能照抄后混比。
本轮只新增离线工具，不修改已有合成与推理实现，不需要重跑完整 159 张实验。

```bash
python diagnose_severity.py \
  --config config/robotcar.yaml --source day-test \
  --profile config/severity.yaml \
  --sample-list runs/rc_stage2_transition_review/samples.txt \
  --conditions LN LNG \
  --levels 2 2.5 2.75 2.875 3 3.125 3.25 3.5 4 \
  --repeats 4 --seed 42 --pairs 8192 --min-pairs 128 --pair-gap 0.1 \
  --failure-delta 0.10 --save-vis --num-vis 12 \
  --defer-summary --out runs/rc_stage2_transition_probe
```

每张图 clean 自动计算一次，外加 2 条件 × 9 剂量 × 4 重复，共 73 次前向。
11 张是 **803 次前向**，不是再跑 35,775 次。实际清单若为 B 张，预算为 73B。
CSV 包含每重复的 clean，因此预期 836 行（76B）。

- S2 保留较低强度锚点，S4 保留高强度/回落检查点。
- S2.875 与 S3.125 加密现有过渡区；其他 7 点与原数据重叠。
- repeat 0/1 检查旧结果能否复现；2/3 提供新的退化实现。
- `qualitative/` 保存所有选中图的第 0 重复可视化；其余重复依靠完整指标记录。
- 同一目录只运行一个进程。若中断，保持同一命令与环境并加 `--resume`。

## 3. 汇总并检查旧结果复现（CPU）

```bash
python tools/summarize_sweep.py \
  --runs runs/rc_stage2_transition_probe \
  --no-fit --bootstrap 1000 --out runs/rc_stage2_transition_probe_analysis

python tools/audit_transition.py \
  --csv runs/rc_stage2_transition_probe_analysis/per_image.csv \
  --reference runs/rc_stage2_analysis/per_image.csv \
  --select-per-group 0 \
  --out runs/rc_stage2_transition_probe_review
```

不以平滑拟合交点决定此次结论。CSV 重叠检查默认容差 1e-6（错误率单位），
比较 GT 排序及增量，同时检查 group、parameter_seed、GT pair 数（若存在）。
11 张图应有 330 条重叠记录。`audit.json` 的 `replay.status` 应为 `pass`；
不符时输出 `replay_differences.csv` 并以非零状态退出，先查数据/参数/软件差异。
此检查只证明所比较的输出一致，不证明输入、权重或素材哈希完全相同。

## 本轮结束时如何决定下一步

1. **旧点无法复现：**先修复可复现性，不解释新增剂量或追加机制模块。
2. **新重复下越阈/回落变化很大：**保留分布与未越阈情况，再区分模板/位置与噪声作用。
3. **若能找到跨重复保持模式的样本：**锁定其中的稳定、过渡、高错误剂量及对照，
   再开发同一 severity 协议下的 clean→degraded 与 degraded→clean 特征干预。

本轮的交付是可复现的小样本、逐重复区间和下一阶段样本依据。不会据此回答
“错误最早在哪一层”，也不会将选中 11 张的统计代替全部 159 张结果。
外部数据集与真实夜间验证仍需要后续独立推进。

运行结束后提供两个文件夹：`rc_stage2_transition_probe_analysis` 与
`rc_stage2_transition_probe_review`，以及 probe 中的 `run.yaml` 和几张 qualitative 图。
训练集无 GT，后续训练或选方法不能把这些已用于探索的测试样本当成独立确认集。

## 软件验证

```bash
python tests/test_transition_audit.py
```

测试覆盖非单调/未越阈、组等权、缺失/重复/非有限值拒绝、配对 delta 校验、
可复现选择与时间间隔，以及重叠指标/元数据不符检测。本地尚未执行 GPU 复测。
