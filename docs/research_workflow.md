# NightTrace 研究执行手册

本研究先冻结 DAV2-L，研究退化剂量与几何错误的关系，再定位错误传播。
当前不训练新模型。输入图已经能显示明显的 L/N/G 渐变，接下来应进入真实 GT
评价，不能继续把“合成看起来够强”当成“已经测得失效边界”。

## 1. 数据与结论的边界

按照现有 RNW 加载与划分使用数据：**训练集没有 GT，测试集有 GT**。

| 数据 | 用途 | 可以报告 | 不可以报告 |
|---|---|---|---|
| day-train，全部合格基图 | 大规模输入与预测稳定性分析 | 亮度/饱和/眩光影响面积、预测漂移、clean 排序一致性 | 真实几何正确率、真实失效临界点 |
| day-test，官方完整测试集 | 配对剂量—真实错误曲线 | GT 排序错误、AbsRel、错误增量、操作性越阈区间 | 把重复合成计作新的独立 GT 场景 |
| night-test，官方完整测试集 | 真实夜间外部检验 | 冻结模型的实际夜间性能 | 给天然夜景赋予未经标定的合成 S 值 |

仓库随附 daytime test 划分为 RobotCar 159、nuScenes 500 条；以服务器实际
加载的 `available` / `samples.txt` 为准。没有新增 GT 时，不能把真实 GT 拟合
声称为“数千张独立基图”。可在大量训练基图研究稳定性，在完整测试集研究正确性。
同图多剂量、多噪声重复提高配对测量精度，不增加独立场景数。

测试集上探索曲线本身可以作为描述性研究，但若根据这些曲线修改阈值/退化参数，
再次运行同一测试集仍是探索，不能称独立确认。确认性结论需要预先冻结设计，
或按真实 sequence/scene 分出独立探索和确认组，并如实报告减少后的有效样本数。

## 2. 阶段、目标与通过条件

| 阶段 | 做什么 | 产物与通过条件 | 当前实现 |
|---|---|---|---|
| 0 协议审计 | 检查真实权重、GT、裁剪、投影和有效范围 | GT audit/overlay、真实模型有限输出；人工确认投影来源 | 已有审计及 smoke 工具，不能自动证明标定正确 |
| 1 输入校准 | 固定基图/模板/位置/种子，检查八条件 S0–S4 | inputs.png 与实际剂量指标；参数冻结 | 已实现；用户提供的图已完成视觉检查，保留对应配置即可 |
| 2 GT 剂量—响应 | 先小批验证管线，再完整 day-test 密集扫描 | 实测点、组 bootstrap、拟合曲线、越阈和联合提前量 | 本次加入密集网格、恢复、分片、独立汇总与拟合 |
| 3 大规模稳定性 | 对全部合格 day-train 基图运行相同设计 | 无 GT 的稳定性分布、异常案例、场景覆盖统计 | 已实现；不输出伪 GT 失效曲线 |
| 4 几何与机制定位 | 按第 2 阶段实测区间取稳定/过渡/高错误样本 | 分几何线索错误与逐层受控干预证据 | 现有 feature/DPT swap 仅是部分探针；系统流程仍需开发 |
| 5 外部检验与报告 | 对真实夜间测试集评估、复核两数据集差异 | 完整样本/分组数、未越阈比例、真实夜间结果、限制 | 基线评价已有；综合研究报告需真实实验后完成 |

顺序：0 → 1 → 2 的小批验证 → 2 的完整实验；3 可独立运行；4 必须依据实测
区间；5 的真实夜间基线可先跑，最终解释在 2–4 后完成。无需先跑百万次推理才发现
GT 配准错误。下面所有命令在仓库根目录执行，输出目录均为新目录。

## 3. 获取代码

```bash
cd /080226/Nighttime_Depth_Estimation/NightTrace
git status --short
git fetch origin
git switch feat/severity-boundary-20261015
git pull --ff-only
git log -1 --oneline
```

如果 switch 因本地配置修改而停止，先保留这些配置再处理；不要用 reset --hard
覆盖数据路径。修改只在上述研究分支，PR #2 基于前一研究分支，尚未合入 main。

## 4. 下一步立即执行：阶段 0 与阶段 2 小批验证

### 4.1 审计 day-test（不推理模型）

```bash
python tools/check_eval_data.py --config config/robotcar.yaml \
  --split day --overlays 12 --out runs/rc_stage0_day_gt
```

检查 `summary.json` 的 checked/available，打开 `*_aligned_sparse_GT.png` 和
`*_raw_crop_window.png`，看稀疏点是否与物体/道路对应。GT/FOV 报错时先处理
数据来源，不能 resize GT 强行通过，也不能仅因尺寸相等就断言标定正确。

若尚未确认当前真实 DAV2-L 权重可正常推理，执行一次：

```bash
python tools/smoke_model.py --config config/robotcar.yaml
```

### 4.2 八张测试基图的端到端验证（有 GT，默认八条件）

```bash
python diagnose_severity.py --config config/robotcar.yaml \
  --source day-test --profile config/severity.yaml \
  --limit 8 --levels 1 2 3 4 --repeats 1 \
  --save-vis --num-vis 2 --failure-delta 0.10 \
  --out runs/rc_stage2_smoke8
```

本步加载模型；默认条件为 clean/L/N/G/LN/LG/NG/LNG。每张基图 29 次前向，
用来确认模型、GT 指标、图表和参数记录全部贯通，**不据此宣布科学结论**。
检查 `per_image.csv` 的 `gt_ordinal_pairs`（具体字段以表头为准）、
`gt_ordinal_error`、`delta_gt_ordinal_error` 与 `gt_abs_rel`；有效像素对不足时
NaN 是缺失，不是零错误。查看 `summary.csv` 的 groups 数与有限值数量。
查看 qualitative 面板；跨条件色标固定，不能用每幅图自动拉伸后的颜色判断失效。

如果此前删除了 pilot16.txt，这些命令不依赖它。程序会新建 `samples.txt`。
旧 day-train 清单不能用于 day-test，程序会拒绝跨 split ID。

## 5. 阶段 2 完整 GT 实验：扫描与统计分开

先冻结 profile、几何范围、剂量步长、seed、repeat 数、分组方式和失效阈值。
`.10` 仅是预先声明的探索阈值：GT 排序错误相对本图 clean 增加 10 个百分点。
它不是普适失效定义。应并报 clean 的绝对错误，避免原本错误的预测看似很稳定。

```bash
python diagnose_severity.py --config config/robotcar.yaml \
  --source day-test --all-split --limit 0 \
  --profile config/severity.yaml --level-step 0.25 \
  --repeats 2 --seed 42 --failure-delta 0.10 \
  --defer-summary --out runs/rc_stage2_full_gt

python tools/summarize_sweep.py \
  --runs runs/rc_stage2_full_gt \
  --bootstrap 1000 --penalty 0.1 --out runs/rc_stage2_analysis
```

第一条命令运行推理并逐图写检查点，不在内存中积攒整个实验。`--all-split`
表示使用完整官方 day-test，不经过时间间隔/缩略图重复过滤。第二条不加载模型，
合并检查点、计算统计和曲线。重新分析 bootstrap/平滑强度时用新的 analysis 目录，
不用重新推理。改变 failure-delta 会影响合同，应建立新的推理实验，不混写。

nuScenes 将 config 改为 `config/nuscenes.yaml`，所有 rc 输出目录改为 ns，
先重复相应 GT 审计和小批验证，再完整扫描。两个数据集分别报告，不直接混均值。

中断后保持同一代码、数据、权重、配置和参数，只在原扫描命令末尾加 `--resume`。
程序核对合同及已完成样本/GT/已用模板哈希，跳过整张完成的基图；半张未完成会重算。
旧版无检查点的目录不能续跑。修改条件/强度/权重/关键代码会拒绝续写，避免混实验。
每个输出目录只允许一个运行进程。

密集扫描默认不保存海量图；输入视觉审查使用已经通过的 S0–S4 图即可。

### 5.1 每份输出回答什么

| 文件（analysis 目录） | 用途 |
|---|---|
| per_image.csv / parameters.csv / parameters.jsonl | 逐图、逐重复、逐条件、逐剂量的指标和实际参数 |
| summary.csv / response_curves.png | 实测剂量点的组均值与点态区间，必须与拟合并看 |
| group_curves.csv | 重复先在图内平均、图片再在组内平均后的数据 |
| boundaries_per_image.csv / boundaries_group.csv | 网格上的首次越阈区间、未越阈及回落标记 |
| fitted_response_curves.png / fitted_curves.csv | 正则化分段线性拟合与组 bootstrap 重拟合区间 |
| fitted_boundaries.csv | 拟合越阈点、回落、bootstrap 中观察到越阈的比例 |
| joint_boundary_shifts.csv | 同组配对的单因素边界 − 含该因素的 LN/LG/NG/LNG 边界；正值表示联合更早 |
| interactions_summary.csv | LN/LG/NG 二阶、LNG 三阶误差尺度交互；与总联合伤害区分 |
| curve_status.csv | 有效完整组数、排除组数、是否剂量/组数不足 |
| analysis.json / input_sha256.json / template_sha256.json | 推理合同、分析参数与代码/输入哈希 |

`--level-step .25` 是 S0–S4 间 16 个非零剂量，不是只有四个等级中心。
S 是指定实验路径坐标；L 曝光、N shot/read、G 增益对应关系见
[合成协议](severity_boundary.md)。同 S 的单因素和联合条件共享对应因素参数。
联合越阈更早不自动等于超加性交互，后者需看完整八条件交互。
目前扫描的是三条轴与四条同步联合路径，不是三维强度空间的完整失效曲面。

拟合不强迫 S 形或单调增长，不超出测量范围。平滑强度为预先指定的 penalty，
不是从测试集挑最漂亮曲线；原始点始终保留。GT 错误增量曲线的 S0 固定为 0。
失效比例是各重复是否越过阈值的指标，经图内、组内平均后的比例，S0 为 0；
其拟合值限制在 [0,1]。10% 与 90% 越阈是操作性的群体过渡标记，
不是自动证明中间有统一“相变”。若未跨越 90%，过渡上界就未观测到；若反复跨越，
必须报告非单调，不能硬凑单一耐受区间。

拟合的区间是点态组 bootstrap 区间。`conditional_p025/p975` 是
**仅在该次 bootstrap 中观察到越阈时的条件分位数**；必须同时报告
`bootstrap_observed_fraction`。有未越阈重采样时，这不是完整无条件临界点置信区间，
不把未越阈者填成 S4。联合提前量同样条件于两个边界都被观察到。
没有同时推断或多重比较校正，边界估计仍依赖设计/阈值/平滑设定。

拟合仅用全部请求条件和剂量都有值的共同组，保障配对比较；`curve_status.csv`
报告排除量。完整组少于 2 时拒绝给出拟合，不把大量帧伪装成很多独立场景。
RobotCar 时间块与 nuScenes recording 是代理分组，未保证独立；有真实 scene/sequence
映射时，用 `--groups-csv mapping.csv`（列 id,group）替换，并在分析前固定。

## 6. 阶段 3：大量无 GT 基图与分片

先用小批真实运行测耗时，再决定 K 和 repeats。假设基图数为 B，非零剂量数 K、
重复数 R，默认七退化条件：前向次数为 `B × (1 + 7KR)`，CSV 行数为
`B × R × (1 + 7K)`。K=16、R=2 时每图 225 次前向：159 张为 35,775 次、
500 张为 112,500 次、5,000 张为 1,125,000 次。分片与恢复解决调度/重算问题，
不会让 DAV2-L 计算免费。先报告基图数和组数，不只报告合成图总量。

单 GPU 可按四片依次执行；不是同时启动四个进程争抢一张卡：

```bash
for shard in 0 1 2 3; do
  python diagnose_severity.py --config config/robotcar.yaml \
    --source day-train --limit 0 --profile config/severity.yaml \
    --level-step 0.25 --repeats 2 --seed 42 \
    --num-shards 4 --shard-index "$shard" --defer-summary \
    --out "runs/rc_stage3_train_part${shard}"
done

python tools/summarize_sweep.py \
  --runs runs/rc_stage3_train_part0 runs/rc_stage3_train_part1 \
         runs/rc_stage3_train_part2 runs/rc_stage3_train_part3 \
  --bootstrap 1000 --out runs/rc_stage3_train_analysis
```

day-train 的 `--limit 0` 是全部经过现有时间间隔/重复过滤的合格基图，
不是所有原始相邻帧。实际数量看 samples.txt；不为凑数关闭相关帧过滤。
多 GPU 可给不同片设置不同 CUDA_VISIBLE_DEVICES，片数、全局清单和配置需一致。
各片目录内 samples.txt 是全局清单，分配按全局索引取模；检查点只包含该片样本。
恢复某片时运行该片原命令并加 `--resume`，不要把整个 for 循环盲目加 resume，
因为还没创建的片不存在合同。合并会拒绝缺片、重复片、合同不一致或缺少检查点。

这里输出的是 `clean_ordinal_disagreement`、`depth_drift` 等稳定性曲线。
训练集没有 GT，所以不会生成 fitted GT 边界。若大量基图稳定、有限 GT 集也正确，
两类证据相互补充，但仍分别说明范围和局限。

## 7. 阶段 4：几何线索和错误形成阶段——还缺什么

当前 GT 全局像素对能检测近远排序错误，不能自动回答“遮挡边界、接地点、
薄物体、平面结构哪一个先坏”。所需下一轮开发应针对真实数据可支持的标注：

1. 按真实 GT 曲线选稳定/过渡/高错误区间，同时保留未失效和回落样本。
2. 定义/审计关系或区域标签；稀疏 GT 不足的区域标记未知，不能插值后冒充密集 GT。
3. 对每类线索建立误差随剂量变化的曲线，并控制 clean 基线是否已错。
4. 在同一批配对样本上记录编码器层及解码阶段，运行 clean→degraded 恢复和
   degraded→clean 扰动、单阶段和联合阶段干预，做必要的干预对照。
5. 比较干预前后的真实 GT 误差，而不只比较 feature cosine 或可视化是否更像 clean。

已有 `diagnose.py` 的特征/DPT swaps 是部分机制工具，其旧合成参数分布不等于
新 severity 路径。尚未实现统一的密集剂量＋几何标签＋全部阶段干预管线，
不能把“F3/F4 换回后恢复”直接解释为“错误最早在 F3/F4 形成”。干预恢复支持
该表示对输出有影响，首次产生错误仍需更细的受控证据。
本阶段不提供假装已实现的执行命令，也不根据预想结论改模型。

## 8. 阶段 5：真实夜间基线与最终报告

先确认配置 `eval.split: night`（test.py 按配置选 split），然后：

```bash
python tools/check_eval_data.py --config config/robotcar.yaml \
  --split night --overlays 12 --out runs/rc_stage5_night_gt
python test.py --config config/robotcar.yaml --out runs/rc_stage5_night_pred
python evaluate.py --config config/robotcar.yaml --pred-dir runs/rc_stage5_night_pred
```

nuScenes 用其配置和独立 ns 目录。天然夜景没有对应 clean 场景/已知 LNG 剂量时，
这一步提供外部性能与误差案例，不构成合成临界点的直接标定。

论文/报告逐条回答：

- 是否观察到耐受/过渡：报告测量范围、阈值、绝对和增量错误、未越阈比例。
- 边界在哪里：同时给经验采样区间、描述性拟合结果、分组不确定性和实际剂量指标。
- 联合是否更早：配对边界位移与完整八条件交互分开报告，不默认答案为是。
- 哪些线索先坏、在哪一阶段：需要阶段 4 的专门结果；当前代码与输入图还不能回答。

## 9. 本次软件验证与尚未运行的实验

CPU 契约测试覆盖八条件默认值、密集剂量、S0 恒等、固定模板/几何、GT 排序与
缺失、非单调越阈、中断恢复、输入/合同变化拒绝、分片与整跑等价、缺片/重片拒绝、
拟合数学与联合配对位移。假模型只验证流程，不能代替服务器 DAV2-L 实验。
开发环境未提供真实服务器数据、权重和 Flare7K++ 素材，尚无实测失效边界。

```bash
python -m pytest tests/test_severity.py tests/test_sweep_execution.py \
  tests/test_diagnosis_protocol.py tests/test_gt_audit_and_flare.py -q
python tests/test_eval_contract.py
python tests/test_dataset.py
python tests/test_model_adapter.py
```
