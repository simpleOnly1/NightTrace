# NightTrace：从弱退化诊断到失效边界扫描

**当前完整步骤、阶段目标和新增分片/拟合命令见 [研究执行手册](research_workflow.md)。**
本文保留合成与指标的详细定义。

`diagnose_severity.py` 是本阶段推荐入口。旧 `diagnose.py` 保留原有参数分布和
DPT oracle swap，用于复现旧实验；不要把旧参数范围当成新的严重度扫描。

## 为什么改

旧 Flare7K 路径把完整模板重采样进 `2 * radius` 的窗口。默认 radius 仅为图像
短边的 .025–.15，因此完整模板只有短边的 5%–30%，模板内部亮核可能更小。
源还可能落到画外。该映射能解释“小光点”的工程风险，但没有原始模板和
parameters.csv，不能断言它是所附样例的唯一原因。

新入口把完整模板的半宽范围改为短边的 .35–.65，默认在画内取 1–2 个源。
这是明确的受控压力实验先验，不是实测夜间光源分布。形状、半宽、位置、方向、
模板、可选反射模板、源数和源间相对振幅在同图同 repeat 的所有剂量中固定；
G 只扫描辐射增益。可在 `config/severity.yaml` 的 geometry 改覆盖范围，
但更换 geometry 要单独运行、单独解释，不能说只改变了亮度。

| 剂量 | S0 | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|---:|
| 线性 RGB 曝光系数 | 1 | .4 | .12 | .025 | .005 |
| shot_peak（越小噪声越强） | 无新增噪声 | 400 | 100 | 25 | 6 |
| read_sigma（线性域） | 0 | .004 | .012 | .035 | .08 |
| glare_gain × 固定源振幅 | 0 | .5 | 2 | 8 | 32 |

S4 是极端压力探针。**这些数值没有用真实夜景频率或 DAV2 错误率标定**。
N 同时改变 shot/read 两项，是预先声明的一维联合噪声路径，不应声称分离了
两类噪声的因果效应。LNG 是同档三因素联合路径，不是 4³ 全网格。

S0 严格返回原图。小数档位在相邻级间按 log exposure、线性 1/shot_peak、
线性 read_sigma 和线性 glare_gain 插值。相同 seed 和独立 read-noise 流可复现；
Poisson 保持正确边际分布，但不同强度/信号下不是逐像素完全相同的光子实现。
跨 CPU/CUDA/PyTorch 版本不保证逐位一致，run.yaml 保存软件版本。

渲染仍是近似线性 RGB + 灰度 Flare7K 形状 + shot/read noise，再转回 sRGB。
模板仍遵循原有 128×128 灰度解码和共同反射变换；没有声称 RAW/ISP 或彩色
眩光的物理精确复现。合成发生在数据集 crop/resize 后、DAV2 预处理前。
旧参数分布不被覆盖，避免旧结果失去可复现性。

## 先检查输入，不加载模型

仓库分支 `feat/severity-boundary-20261015` 基于 PR #1 的 `3560691`。
从服务器已有仓库运行（有本地改动时先保留，不要覆盖）：

```bash
git fetch origin
git switch feat/severity-boundary-20261015
git pull --ff-only

# 复用已有16张正常基图清单；清单会被完整使用，不会再按limit截断。
python diagnose_severity.py --config config/robotcar.yaml \
  --profile config/severity.yaml \
  --sample-list runs/rc_select256/pilot16.txt \
  --render-only --save-vis --num-vis 16 \
  --out runs/rc_severity_inputs16
```

没有 pilot16.txt 时，改为 `--limit 16`，程序会从允许的 day_train 中分组抽样。
nuScenes 使用 `config/nuscenes.yaml` 和独立输出目录；可复用它已有的 pilot 清单。
配置中的 Flare7K 路径必须存在，否则直接报错。`--analytic-flare` 是明确选择
另一个后端，仅用于工程调试，不应拿它冒充 Flare7K 实验。

查看 `qualitative/*/inputs.png`：每行一个条件，每列 S0–S4，旁边显示亮度中位数、
近黑面积、饱和面积、可见 flare 覆盖、噪声 RMS。输入以 float 送入网络，
PNG 仅是量化到 8-bit 的预览。`--prepare-only` 不加载 flare/模型；若选择
day-test，会检查真实 GT 的存在与 FOV。

`per_image.csv` 的输入审计还包括亮度 1/10/50/90/99 分位、RGB MAE、
饱和面积变化、线性 flare 支持面积。噪声 RMS 来自实际输出与同 L/G 参数的
无新增噪声输出之差，是经过 sRGB 与截断后的残差，不是传感器读噪标定值。
`glare_visible_area` 比较无噪声、有/无 G 的图像，RGB 平均差超过 .02 的面积，
不让随机噪声冒充眩光影响。

若最高请求剂量的可见 flare 覆盖 <1%、N 的 RMS <.02、L 的亮度中位数 >.1，
或近黑/饱和面积 >95%，生成 `audit_warnings.csv`。这些阈值是审查提示，
不删除样本、不自动调参、不重抽模板；无提示时该文件为空。

## 再跑预测漂移或真实 GT 评价

不带 GT 的日间训练基图，只能测预测变化：

```bash
python diagnose_severity.py --config config/robotcar.yaml \
  --profile config/severity.yaml \
  --sample-list runs/rc_severity_inputs16/samples.txt \
  --save-vis --num-vis 4 --out runs/rc_severity_drift16
```

**GT 路径必须另选 day-test**。不能把训练基图的 samples.txt 塞进 day-test；
程序会拒绝不属于该 split 的 ID。先检查数据和投影来源，再运行：

```bash
python diagnose_severity.py --config config/robotcar.yaml \
  --source day-test --prepare-only --limit 32 \
  --out runs/rc_boundary_select32

python diagnose_severity.py --config config/robotcar.yaml \
  --source day-test --profile config/severity.yaml \
  --sample-list runs/rc_boundary_select32/samples.txt \
  --repeats 2 --save-vis --num-vis 4 \
  --failure-delta 0.10 --out runs/rc_boundary32
```

RobotCar 使用 daytime package 的 test_split 和对应 depth；nuScenes 使用
day_test_split 与 test/color、test/gt。缺失 GT 或 FOV 不符直接失败，没有
clean-prediction 冒充 GT 的 fallback。RobotCar 仍是中心裁剪 1152×640、
0.1–40m，nuScenes 仍是原始 RGB、现有配置的 60m。GT 不 resize；预测按
既有协议最近邻映射到已对齐的 GT 网格。`gt_geometry.json` 记录选中数据的
裁剪与有效点数，形状一致仍不代表标定正确，需结合已有 GT overlay 审查。

## “失效”具体测什么

- `depth_drift`：相对 clean q 的正斜率仿射误差，仅是预测漂移。
- `raw_q_mae` / `fixed_scale_q_mae`：未重新拟合尺度的 q 变化；后者除以
  同图 clean q 的 MAD。clean MAD 很小时该归一化会敏感，应并看原始 MAE。
- `clean_ordinal_disagreement`：与干净预测的排序不一致，不是 GT 错误。
- `gt_ordinal_error`：在 GT 有效像素抽取固定像素对，要求 GT 深度差 /
  较小 GT 深度 > `--pair-gap .1`，比较未经每档 median scale 或深度 cap 截断
  的预测深度排序。预测完全相等计 .5 错误。默认尝试 8192 对，去除近深度对
  和重复无序像素对后不足
  `--min-pairs 128` 则标为 NaN，不猜测失效。
- `gt_abs_rel`、`gt_a1` 等：沿用既有 7 指标及逐图 median scaling。
- `delta_gt_ordinal_error`：同图、同固定 GT 像素对上的退化错误率减 clean
  错误率。baseline 本身可能很差，因此同时报告绝对错误。

固定 GT 像素对覆盖全局排序，**不是遮挡边界、物体接地点标注**，也不是完整
几何能力的判定。后续若要声称某类线索失效，仍需独立区域/人工关系标签。

`--failure-delta .10` 是探索用的 10 个百分点增量，不能包装为普适失效标准。
确认实验前要冻结该阈值、剂量和数据分组。程序给出**采样网格上的第一次越阈
区间**，不是精确临界点：例如 `(S2,S3]`。允许后续回落，`recrossed=True`
会明确记录；未越阈记 `not_observed`，缺少有效统计记 `insufficient_data`。

`boundaries_per_image.csv` 是逐图/重复的经验越阈；`boundaries_group.csv`
分别给出 group 均值及 pointwise 95% CI 下界的越阈。**这些不是阈值本身的
置信区间，也没有对搜索多个剂量做同时推断或多重比较校正**。
阴性结果、非单调结果、只有均值越阈但区间证据不足，都应如实报告。

可在探索样本上细化某个区间：

```bash
python diagnose_severity.py --config config/robotcar.yaml \
  --source day-test --sample-list runs/rc_boundary_select32/samples.txt \
  --profile config/severity.yaml --levels 2 2.25 2.5 2.75 3 \
  --conditions L N G LNG --repeats 2 \
  --save-vis --out runs/rc_boundary_refine32
```

细化后的区间仍是探索结果，不能拿同一批图作为独立确认。

现在默认 `--conditions L N G LN LG NG LNG`，加自动 clean 即完整八条件，
输出 GT ordering/AbsRel 的逐档交互表。若手动删掉部分条件，则不能计算完整因子交互。
每个指标的交互是该误差尺度上的差分，不是联合总效应。

## 分组、开销和输出

同图 repeats 先平均、同组图片再平均，然后按组 bootstrap。RobotCar 时间块
是 proxy；nuScenes day-test 文件中的 recording 前缀用于保守聚合，不伪装成
scene token。可传 `--groups-csv`（列 `id,group`，ID 与 samples.txt 一致）
给已选样本覆盖统计分组；该映射在选样后应用，不改变原有抽样过程。
只有一个组时 CI 为 NaN；不要把像素对或 repeats 数当成独立场景数。
程序不自动建立 fit/exploration/confirmation 的 scene 隔离。

默认每图每 repeat 29 条记录，clean 推理跨 repeat 缓存，所以实际前向为
`1 + 28*repeats`。一般为 `1 + 7*K*repeats`，K 为非零剂量数。
`--level-step .25` 得到 16 个非零剂量。本扫描不执行多次 DPT swap；
发现可复现区间后再用既有机制探针定位，不预设某层一定损坏。

输出包含 samples、run.yaml、parameters.csv/jsonl、per_image.csv、
summary.csv、response_curves.png，以及有 GT 才生成的边界表。
run.yaml 保存全部剂量、参数、软件版本和实现/权重 SHA256；
input_sha256.json、template_sha256.json 保存实际输入/GT/用到的模板哈希。
每完成一张基图原子写入 checkpoints；中断时 status 保留 running。
相同命令加 `--resume` 会核对合同及已完成输入哈希再续跑；未完成基图重算。
未指定 resume 时拒绝非空输出目录。旧格式无 contract/checkpoints 的目录不能续跑。
小批运行结束自动汇总；大规模使用 `--defer-summary`，然后通过
`tools/summarize_sweep.py` 流式合并和拟合。分片必须齐全且合同一致。
拟合输出与条件性越阈区间的解释见研究执行手册，不能与上述经验区间混淆。

新 q 面板同图跨条件、跨剂量共享 clean P5–P95；误差图统一显示
`abs(q-q_clean)/clean_MAD` 的 [0,1] 区间并带 colorbar，超过1会饱和着色，
数值指标不截断。旧八宫格也改为 clean MAD 决定误差色标，不再逐条件 P95
拉伸。不要跨不同图比较原始 q 色值。

## 验证范围

本地 CPU 测试覆盖 S0 恒等、条件顺序不影响结果、固定几何/模板、小数档位、
Poisson 矩、强度实际可见性、GT 顺序反转/常数预测/缺失 GT、非单调越阈，
以及带假模型的双样本/双 repeat/七条件 CLI 全流程和图表生成。
这些是软件与数学契约测试；真实 DAV2-L checkpoint、服务器数据、Flare7K++
素材没有在开发环境提供，**没有测得真实模型的失效阈值或性能改善**。

```bash
python -m unittest discover -s tests -p 'test_severity.py'
python tests/test_diagnosis_protocol.py
python tests/test_gt_audit_and_flare.py
python tests/test_eval_contract.py
python tests/test_dataset.py
python tests/test_model_adapter.py
```
