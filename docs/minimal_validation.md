# NightTrace 最小前期验证（2026-10-12）

目标：冻结原版 DAV2，检查联合成像退化、局部传播与 DPT 恢复潜力。
先做受控合成，不训练 PoseNet、不重训 DPT、不强制先收集真实夜间标注。
真实夜间是机制外部验证和真实深度误差结论的必要后续步骤，不是第一轮代码试跑的前置条件。
nuScenes 主输入保持原 RGB，不增加去噪。

## 1. 先审查采样，不加载模型

```bash
python diagnose.py --config config/robotcar.yaml --limit 512 \
  --prepare-only --out runs/rc_selection
python diagnose.py --config config/nuscenes.yaml --limit 512 \
  --prepare-only --out runs/ns_selection
```

复用现有 STEPS `train_split.txt` / `day_train_split.txt`，不改官方训练/测试划分。
RobotCar 用整段时间范围的 30 秒块；nuScenes 用已有 clip 目录分组。
组内固定种子打乱，轮流从各组取样；已识别的微秒时间戳至少间隔 1 秒，
没有时间戳时按已有帧列表至少间隔 10 项。再用小缩略图 MAE 剔除同序列高度相似帧。
这些参数均在 YAML 中可调；不足预算会报告实际数量，不重复填满。
没有用模型误差挑样，也不声称缩略图相似度等价于“静止检测”。

检查 `samples.txt`、`samples.csv` 和 `sampling_coverage.csv`。前者是可复用的普通
split 样式 TXT，后两者记录分组、时间和筛选计数，无 `mainfast.json` / 自定义 manifest 前置流程。
`--sample-list` 重用名单时以完整名单为准，忽略 `--limit`，仍检查是否属于允许的白天 split。
默认 512 是预算，不是代表性保证；用相同种子和筛选参数重跑 256/512/1024，较小预算是较大预算的前缀。
`--limit 0` 是全部符合筛选条件的样本。128 仅可作为用户显式的小规模调试预算。

注意：RobotCar 的同一 traversal 切块不是独立采集；nuScenes clip 目录也不保证是独立 scene。
当前分组区间只描述这些块/clip 的变异。跨采集或跨场景确认需要真实 scene/route 元数据。
缩略图筛选会改变抽样总体；不把结果当成自然分布总体误差，不伪造入样概率。

## 2. 一次完整的配对实验

```bash
python diagnose.py --config config/robotcar.yaml \
  --sample-list runs/rc_selection/samples.txt --repeats 2 \
  --save_vis --out runs/rc_factorial
python diagnose.py --config config/nuscenes.yaml \
  --sample-list runs/ns_selection/samples.txt --repeats 2 \
  --save_vis --out runs/ns_factorial
```

同一基图、同一次参数采样，生成 `clean / L / N / G / LN / LG / NG / LNG` 八组。
默认同时含 1–3 个随机位置、椭圆光晕和旋转条纹的白色光源贡献；所有参数范围可在
`diagnosis.degradation_ranges` 修改。每图/重复的种子由样本 ID 稳定计算，条件顺序不改变结果。
`parameters.csv` 保存所有连续参数；`run.yaml` 保存实际配置和启动参数。
所有条件都从相同裁剪、相同尺寸的基图生成，随后走同一个 DAV2 预处理。

这是简化 RGB 成像近似，采样区间是可检查的起始设置，不是已标定的道路夜景分布，也不冒充论文原参数。
没有可信光源米制深度时，取消原来的“近/中/远距离”标签。光斑尺寸、物体 GT 深度、评测上限是三个不同概念。
不借 DAV2 自己的相对深度推测光源米制距离，也不因 RobotCar 40m 评测上限排除画外亮源。

每组测量 F1–F4 局部关系相关性、正斜率仿射漂移、近远次序不一致率、欠曝/饱和面积。
漂移在 **DAV2 原始相对逆深度 q** 上计算，参照是同图干净预测，不是米制 GT。
可视化使用共同的干净 q 色标；误差面板只是未对齐的 q 绝对差，不等同 CSV 的仿射误差。

四层分别用干净特征替换，并保留 F3+F4 组合。输出绝对恢复量 `delta_F*`；
退化漂移接近零时相对恢复率记为 NaN。clean→clean、degraded→degraded 是 sham；
上一幅不同图像的 F4 是错图负对照，首图无此项，记 NaN。错图控制未匹配内容，不是充分因果控制。
这些都是 **DPT 端 oracle 替换**，不能据此宣称某编码层或 attention 是根因。

`interactions_per_image.csv` 先在每图每重复上算差分：
- `L/N/G` 是其他因素关闭时的单因素效应。
- `LN = E(LN)-E(L)-E(N)+E(clean)`，`LG/NG` 类似。
- `LNG` 是三阶差分，不是联合退化总误差。

`summary.csv` 是先平均同图重复、再平均组内图像、最后对组等权的结果。
`group_intervals.csv` 和 `interactions_summary.csv` 按组做 1000 次 bootstrap，
不会把同图多种退化、多个随机实例或 token 当成独立样本。只有一个组时置信区间记 NaN。
交互效应是所选指标和强度区间内的统计结果，不能直接等价为某条网络路径的物理交互。
条件不完整时不会生成完整八组交互表。每次实验使用新的输出目录，避免旧文件混入。

## 3. 局部传播检查

```bash
python diagnose_spread.py --config config/robotcar.yaml \
  --sample-list runs/rc_selection/samples.txt --repeats 2 --out runs/rc_spread
python diagnose_spread.py --config config/nuscenes.yaml \
  --sample-list runs/ns_selection/samples.txt --repeats 2 --out runs/ns_spread
```

使用相同成像后端的紧支撑 G-only 探针。它不是大范围联合退化的替代实验。
变化区域从**实际预处理后**的干净/扰动输入差分计算；14×14 patch 内任意像素变化即受影响。
增加 attention 之前的 `patch_embed` 对照。若该层未受影响 patch 已出现明显变化，
先排查数值容差、mask 和预处理；F1–F4 的非局部漂移本身不能证明 attention 因果传播。

### Flare7K 模板选项（默认无需素材即可调试）

若已下载官方 Flare7K 光源/光学伪影图片，建议优先运行：

```bash
python diagnose.py --config config/robotcar.yaml --flare-dir /ABS/PATH/Flare7K/images \\
  --sample-list runs/rc_selection/samples.txt --out runs/rc_factorial_flare7k
```

参数种子不仅固定二维位置、半径、强度，也固定模板路径；按文件名排序读取本地 bank。
模板转为亮度形状后参与近似线性成像和 shot/read 噪声，当前尚不模拟色彩互补或三维反射。
若不提供 `--flare-dir` / `diagnosis.flare_dir`，仍使用已有椭圆光晕+条纹后备模式，
并在 `run.yaml` 中明确记录；指定的模板目录不存在或无图片会直接报错。
文件中的位置与大小为图像域参数，不是米制光源距离。

## 4. 真实夜间评测与附件中的 GT 坐标问题

```bash
python tools/check_eval_data.py --config config/robotcar.yaml --overlays 12 --out runs/gt_audit_robotcar
python test_robotcar_disp.py night --config config/robotcar.yaml --out evaluation/rc_result
python evaluate.py --config config/robotcar.yaml --pred-dir evaluation/rc_result
python tools/check_eval_data.py --config config/nuscenes.yaml --overlays 12 --out runs/gt_audit_nuscenes
python test_nuscenes_disp.py night --config config/nuscenes.yaml --out evaluation/ns_result
python evaluate.py --config config/nuscenes.yaml --pred-dir evaluation/ns_result
```

保留完整官方/约定测试名单，既不应用上述去重也不取前 128 张。RobotCar 默认 `(0.1,40m)`，
nuScenes 默认 `(1e-5,60m)`，边界采用严格大于/小于，median scaling。
STEPS 的原 RobotCar evaluator 使用 `(1e-5,60m)`，需要比较时用独立输出：

```bash
python evaluation/eval_robotcar.py night --pred_dir evaluation/rc_result \
  --min_depth 0.00001 --max_depth 60 --output_file_name evaluation/rc_result/metrics_steps60.json
```

`test.py` 只读取 RGB，记录原始尺寸、精确中心裁剪窗口、网络输入尺寸和 `gt_space`。
`index.json` 是既有预测索引，新增坐标元数据，不是额外样本生成依赖。

- 默认 `data.gt_space: raw`：GT 必须与原始 RGB 网格完全一致，再按同一个 RGB 窗口裁剪。
- 仅当你已核实 GT 预裁剪的来源时，设置 `data.gt_space: cropped`：GT 必须恰好等于该裁剪窗口尺寸。
- 不从尺寸自动猜 raw/cropped，不 resize GT；视场对齐后才允许把低分辨率预测最近邻上采样到 GT。
- 旧预测缺少坐标元数据会明确报错，需重新运行预测。不能给旧预测盲填元数据：旧 RGB 预处理可能不同。

RobotCar 的 RGB 在训练、诊断、测试中均用配置中的 1152×640 中心裁剪；K 对应减去裁剪偏移再缩放。
形状一致仍不能证明 LiDAR 投影、时间同步或已裁剪 GT 的坐标来源正确，需要真实样本目视核验。
`check_eval_data.py --overlays 12` 会在原图上画出裁剪窗口，并将有效稀疏GT彩色点覆盖到裁剪后的RGB上，
同时输出 `audit.csv`。彩色点仅用于视觉检查，**不会在指标计算中插值GT**。
原图与GT的投影来源未核实前，不能据此宣称评测协议已经通过真实数据验证。

## 5. 已实现与未声称完成

本提交实现上述代码和 CPU 回归测试，包括小型假模型的两个诊断入口贯通测试。
当前执行环境没有真实 RGB/GT 或 DAV2-L checkpoint，未运行真实深度实验。
尚未实现/验证语义灯源定位、相机 RAW 标定、Flare7K 资产合成、米制灯源深度、
冻结几何 probe、编码器内部残差/attention/MLP 修补及真实 GT 上的机制干预。
这些不阻塞第一轮合成对照；只有发现可重复的候选效应后，才值得扩展对应模块。

参考核对：STEPS 官方 [evaluation/eval_robotcar.py](https://github.com/ucaszyp/STEPS/blob/master/evaluation/eval_robotcar.py) 与 [transforms/cv2_transforms.py](https://github.com/ucaszyp/STEPS/blob/master/transforms/cv2_transforms.py)
的中心裁剪及评测次序。参数简化原则参考随附审计中的 TDDC/DepthDark 讨论；
本提交没有声称复现两篇论文的完整生成器。
