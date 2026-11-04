# 冻结基础模型：真实夜间测试与七指标

更新：2026-11-04。

当前 `diagnose_severity.py --source day-test` 测量白天基图经过受控合成后的响应；本页补齐冻结模型在真实夜间完整 test split 上的七指标基线。两者分别回答退化失效规律与真实夜间性能。本页不运行训练，不添加合成退化。

## 1. 统一的主评估协议

`test.py`、`test_robotcar_disp.py`、`test_nuscenes_disp.py` 现在按 `model.name` 选择 DAV2 或 UniDepth V2。`--evaluate` 在预测全部保存后调用已有 `evaluation.protocol.evaluate_records`，不另写一套指标。

| 项目 | RobotCar-Night | nuScenes-Night |
|---|---|---|
| 名单 | night_package/test_split.txt，全量、排序 | splits/night_test_split.txt，全量、排序 |
| RGB 输入网格，与当前 DAV2 相同 | 高392×宽672 | 高448×宽784 |
| RGB/GT 视场 | 同一宽1152×高640中心裁剪 | 原视场，不额外裁剪 |
| 主结果有效 GT 范围 | 0.1 < GT < 40m | 1e-5 < GT < 60m |
| 尺度对齐 | 逐图 median scaling | 逐图 median scaling |
| 汇总 | 七指标先逐图计算，再对图像等权平均 | 相同 |

上述为当前配置；实际配置、split 哈希、RGB 哈希、每图坐标元数据和模型身份均保存到输出。服务器若修改过数据根路径或已验证的 gt_space，请同步到 UniDepth 配置中，保留其 model 块。主结果应使用与 DAV2 相同的 GT 文件、范围和名单。

七项为 `abs_rel`、`sq_rel`、`rmse`、`rmse_log`（越低越好）及 `a1`、`a2`、`a3`（越高越好；δ 阈值为 1.25、1.25²、1.25³，保存为0到1的比例）。

评价次序固定：按显式 raw/cropped 来源对齐 GT 与 RGB 视场；用 GT 形成严格上下界有效 mask；将预测最近邻 resize 到对齐后的 GT 网格；在有效像素上计算 median scale；缩放后裁剪预测；逐图计算七指标；对图像等权平均。GT 不做插值。预测在有效 GT 像素上非正或非有限时明确报错，不删除困难像素。

DAV2 存 `1/max(q,epsilon)`；UniDepth 存原生正 metric z-depth，不能套用 DAV2 的逆深度 epsilon，也不能使用显示归一化图求指标。为了与当前 DAV2、RNW/STEPS 的相对深度评价比较，UniDepth 主结果也逐图中值对齐，因此表格应标为 **median-aligned**，不能写成未经对齐的米制性能。

## 2. 先完成正在运行的扫描，再依次执行两个夜间基线

继续使用 nighttime_ds。两项新评测都使用 GPU02，但不与同卡的全量扫描并发，以免重复出现显存不足。这次更新仅修改测试入口、测试和文档，不修改运行中 severity 扫描所哈希的模型、数据、合成或评估实现，因此保持其已有恢复合同。

```bash
git pull --ff-only
```

### nuScenes 全量真实夜间

```bash
DISABLE_ADDMM_CUDA_LT=1 CUDA_VISIBLE_DEVICES=1 python test_nuscenes_disp.py night \
  --config config/nuscenes_unidepthv2.yaml \
  --out evaluation/uni_ns_night --evaluate
```

### RobotCar 全量真实夜间

```bash
DISABLE_ADDMM_CUDA_LT=1 CUDA_VISIBLE_DEVICES=1 python test_robotcar_disp.py night \
  --config config/robotcar_unidepthv2.yaml \
  --out evaluation/uni_rc_night --evaluate
```

默认 `--limit 0` 即全部名单，不默认取小样本。预测阶段仅向模型传 RGB；GT 数值在预测保存后的独立评价阶段加载。已有非空输出目录会拒绝覆盖，请使用新的结果目录。

每个目录输出：

- `metrics.json`：七项汇总、每图结果、有效像素数和逐图尺度。
- `predictions.npy`：未经尺度对齐或范围裁剪的浮点深度。
- `raw_inverse_depth.npy`：DAV2 为原始 q；UniDepth 为原生深度的倒数，语义在合同中明确。
- `index.json`：排序样本 ID、RGB 路径/哈希、GT 路径和坐标元数据。
- `prediction_contract.json`：实际配置、名单哈希、模型/权重身份、代码哈希和评估协议。

已有 DAV2 预测若包含同样的坐标元数据，可直接用现有 `evaluate.py` 按相同配置重评分，不必重新 GPU 推理。例如：

```bash
python evaluate.py --config config/robotcar.yaml \
  --pred-dir evaluation/rc_result --output evaluation/rc_result/metrics_current_protocol.json
python evaluate.py --config config/nuscenes.yaml \
  --pred-dir evaluation/ns_result --output evaluation/ns_result/metrics_current_protocol.json
```

上面 DAV2 路径为项目旧默认值；如果此前实际使用其他目录，以已有有效预测目录为准。

## 3. RNW/STEPS 官方默认范围的差异必须标清

四个官方脚本的七指标、nearest resize、median scaling、clip 与逐图平均流程相同，但 RobotCar 默认范围不完全相同：

| 协议 | RobotCar 最小/最大 GT 深度 | nuScenes 最小/最大 GT 深度 |
|---|---|---|
| 当前 NightTrace DAV2/UniDepth 主结果 | 0.1 / 40m | 1e-5 / 60m |
| RNW 官方脚本默认 | 1e-5 / 40m | 1e-5 / 60m |
| STEPS 官方脚本默认 | 1e-5 / 60m | 1e-5 / 60m |

不能把不同 mask/上限的结果直接放进同一列比较。若需要和官方默认口径比较，复用相同 RobotCar 原始预测另算以下两份报告，不增加 GPU 前向：

```bash
python evaluation/eval_robotcar.py night --pred_dir evaluation/uni_rc_night \
  --min_depth 0.00001 --max_depth 40 \
  --output_file_name evaluation/uni_rc_night/metrics_rnw40.json
python evaluation/eval_robotcar.py night --pred_dir evaluation/uni_rc_night \
  --min_depth 0.00001 --max_depth 60 \
  --output_file_name evaluation/uni_rc_night/metrics_steps60.json
```

DAV2 或其他方法也必须对同一实际测试名单和 GT 使用相同范围重评分，才能作相应对比。无需改动训练或合成扫描配置。

官方来源：

- [RNW RobotCar](https://github.com/w2kun/RNW/blob/master/evaluation/eval_robotcar.py)、[RNW nuScenes](https://github.com/w2kun/RNW/blob/master/evaluation/eval_nuscenes.py)。
- [STEPS RobotCar](https://github.com/ucaszyp/STEPS/blob/master/evaluation/eval_robotcar.py)、[STEPS nuScenes](https://github.com/ucaszyp/STEPS/blob/master/evaluation/eval_nuscenes.py)。

2026-11-04 已从四个官方脚本提取纯函数独立核对。每份源码比较20组数组的七指标，并比较3种范围×5幅合成图的 mask、nearest、median、clip、逐图平均流程；float64 合成输入最大绝对差为0。几何部分使用缩小尺寸的显式中心裁剪，仅验证计算次序，不代替真实 LiDAR 投影、GT 来源及真实名单的核验。对应源码 blob 身份与结果见 [校验记录](validation/night_metric_parity_20261104.json)。

本次测试以小型假模型验证两个真实数据加载器和评价器的完整流程、UniDepth 原生深度、DAV2 checkpoint 路径、输出保护和全量/子集标签；不替代服务器真实权重夜间测量。

## 4. 完成后回传

```bash
tar -czf runs/uni_real_night_metrics.tar.gz \
  evaluation/uni_ns_night/*.json evaluation/uni_rc_night/*.json
```

本轮不需要传大体积 predictions.npy。先确认完整名单、GT范围、每图有效像素和七指标，再与现有 DAV2 结果组成同口径基线表。
