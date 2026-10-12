# GT视野审计与Flare7K可选输入（2026-10-12）

本补丁建立在主分支已经实现的成组抽样、L/N/G配对和 GT geometry 上，
不改变 STEPS/RNW 的数据划分，也不依赖 `mainfast.json`。

## 核对裁剪

STEPS 官方 RobotCar evaluator 明确将 GT 中心裁剪后再评价；NightTrace 
`evaluation/protocol.py::prepare_gt` 现要求 provenance：

- `data.gt_space: raw`：GT需为原始RGB网格，按记录的中心裁剪 `crop_xywh` 截取；
- `data.gt_space: cropped`：GT需与该裁剪窗口完全同形且来源确为已裁剪GT；
- 未知/不匹配的 GT 坚决报错，绝不把GT自动resize到预测形状。
- 视野一致后，将低分辨率**预测**最近邻resize至裁剪后的GT进行median alignment。
- 在GT有效深度范围内，模型若有非正值/NaN/Inf，评价直接失败，不再剔除此类像素降低误差。

```bash
python tools/check_eval_data.py --config config/robotcar.yaml \
  --overlays 12 --out runs/gt_audit_robotcar
python tools/check_eval_data.py --config config/nuscenes.yaml \
  --overlays 12 --out runs/gt_audit_nuscenes
```

输出 `audit.csv`、`summary.json`、原始RGB裁剪框以及有效稀疏深度彩点叠加。
叠加仅用于目视审计，不插值稀疏GT、不证明LiDAR标定及时间同步正确；
必须在服务器确认真实深度数组来源后，才报告“协议已核验”。

## 眩光模板（可选）

从官方 Flare7K 解压一组真正的flare模板（而非白天输入照片）
到服务器路径，显式提供给诊断脚本：

```bash
python diagnose.py --config config/robotcar.yaml \
  --sample-list runs/rc_selection/samples.txt \
  --flare-dir /ABS/PATH/Flare7K/templates \
  --out runs/rc_factorial_flare7k
python diagnose_spread.py --config config/robotcar.yaml \
  --sample-list runs/rc_selection/samples.txt \
  --flare-dir /ABS/PATH/Flare7K/templates \
  --out runs/rc_spread_flare7k
```

无需下载库或训练检测器，模板从目录递归枚举，模板选择、强度、旋转、
图像坐标和缩放均由每图/重复的稳定随机种子决定。
本版本只取模板亮度形状（achromatic），不声称复现颜色效应、
真实相机ISP、米制源深度、湿路反射或 TDDC/DepthDark 完整生成器。
未提供模板目录时后备为原本的椭圆halo+streak，并且在run.yaml中注明，
**因此仅在指定Flare7K bank的实验中才可声称使用了Flare7K模板**。

## 测试

```bash
python tests/test_gt_audit_and_flare.py
python tests/test_diagnosis_protocol.py
python tests/test_eval_contract.py
```

仍需服务器上的真实数据、权重完成物理与深度实验，这些CPU测试不替代真实GT的核验。
