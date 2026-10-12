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

## 服务器已有的 Flare7K++：默认启用

RobotCar、nuScenes 的 `diagnosis.flare_dir` 都设置为以下绝对路径：

```text
/080226/Nighttime_Depth_Estimation/datasets/Flare7Kpp/Flare7K
├── Scattering_Flare/
│   ├── Compound_Flare/   # 完整散射眩光主模板
│   ├── Core/             # 分解成分，不当完整模板抽样
│   ├── Light_Source/     # 同上
│   └── Streak/           # 同上
└── Reflective_Flare/      # 可选叠加的反射模板
```

参考官方 [Flare7K](https://github.com/ykdai/Flare7K)
`options/uformer_flare7kpp_baseline_option.yml` 的 `scattering_dict` 和 `reflective_dict`：
散射主素材来自 `Scattering_Flare/Compound_Flare`，反射素材另行叠加，
不把五种不同语义子目录全部混作完整flare模板。

```bash
python diagnose.py --config config/robotcar.yaml --prepare-only --limit 512 --out runs/rc_selection
python diagnose.py --config config/robotcar.yaml --sample-list runs/rc_selection/samples.txt \
  --out runs/rc_factorial_flare7k
python diagnose_spread.py --config config/nuscenes.yaml --out runs/ns_spread_flare7k
```

现在不必再指定 `--flare-dir`、下载或复制任何素材。读取目录只在实际生成扰动时执行；
`--prepare-only` 不需要 Flare7K 资产。每个源按固定种子选择一个 Compound_Flare，
若 Reflective_Flare 存在则以0.5的**实验概率**增加反射模板（非道路真实概率），
两者按共同二维变换叠加，生成器继续按照L/N/G条件配对。
`run.yaml` / `parameters.csv` 留存模板来源。

指定目录不存在、Compound_Flare不存在或为空时会**直接报错**，不会悄悄降级为椭圆高斯。
若在另一台机器无素材，必须显式删掉YAML中的 `diagnosis.flare_dir` 才会走解析后备路径。
当前仍然是灰度形状＋近似RGB成像，并非完整复现 Flare7K++ 的RGB/ISP/颜色模型；
GT裁剪与深度评价也没有因为增加素材而改变。

## 测试

```bash
python tests/test_gt_audit_and_flare.py
python tests/test_diagnosis_protocol.py
python tests/test_eval_contract.py
```

仍需服务器上的真实数据、权重完成物理与深度实验，这些CPU测试不替代真实GT的核验。
