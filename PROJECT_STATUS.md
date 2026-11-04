# NightTrace build status — 2026-11-04

当前步骤见 [UniDepth V2 执行说明](docs/unidepth_v2.md)。已收到并分析 DAV2 / nuScenes 全量结果，现接入第二个基础模型。

- 按用户要求继续使用 Python 3.8 的 nighttime_ds：固定官方源码的类型标注回移、去除非推理可视化自动导入，补充依赖仅安装到项目内 `.deps/`，不升级 DAV2 环境的核心包。详见上述执行说明第 3 节。
- 用户提供的 uni_env_check.json 与 smoke.json 已验收：现有 nighttime_ds / Python 3.8.12，核心包版本检查前后一致；本地权重完成 CUDA 单张推理，输出 448×784 正有限深度；157 个源码文件哈希与固定版本兼容源码全部一致。随后全量扫描在首张 clean 的 cuBLASLt 算法选择处失败；现先按执行说明第 4b 节设置进程开关并检查，不将运行时异常记为模型失效。

- UniDepth V2 ViT-L/14 已接入冻结强度扫描，使用官方 infer、RGB-only、本地严格权重加载；原生 metric depth 直接评价，倒数仅用于漂移/可视化。
- 新增独立配置、固定版本源码准备工具、一张真实权重检查入口及离线下载指南。权重与源码/配置身份纳入推理合同。
- CPU 测试覆盖新适配器与已有扫描/恢复/分片/指标回归。真实 checkpoint 已在用户服务器完成单张 CUDA 推理；GT 退化扫描结果尚待生成，尚无 UniDepth 的精度或失效规律结论。
- 新增 BLAS/CUDA 环境设置的模型身份记录和定向错误提示；不自动改精度或忽略失败图。`DISABLE_ADDMM_CUDA_LT=1` 的绕行效果仍待用户服务器确认。
- 各模型允许具有相同或不同的失效原因；先测各自响应，再分析并针对性优化。Marigold V2 与下游迁移仍待后续推进。

## 2026-10-27 阶段记录

当前步骤见 [nuScenes 小批执行说明](docs/nuscenes_pilot.md)。研究允许不同基础模型具有不同失效原因；共同原因是可能的结果，不是前提或验收条件。

- 已有用户提供并分析的 RobotCar＋DAV2 先导扫描及过渡区复核；其解释范围限于对应模型、数据和合成路径。
- 本轮新增 `tools/run_nuscenes_pilot.py`：显式 audit / pilot / summarize 阶段，八条件、最多八张默认基图、无拟合汇总、恢复及执行状态记录。软件测试通过不等价于 nuScenes GPU 实验完成。
- nuScenes 尚待服务器运行和核验；UniDepth V2、Marigold V2 尚未接入统一实验入口。只改配置模型名称不能替代适配。
- 后续按各模型实测结果分析各自原因，进行针对性优化、真实夜间评价，再验证向一个下游方法迁移；不预设统一机制。

## 历史实现记录（当时状态）

# NightTrace build status — 2026-10-15

最新执行顺序见 [研究执行手册](docs/research_workflow.md)。严重度入口现在默认八条件，
支持小数剂量网格、完整 day-test、逐图原子检查点、严格恢复和确定性分片。
独立汇总实现组 bootstrap、非强制单调的描述性曲线拟合、越阈观测比例与配对边界位移。
数千训练基图仅用于无 GT 稳定性；测试集规模不因合成或重复而增加。
新增断点/分片/拟合测试是软件契约验证，尚未在服务器测得真实模型失效边界。

## 2026-10-12 基础实现

本次实现和运行方式见 [最小前期验证](docs/minimal_validation.md)。

已实现：显式 RGB/GT 坐标协议、RobotCar 40m / nuScenes 60m 默认评测、
跨完整白天 split 分组抽样和轻量去重、统一近似线性成像后端、连续均匀参数、
L/N/G 八组配对与交互统计、正斜率仿射/次序指标、F1–F4 对称 DPT swap 和负对照、
模型输入空间 patch 支持域及 patch embedding 对照。无 mainfast.json 前置流程。

本地验证：CPU 回归测试与小型假模型入口贯通；不等价于真实 DAV2 实验。
真实 RGB/GT、DAV2-L checkpoint 不在本环境，数据坐标来源、真实帧覆盖、
实际 GPU forward 和合成到真实夜景的机制一致性仍需在数据服务器验证。

没有训练新网络、PoseNet 或 DPT。未实现冻结几何 probe 和编码器内因果路径干预。
`train.py --stage night` 继续拒绝尚未定义的夜间训练。

