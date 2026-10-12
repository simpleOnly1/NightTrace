# NightTrace build status — 2026-10-12

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
