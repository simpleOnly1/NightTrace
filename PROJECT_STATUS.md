# NightTrace build status — 2026-10-08

**Included and locally tested without external checkpoints:** Native STEPS-like
RobotCar/nuScenes dataset loading and split ordering, center crop, outputs
`predictions.npy`, seven STEPS depth metrics, median alignment, visualization,
ACDepth-style day trainer, DAV2 wrapper, geometric reprojection, degradation
and layer diagnostic scripts, plus offline one-time source import.

**Still required on the user server:** Import official DAV2 Python files into
`models/depth_anything_v2/` and real STEPS splits into `split_files/` (via
`tools/prepare_sources.py`); place `depth_anything_v2_vitl.pth` under
`models/checkpoints/`; run the real checkpoint forward and validate data/GT
ordering, RobotCar GT region/mask and STEPS numerical parity on real data.

**Scientific boundary:** Existing night-loss/network changes are not included.
`train.py --stage night` refuses execution pending causal degradation analysis.
This prevents an unvalidated method from being presented as the final model.

**No `*_original`/`third_party` directories**, no STEPS/RNW sibling repository,
no Git dependency.
