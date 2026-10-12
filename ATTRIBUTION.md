# Attribution and code provenance

NightTrace is an independent, single-root research project. The source tree
contains no nested upstream projects and does not require Git for training.

- **STEPS** (Zheng et al., ICRA 2023, MIT): the RobotCar/nuScenes directory and
  split conventions, test-script names, prediction-array ordering, seven depth
  metrics, per-image median scaling and formatted depth reporting inform the
  native NightTrace `datasets/`, `evaluation/`, `test_*_disp.py`, and
  `show_eval_result.py`. The `evaluation/metrics.py` function implements STEPS's
  seven published metrics and argument order. See https://github.com/ucaszyp/STEPS .
  The official split files are imported without renaming using
  `tools/prepare_sources.py` into the project's `split_files/` directory.
- **ACDepth** (Jiang et al., IJCAI 2025, MIT): the root-level `train.py`,
  `trainer.py`, and `config/datasets/models/losses/evaluation` separation is
  adopted as project organization, without copying an old training framework.
  See https://github.com/msscao/ACDepth .
- **md4all** (Gasperini et al., ICCV 2023): architectural precedent for keeping
  the data and self-supervised geometry infrastructure separate from a method's
  depth network and losses. See https://github.com/md4all/md4all .
- **Depth Anything V2** (Yang et al., NeurIPS 2024, Apache-2.0): official model
  implementation imported intact into `models/depth_anything_v2/` using
  `tools/prepare_sources.py`. The official LICENSE is imported into
  `licenses/DAV2_LICENSE`. The ViT-L checkpoint is user supplied.

The STEPS evaluation is adapted, **not asserted byte-identical**. RobotCar GT
preprocessing/masking and test-set parity must be checked on actual datasets
before reporting comparison numbers. `tools/check_parity.py` verifies native
seven-metric numerical equivalence against the official source function from
an upstream ZIP, without storing a second source tree.
