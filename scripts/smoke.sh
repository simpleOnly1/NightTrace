#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/test_core.py
python tests/test_geometry.py
python tests/test_dataset.py
python tests/test_model_adapter.py
python tests/test_eval_contract.py
python tools/check_setup.py --config config/robotcar.yaml --stage test
python test_robotcar_disp.py night --config config/robotcar.yaml \
  --limit 2 --out evaluation/rc_smoke --vis
python evaluate.py --config config/robotcar.yaml --pred-dir evaluation/rc_smoke
