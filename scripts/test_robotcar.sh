#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-}"
ARGS=()
if [[ -n "$CKPT" ]]; then ARGS+=(--checkpoint "$CKPT"); fi
python tools/check_eval_data.py --config config/robotcar.yaml --limit 3
python test_robotcar_disp.py night \
  --config config/robotcar.yaml --out evaluation/rc_result --vis "${ARGS[@]}"
python evaluate.py --config config/robotcar.yaml --pred-dir evaluation/rc_result
python show_eval_result.py --file evaluation/rc_result/metrics.json
