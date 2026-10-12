#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-}"
ARGS=()
if [[ -n "$CKPT" ]]; then ARGS+=(--checkpoint "$CKPT"); fi
python tools/check_eval_data.py --config config/nuscenes.yaml --limit 3
python test_nuscenes_disp.py night \
  --config config/nuscenes.yaml --out evaluation/ns_result --vis "${ARGS[@]}"
python evaluate.py --config config/nuscenes.yaml --pred-dir evaluation/ns_result
python show_eval_result.py --file evaluation/ns_result/metrics.json
