#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py --stage day --config config/robotcar.yaml "$@"
