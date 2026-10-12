#!/usr/bin/env bash
set -euo pipefail
python diagnose.py --config config/robotcar.yaml --limit 512 --repeats 2 --out runs/diagnose_robotcar "$@"
