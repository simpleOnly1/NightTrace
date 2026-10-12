#!/usr/bin/env bash
set -euo pipefail
python diagnose.py --config config/robotcar.yaml --limit 128 --out runs/diagnose_robotcar
