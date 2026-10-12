#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# With no arguments, download official source ZIPs and import them once.
# Offline: bash scripts/prepare.sh --steps-zip /path/STEPS-master.zip --dav2-zip /path/DAV2-main.zip
python tools/prepare_sources.py "$@"
python tests/test_parity_local.py
python -m compileall -q train.py trainer.py test.py datasets models evaluation losses analysis
