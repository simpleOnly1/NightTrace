import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
"""Numerically compare NightTrace's native metrics with STEPS's official function.

The original STEPS archive stays outside NightTrace (or is downloaded to memory).
No steps_original folders are created.
"""
import argparse
import ast
import io
import zipfile
import urllib.request
from pathlib import Path
import numpy as np
from evaluation.metrics import compute_metrics

URL = 'https://codeload.github.com/ucaszyp/STEPS/zip/refs/heads/master'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps-zip', help='Official STEPS-master.zip (offline option)')
    args = ap.parse_args()
    if args.steps_zip:
        blob = Path(args.steps_zip).read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=120) as response:
            blob = response.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        candidates = [p for p in zf.namelist() if p.endswith('/evaluation/eval_nuscenes.py')]
        if len(candidates) != 1:
            raise RuntimeError('STEPS evaluator not found exactly once')
        text = zf.read(candidates[0]).decode('utf-8')
    module = ast.parse(text)
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_metrics')
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<STEPS compute_metrics>', 'exec')
    scope = {'np': np}
    exec(code, scope)
    original = scope['compute_metrics']
    rng = np.random.RandomState(17)
    for _ in range(20):
        gt = rng.uniform(.2, 60, (40, 60))
        pred = gt * rng.uniform(.6, 1.5, gt.shape)
        a = original(pred, gt)
        b = compute_metrics(pred, gt)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)
    print('PASS: 20 synthetic arrays x 7 metrics equal the STEPS source function')
    print('NOTE: GT masks/crops/split ordering require verification on your actual dataset.')


if __name__ == '__main__':
    main()
