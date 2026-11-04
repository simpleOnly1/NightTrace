"""One real checkpoint forward for severity adapters; saves unnormalized float depth."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from datasets import make_dataset
from models.inference import build_inference_model, model_name, prediction_depths
from utils.config import load_config, project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', choices=['day', 'night'], default='day')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    target = project_path(args.out)
    if target.exists() and any(target.iterdir()):
        raise ValueError('Choose a new smoke output directory')
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    frame = make_dataset(cfg, split=args.split, training=False)[0]
    net = build_inference_model(cfg['model']).to(device).eval().requires_grad_(False)
    with torch.no_grad():
        q, depth = prediction_depths(net(frame['image'][None].to(device)), cfg['model'])
    target.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, tensor in [('inverse_depth', q), ('depth', depth)]:
        array = tensor.detach().float().cpu().numpy()[0, 0]
        np.save(str(target / (name + '.npy')), array, allow_pickle=False)
        stats[name] = dict(shape=list(array.shape), min=float(array.min()),
                           median=float(np.median(array)), max=float(array.max()))
    metadata = dict(model=model_name(cfg['model']), config=cfg, sample=frame['id'], split=args.split,
                    input_shape=list(frame['image'].shape), device=device, stats=stats,
                    model_identity=getattr(net, 'inference_metadata', None),
                    note='Float depth arrays, no visualization normalization; inspect near/far before a full sweep')
    (target / 'smoke.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    print('PASS: one real checkpoint forward. This is an interface check, not a failure-law experiment.')


if __name__ == '__main__':
    main()
