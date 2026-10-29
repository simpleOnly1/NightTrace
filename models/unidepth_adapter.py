"""Offline UniDepth V2 ViT-L RGB-only inference using the official infer API."""
import hashlib
import json
from importlib import metadata
from pathlib import Path
import sys

import torch
from torch import nn


UPSTREAM_COMMIT = '8d8cfe4c7ee15297099983607febf0d4f32eb3d6'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_files(directory):
    directory = Path(directory)
    config = directory / 'config.json'
    weights = directory / 'model.safetensors'
    if not weights.is_file():
        weights = directory / 'pytorch_model.bin'
    if not config.is_file() or not weights.is_file():
        raise FileNotFoundError('Need local config.json and model.safetensors (or pytorch_model.bin) in ' + str(directory))
    return config, weights


class UniDepthV2Adapter(nn.Module):
    def __init__(self, directory, resolution_level=9):
        super().__init__()
        if sys.version_info < (3, 10):
            raise RuntimeError('Official UniDepth V2 requires Python >=3.10; use a separate environment')
        if type(resolution_level) is not int or not 0 <= resolution_level <= 9:
            raise ValueError('resolution_level must be an integer in [0,9]')
        config_path, weights_path = checkpoint_files(directory)
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if (config.get('model', {}).get('name') != 'UniDepthV2' or
                config['model'].get('pixel_encoder', {}).get('name') != 'dinov2_vitl14'):
            raise ValueError('This adapter requires the official UniDepthV2 ViT-L/14 config')
        # The full local checkpoint contains its encoder: forbid hidden backbone downloads.
        if config['model']['pixel_encoder'].get('pretrained') is not None:
            raise ValueError('Set upstream pixel_encoder.pretrained to null for offline full-checkpoint loading')
        root = Path(__file__).resolve().parent
        package = root / 'unidepth'
        if not (package / 'models/unidepthv2/unidepthv2.py').is_file():
            raise RuntimeError('Official source missing. Run python tools/prepare_unidepth.py')
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from unidepth.models import UniDepthV2
        import unidepth
        imported = {Path(p).resolve() for p in unidepth.__path__}
        if imported != {package.resolve()}:
            raise RuntimeError('A different unidepth package is already imported; start a fresh process')
        self.net = UniDepthV2(config)
        if weights_path.suffix == '.safetensors':
            from safetensors.torch import load_file
            state = load_file(str(weights_path), device='cpu')
        else:
            state = torch.load(str(weights_path), map_location='cpu', weights_only=True)
        self.net.load_state_dict(state, strict=True)
        del state
        self.net.resolution_level = resolution_level
        self.net.eval().requires_grad_(False)
        self.eval()
        source_hashes = {p.relative_to(root).as_posix(): sha256(p)
                         for p in sorted(package.rglob('*.py'))}
        versions = {}
        for name in ('torch', 'torchvision', 'numpy', 'opencv-python', 'timm',
                     'huggingface-hub', 'safetensors', 'einops', 'xformers', 'pillow', 'scipy'):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        self.inference_metadata = dict(
            name='unidepth_v2_vitl14', input='RGB float [0,1]; official infer receives RGB*255 without quantization',
            output='native metric z-depth; reciprocal only for drift/visualization',
            camera='RGB-only; no supplied intrinsics', resolution_level=resolution_level,
            gt_distance_alignment='median (same relative-depth comparison as DAV2)',
            weights_file=str(weights_path), weights_sha256=sha256(weights_path),
            config_sha256=sha256(config_path), upstream_expected_commit=UPSTREAM_COMMIT,
            source_sha256=source_hashes, package_versions=versions)

    def train(self, mode=True):
        # This adapter deliberately exposes no fine-tuning or DAV2 feature-swap API.
        return super().train(False)

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
            raise ValueError('UniDepth input must be finite Bx3xHxW RGB in [0,1]')
        with torch.no_grad():
            result = self.net.infer(rgb * 255.0, camera=None, normalize=True)
        depth = result['depth'].float()
        if (depth.shape != (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]) or
                not torch.isfinite(depth).all() or (depth <= 0).any()):
            raise ValueError('Official infer must return positive finite metric depth on the input grid')
        inverse = depth.reciprocal()
        if not torch.isfinite(inverse).all():
            raise ValueError('UniDepth depth reciprocal is nonfinite')
        return dict(raw_relative=inverse, relative_inverse_depth=inverse,
                    relative_depth=depth, metric_depth=depth)
