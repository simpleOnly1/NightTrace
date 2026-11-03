"""Project-local extras for UniDepth only; never imported by the DAV2 factory."""
from importlib import metadata
from pathlib import Path
import sys

PROTECTED = ('torch', 'torchvision', 'numpy', 'opencv-python', 'scipy')


def extras_path():
    root = Path(__file__).resolve().parents[1]
    return root / '.deps' / ('unidepth-py%d%d' % sys.version_info[:2])


def activate_extras():
    path = extras_path()
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def protected_versions():
    versions = {}
    for name in PROTECTED:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def check_base_runtime():
    if sys.version_info < (3, 8):
        raise RuntimeError('NightTrace UniDepth compatibility requires Python >=3.8')
    import torch
    import torchvision.transforms.v2.functional as transforms
    import torch.nn.functional as functional
    if not hasattr(functional, 'scaled_dot_product_attention') or not hasattr(torch, 'autocast'):
        raise RuntimeError('UniDepth needs PyTorch >=2.0 APIs. Keep the DAV2 environment; share the environment check output.')
    if not callable(transforms.normalize):
        raise RuntimeError('Missing torchvision v2 normalize API')
    return protected_versions()
