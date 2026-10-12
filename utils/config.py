"""YAML config with ACDepth-like --config interface; no mmcv dependency."""
from pathlib import Path
import yaml


def load_config(path):
    p = Path(path).expanduser().resolve()
    with p.open('r', encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('Config must be a YAML mapping: %s' % p)
    return cfg


def project_path(p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else Path(__file__).resolve().parents[1] / p
