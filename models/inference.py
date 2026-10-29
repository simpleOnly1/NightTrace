"""Frozen severity-sweep model selection and explicit depth semantics."""
from utils.config import project_path


DAV2 = 'dav2_vitl'
UNIDEPTH = 'unidepth_v2_vitl14'


def model_name(config):
    name = config.get('name', DAV2)
    if name not in (DAV2, UNIDEPTH):
        raise ValueError('Unsupported severity model: ' + str(name))
    return name


def build_inference_model(config):
    name = model_name(config)
    if name == DAV2:
        from models.depth_model import DepthModel
        return DepthModel(str(project_path(config['weights'])),
                          input_size=config.get('input_size', 518),
                          freeze_encoder=True, train_dpt=False,
                          inverse_epsilon=config.get('inverse_epsilon', 1e-3))
    from models.unidepth_adapter import UniDepthV2Adapter
    return UniDepthV2Adapter(project_path(config['weights']),
                            resolution_level=config.get('resolution_level', 9))


def prediction_depths(prediction, config):
    """Return inverse depth for drift/plots and native depth for GT evaluation.

    DAV2 keeps its exact historical q -> 1/max(q, eps) conversion. UniDepth
    is evaluated from its native metric depth, never by a second reciprocal
    or the DAV2 inverse-epsilon floor.
    """
    import torch
    name = model_name(config)
    q = prediction['raw_relative']
    if q.ndim != 4 or q.shape[1] != 1 or not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError('Expected finite nonnegative Bx1xHxW inverse depth')
    if name == DAV2:
        depth = q.clamp_min(config.get('inverse_epsilon', 1e-3)).reciprocal()
    else:
        depth = prediction['metric_depth']
        if depth.shape != q.shape or not torch.isfinite(depth).all() or (depth <= 0).any():
            raise ValueError('Expected finite positive UniDepth metric depth matching inverse depth')
    return q, depth
