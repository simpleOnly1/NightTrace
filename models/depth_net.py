"""Depth Anything V2-L wrapper. Raw output is *relative inverse depth*, not metric depth.

No per-image min-max, z-score or affine remapping is permitted in the forward path:
such remapping changes the effect of night corruptions and invalidates diagnosis.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

DAV2_HOME = Path(__file__).resolve().parent
if str(DAV2_HOME) not in sys.path:
    sys.path.insert(0, str(DAV2_HOME))


def preprocess(rgb, size=518):
    """Follow official DAV2 Resize(lower_bound, multiple_of=14) + ImageNet norm.

    Assumes rgb is float in [0,1], BCHW. Intermediate interpolation is bicubic;
    official inference transform uses OpenCV cubic interpolation. The small
    sampling difference is documented; use official transform for exact parity.
    """
    h, w = rgb.shape[-2:]
    scale = float(size) / min(h, w)
    hh = max(14, int(round(h * scale / 14)) * 14)
    ww = max(14, int(round(w * scale / 14)) * 14)
    rgb = F.interpolate(rgb, (hh, ww), mode='bicubic', align_corners=False)
    mean = rgb.new_tensor([.485,.456,.406])[None,:,None,None]
    std = rgb.new_tensor([.229,.224,.225])[None,:,None,None]
    return (rgb - mean) / std


def inverse_to_depth(q, epsilon=1e-3):
    """Scale-free monotone conversion for evaluation. NOT STEPS disp_to_depth."""
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    return q.clamp_min(epsilon).reciprocal()


class DepthNet(nn.Module):
    TAPS = (4,11,17,23)

    def __init__(self, weights='models/checkpoints/depth_anything_v2_vitl.pth',
                 input_size=518, freeze_encoder=True, train_dpt=True,
                 inverse_epsilon=1e-3):
        super().__init__()
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except Exception as exc:
            raise RuntimeError('Official DAV2 package absent. Run: python tools/prepare_sources.py --dav2-only') from exc
        self.net = DepthAnythingV2(encoder='vitl', features=256,
                  out_channels=[256,512,1024,1024], use_bn=False, use_clstoken=False)
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError('DAV2-L checkpoint missing: %s' % weights)
        sd = torch.load(str(weights), map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd=sd['state_dict']
        self.net.load_state_dict(sd, strict=True)
        self.input_size=input_size
        self.inverse_epsilon=inverse_epsilon
        self.freeze_encoder=freeze_encoder
        if freeze_encoder:
            self.net.pretrained.requires_grad_(False)
        if not train_dpt:
            self.net.depth_head.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.net.pretrained.eval()
        return self

    def features(self, x):
        xx=preprocess(x,self.input_size)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            fs=self.net.pretrained.get_intermediate_layers(xx,list(self.TAPS),return_class_token=True)
        return xx,fs

    # Keep the Stage-0 diagnostic API stable.
    def extract(self, rgb):
        return self.features(rgb)

    def decode(self, preprocessed, features, target_size):
        ph, pw = preprocessed.shape[-2] // 14, preprocessed.shape[-1] // 14
        raw = self.net.depth_head(features, ph, pw)
        if raw.dim() == 3:
            raw = raw.unsqueeze(1)
        return F.interpolate(raw, size=target_size, mode='bilinear', align_corners=False)

    def forward(self, rgb, return_features=False):
        if rgb.dim()!=4 or rgb.shape[1]!=3:
            raise ValueError('RGB image must be Bx3xHxW tensor, scaled into [0,1]')
        prep,fs=self.features(rgb)
        ph,pw=prep.shape[-2]//14,prep.shape[-1]//14
        raw=self.net.depth_head(fs,ph,pw)
        if raw.dim()==3:
            raw=raw.unsqueeze(1)
        if raw.dim()!=4 or raw.shape[1]!=1:
            raise ValueError('Unexpected DAV2 DPT output shape: %s' % (tuple(raw.shape),))
        raw=F.interpolate(raw, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        # DPT includes terminal ReLU. Keep exact q, including zeros.
        result={'raw_relative':raw,'relative_inverse_depth':raw,
                'relative_depth':inverse_to_depth(raw, self.inverse_epsilon),
                'depth_for_warp':inverse_to_depth(raw, self.inverse_epsilon),
                'disp':raw}
        if return_features:
            result['features']=fs
        return result

# Compatibility alias for existing diagnostic/train implementations.
DepthModel=DepthNet
