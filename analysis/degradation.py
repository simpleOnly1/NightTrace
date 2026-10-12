"""Legacy names routed through the same approximate imaging backend."""
import warnings
import torch
from .degradation_v2.pipeline import to_linear, to_srgb, degrade, sample_parameters


def corrupt(rgb,kind='low',gain=.12,shot_peak=350,read_sigma=.003,
            glare_strength=2.,generator=None):
    mapping={'low':'L','low_noise':'LN','low_colored_glare':'LG','low_white_glare':'LG',
             'low_noise_colored_glare':'LNG','low_noise_white_glare':'LNG',
             'colored_glare_only':'G','white_glare_only':'G'}
    if kind not in mapping:raise ValueError(kind)
    if 'colored' in kind:
        warnings.warn('Legacy colored conditions now use achromatic G; use explicit L/N/G labels',
                      DeprecationWarning,stacklevel=2)
    seed=int(torch.randint(0,2**31-1,()).item()) if generator is None else generator.initial_seed()
    p=sample_parameters(seed,{'exposure':(gain,gain),'shot_peak':(shot_peak,shot_peak),
                             'read_sigma':(read_sigma,read_sigma),'amplitude':(glare_strength,glare_strength)})
    return degrade(rgb,mapping[kind],p,compact=kind.endswith('_only'),generator=generator)
