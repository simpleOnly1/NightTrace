"""Paired severity paths. Dose is an experimental setting, not a night frequency.

Geometry/templates stay fixed across doses. Poisson marginals are exact; using
the same seed does NOT imply matched photon events across different rates.
"""
from copy import deepcopy
import math
import numpy as np
import torch
from .pipeline import sample_parameters, degrade

VERSION = 'severity-v1'
DEFAULT_PROFILE = {
    'exposure': [1., .4, .12, .025, .005],
    'shot_peak': [None, 400., 100., 25., 6.],
    'read_sigma': [0., .004, .012, .035, .08],
    'glare_gain': [0., .5, 2., 8., 32.],
}
DEFAULT_GEOMETRY = {
    'source_count': [1, 2], 'u': [.15, .85], 'v': [.15, .75],
    # Whole-template half extent, NOT the bright core radius or distance.
    'radius': [.35, .65], 'amplitude': [.8, 1.2],
    'aspect': [.8, 1.5], 'angle': [0., math.pi], 'streak_weight': [0., .5],
}


def validate_profile(profile):
    if set(profile) != set(DEFAULT_PROFILE):
        raise ValueError('Profile requires exposure, shot_peak, read_sigma, glare_gain')
    for k, v in profile.items():
        if len(v) != 5:
            raise ValueError(k + ' requires S0..S4 (five entries)')
        checked = v[1:] if k == 'shot_peak' else v
        if not np.isfinite(checked).all():
            raise ValueError('Nonfinite profile: ' + k)
    e, p, n, g = (profile[k] for k in DEFAULT_PROFILE)
    if e[0] != 1 or min(e) <= 0 or any(a < b for a, b in zip(e, e[1:])):
        raise ValueError('Exposure must decrease from 1 and stay positive')
    if p[0] is not None or min(p[1:]) <= 0 or any(a < b for a, b in zip(p[1:], p[2:])):
        raise ValueError('shot_peak must start null and then decrease positively')
    for k, v in [('read_sigma', n), ('glare_gain', g)]:
        if v[0] != 0 or min(v) < 0 or any(a > b for a, b in zip(v, v[1:])):
            raise ValueError(k + ' must increase from zero')
    return deepcopy(profile)


def sample_plan(seed, templates=None, geometry=None):
    spec = dict(DEFAULT_GEOMETRY)
    if geometry:
        if set(geometry) - set(spec):
            raise ValueError('Only source geometry/amplitude ranges belong in geometry')
        spec.update(geometry)
    return sample_parameters(seed, spec, templates)


def at_severity(plan, severity, profile=None):
    profile = validate_profile(profile or DEFAULT_PROFILE)
    s = float(severity)
    if not np.isfinite(s) or not 0 <= s <= 4:
        raise ValueError('Severity must be finite in [0,4]')
    lo = min(int(s), 3); t = s - lo
    def interp(v, log=False):
        a, b = v[lo:lo+2]
        return float(math.exp((1-t)*math.log(a)+t*math.log(b)) if log else (1-t)*a+t*b)
    p = deepcopy(plan)
    p['exposure'] = interp(profile['exposure'], True)
    # Variance coefficient 1/peak goes continuously to zero at S0.
    inv_peak = interp([0.] + [1/x for x in profile['shot_peak'][1:]])
    p['shot_peak'] = 1/inv_peak if inv_peak else None
    p['read_sigma'] = interp(profile['read_sigma'])
    gain = interp(profile['glare_gain'])
    for source in p['sources']:
        source['amplitude'] *= gain
    p.update(severity=s, severity_version=VERSION, glare_gain=gain)
    return p


def render(img, condition, params):
    if params['severity'] == 0:
        return img.clone(), torch.zeros_like(img[:, :1])
    return degrade(img, condition, params)


def audit_input(clean, observed, condition, params, flare):
    """Actual float model-input statistics; never reject/select by model error."""
    def lum(x):
        return (x*x.new_tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1)
    l = lum(observed)
    quantiles = torch.quantile(l.flatten(), l.new_tensor([.01, .1, .5, .9, .99]))
    noiseless, _ = render(clean, condition.replace('N', '') or 'clean', params)
    no_glare, _ = render(clean, 'L' if 'L' in condition else 'clean', params)
    noise_rms = float((observed-noiseless).square().mean().sqrt())
    glare_delta = (noiseless-no_glare).abs().mean(1)
    return {
        **{f'luma_p{q:02d}': float(v) for q, v in zip([1, 10, 50, 90, 99], quantiles)},
        'luma_mean': float(l.mean()),
        'underexposed_area': float((l < .05).float().mean()),
        'black_area': float((l < .01).float().mean()),
        'saturated_area': float((observed.amax(1) >= .99).float().mean()),
        'saturation_increase': float(((observed.amax(1) >= .99).float() -
                                      (clean.amax(1) >= .99).float()).mean()),
        'rgb_mae': float((observed-clean).abs().mean()),
        'flare_linear_area': float((flare > .01).float().mean()),
        'glare_visible_area': float((glare_delta > .02).float().mean()),
        'glare_rgb_mae': float(glare_delta.mean()),
        'noise_rgb_rms': noise_rms,
        'noise_snr_db': (20*math.log10(max(float(noiseless.square().mean().sqrt()), 1e-12)/noise_rms)
                         if noise_rms > 0 else float('nan')),
    }
