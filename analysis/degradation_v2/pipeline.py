"""Paired L/N/G interventions on an approximate linear RGB signal.

Uniform ranges are experiment settings, not calibrated sensor distributions.
Flare size is image-plane extent, never a metric source distance. All conditions
share one sampled parameter set; seeds do not depend on condition iteration.
"""
import math
import warnings
import numpy as np
import torch

CONDITIONS=('clean','L','N','G','LN','LG','NG','LNG')
DEFAULT_RANGES={
    'exposure':(.08,.4), 'shot_peak':(100.,500.), 'read_sigma':(.001,.008),
    'source_count':(1,3), 'u':(-.1,1.1), 'v':(.1,.7),
    'radius':(.025,.15), 'amplitude':(.2,1.2), 'aspect':(.6,1.8),
    'angle':(0.,math.pi), 'streak_weight':(0.,.5),
}


def to_linear(rgb):
    return torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).clamp_min(0)**2.4)


def to_srgb(x):
    x=x.clamp_min(0)
    return torch.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)


def sample_parameters(seed, ranges=None):
    spec=dict(DEFAULT_RANGES)
    if ranges:
        unknown=set(ranges)-spec.keys()
        if unknown:raise ValueError('Unknown degradation parameters: '+str(unknown))
        spec.update(ranges)
    for key,bounds in spec.items():
        if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>bounds[1]:
            raise ValueError('Invalid range: '+key)
    for key in ('exposure','shot_peak','radius','aspect'):
        if spec[key][0]<=0:raise ValueError(key+' must be positive')
    for key in ('read_sigma','amplitude','streak_weight'):
        if spec[key][0]<0:raise ValueError(key+' must be nonnegative')
    if spec['exposure'][1]>1:raise ValueError('L exposure must not exceed one')
    lo,hi=spec['source_count']
    if lo<1 or int(lo)!=lo or int(hi)!=hi:raise ValueError('source_count requires positive integer bounds')
    rng=np.random.default_rng(seed)
    def draw(k):return float(rng.uniform(*spec[k]))
    params={k:draw(k) for k in ('exposure','shot_peak','read_sigma')}
    params['seed']=int(seed)
    params['sources']=[{k:draw(k) for k in ('u','v','radius','amplitude','aspect','angle','streak_weight')}
                       for _ in range(int(rng.integers(lo,hi+1)))]
    return params


def flare_field(img, sources, compact=False):
    """Elliptical halo + rotated streak; no scene-reflection/semantic claim."""
    h,w=img.shape[-2:];scale=min(h,w)
    yy,xx=torch.meshgrid(torch.arange(h,device=img.device,dtype=img.dtype),
                         torch.arange(w,device=img.device,dtype=img.dtype),indexing='ij')
    field=torch.zeros_like(img[:,:1])
    for s in sources:
        dx=xx-s['u']*(w-1);dy=yy-s['v']*(h-1)
        c=math.cos(s['angle']);sn=math.sin(s['angle'])
        xr=c*dx+sn*dy;yr=-sn*dx+c*dy;r=s['radius']*scale
        radius2=(xr/(r*s['aspect']))**2+(yr/r)**2
        halo=torch.exp(-.5*radius2)
        streak=torch.exp(-.5*((xr/(3*r))**2+(yr/(.12*r))**2))
        psf=halo+s['streak_weight']*streak
        if compact:psf=torch.where(radius2<=9,psf,torch.zeros_like(psf))
        field=field+s['amplitude']*psf[None,None]
    return field


def degrade(img, condition, params, compact=False, generator=None):
    if condition not in CONDITIONS:raise ValueError(condition)
    if condition=='clean':return img.clone(),torch.zeros_like(img[:,:1])
    signal=to_linear(img)
    if 'L' in condition:signal=signal*params['exposure']
    flare=torch.zeros_like(img[:,:1])
    if 'G' in condition:
        flare=flare_field(img,params['sources'],compact)
        signal=signal+flare
    if 'N' in condition:
        if generator is None:
            generator=torch.Generator(device=img.device).manual_seed(params['seed'])
        # Both background AND added optical radiance enter shot noise.
        signal=torch.poisson(signal.clamp_min(0)*params['shot_peak'],generator=generator)/params['shot_peak']
        # A separate stream preserves the same read-noise draw even when the
        # signal-dependent Poisson sampler consumes different random numbers.
        read_gen=torch.Generator(device=img.device).manual_seed(params['seed'] ^ 0x5DEECE66D)
        signal=signal+torch.randn(signal.shape,device=img.device,dtype=img.dtype,generator=read_gen)*params['read_sigma']
    out=to_srgb(signal.clamp(0,1)).clamp(0,1)
    if compact and condition=='G':
        out=torch.where(flare>0,out,img)  # exact unchanged pixels outside support
    return out,flare


def nighttime_degradation(img, illumination='medium', noise='medium',
                          flare_distance=None, flare_color='white'):
    """Deprecated call signature; old distance names only select radius ranges."""
    warnings.warn('Use sample_parameters/degrade; legacy near/far were not metric distances',DeprecationWarning,stacklevel=2)
    exposure={'mild':(.35,.6),'medium':(.12,.35),'extreme':(.03,.12)}
    peak={'mild':(400.,800.),'medium':(100.,400.),'extreme':(30.,100.)}
    radii={'near':(.12,.25),'medium':(.06,.12),'far':(.02,.06)}
    ranges={}
    if illumination:ranges['exposure']=exposure[illumination]
    if noise:ranges['shot_peak']=peak[noise]
    if flare_distance:ranges['radius']=radii[flare_distance]
    if flare_color!='white':raise ValueError('Chromatic legacy flare is retired; use the achromatic paired protocol')
    params=sample_parameters(int(torch.randint(0,2**31-1,()).item()),ranges)
    condition=('L' if illumination else '')+('N' if noise else '')+('G' if flare_distance else '')
    return degrade(img,condition or 'clean',params)


def generate_condition(img,condition,seed=42):
    return degrade(img,condition,sample_parameters(seed))
