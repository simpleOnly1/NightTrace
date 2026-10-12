"""Paired L/N/G interventions on an approximate linear RGB signal.

Uniform ranges are experiment settings, not calibrated sensor distributions.
Flare size is image-plane extent, never a metric source distance. All conditions
share one sampled parameter set; seeds do not depend on condition iteration.
"""
import math
import warnings
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

CONDITIONS=('clean','L','N','G','LN','LG','NG','LNG')
DEFAULT_RANGES={
    'exposure':(.08,.4), 'shot_peak':(100.,500.), 'read_sigma':(.001,.008),
    'source_count':(1,3), 'u':(-.1,1.1), 'v':(.1,.7),
    'radius':(.025,.15), 'amplitude':(.2,1.2), 'aspect':(.6,1.8),
    'angle':(0.,math.pi), 'streak_weight':(0.,.5),
}


def list_flare_templates(folder):
    """Optional local Flare7K image bank; no download or hidden dependency.

    The fallback analytic halo is used only when folder is None. A specified
    empty/missing bank is a configuration error, not a silent fallback.
    """
    if not folder:
        return []
    path=Path(folder).expanduser()
    if not path.is_dir():
        raise FileNotFoundError('Flare bank folder not found: '+str(path))
    extensions={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
    def images(directory):
        return sorted(str(p.resolve()) for p in directory.rglob('*')
                      if p.is_file() and p.suffix.lower() in extensions)
    # Official Flare7K++ uses scattering Compound_Flare, with optional
    # Reflective_Flare. Core, Streak and Light_Source are components, not
    # independent complete scattering templates.
    scattering=path/'Scattering_Flare'
    reflection=path/'Reflective_Flare'
    if scattering.is_dir() or reflection.is_dir():
        compound=scattering/'Compound_Flare'
        if not compound.is_dir():
            raise FileNotFoundError('Flare7K++ Compound_Flare folder missing: '+str(compound))
        scatter=images(compound)
        if not scatter:
            raise ValueError('No Compound_Flare images found: '+str(compound))
        reflected=images(reflection) if reflection.is_dir() else []
        if reflection.is_dir() and not reflected:
            raise ValueError('Reflective_Flare folder contains no images: '+str(reflection))
        return scatter+reflected
    files=images(path)  # Backward compatible with a custom flat image bank.
    if not files:
        raise ValueError('Flare bank contains no image templates: '+str(path))
    return files


@lru_cache(maxsize=256)
def _flare_gray(path):
    """Decode a template once to 128x128 luminance, preserving original shape."""
    a=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if a is None:
        raise FileNotFoundError('Cannot decode flare template: '+str(path))
    a=cv2.resize(a,(128,128),interpolation=cv2.INTER_AREA)
    a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    if not np.isfinite(a).all() or float(a.max())<=0:
        raise ValueError('Flare template is empty/invalid: '+str(path))
    return a


def to_linear(rgb):
    return torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).clamp_min(0)**2.4)


def to_srgb(x):
    x=x.clamp_min(0)
    return torch.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)


def sample_parameters(seed, ranges=None, flare_templates=None):
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
    if flare_templates:
        bank=sorted(str(p) for p in flare_templates)
        scatter=[p for p in bank if 'Reflective_Flare' not in Path(p).parts]
        reflected=[p for p in bank if 'Reflective_Flare' in Path(p).parts]
        if not scatter:
            raise ValueError('At least one scattering flare template is required')
        for source in params['sources']:
            source['template']=scatter[int(rng.integers(len(scatter)))]
            # Optional reflection: 50% is an experimental knob, not an
            # estimated frequency of reflections in real road scenes.
            if reflected and rng.random()<0.5:
                source['reflective_template']=reflected[int(rng.integers(len(reflected)))]
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
        if 'template' in s:
            # Flare7K already supplies the optical shape: avoid re-imposing a
            # Gaussian/streak on it. Image-plane radius is NOT distance in m.
            template=torch.as_tensor(_flare_gray(s['template']),device=img.device,dtype=img.dtype)[None,None]
            grid=torch.stack((xr/(r*s['aspect']),yr/r),dim=-1)[None]
            psf=F.grid_sample(template,grid,mode='bilinear',padding_mode='zeros',align_corners=True)[0,0]
            if 'reflective_template' in s:
                reflection=torch.as_tensor(_flare_gray(s['reflective_template']),
                                           device=img.device,dtype=img.dtype)[None,None]
                reflection=F.grid_sample(reflection,grid,mode='bilinear',
                                         padding_mode='zeros',align_corners=True)[0,0]
                psf=(psf+0.5*reflection).clamp(0,1)
            if compact:
                psf=torch.where(radius2<=1,psf,torch.zeros_like(psf))
        else:
            # Explicit analytic fallback when external assets are not supplied.
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
