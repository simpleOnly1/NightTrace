"""Controlled physical factors for NightTrace diagnosis, not a restoration network.

The controlled conditions separate photometric attenuation, read/shot noise,
chromatic veiling and achromatic overexposure. Synthetic ≠ real sensor RAW.
"""
import torch
import torch.nn.functional as F


def to_linear(rgb):
    return torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).clamp_min(0)**2.4)

def to_srgb(x):
    x=x.clamp_min(0)
    return torch.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)

def glare_mask(x,center=(.68,.32),sigma=.09):
    _,_,h,w=x.shape
    yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=x.device),
                         torch.linspace(0,1,w,device=x.device))
    dist=(xx-center[0])**2+(yy-center[1])**2
    mask=torch.exp(-dist/(2*sigma**2))
    # Compact support so non-exposed pixels are EXACTLY unchanged for propagation tests.
    mask=torch.where(dist <= (3*sigma)**2,mask,torch.zeros_like(mask))
    return mask.view(1,1,h,w)

def corrupt(rgb,kind='low',gain=.12,shot_peak=350,read_sigma=.003,
            glare_strength=2.0,generator=None):
    kinds=('low','low_noise','low_colored_glare','low_white_glare',
           'low_noise_colored_glare','low_noise_white_glare',
           'colored_glare_only','white_glare_only')
    if kind not in kinds:raise ValueError(kind)
    lin=to_linear(rgb)
    use_low=kind.startswith('low')
    lin=lin*(gain if use_low else 1.0)
    mask=torch.zeros_like(lin[:,:1])
    if 'glare' in kind:
        mask=glare_mask(rgb)
        # additive radiance creates veiling and saturation; not a polygon overlay
        gains=[1.0,.18,.09] if 'colored' in kind else [1.,1.,1.]
        g=lin.new_tensor(gains).view(1,3,1,1)
        lin=lin+glare_strength*mask*g
    if 'noise' in kind:
        # Sensor-domain rough approximation; intensity-dependent Poisson noise.
        lin=torch.poisson(lin.clamp_min(0)*shot_peak)/shot_peak
        lin=lin+torch.randn(lin.shape,device=lin.device,dtype=lin.dtype,
                             generator=generator)*read_sigma
    out=to_srgb(lin.clamp(0,1)).clamp(0,1)
    return out,mask
