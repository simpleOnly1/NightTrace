"""Nonlocal ViT perturbation propagation in unmodified regions (hypothesis test).

Run *glare_only* to guarantee identical input pixels outside the exposure mask.
Reports descriptive token drift by distance to changed patches; this alone
cannot identify attention as the causal carrier.
"""
from scipy.ndimage import distance_transform_edt
import numpy as np
import torch
import torch.nn.functional as F

def spatial_drift(clean_features,dark_features,patch_hw,pixel_change_mask):
    h,w=patch_hw
    changed=F.interpolate(pixel_change_mask.float(),size=(h,w),mode='nearest')[0,0].cpu().numpy()>.05
    if not changed.any():return []
    # Distance in tokens from the nearest changed patch.
    dt=distance_transform_edt(~changed)
    bands=[('affected',dt==0),('1-2tokens',(dt>0)&(dt<=2)),
           ('3-5tokens',(dt>2)&(dt<=5)),('6+tokens',dt>5)]
    rows=[]
    for layer,(fc,fd) in enumerate(zip(clean_features,dark_features),start=1):
        c=fc[0] if isinstance(fc,tuple) else fc
        d=fd[0] if isinstance(fd,tuple) else fd
        c=F.normalize(c.float(),dim=-1)
        d=F.normalize(d.float(),dim=-1)
        drift=(1-(c*d).sum(-1)).reshape(h,w).cpu().numpy()
        for label,mask in bands:
            if mask.any():rows.append(dict(layer='F%d'%layer,band=label,drift=float(np.mean(drift[mask])),n=int(mask.sum())))
    return rows
