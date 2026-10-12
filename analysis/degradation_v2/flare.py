"""Deprecated radius categories: these never represented metric distances."""
import warnings
import torch
from .pipeline import sample_parameters,degrade


def apply_flare(img,distance='medium',color='white',position=None):
    warnings.warn('distance is a legacy radius label, not metres',DeprecationWarning,stacklevel=2)
    if color!='white':raise ValueError('Use achromatic paired experiments')
    radius={'near':(.12,.25),'medium':(.06,.12),'far':(.02,.06)}[distance]
    ranges={'radius':radius,'source_count':(1,1)}
    if position is not None:
        u=position[0]/max(img.shape[-1]-1,1);v=position[1]/max(img.shape[-2]-1,1)
        ranges.update(u=(u,u),v=(v,v))
    params=sample_parameters(int(torch.randint(0,2**31-1,()).item()),ranges)
    return degrade(img,'G',params)
