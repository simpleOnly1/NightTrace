
import torch

def apply_illumination(img, level="medium"):
    params={"mild":(0.85,0.6),"medium":(1.2,0.35),"extreme":(1.6,0.12)}
    gamma,scale=params[level]
    return torch.clamp(img**gamma*scale,0,1)
