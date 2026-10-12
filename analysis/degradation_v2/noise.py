
import torch

def apply_shot_read_noise(img, level="medium"):
    photons={"mild":80.,"medium":25.,"extreme":8.}[level]
    signal=torch.clamp(img*photons,0,None)
    shot=torch.poisson(signal)/photons
    read=torch.randn_like(img)*{"mild":0.005,"medium":0.01,"extreme":0.02}[level]
    return torch.clamp(shot+read,0,1)
