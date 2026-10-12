
import torch

def apply_flare(img,distance="medium",color="white",position=None):
    B,C,H,W=img.shape
    device=img.device
    if position is None:
        cx,cy=W//2,H//2
    else:
        cx,cy=position
    sigma={"near":min(H,W)*0.25,
           "medium":min(H,W)*0.12,
           "far":min(H,W)*0.045}[distance]
    amp={"near":1.5,"medium":0.9,"far":0.5}[distance]

    y,x=torch.meshgrid(torch.arange(H,device=device),
                       torch.arange(W,device=device),
                       indexing="ij")
    psf=torch.exp(-((x-cx)**2+(y-cy)**2)/(2*sigma*sigma))
    psf=(psf/psf.max()).view(1,1,H,W)

    if color=="white":
        rgb=torch.ones(1,3,1,1,device=device)
    elif color=="red":
        rgb=torch.tensor([1.,0.25,0.15],device=device).view(1,3,1,1)
    else:
        rgb=torch.tensor([0.2,0.35,1.],device=device).view(1,3,1,1)

    flare_img = torch.clamp(
        img + amp * psf * rgb,
        0,
        1
    )

    flare_mask = psf

    return flare_img, flare_mask
