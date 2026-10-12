"""Monodepth2/STEPS-style projection, minimum reprojection, automask, SSIM, smoothness."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def warp_source(source,depth,K,T):
    B,C,H,W=source.shape
    ys,xs=torch.meshgrid(torch.arange(H,device=source.device,dtype=source.dtype),
                         torch.arange(W,device=source.device,dtype=source.dtype))
    pix=torch.stack((xs,ys,torch.ones_like(xs)),0).reshape(3,-1)[None].expand(B,-1,-1)
    cam=torch.bmm(torch.inverse(K),pix)*depth.reshape(B,1,-1)
    homo=torch.cat((cam,torch.ones(B,1,H*W,device=source.device,dtype=source.dtype)),1)
    cam_s=torch.bmm(T,homo)[:,:3]
    proj=torch.bmm(K,cam_s)
    z=proj[:,2:3].clamp(min=1e-6)
    u=(proj[:,0:1]/z).reshape(B,H,W)
    v=(proj[:,1:2]/z).reshape(B,H,W)
    grid=torch.stack((2*u/max(W-1,1)-1,2*v/max(H-1,1)-1),-1)
    warped=F.grid_sample(source,grid,mode='bilinear',padding_mode='border',align_corners=True)
    valid=(grid.abs()<=1).all(-1).unsqueeze(1)
    return warped,valid

class SSIM(nn.Module):
    def __init__(self):super().__init__()
    def forward(self,x,y):
        x=F.pad(x,(1,1,1,1),mode='reflect'); y=F.pad(y,(1,1,1,1),mode='reflect')
        mu_x=F.avg_pool2d(x,3,1);mu_y=F.avg_pool2d(y,3,1)
        s_x=F.avg_pool2d(x*x,3,1)-mu_x*mu_x
        s_y=F.avg_pool2d(y*y,3,1)-mu_y*mu_y
        s_xy=F.avg_pool2d(x*y,3,1)-mu_x*mu_y
        return torch.clamp((1-(2*mu_x*mu_y+.01**2)*(2*s_xy+.03**2)/
                            ((mu_x*mu_x+mu_y*mu_y+.01**2)*(s_x+s_y+.03**2))),0,2)/2

def smoothness(disp,img):
    d=disp/(disp.mean(dim=(2,3),keepdim=True)+1e-6)
    dx=(d[:,:,:,1:]-d[:,:,:,:-1]).abs();dy=(d[:,:,1:,:]-d[:,:,:-1,:]).abs()
    gx=(img[:,:,:,1:]-img[:,:,:,:-1]).abs().mean(1,keepdim=True)
    gy=(img[:,:,1:,:]-img[:,:,:-1,:]).abs().mean(1,keepdim=True)
    return (dx*torch.exp(-gx)).mean()+(dy*torch.exp(-gy)).mean()

def reprojection(target,sources,depth,disp,K,pose_model,ssim,lam_s=0.001):
    losses=[];identity=[]
    for src in sources:
        pose=pose_model(target,src)
        w,valid=warp_source(src,depth,K,pose)
        l=.85*ssim(w,target).mean(1,keepdim=True)+.15*(w-target).abs().mean(1,keepdim=True)
        l=torch.where(valid,l,torch.full_like(l,1e3))
        losses.append(l)
        identity.append(.85*ssim(src,target).mean(1,keepdim=True)+.15*(src-target).abs().mean(1,keepdim=True))
    projected=torch.cat(losses,1)
    # Monodepth2 automasking: identity error acts as a competing candidate.
    ident=torch.cat(identity,1)
    stacked=torch.cat([projected,ident+torch.randn_like(ident)*1e-5],dim=1)
    best=stacked.min(1).values.mean()
    return best+lam_s*smoothness(disp,target),{'photo':float(best.detach()),'smooth':float(smoothness(disp,target).detach())}
