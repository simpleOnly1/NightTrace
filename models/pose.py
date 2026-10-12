"""Compact Monodepth-style 6DoF PoseNet. Relative poses, no GT supervision."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def axis_angle_to_matrix(v):
    angle=torch.norm(v,dim=-1,keepdim=True).clamp_min(1e-8)
    axis=v/angle
    x,y,z=axis.unbind(-1)
    zeros=torch.zeros_like(x)
    K=torch.stack([zeros,-z,y,z,zeros,-x,-y,x,zeros],dim=-1).reshape(-1,3,3)
    eye=torch.eye(3,device=v.device,dtype=v.dtype)[None].expand(v.size(0),-1,-1)
    return eye+torch.sin(angle)[...,None]*K+(1-torch.cos(angle))[...,None]*torch.bmm(K,K)


def pose_matrix(v):
    R=axis_angle_to_matrix(v[:,:3])
    T=torch.eye(4,device=v.device,dtype=v.dtype).unsqueeze(0).repeat(v.shape[0],1,1)
    T[:,:3,:3]=R
    T[:,:3,3]=v[:,3:]
    return T


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths=[6,32,64,128,256]
        layers=[]
        for idx in range(4):
            layers.extend([nn.Conv2d(widths[idx],widths[idx+1],3,stride=2,padding=1),nn.ReLU(inplace=True)])
        self.encoder=nn.Sequential(*layers)
        self.fc=nn.Linear(256,6)
        nn.init.normal_(self.fc.weight,mean=0,std=1e-4)
        nn.init.zeros_(self.fc.bias)

    def forward(self,target,source):
        x=torch.cat([target,source],1)
        feat=self.encoder(x)
        v=self.fc(feat.mean(dim=(2,3)))*.01
        return pose_matrix(v)
