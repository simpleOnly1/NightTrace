import numpy as np
import torch
import torch.nn.functional as F

def as_grid(feat):
    patches=feat[0] if isinstance(feat,(list,tuple)) else feat
    if patches.ndim==4:return patches
    B,N,C=patches.shape
    H=int(N**.5)
    if H*H!=N:raise ValueError('Need token H,W; use pass grid_size for non-square tokens')
    return patches.transpose(1,2).reshape(B,C,H,H)

def relation_vector(patch_tokens, grid_hw):
    """Local cosine-distance, 4 positive offsets = 8-neighborhood undirected."""
    tokens=patch_tokens[0] if isinstance(patch_tokens,(tuple,list)) else patch_tokens
    B,N,C=tokens.shape;h,w=grid_hw
    if N!=h*w:raise ValueError('Token count does not match patch grid')
    f=F.normalize(tokens.float(),dim=-1).reshape(B,h,w,C)
    vals=[]
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        aa=f[:,max(0,-dy):h-max(0,dy),max(0,-dx):w-max(0,dx)]
        bb=f[:,max(0,dy):h-max(0,-dy),max(0,dx):w-max(0,-dx)]
        vals.append((1-(aa*bb).sum(-1)).flatten(1))
    return torch.cat(vals,dim=1)

def spearman(x,y):
    # no scipy dependence; average ranks for repeated values
    from scipy.stats import rankdata
    x=rankdata(x);y=rankdata(y)
    if np.std(x)<1e-14 or np.std(y)<1e-14:return float('nan')
    return float(np.corrcoef(x,y)[0,1])

def affine_fit(pred,target):
    p=pred.detach().double().reshape(-1).cpu().numpy()
    t=target.detach().double().reshape(-1).cpu().numpy()
    valid=np.isfinite(p)&np.isfinite(t)
    p=p[valid];t=t[valid]
    if not len(p):raise ValueError('No finite values for affine fit')
    pp=p-p.mean();tt=t-t.mean()
    a=max(float(np.dot(pp,tt)/max(np.dot(pp,pp),1e-20)),1e-12)
    b=float(t.mean()-a*p.mean())
    denom=np.median(np.abs(t-np.median(t)))+1e-6
    return dict(error=float(np.mean(np.abs(a*p+b-t))/denom),scale=a,shift=b)


def affine_error(pred,target):
    """Positive-slope affine consistency; target may be a clean prediction, NOT GT."""
    return affine_fit(pred,target)['error']


def ordinal_error(pred,target,seed=0,pairs=4096):
    """Order disagreement in the same representation; half penalty for pred ties."""
    p=pred.detach().float().reshape(-1).cpu().numpy()
    t=target.detach().float().reshape(-1).cpu().numpy()
    rng=np.random.default_rng(seed)
    i=rng.integers(len(p),size=pairs);j=rng.integers(len(p),size=pairs)
    dp=p[i]-p[j];dt=t[i]-t[j]
    good=np.isfinite(dp)&np.isfinite(dt)&(np.abs(dt)>1e-6)
    if not good.any():return float('nan')
    return float(np.mean(np.where(np.abs(dp[good])<=1e-6,.5,(dp[good]*dt[good]<0).astype(float))))
