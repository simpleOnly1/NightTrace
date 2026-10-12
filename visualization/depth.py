"""STEPS-style percentile normalized disparity colormap, never metric depth."""
from pathlib import Path
import numpy as np
import cv2

def save_relative_disp(relative,path,percentile=95):
    a=np.asarray(relative,dtype=np.float32)
    if a.ndim!=2:raise ValueError('Expected HxW')
    valid=a[np.isfinite(a)]
    if valid.size==0:raise ValueError('No finite pixels to render')
    low=np.percentile(valid,5);high=np.percentile(valid,percentile)
    u=np.uint8(255*np.clip((np.nan_to_num(a,nan=low)-low)/(high-low+1e-9),0,1))
    color=cv2.applyColorMap(u,cv2.COLORMAP_MAGMA)
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(p),color)
