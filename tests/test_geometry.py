from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from losses.photometric import warp_source,SSIM,reprojection
from models.pose import PoseNet

def run():
    B,H,W=1,32,48
    x=torch.rand(B,3,H,W,requires_grad=True)
    d=torch.full((B,1,H,W),8.)
    K=torch.tensor([[[30.,0,24],[0,30,16],[0,0,1]]])
    T=torch.eye(4)[None]
    warped,mask=warp_source(x,d,K,T)
    assert torch.allclose(x,warped,atol=2e-5)
    pose=PoseNet()
    loss,_=reprojection(x,[x.detach(),x.detach()],d,torch.rand(B,1,H,W),K,pose,SSIM())
    assert torch.isfinite(loss)
    loss.backward()
    print('GEOMETRY TESTS PASS')
if __name__=='__main__':run()
