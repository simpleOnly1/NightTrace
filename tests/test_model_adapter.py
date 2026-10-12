"""Interface-only fake DAV2 test; this is NOT a real ViT-L checkpoint test."""
from pathlib import Path
import sys,tempfile,types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import torch.nn.functional as F

mod=types.ModuleType('depth_anything_v2')
dpt=types.ModuleType('depth_anything_v2.dpt')
class FakeEncoder(nn.Module):
    def __init__(self):super().__init__();self.project=nn.Conv2d(3,8,14,14)
    def get_intermediate_layers(self,x,taps,return_class_token=True):
        f=self.project(x).flatten(2).transpose(1,2)
        return [(f*(i+1),f[:,0]) for i in range(4)]
class FakeDPT(nn.Module):
    def __init__(self):super().__init__();self.proj=nn.Conv2d(8,1,1)
    def forward(self,features,ph,pw):
        x=features[3][0].transpose(1,2).reshape(features[3][0].shape[0],8,ph,pw)
        return self.proj(x)
class FakeDAV2(nn.Module):
    def __init__(self,**kw):super().__init__();self.pretrained=FakeEncoder();self.depth_head=FakeDPT()
dpt.DepthAnythingV2=FakeDAV2
sys.modules['depth_anything_v2']=mod
sys.modules['depth_anything_v2.dpt']=dpt
from models.depth_model import DepthModel

def run():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'fake.pth'
        torch.save(FakeDAV2().state_dict(),p)
        model=DepthModel(str(p),freeze_encoder=True,train_dpt=True,input_size=28)
        x=torch.rand(1,3,32,48)
        out=model(x,return_features=True)
        assert out['disp'].shape==(1,1,32,48)
        assert out['depth_for_warp'].shape==(1,1,32,48)
        assert len(out['features'])==4
        y=out['disp'].mean();y.backward()
        assert model.net.pretrained.project.weight.grad is None
    print('MODEL ADAPTER INTERFACE TEST PASS (fake encoder)')
if __name__=='__main__':run()
