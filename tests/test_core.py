from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from evaluation.metrics import evaluate_single
from analysis.degradation import corrupt
from analysis.relations import relation_vector,affine_error
from models.pose import axis_angle_to_matrix,pose_matrix

def run():
    gt=np.array([[2.,4.],[8.,16.]])
    pred=1/(gt*5.)
    met,aligned,mask=evaluate_single(pred,gt,'median','relative_disparity')
    assert met['abs_rel']<1e-8,met
    met2,_,_=evaluate_single(gt*2+1,gt,'scale_shift','relative_depth')
    assert met2['abs_rel']<1e-7
    a=torch.rand(1,3,64,96)
    for cond in ['low','low_noise','low_colored_glare','low_white_glare']:
        b,g=corrupt(a,cond)
        assert b.shape==a.shape and torch.isfinite(b).all()
    I=axis_angle_to_matrix(torch.zeros(2,3))
    assert torch.allclose(I,torch.eye(3)[None].expand(2,-1,-1),atol=1e-6)
    f=torch.randn(2,6*8,32)
    r=relation_vector(f,(6,8))
    assert r.shape[0]==2 and torch.allclose(r,relation_vector(f,(6,8)))
    assert affine_error(a,a)<1e-5
    print('CORE TESTS PASS')

if __name__=='__main__':run()
