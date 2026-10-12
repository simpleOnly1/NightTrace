from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from evaluation.protocol import eval_pair
from evaluation.metrics import relative_to_depth

def run():
    q=np.array([[2.,4.],[8.,16.]],dtype=np.float32)
    d=relative_to_depth(q)
    assert np.allclose(d,np.array([[.5,.25],[.125,.0625]])),d
    g=d*7.
    a=eval_pair(d,g,min_depth=1e-5,max_depth=60)
    assert a['abs_rel']<1e-7,a
    q2=q*5.
    d2=relative_to_depth(q2)
    b=eval_pair(d2,g,min_depth=1e-5,max_depth=60)
    assert b['abs_rel']<1e-7,b
    resized=cv2.resize(d,(4,4),interpolation=cv2.INTER_NEAREST)
    c=eval_pair(d,resized*3,min_depth=1e-5,max_depth=60)
    assert c['abs_rel']<1e-7,c
    print('PASS: q->relative depth, median scaling, nearest pred resize')
if __name__=='__main__':run()
