from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from evaluation.metrics import compute_metrics
from evaluation.protocol import eval_pair

def main():
    gt = np.array([[1.,2.],[3.,4.]],dtype=np.float32)
    pred = np.array([[1.,2.],[3.,4.]],dtype=np.float32)
    scores = compute_metrics(pred,gt)
    assert scores['abs_rel']==0
    assert scores['rmse']==0
    assert scores['a1']==1
    scores2 = eval_pair(pred*2.,gt,min_depth=.1,max_depth=100.,align='median')
    assert scores2['abs_rel'] < 1e-6
    print('PASS: native STEPS metrics and median alignment')

if __name__=='__main__':main()
