"""Paired effects and group bootstrap. Replicates/tokens are not independent N."""
import csv
from collections import defaultdict
import numpy as np
from .degradation_v2.pipeline import CONDITIONS


def write_csv(path, rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def factorial_effects(rows, metric='depth_drift'):
    paired=defaultdict(dict)
    for r in rows:paired[(r['id'],r['group'],r['repeat'])][r['condition']]=r[metric]
    result=[]
    for (sid,group,repeat),v in paired.items():
        if not set(CONDITIONS)<=v.keys():continue
        e0=v['clean']
        result.append(dict(id=sid,group=group,repeat=repeat,
            L=v['L']-e0,N=v['N']-e0,G=v['G']-e0,
            LN=v['LN']-v['L']-v['N']+e0,
            LG=v['LG']-v['L']-v['G']+e0,
            NG=v['NG']-v['N']-v['G']+e0,
            LNG=v['LNG']-v['LN']-v['LG']-v['NG']+v['L']+v['N']+v['G']-e0))
    return result


def group_estimate(rows, key, seed=42, n_boot=1000):
    images=defaultdict(list)
    for r in rows:
        if np.isfinite(r[key]):images[(r['group'],r['id'])].append(r[key])
    groups=defaultdict(list)
    for (group,_),vals in images.items():groups[group].append(np.mean(vals))
    vals=np.array([np.mean(v) for _,v in sorted(groups.items())])
    if not len(vals):return dict(mean=float('nan'),low=float('nan'),high=float('nan'),groups=0,images=0)
    if len(vals)>1:
        rng=np.random.default_rng(seed)
        boot=vals[rng.integers(len(vals),size=(n_boot,len(vals)))].mean(1)
        low,high=np.quantile(boot,[.025,.975])
    else:low=high=float('nan')
    return dict(mean=float(vals.mean()),low=float(low),high=float(high),groups=len(vals),images=len(images))


def save_statistics(out, rows, seed):
    metadata={'id','group','repeat','condition','parameter_seed'}
    numeric=[k for k in rows[0] if k not in metadata]
    summary=[];intervals=[]
    for condition in dict.fromkeys(r['condition'] for r in rows):
        subset=[r for r in rows if r['condition']==condition]
        record=dict(condition=condition,n=len({r['id'] for r in subset}),replicates=len(subset))
        for key in numeric:
            est=group_estimate(subset,key,seed)
            record[key]=est['mean']
            if key in ('depth_drift','ordinal_error'):
                intervals.append(dict(condition=condition,metric=key,**est))
        summary.append(record)
    write_csv(out/'summary.csv',summary)
    write_csv(out/'group_intervals.csv',intervals)
    effects=factorial_effects(rows)
    write_csv(out/'interactions_per_image.csv',effects)
    if effects:
        write_csv(out/'interactions_summary.csv',[
            dict(effect=k,**group_estimate(effects,k,seed)) for k in ('L','N','G','LN','LG','NG','LNG')])
