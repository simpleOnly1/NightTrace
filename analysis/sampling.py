"""Small, reproducible mechanism subsets from existing STEPS training splits.

No GT, predicted error, pose model, or JSON manifest is needed. The official
night test split is never filtered. Visual similarity is not a motion label.
"""
import csv
import hashlib
import re
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from datasets.common import image_file, read_tokens, rgb
from utils.config import project_path


def stable_seed(seed, sample_id, repeat=0):
    return int.from_bytes(hashlib.sha256(f'{seed}|{sample_id}|{repeat}'.encode()).digest()[:8], 'big') % (2**31)


def candidates(cfg, block_seconds=30):
    d=cfg['data'];root=project_path(d['root'])
    if block_seconds<=0:raise ValueError('block_seconds must be positive')
    if d['name']=='robotcar':
        base=root/d['day_package']
        paths=[image_file(base/'rgb',Path(t).name) for t in read_tokens(base/'train_split.txt')]
    else:
        from datasets.nuscenes import make_nuscenes
        # No temporal neighbours are required for frozen single-image diagnosis.
        ds=make_nuscenes(dict(d,offsets=[]),'day',True)
        paths=[r['target'] for r in ds.records]
    rows=[]
    for path in sorted(set(paths)):
        sid=path.relative_to(root).as_posix()
        seq=path.parent.relative_to(root).as_posix()
        matches=re.findall(r'(?<!\d)(\d{16})(?!\d)',path.stem)
        timestamp=int(matches[-1])/1e6 if matches else None
        # nuScenes directory is a clip proxy, not a guaranteed independent scene.
        group=seq
        if d['name']=='robotcar' and timestamp is not None:
            group+=f'/block_{int(timestamp//block_seconds)}'
        rows.append(dict(path=path,id=sid,sequence=seq,group=group,timestamp=timestamp))
    ranks=defaultdict(int)
    for row in rows:
        row['frame_rank']=ranks[row['sequence']];ranks[row['sequence']]+=1
    if not rows:raise ValueError('Empty daytime mechanism split')
    return rows


def select_samples(cfg, limit=512, seed=42, sample_list=None):
    settings=cfg.get('diagnosis',{})
    rows=candidates(cfg,settings.get('block_seconds',30))
    if limit<0:raise ValueError('limit must be nonnegative; 0 means all eligible samples')
    if sample_list:
        by_id={r['id']:r for r in rows}
        ids=[s.strip() for s in Path(sample_list).read_text().splitlines() if s.strip() and not s.startswith('#')]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate entries in sample list')
        missing=set(ids)-by_id.keys()
        if missing:raise ValueError('Sample list contains entries outside the allowed day split: '+str(sorted(missing)[:3]))
        selected=[by_id[s] for s in ids]
        if not selected:raise ValueError('Empty sample list')
        return selected,[dict(group=g,candidates=sum(r['group']==g for r in rows),
                              selected=sum(r['group']==g for r in selected),time_rejected=0,duplicate_rejected=0)
                         for g in sorted({r['group'] for r in rows})]
    rng=np.random.default_rng(seed)
    buckets=defaultdict(list)
    for row in rows:buckets[row['group']].append(row)
    audit={g:dict(group=g,candidates=len(v),selected=0,time_rejected=0,duplicate_rejected=0) for g,v in buckets.items()}
    for v in buckets.values():rng.shuffle(v)
    groups=sorted(buckets);rng.shuffle(groups)
    accepted=defaultdict(list);thumbs=defaultdict(list);selected=[]
    min_seconds=settings.get('min_gap_seconds',1.0)
    min_frames=settings.get('min_gap_frames',10)
    threshold=settings.get('duplicate_mae',.015)
    if min(min_seconds,min_frames,threshold)<0:raise ValueError('Sampling thresholds must be nonnegative')
    d=cfg['data'];crop=(d.get('crop_width',1152),d.get('crop_height',640)) if d['name']=='robotcar' else None
    while groups and (not limit or len(selected)<limit):
        remaining=[]
        for group in groups:
            bucket=buckets[group]
            while bucket:
                row=bucket.pop();seq=row['sequence'];ts=row['timestamp']
                too_close=any(abs(ts-r['timestamp'])<min_seconds if ts is not None and r['timestamp'] is not None
                              else abs(row['frame_rank']-r['frame_rank'])<min_frames for r in accepted[seq])
                if too_close:audit[group]['time_rejected']+=1;continue
                if threshold>0:
                    im,_=rgb(row['path'],(18,32),crop=crop)
                    thumb=im.mean(0).numpy()
                    if any(float(np.abs(thumb-old).mean())<threshold for old in thumbs[seq]):
                        audit[group]['duplicate_rejected']+=1;continue
                    thumbs[seq].append(thumb)
                accepted[seq].append(row);selected.append(row);audit[group]['selected']+=1
                break
            if bucket:remaining.append(group)
            if limit and len(selected)>=limit:break
        groups=remaining
    if not selected:raise ValueError('No eligible samples after filtering')
    if limit and len(selected)<limit:warnings.warn(f'Requested {limit}, selected {len(selected)} eligible samples; no duplicate padding')
    return selected,list(audit.values())


def save_selection(out, rows, audit):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'samples.txt').write_text('\n'.join(r['id'] for r in rows)+'\n')
    for name,records in [('sampling_coverage.csv',audit),('samples.csv',[{k:v for k,v in r.items() if k!='path'} for r in rows])]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
