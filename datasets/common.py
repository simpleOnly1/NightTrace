"""RNW/STEPS-shaped inputs. No GT is read by a training dataset."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS=('.png','.jpg','.jpeg')

def read_tokens(txt):
    """Read split entries, ignoring blank lines, comments and separators."""
    import re

    tokens = []
    with Path(txt).open(encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            token = line.split()[0]

            # Split-file visual separators, not image names.
            if re.fullmatch(r'[-=_*]{3,}', token):
                continue

            tokens.append(token)

    return tokens

def image_file(root, item):
    p=Path(root)/str(item)
    if p.is_file():return p
    if p.suffix:
        raise FileNotFoundError(p)
    for ext in IMG_EXTS:
        candidate=p.with_suffix(ext)
        if candidate.is_file():return candidate
    raise FileNotFoundError(p)

def rgb(path, size=None, crop=None):
    img=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if img is None:raise FileNotFoundError(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if crop is not None:
        cw,ch=crop
        h,w=img.shape[:2]
        if h<ch or w<cw:raise ValueError('Requested crop (%d,%d) larger than %s (%d,%d)'%(cw,ch,path,w,h))
        sx=(w-cw)//2;sy=(h-ch)//2
        img=img[sy:sy+ch,sx:sx+cw]
    orig=img.shape[:2]
    if size is not None:
        img=cv2.resize(img,(size[1],size[0]),interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2,0,1).float()/255.,orig

def read_K(path, original_hw, new_hw, crop=None):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError('Intrinsic required: '+str(path))
    K=np.load(str(path)).astype(np.float32)
    if K.shape not in [(3,3),(4,4)]:raise ValueError('K should be 3x3 or 4x4: '+str(path))
    K=K[:3,:3].copy()
    oh,ow=original_hw;nh,nw=new_hw
    # K in raw image pixel coordinates; crop BEFORE resizing.
    if crop is not None:
        cw,ch=crop
        K[0,2]-=(ow-cw)//2
        K[1,2]-=(oh-ch)//2
        oh,ow=ch,cw
    K[0,:]*=nw/ow
    K[1,:]*=nh/oh
    return torch.tensor(K,dtype=torch.float32)

def original_size(path):
    img=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if img is None:raise FileNotFoundError(path)
    return img.shape[:2]

class ClipDataset(Dataset):
    def __init__(self,records,height,width,offsets=(-1,1),crop=None):
        self.records=records
        self.height,self.width=height,width
        self.offsets=tuple(offsets)
        self.crop=crop
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        rec=self.records[i]
        target=rec['target']
        x,_=rgb(target,(self.height,self.width),crop=self.crop)
        K=read_K(rec['K'],original_size(target),(self.height,self.width),self.crop)
        samples={'image':x,'K':K,'id':rec['id']}
        for offset in self.offsets:
            source=rec['sources'][offset]
            sx,_=rgb(source,(self.height,self.width),crop=self.crop)
            samples['source_%+d'%offset]=sx
        return samples

class FrameDataset(Dataset):
    """Eval returns unprocessed center-cropped rgb and optional GT path (no GT loaded)."""
    def __init__(self,records,crop=None,size=None):self.records=records;self.crop=crop;self.size=size
    def __len__(self):return len(self.records)
    def __getitem__(self,idx):
        rec=self.records[idx]
        x,hw=rgb(rec['target'],size=self.size,crop=self.crop)
        return {'image':x,'id':rec['id'],'gt_path':str(rec.get('gt','')),'height':hw[0],'width':hw[1]}

def seq_records(frames,K,split_keys=None,offsets=(-1,1),strict=True):
    """Never construct temporal pairs across recording boundaries."""
    frames=[Path(x) for x in frames]
    if split_keys is None:inds=range(len(frames))
    else:
        keys={Path(k).stem for k in split_keys}
        inds=[i for i,x in enumerate(frames) if x.stem in keys]
        if strict and len(inds)==0:raise ValueError('Split did not match any frames')
    result=[]
    for i in inds:
        if any(i+d<0 or i+d>=len(frames) for d in offsets):continue
        rec={'target':frames[i],'sources':{d:frames[i+d] for d in offsets},
             'K':K,'id':frames[i].stem}
        result.append(rec)
    if not result:raise ValueError('No valid temporal triplets in data/split')
    return result
