"""Check config/files/splits without requesting a GPU or loading the 335M DAV2."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.config import load_config,project_path
from datasets import make_dataset

def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True)
    a.add_argument('--stage',default='day',choices=['day','night','test'])
    x=a.parse_args();cfg=load_config(x.config)
    w=project_path(cfg['model']['weights'])
    dav=project_path('models/depth_anything_v2/dpt.py')
    print('Weights:',w,'EXISTS' if w.is_file() else 'MISSING')
    print('Official DAV2 source:',dav,'EXISTS' if dav.is_file() else 'MISSING (run tools/prepare_sources.py --dav2-only)')
    stage='night' if x.stage=='test' else x.stage
    ds=make_dataset(cfg,stage,training=(x.stage!='test'))
    print('Dataset length:',len(ds),'Sample #0:')
    sample=ds[0]
    for key,val in sample.items():
        print(' ',key,tuple(val.shape) if hasattr(val,'shape') else str(val)[:180])
    if not w.is_file() or not dav.is_file():
        print('WARNING: runtime model load will fail until missing sources/weights are supplied')

if __name__=='__main__':main()
