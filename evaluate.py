import argparse,json
from evaluation.protocol import evaluate_records
from utils.config import load_config

def main():
    p=argparse.ArgumentParser(description='STEPS-like per-image median-aligned relative depth evaluation')
    p.add_argument('--config',required=True)
    p.add_argument('--pred-dir',required=True)
    p.add_argument('--align',choices=['median','scale_shift','none'])
    p.add_argument('--output')
    a=p.parse_args()
    cfg=load_config(a.config);c=cfg['eval']
    result=evaluate_records(a.pred_dir,a.output,a.align or c.get('align','median'),
                            c['min_depth'],c['max_depth'],dataset=cfg['data']['name'])
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
