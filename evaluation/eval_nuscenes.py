from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Adapted STEPS evaluation entrypoint (native STEPS protocol)."""
import argparse,json
from evaluation.protocol import evaluate_records

def main():
    p=argparse.ArgumentParser()
    p.add_argument('weather',choices=['day','night'],nargs='?',default='night')
    p.add_argument('--pred_dir',default='evaluation/ns_result')
    p.add_argument('--max_depth',type=float,default=60.0)
    p.add_argument('--min_depth',type=float,default=1e-5)
    p.add_argument('--output_file_name')
    p.add_argument('--align',choices=['median','scale_shift','none'],default='median')
    a=p.parse_args()
    x=evaluate_records(a.pred_dir,a.output_file_name,a.align,a.min_depth,a.max_depth,dataset='nuscenes')
    print(json.dumps(x,indent=2))
if __name__=='__main__':main()
