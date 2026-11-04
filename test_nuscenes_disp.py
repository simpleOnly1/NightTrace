"""nuScenes test entry for DAV2 or UniDepth V2, selected by model.name."""
import argparse
from test import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('weather',choices=['day','night'],nargs='?',default='night')
    p.add_argument('--config',default='config/nuscenes.yaml')
    p.add_argument('--checkpoint')
    p.add_argument('--out',default='evaluation/ns_result')
    p.add_argument('--vis',action='store_true')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--evaluate',action='store_true')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    run(a.config,a.out,a.checkpoint,a.vis,a.limit,split_override=a.weather,
        evaluate=a.evaluate,device=a.device)
if __name__=='__main__':main()

