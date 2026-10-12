"""STEPS-compatible RobotCar test entry, using DAV2 relative-depth output contract."""
import argparse
from test import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('weather', choices=['day','night'], nargs='?', default='night')
    p.add_argument('--config',default='config/robotcar.yaml')
    p.add_argument('--checkpoint')
    p.add_argument('--out',default='evaluation/rc_result')
    p.add_argument('--vis',action='store_true')
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    run(a.config,a.out,a.checkpoint,a.vis,a.limit,split_override=a.weather)
if __name__=='__main__':main()
