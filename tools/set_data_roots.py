"""Set local dataset roots inside NightTrace configs without moving datasets."""
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robotcar', help='Absolute directory containing RobotCar packages')
    parser.add_argument('--nuscenes', help='Absolute nuScenes-Night directory')
    args = parser.parse_args()
    if not args.robotcar and not args.nuscenes:
        parser.error('Specify --robotcar and/or --nuscenes')
    for name, path in [('robotcar', args.robotcar), ('nuscenes', args.nuscenes)]:
        if not path:
            continue
        absolute = Path(path).resolve()
        if not absolute.is_dir():
            raise FileNotFoundError(absolute)
        conf = ROOT / 'config' / ('%s.yaml' % name)
        data = yaml.safe_load(conf.read_text(encoding='utf-8'))
        data['data']['root'] = str(absolute)
        conf.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
        print('%s dataset.root = %s' % (name, absolute))

if __name__ == '__main__':
    main()
