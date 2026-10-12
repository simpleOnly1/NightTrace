"""NightTrace train.py — ACDepth-style config entrypoint."""
import argparse
from trainer import fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['day', 'night'], default='day')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    fit(args.config, args.stage, args.smoke)


if __name__ == '__main__':
    main()
