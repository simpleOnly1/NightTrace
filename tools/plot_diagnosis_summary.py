import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


LAYER_KEYS = ['LRR_F1', 'LRR_F2', 'LRR_F3', 'LRR_F4']


def plot_lrr(df, out_dir):
    plt.figure(figsize=(10, 6))
    for _, row in df.iterrows():
        y = [row[k] for k in LAYER_KEYS]
        plt.plot(['F1', 'F2', 'F3', 'F4'], y, marker='o', label=row['condition'])
    plt.ylim(0, 1.05)
    plt.ylabel('LRR')
    plt.title('Layer-wise structural consistency')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / 'lrr_curves.png', dpi=200)
    plt.close()


def plot_depth_drift(df, out_dir):
    plt.figure(figsize=(10, 5))
    plt.bar(df['condition'], df['depth_drift'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Depth drift')
    plt.title('Depth prediction drift under different degradations')
    plt.tight_layout()
    plt.savefig(out_dir / 'depth_drift.png', dpi=200)
    plt.close()


def plot_recovery(df, out_dir):
    x = range(len(df))
    w = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - w/2 for i in x], df['recovery_F4'], width=w, label='Swap F4')
    plt.bar([i + w/2 for i in x], df['recovery_F34'], width=w, label='Swap F3+F4')
    plt.xticks(list(x), df['condition'], rotation=45, ha='right')
    plt.ylabel('Recovery')
    plt.title('Oracle feature swap recovery')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'recovery.png', dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    plot_lrr(df, out_dir)
    plot_depth_drift(df, out_dir)
    plot_recovery(df, out_dir)

    print('Saved figures to', out_dir)


if __name__ == '__main__':
    main()