"""Dose grids with one clean-reference q scale and one fixed error scale."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .qualitative import tensor_rgb_to_uint8, depth_to_numpy


def save_sweep_panels(out, frames, predictions, rows, conditions, levels, title):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    nrows, ncols = len(conditions), len(levels)+1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.8*nrows), squeeze=False)
    for i, c in enumerate(conditions):
        for j, s in enumerate([0.] + list(levels)):
            key = ('clean', 0.) if s == 0 else (c, s)
            row = rows[key]; ax = axes[i, j]
            image = tensor_rgb_to_uint8(frames[key]); ax.imshow(image); ax.axis('off')
            ax.set_title(f'{c} / S{s:g} | Y50={row["luma_p50"]:.3f}\n'
                         f'black={row["black_area"]:.1%}  sat={row["saturated_area"]:.1%}\n'
                         f'glare={row["glare_visible_area"]:.1%}  noise RMS={row["noise_rgb_rms"]:.3f}', fontsize=9)
            if i == 0 or s != 0:
                plt.imsave(out/f'{key[0]}_S{s:g}_rgb.png', image)
    fig.suptitle(title+' | float input audit (PNG previews quantized to 8-bit)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(out/'inputs.png', dpi=130); plt.close(fig)
    if not predictions:
        return
    clean = depth_to_numpy(predictions[('clean', 0.)])
    lo, hi = np.percentile(clean, [5, 95]); hi = max(hi, lo+1e-6)
    scale = max(float(np.median(np.abs(clean-np.median(clean)))), 1e-6)
    (out/'scales.txt').write_text(f'q_min={lo}\nq_max={hi}\nerror=abs(q-q_clean)/clean_MAD\n'
                                 f'clean_MAD={scale}\nerror_min=0\nerror_max=1\n', encoding='utf-8')
    for kind in ('q', 'error'):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.7*nrows), squeeze=False, constrained_layout=True)
        for i, c in enumerate(conditions):
            for j, s in enumerate([0.] + list(levels)):
                key = ('clean', 0.) if s == 0 else (c, s)
                q = depth_to_numpy(predictions[key]); row = rows[key]
                if kind == 'q':
                    data, cmap, norm = q, 'magma', Normalize(lo, hi, clip=True)
                else:
                    data, cmap, norm = np.abs(q-clean)/scale, 'viridis', Normalize(0, 1, clip=True)
                artist = axes[i, j].imshow(data, cmap=cmap, norm=norm)
                axes[i, j].axis('off')
                label = f'{c} / S{s:g} | drift={row["depth_drift"]:.3f}'
                if 'gt_ordinal_error' in row:
                    label += f'\nGT order={row["gt_ordinal_error"]:.1%} AbsRel={row["gt_abs_rel"]:.3f}'
                axes[i, j].set_title(label, fontsize=9)
        fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=.7,
                     label='raw inverse depth q' if kind == 'q' else '|q-q_clean| / clean MAD (clipped at 1)')
        fig.suptitle(title+' | same scale across all conditions and doses')
        fig.savefig(out/f'{kind}.png', dpi=130); plt.close(fig)


def save_response_curves(out):
    out = Path(out)
    with (out/'summary.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    keys = ['luma_p50', 'noise_rgb_rms', 'glare_visible_area', 'depth_drift',
            'gt_ordinal_error', 'delta_gt_ordinal_error', 'gt_abs_rel', 'gt_a1']
    keys = [k for k in keys if any(r['metric'] == k for r in rows)]
    fig, axes = plt.subplots((len(keys)+1)//2, 2, figsize=(12, 3.3*((len(keys)+1)//2)), squeeze=False)
    conditions = list(dict.fromkeys(r['condition'] for r in rows if r['condition'] != 'clean'))
    for ax, key in zip(axes.ravel(), keys):
        baseline = next((r for r in rows if r['condition'] == 'clean' and r['metric'] == key), None)
        for c in conditions:
            subset = sorted([r for r in rows if r['condition'] == c and r['metric'] == key], key=lambda r: float(r['severity']))
            if baseline:
                subset = [baseline] + subset
            x = [float(r['severity']) for r in subset]
            y = [float(r['mean']) for r in subset]
            line, = ax.plot(x, y, marker='o', label=c)
            ax.fill_between(x, [float(r['low']) for r in subset], [float(r['high']) for r in subset], alpha=.12, color=line.get_color())
        ax.set_title(key); ax.set_xlabel('Experimental severity (not calibrated night frequency)'); ax.grid(alpha=.2); ax.legend()
    for ax in axes.ravel()[len(keys):]:
        ax.axis('off')
    fig.suptitle('Group-balanced means and pointwise group-bootstrap 95% intervals')
    fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(out/'response_curves.png', dpi=150); plt.close(fig)
