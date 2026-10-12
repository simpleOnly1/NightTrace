from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


def tensor_rgb_to_uint8(x):
    """
    x: [1,3,H,W] or [3,H,W], torch tensor in [0,1]
    """
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().permute(1, 2, 0).numpy()
    x = np.clip(x, 0, 1)
    return (x * 255.0).astype(np.uint8)


def depth_to_numpy(depth):
    """
    depth: [1,1,H,W] or [1,H,W]
    """
    if depth.dim() == 4:
        depth = depth[0, 0]
    elif depth.dim() == 3:
        depth = depth[0]
    return depth.detach().cpu().numpy()


def colorize_depth(depth, vmax_percentile=95, cmap='magma'):
    d = depth_to_numpy(depth)
    vmax = np.percentile(d, vmax_percentile)
    vmin = np.percentile(d, 5)
    if vmax <= vmin:
        vmax = d.max() + 1e-6
        vmin = d.min()
    d = np.clip((d - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colored = plt.get_cmap(cmap)(d)[..., :3]
    return (colored * 255.0).astype(np.uint8)


def colorize_error(err, vmax_percentile=95, cmap='viridis'):
    e = depth_to_numpy(err)
    vmax = np.percentile(e, vmax_percentile)
    vmin = 0.0
    e = np.clip((e - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colored = plt.get_cmap(cmap)(e)[..., :3]
    return (colored * 255.0).astype(np.uint8)


def save_image(arr, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, arr)


def save_panel(
    save_dir,
    clean_rgb,
    degraded_rgb,
    clean_depth,
    degraded_depth,
    swap4_depth,
    swap34_depth,
    degraded_err,
    swap34_err,
    title=''
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    clean_rgb_np = tensor_rgb_to_uint8(clean_rgb)
    degraded_rgb_np = tensor_rgb_to_uint8(degraded_rgb)

    clean_depth_np = colorize_depth(clean_depth)
    degraded_depth_np = colorize_depth(degraded_depth)
    swap4_depth_np = colorize_depth(swap4_depth)
    swap34_depth_np = colorize_depth(swap34_depth)

    degraded_err_np = colorize_error(degraded_err)
    swap34_err_np = colorize_error(swap34_err)

    # 单独保存
    save_image(clean_rgb_np, save_dir / 'clean_rgb.png')
    save_image(degraded_rgb_np, save_dir / 'degraded_rgb.png')
    save_image(clean_depth_np, save_dir / 'clean_depth.png')
    save_image(degraded_depth_np, save_dir / 'degraded_depth.png')
    save_image(swap4_depth_np, save_dir / 'swap4_depth.png')
    save_image(swap34_depth_np, save_dir / 'swap34_depth.png')
    save_image(degraded_err_np, save_dir / 'degraded_error.png')
    save_image(swap34_err_np, save_dir / 'swap34_error.png')

    # 拼图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    panels = [
        (clean_rgb_np, 'Clean RGB'),
        (degraded_rgb_np, 'Degraded RGB'),
        (clean_depth_np, 'Clean Depth'),
        (degraded_depth_np, 'Degraded Depth'),
        (swap4_depth_np, 'Swap F4'),
        (swap34_depth_np, 'Swap F3+F4'),
        (degraded_err_np, 'Error (Degraded)'),
        (swap34_err_np, 'Error (Swap F3+F4)')
    ]

    for ax, (img, name) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    fig.savefig(save_dir / 'panel.png', dpi=200)
    plt.close(fig)