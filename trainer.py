"""ACDepth-style training engine.

Day stage is the independently usable self-supervised DAV2-L baseline. The
experimental night objective belongs to losses/nighttrace_loss.py *after*
mechanism studies; the current program refuses to pretend it is implemented.
"""
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.config import load_config, project_path
from datasets import make_dataset
from models.depth_net import DepthNet
from models.pose import PoseNet
from losses.photometric import SSIM, reprojection


def fit(config, stage='day', smoke=False):
    if stage != 'day':
        raise NotImplementedError('NightTrace night-stage loss is not specified yet. '
                                  'Run diagnose.py and diagnose_spread.py first.')
    cfg = load_config(config)
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tr, mc = cfg['train'], cfg['model']
    ds = make_dataset(cfg, split='day', training=True)
    dl = DataLoader(ds, batch_size=tr['batch_size'], shuffle=True,
                    num_workers=cfg['data']['num_workers'], drop_last=True)
    if len(dl) == 0:
        raise ValueError('Dataset too small for batch size %d' % tr['batch_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth = DepthNet(weights=str(project_path(mc['weights'])),
                     input_size=mc.get('input_size', 518),
                     freeze_encoder=mc.get('encoder_frozen', True),
                     train_dpt=mc.get('train_dpt', True),
                     inverse_epsilon=mc.get('inverse_epsilon', .001)).to(device)
    pose = PoseNet().to(device)
    params = [p for p in depth.parameters() if p.requires_grad] + list(pose.parameters())
    if not params:
        raise ValueError('No trainable parameters')
    optim = torch.optim.AdamW(params, lr=tr['lr'], weight_decay=tr['weight_decay'])
    start_epoch = 0
    save_dir = project_path(tr['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    if tr.get('resume'):
        ckpt = torch.load(str(project_path(tr['resume'])), map_location='cpu')
        depth.load_state_dict(ckpt['model'], strict=True)
        pose.load_state_dict(ckpt['pose'], strict=True)
        optim.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
    ssim = SSIM().to(device)
    for epoch in range(start_epoch, tr['epochs']):
        depth.train()
        pose.train()
        run_losses = []
        for step, batch in enumerate(tqdm(dl, desc='Day epoch %d' % epoch)):
            image = batch['image'].to(device)
            intrinsic = batch['K'].to(device)
            sources = [batch['source_%+d' % i].to(device)
                       for i in cfg['data'].get('offsets', [-1, 1])]
            result = depth(image)
            loss, details = reprojection(image, sources, result['depth_for_warp'],
                                         result['disp'], intrinsic, pose, ssim,
                                         tr['smoothness_weight'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite day loss at epoch %d batch %d' % (epoch, step))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tr['grad_clip'])
            optim.step()
            run_losses.append(float(loss.detach()))
            if step % tr.get('log_every', 50) == 0:
                print('epoch=%d step=%d loss=%.5f photo=%.5f smooth=%.5f' %
                      (epoch, step, float(loss.detach()), details['photo'], details['smooth']))
            if smoke and step >= 1:
                break
        torch.save({'epoch': epoch, 'model': depth.state_dict(),
                    'pose': pose.state_dict(), 'optimizer': optim.state_dict(),
                    'config': cfg}, save_dir / 'latest.pt')
        print('Checkpoint:', save_dir / 'latest.pt', 'mean loss:', float(np.mean(run_losses)))
        if smoke:
            break
