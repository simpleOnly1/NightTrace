"""nuScenes STEPS video clip file_list + global day/night split protocol.

A split may contain clip IDs or image names/clip/image tuples. We resolve
both, but never silently train on frames outside a requested day/night split.
"""
from pathlib import Path
from .common import ClipDataset,FrameDataset,read_tokens,image_file,seq_records,IMG_EXTS
from utils.config import project_path


def make_nuscenes(cfg,split='day',training=True):
    root=project_path(cfg['root']); sequences=root/'sequences'
    if training:
        split_path=root/cfg.get('split_dir','splits')/('%s_train_split.txt'%split)
        if not split_path.exists():raise FileNotFoundError(split_path)
        tokens=read_tokens(split_path)
        specs=set(str(s).replace('\\','/') for s in tokens)
        records=[]
        for clip in sorted(p for p in sequences.iterdir() if p.is_dir()):
            manifest=clip/'file_list.txt'
            if not manifest.is_file():continue
            paths=[image_file(clip,Path(s).name if not (clip/s).exists() else s) for s in read_tokens(manifest)]
            if not paths:continue
            include_clip=clip.name in specs or 'sequences/'+clip.name in specs
            if include_clip:
                keys=None
            else:
                keys=[p.stem for p in paths if any(key in specs for key in
                      [p.stem,p.name,clip.name+'/'+p.name,clip.name+'/'+p.stem,
                       'sequences/'+clip.name+'/'+p.name])]
                if not keys:continue
            records.extend(seq_records(paths,clip/'intrinsic.npy',keys,cfg.get('offsets',[-1,1])))
        if not records:
            raise ValueError('No matched nuScenes video clips. Inspect split format: '+str(split_path))
        return ClipDataset(records,cfg['input_height'],cfg['input_width'],cfg.get('offsets',[-1,1]))
    # STEPS test ordering is *sorted day/night_test_split.txt*, never os.listdir().
    color = root / 'test' / 'color'
    gt = root / 'test' / 'gt'
    split_path = root / cfg.get('split_dir', 'splits') / ('%s_test_split.txt' % split)
    if not split_path.is_file():
        raise FileNotFoundError('STEPS evaluation requires the split file: %s' % split_path)
    if not color.is_dir() or not gt.is_dir():
        raise FileNotFoundError('nuScenes requires test/color and test/gt')
    tokens = sorted(read_tokens(split_path))
    records = []
    for token in tokens:
        stem = Path(token).stem
        image = image_file(color, stem)
        label = gt / (stem + '.npy')
        if not label.is_file():
            raise FileNotFoundError('GT mismatch for test token %s: %s' % (token, label))
        records.append({'target': image, 'gt': label, 'id': stem})
    if not records:
        raise ValueError('Empty nuScenes test split: ' + str(split_path))
    return FrameDataset(records, size=(cfg['input_height'], cfg['input_width']))
