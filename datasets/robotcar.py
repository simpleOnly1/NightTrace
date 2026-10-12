from pathlib import Path
from .common import ClipDataset, FrameDataset, IMG_EXTS, read_tokens, seq_records, image_file
from utils.config import project_path


def make_robotcar(cfg,split='day',training=True):
    root=project_path(cfg['root']); package=cfg['day_package'] if split=='day' else cfg['night_package']
    pkg=root/package
    if not pkg.is_dir():raise FileNotFoundError('RobotCar package: '+str(pkg))
    img_dir=pkg/'rgb'
    if not img_dir.is_dir():raise FileNotFoundError(img_dir)
    names=read_tokens(pkg/('train_split.txt' if training else 'test_split.txt'))
    if not training:
        names=sorted(names)  # STEPS test_robotcar_disp.py sorts test_split entries
    frames=sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if training:
        entries=seq_records(frames,pkg/'intrinsic.npy',names,cfg.get('offsets',[-1,1]))
        return ClipDataset(entries,cfg['input_height'],cfg['input_width'],cfg.get('offsets',[-1,1]))
    eval_records=[]
    for token in names:
        filename=Path(token).name
        if filename.startswith('rgb/'):
            filename=filename[4:]
        path=image_file(img_dir,filename)
        gt_suffix = cfg.get('gt_suffix', '_depth')
        gt_path = pkg / 'depth' / (path.stem + gt_suffix + '.npy')
        eval_records.append({'target':path,'gt':gt_path,'id':path.stem})
    crop=(cfg.get('crop_width',1152),cfg.get('crop_height',640))
    return FrameDataset(eval_records,crop=crop,size=(cfg['input_height'],cfg['input_width']))
