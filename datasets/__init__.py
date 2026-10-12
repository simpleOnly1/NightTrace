from .robotcar import make_robotcar
from .nuscenes import make_nuscenes

def make_dataset(config,split='day',training=True):
    ds=config['data'];name=ds['name'].lower()
    if name=='robotcar':return make_robotcar(ds,split,training)
    if name=='nuscenes':return make_nuscenes(ds,split,training)
    raise ValueError('Unknown dataset: '+name)
