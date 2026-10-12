from pathlib import Path
import sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,json
from datasets import make_dataset
from evaluation.protocol import align_gt_geometry,run_evaluation

def run():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'RobotCar-Night'/'day'
        (p/'rgb').mkdir(parents=True)
        (p/'depth').mkdir()
        np.save(p/'intrinsic.npy',np.array([[70.,0.,32],[0,70.,24],[0,0,1]],dtype=np.float32))
        for i in range(5):
            cv2.imwrite(str(p/'rgb'/('%04d.png'%i)),np.full((48,64,3),i*20,dtype=np.uint8))
        (p/'train_split.txt').write_text('\n'.join('%04d'%i for i in range(5)))
        (p/'test_split.txt').write_text('0002\n')
        np.save(p/'depth'/'0002.npy',np.ones((48,64)))
        cfg={'data':{'name':'robotcar','root':str(Path(d)/'RobotCar-Night'),'day_package':'day',
          'night_package':'day','input_height':32,'input_width':48,'crop_width':64,'crop_height':48,
          'offsets':[-1,1]}}
        ds=make_dataset(cfg,'day',True)
        assert len(ds)==3
        sample=ds[0];assert sample['image'].shape==(3,32,48)
        ev=make_dataset(cfg,'night',False)
        assert len(ev)==1 and ev[0]['image'].shape==(3,32,48)
        assert align_gt_geometry(np.ones((60,80)),(48,64)).shape==(48,64)
        folder=Path(d)/'results';folder.mkdir()
        np.save(folder/'predictions.npy',np.ones((1,16,24),dtype=np.float32)*3)
        (folder/'index.json').write_text(json.dumps([{'id':'0002','gt_path':str(p/'depth'/'0002.npy')}]))
        result=run_evaluation(folder,kind='relative_depth',min_depth=.01,max_depth=100)
        assert result['a1']==1.0, result
    print('DATASET TESTS PASS')
if __name__=='__main__':run()
