# My 3D Work
* `/Example`: smoe useful function for 3d point cloud.
* `/YoPCNet`: Good 3D point cloud Classfication.

# Some Note
```
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

python train.py --mtype=P
python train.py --mtype=P --feature_transform
python train.py --mtype=PI
python train.py --mtype=PI --feature_transform
python train.py --mtype=PH
python train.py --mtype=PH --feature_transform

python train.py --mtype=P --ltype=NL
python train.py --mtype=P --ltype=NL --feature_transform
python train.py --mtype=PI --ltype=NL
python train.py --mtype=PI --ltype=NL --feature_transform
python train.py --mtype=PH --ltype=NL
python train.py --mtype=PH --ltype=NL --feature_transform
```
