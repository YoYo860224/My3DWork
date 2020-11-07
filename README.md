# My 3D Work
* `/Example`: smoe useful function for 3d point cloud.
* `/YoPCNet`: Good 3D point cloud Classfication.

# Some Note
```
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

python train.py --feature_transform
python test.py --feature_transform --model=./pth/cls_model_1.pth
python train.py --dataroot=/media/yoyo/harddisk/kitti_npc
python test.py --dataroot=/media/yoyo/harddisk/kitti_npc --model=./pth/cls_model_1.pth
```
