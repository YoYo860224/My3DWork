import os
import numpy as np
from pypcd import pypcd

fromPath = "D:\\Downloads\\KITTI\\2011_09_28_drive_0016_extract\\velodyne_points\\pcd\\"
toPath = "D:\\Downloads\\KITTI\\2011_09_28_drive_0016_extract\\velodyne_points\\pcdb\\"

if not os.path.exists(toPath):
    os.mkdir(toPath)

for filename in os.listdir(fromPath):
    loadfilepath = os.path.join(fromPath, filename)
    savefilepath = os.path.join(toPath, filename)
    pcdata = pypcd.PointCloud.from_path(loadfilepath)
    pypcd.save_point_cloud_bin(pcdata, savefilepath)
    print(loadfilepath, " OK!")
