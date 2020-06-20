import os
import natsort
import numpy as np
from pypcd import pypcd

fromPath = "/media/yoyo/harddisk/kitti_npc/N_ori"
toPath = "/media/yoyo/harddisk/kitti_npc/N"

if not os.path.exists(toPath):
    os.mkdir(toPath)

i = 0
for filename in natsort.natsorted(os.listdir(fromPath)):
    loadfilepath = os.path.join(fromPath, filename)
    pcdata = pypcd.PointCloud.from_path(loadfilepath)

    i+=1
    savename = '{:04d}.pcd'.format(i)
    savefilepath = os.path.join(toPath, savename)
    pypcd.save_point_cloud(pcdata, savefilepath)

    print(loadfilepath, " OK!")
