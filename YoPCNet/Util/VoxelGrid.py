import os
import natsort
import math
import numpy as np
import cv2
from pypcd import pypcd
from matplotlib import pyplot as plt

from .PcdRead import ReadPCD_XYZI


def getVoxel(pc):
    VoxelSize = 30

    maxXYZ = np.max(pc, 0)[0:3]
    maxXYZ += 0.0001
    minXYZ = np.min(pc, 0)[0:3]
    interval = (maxXYZ-minXYZ)/VoxelSize

    voxel = np.zeros([VoxelSize, VoxelSize, VoxelSize], dtype=np.float32)
    nums = np.zeros([VoxelSize, VoxelSize, VoxelSize], dtype=np.float32)
    for i in pc:
        p_x = int((i[0] - minXYZ[0]) / interval[0])
        p_y = int((i[1] - minXYZ[1]) / interval[1])
        p_z = int((i[2] - minXYZ[2]) / interval[2])

        voxel[p_x, p_y, p_z] += i[3]
        nums += 1

    for x in range(30):
        for y in range(30):
            for z in range(30):
                if(voxel[x, y, z] > 0):
                    voxel[x, y, z] = voxel[x, y, z] / nums[x, y, z]

    return voxel


if __name__ == "__main__":
    fromPath = "/media/yoyo/harddisk/kitti_npc/N/"
    toPath = "/media/yoyo/harddisk/kitti_npc/N_voxel/"

    if not os.path.exists(toPath):
        os.mkdir(toPath)

    i = 0

    for filename in natsort.natsorted(os.listdir(fromPath)):
        loadfilepath = os.path.join(fromPath, filename)
        pc = ReadPCD_XYZI(loadfilepath)

        voxel = getVoxel(pc)

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(voxel, edgecolor='k')
        # plt.show()

        i+=1
        savename = '{:04d}.npy'.format(i)
        savefilepath = os.path.join(toPath, savename)
        np.save(savefilepath, voxel)

        print(loadfilepath, " OK!", end="\r")
