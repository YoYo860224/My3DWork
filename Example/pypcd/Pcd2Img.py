import open3d
import os
import natsort
import math
import numpy as np
import cv2.cv2 as cv2
from pypcd import pypcd

def ReadPCD_XYZ(filename):
    pcdata = pypcd.PointCloud.from_path(filename)
    pc = np.asarray(pcdata.pc_data[['x', 'y', 'z']].tolist(), dtype=float)
    return pc

def RotateZ(pc):
    avgpc = np.average(pc, 0)
    theta = math.atan2(avgpc[0], avgpc[1])

    rz = np.array([[ math.cos(theta), math.sin(theta), 0],
                   [-math.sin(theta), math.cos(theta), 0],
                   [               0,               0, 1]])

    return pc.dot(rz), math.hypot(avgpc[0], avgpc[1])

def GetImage(pc, res=100, sqrSize=224):
    pcxyz, dis = RotateZ(pc)

    maxXYZ = np.max(pcxyz, 0)
    minXYZ = np.min(pcxyz, 0)

    xMin = minXYZ[0]
    zMin = minXYZ[2]
    xWidth = maxXYZ[0] - minXYZ[0]
    zWidth = maxXYZ[2] - minXYZ[2]

    imH = int(zWidth * res)
    imW = int(xWidth * res)

    img1 = np.zeros((imH, imW, 3))

    for i in pcxyz:
        ty = int((i[2]-zMin) * res)
        tx = int((i[0]-xMin) * res)
        ty = (imH - 1) - int(min(ty, imH-1))
        tx = int(min(tx, imW-1))

        img1[ty, tx] = (255, 255, 255)

    if imH > imW:
        left, top = round(imH * 0.1), round((1.2 * imH - imW) / 2)
    else:
        left, top = round(imW * 0.1), round((1.2 * imW - imH) / 2)

    img1 = cv2.copyMakeBorder(img1, top, top, left, left, cv2.BORDER_CONSTANT)
    img1 = cv2.resize(img1, (sqrSize, sqrSize))

    return img1, dis


if __name__ == "__main__":
    fromPath = "/media/yoyo/harddisk/kitti_npc/C/"
    toPath = "/media/yoyo/harddisk/kitti_npc/C_img/"

    if not os.path.exists(toPath):
        os.mkdir(toPath)

    i = 0

    for filename in natsort.natsorted(os.listdir(fromPath)):
        loadfilepath = os.path.join(fromPath, filename)
        pc = ReadPCD_XYZ(loadfilepath)
        img1, dis = GetImage(pc)
        cv2.threshold(img1, 30, 255, cv2.THRESH_BINARY)


        img1 = cv2.dilate(img1, np.ones((3, 3)))
        if (dis > 10):
            img1 = cv2.dilate(img1, np.ones((5, 5)))
        if (dis > 20):
            img1 = cv2.dilate(img1, np.ones((7, 7)))

        i+=1
        savename = '{:04d}_{:0>5.2f}.png'.format(i, dis)
        savefilepath = os.path.join(toPath, savename)
        cv2.imwrite(savefilepath, img1)

        print(loadfilepath, " OK!")
