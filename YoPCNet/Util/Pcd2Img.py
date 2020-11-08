import os
import natsort
import math
import numpy as np
import cv2
from pypcd import pypcd

from .PcdRead import ReadPCD_XYZI


def RotateZ(pc):
    avgpc = np.average(pc, 0)
    theta = math.atan2(avgpc[0], avgpc[1])

    rz = np.array([[ math.cos(theta), math.sin(theta), 0],
                   [-math.sin(theta), math.cos(theta), 0],
                   [               0,               0, 1]])

    return pc.dot(rz), math.hypot(avgpc[0], avgpc[1])

def GetImage(pc, res=100):
    pcxyz, pcI = pc[:, 0:3], pc[:, 3:4]
    pcxyz, dis = RotateZ(pcxyz)

    pcDis = np.linalg.norm(pcxyz, axis=1)
    disF = np.max(pcDis, axis=0)
    disN = np.min(pcDis, axis=0)
    disInterval = disF - disN
    pcDis = (pcDis - disN) / disInterval * 255
    pcDis = pcDis.reshape(-1, 1)
    pcI = pcI * 255
    newPC = np.concatenate([pcxyz, pcI, pcDis], axis=1)

    # 算投影圖大小，並建立投影圖
    maxXYZ = np.max(pcxyz, 0)
    minXYZ = np.min(pcxyz, 0)
    xMin = minXYZ[0]
    zMin = minXYZ[2]
    xWidth = maxXYZ[0] - minXYZ[0]
    zWidth = maxXYZ[2] - minXYZ[2]
    imH = int(max(1, zWidth * res))
    imW = int(max(1, xWidth * res))
    img1 = np.zeros((imH, imW, 3), dtype=np.uint8)

    for i in newPC:
        ty = int((i[2]-zMin) * res)
        tx = int((i[0]-xMin) * res)
        ty = (imH - 1) - int(min(ty, imH-1))
        tx = int(min(tx, imW-1))

        img1[ty, tx] = (int(i[4]), 255, int(i[3]))

    if imH > imW:
        top, left = round(imH * 0.1), round((1.2 * imH - imW) / 2)
    else:
        left, top = round(imW * 0.1), round((1.2 * imW - imH) / 2)

    img1 = cv2.copyMakeBorder(img1, top, top, left, left, cv2.BORDER_CONSTANT)

    return img1, dis

def DilateImage(img, dis):
    dkSize = 3 + int(dis//3)*2
    dkEle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dkSize*2, dkSize * 3))
    img = cv2.dilate(img, dkEle)

    return img

def GaussionImage(img):
    img = cv2.GaussianBlur(img, (3, 3), 5)
    return img

def FillContour(img, dImg):
    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bImg = cv2.threshold(gImg, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    fillContour = dImg.copy()
    cv2.drawContours(fillContour, contours, -1, (0, 255, 0), -1)
    goodImg = dImg + fillContour
    return goodImg

def ImgFlow(pcXYZI):
    img1, dis = GetImage(pcXYZI)
    imgD = DilateImage(img1, dis)
    imgG = GaussionImage(imgD)
    img = FillContour(imgG, imgD)
    img = cv2.resize(img, (224, 224))
    return img


if __name__ == "__main__":
    fromPath = "/media/yoyo/harddisk/kitti_npc/N/"
    toPath = "/media/yoyo/harddisk/kitti_npc/N_img/"

    if not os.path.exists(toPath):
        os.makedirs(toPath)

    for filename in natsort.natsorted(os.listdir(fromPath)):
        loadfilepath = os.path.join(fromPath, filename)
        pc = ReadPCD_XYZI(loadfilepath)

        img = ImgFlow(pc)

        savename = filename.split('.')[0] + ".png"
        savefilepath = os.path.join(toPath, savename)
        cv2.imwrite(savefilepath, img)

        print(loadfilepath, " OK!", end="\r")
