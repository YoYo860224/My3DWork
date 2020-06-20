import sys
import os
import math
import natsort
import numpy as np
import cv2.cv2 as cv2
import pypcd
from skimage.transform import integral_image
from skimage import feature as ft

sys.path.append("../")
from Example.pypcd.PcdRead import ReadPCD_XYZI

dataRoot = "/media/yoyo/harddisk/kitti_npc"

pcd_path = os.path.join(dataRoot, "C")
img_path = os.path.join(dataRoot, "C_img")
saveFeature = os.path.join(dataRoot, "C_feature.npy")

pcd_list = [os.path.join(pcd_path, f) for f in natsort.natsorted(os.listdir(pcd_path))]
img_list = [os.path.join(img_path, f) for f in natsort.natsorted(os.listdir(img_path))]
dlen = len(pcd_list)

# region Features
def getTheta(pc):
    avgpc = np.average(pc, 0)
    theta = math.atan2(avgpc[0], avgpc[1])
    return theta

def f4_covM(pc, length):
    avgpc = np.average(pc, axis=0)
    f41 = (pc[:, 0] - avgpc[0]).dot(pc[:, 0] - avgpc[0]) / length #xx
    f42 = (pc[:, 1] - avgpc[1]).dot(pc[:, 1] - avgpc[1]) / length #yy
    f43 = (pc[:, 2] - avgpc[2]).dot(pc[:, 2] - avgpc[2]) / length #zz
    f44 = (pc[:, 0] - avgpc[0]).dot(pc[:, 1] - avgpc[1]) / length #xy
    f45 = (pc[:, 0] - avgpc[0]).dot(pc[:, 2] - avgpc[2]) / length #xz
    f46 = (pc[:, 1] - avgpc[1]).dot(pc[:, 2] - avgpc[2]) / length #yz

    return np.array([f41, f42, f43, f44, f45, f46])

def f5_momentOfInertia(pc):
    f51 = np.average(np.square(pc[:, 0]) + np.square(pc[:, 0]), axis=0) #x^2 + y^2
    f52 = np.average(np.square(pc[:, 0]) + np.square(pc[:, 2]), axis=0) #x^2 + z^2
    f53 = np.average(np.square(pc[:, 1]) + np.square(pc[:, 2]), axis=0) #y^2 + z^2
    f54 = np.average(np.multiply(pc[:, 0], pc[:, 1]), axis=0) #xy
    f55 = np.average(np.multiply(pc[:, 0], pc[:, 2]), axis=0) #xz
    f56 = np.average(np.multiply(pc[:, 1], pc[:, 2]), axis=0) #yz

    return np.array([f51, f52, f53, f54, f55, f56])

def f6_2Din3Zone(pc, theta):
    rz = np.array([[ math.cos(theta), math.sin(theta), 0],
                   [-math.sin(theta), math.cos(theta), 0],
                   [               0,               0, 1]])

    mainPlane = pc[:, 0:3].dot(rz)
    maxXYZ = np.max(mainPlane, 0)
    minXYZ = np.min(mainPlane, 0)
    maxX = maxXYZ[0]
    minX = minXYZ[0]
    maxZ = maxXYZ[2]
    minZ = minXYZ[2]
    midX = (maxX + minX) / 2
    midZ = (maxZ + minZ) / 2
    zone1 = mainPlane[np.where(mainPlane[:, 2] > midZ)]
    zoneb = mainPlane[np.where(mainPlane[:, 2] < midZ)]
    zone2 = mainPlane[np.where(zoneb[:, 0] < midX)]
    zone3 = mainPlane[np.where(zoneb[:, 0] > midX)]
    f6 = []
    zones = [zone1, zone2, zone3]
    for z in zones:
        length = z.shape[0]
        if length==0:
            f6 += [0, 0, 0]
        else:
            avgpc = np.average(z, axis=0)
            f6 += [(z[:, 0] - avgpc[0]).dot(z[:, 0] - avgpc[0]) / length]
            f6 += [(z[:, 0] - avgpc[0]).dot(z[:, 1] - avgpc[1]) / length]
            f6 += [(z[:, 1] - avgpc[1]).dot(z[:, 1] - avgpc[1]) / length]

    return f6

def f7_14x7Bin(pc, theta):    
    rz = np.array([[ math.cos(theta), math.sin(theta), 0],
                   [-math.sin(theta), math.cos(theta), 0],
                   [               0,               0, 1]])

    mainPlane = pc[:, 0:3].dot(rz)
    maxXYZ = np.max(mainPlane, 0)
    minXYZ = np.min(mainPlane, 0)
    maxX = maxXYZ[0]
    minX = minXYZ[0]
    maxZ = maxXYZ[2]
    minZ = minXYZ[2]
    intX = np.arange(minX, maxX+((maxX-minX)/8), (maxX-minX)/7)
    intZ = np.arange(minZ, maxZ+((maxZ-minZ)/15), (maxZ-minZ)/14)
    intX = intX[1:]
    intZ = intZ[1:]

    f7 = []
    reserve = mainPlane
    for i in intX:
        zone = reserve[np.where(reserve[:, 0] <= i)]
        reserve = reserve[np.where(reserve[:, 0] > i)]

        for j in intZ:
            szone = zone[np.where(zone[:, 2] <= j)]
            zone = zone[np.where(zone[:, 2] > j)]
            f7 += [szone.shape[0]]
    
    return f7

def f8_9x5Bin(pc, theta):
    theta += math.radians(90)
    rz = np.array([[ math.cos(theta), math.sin(theta), 0],
                   [-math.sin(theta), math.cos(theta), 0],
                   [               0,               0, 1]])

    secPlane = pc[:, 0:3].dot(rz)
    maxXYZ = np.max(secPlane, 0)
    minXYZ = np.min(secPlane, 0)
    maxX = maxXYZ[0]
    minX = minXYZ[0]
    maxZ = maxXYZ[2]
    minZ = minXYZ[2]
    intX = np.arange(minX, maxX+((maxX-minX)/6), (maxX-minX)/5)
    intZ = np.arange(minZ, maxZ+((maxZ-minZ)/10), (maxZ-minZ)/9)
    intX = intX[1:]
    intZ = intZ[1:]

    f8 = []
    reserve = secPlane
    for i in intX:
        zone = reserve[np.where(reserve[:, 0] <= i)]
        reserve = reserve[np.where(reserve[:, 0] > i)]

        for j in intZ:
            szone = zone[np.where(zone[:, 2] <= j)]
            zone = zone[np.where(zone[:, 2] > j)]
            f8 += [szone.shape[0]]
    
    return f8

def f9_SliceF(pc, theta):
    rz = np.array([[ math.cos(theta), math.sin(theta), 0],
                   [-math.sin(theta), math.cos(theta), 0],
                   [               0,               0, 1]])

    mainPlane = pc[:, 0:3].dot(rz)
    maxXYZ = np.max(mainPlane, 0)
    minXYZ = np.min(mainPlane, 0)
    maxZ = maxXYZ[2]
    minZ = minXYZ[2]
    intZ = np.arange(minZ, maxZ+((maxZ-minZ)/10), (maxZ-minZ)/9)
    
    f9 = []
    for i in intZ:
        zone = mainPlane[np.where(mainPlane[:, 2] <= i)]
        mainPlane = mainPlane[np.where(mainPlane[:, 2] > i)]
        if zone.shape[0] == 0:
            f9 += [0, 0]
        else:
            maxXYZ = np.max(zone, 0)
            minXYZ = np.min(zone, 0)
            maxX = maxXYZ[0]
            minX = minXYZ[0]
            maxZ = maxXYZ[2]
            minZ = minXYZ[2]
            f9 += [maxX-minX, maxZ-minZ]

    return f9

def f10_DisOfR(pc):
    pcI = pc[:, 3]
    f10, _ = np.histogram(pcI, 25)

    f10 = np.append(f10, np.average(pcI))
    f10 = np.append(f10, np.std(pcI))

    return f10

def f11_static(pc):
    maxXYZ = np.max(pc, 0)
    minXYZ = np.min(pc, 0)
    maxZ = maxXYZ[2]
    minZ = minXYZ[2]
    
    f11 = [maxZ - minZ]
    f11 += [np.std(pc[:, 0])]
    f11 += [np.std(pc[:, 1])]
    f11 += [np.std(pc[:, 2])]
    uml, _ = np.histogram(pc[:, 2], 3)
    uml_p = uml / pc.shape[0]
    f11 += [np.max(uml)]
    f11 += [np.max(uml_p)]
    f11 += [uml_p[0]]
    f11 += [uml_p[1]]
    f11 += [uml_p[2]]
    
    return f11

def f12_HistLine(pc, theta):
    rz = np.array([[ math.cos(theta), math.sin(theta), 0],
                   [-math.sin(theta), math.cos(theta), 0],
                   [               0,               0, 1]])

    mainPlane = pc[:, 0:3].dot(rz)
    maxXYZ = np.max(mainPlane, 0)
    minXYZ = np.min(mainPlane, 0)
    maxX = maxXYZ[0]
    minX = minXYZ[0]
    intX = np.arange(minX, maxX+((maxX-minX)/10), (maxX-minX)/9)
    
    f12 = []
    for i in intX:
        zone = mainPlane[np.where(mainPlane[:, 0] <= i)]
        mainPlane = mainPlane[np.where(mainPlane[:, 0] > i)]
        if zone.shape[0] == 0:
            f12 += [0]
        else:
            avgZ = np.max(zone[:, 2])
            f12 += [avgZ]

    return f12

def f13_HOG(img):
    features = ft.hog(img,                  # input image
                  orientations=9,           # number of bins
                  pixels_per_cell=(16, 16),   # pixel per cell
                  cells_per_block=(1, 1),   # cells per blcok
                  block_norm = 'L2-Hys',    # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                  transform_sqrt = True,    # power law compression (also known as gamma correction)
                  feature_vector=True,      # flatten the final vectors
                  visualize=False)          # return HOG map)
    return features

def f14_LBP(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbpImg = ft.local_binary_pattern(img, P=8, R=1)
    lbpImg = lbpImg.astype(np.uint8)
    f14 = cv2.calcHist([lbpImg], [0], None, [256], [0, 256])
    f14 = f14.reshape(-1)
    return f14

def f15_Haar(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img_ii = integral_image(img)
    f15 = ft.haar_like_feature(img_ii, 0, 0, 8, 8, 'type-4')
    return f15

def getFeatures(pc, img):
    f1 = pcd.shape[0]
    f2 = np.min(np.linalg.norm(pcd, axis=1))
    f3 = getTheta(pcd)
    f4 = f4_covM(pcd, f1)
    f5 = f5_momentOfInertia(pcd)
    f6 = f6_2Din3Zone(pcd, f3)
    f7 = f7_14x7Bin(pcd, f3)
    f8 = f8_9x5Bin(pcd, f3)
    f9 = f9_SliceF(pcd, f3)
    f10 = f10_DisOfR(pcd)
    f11 = f11_static(pcd)
    f12 = f12_HistLine(pcd, f3)
    f13 = f13_HOG(img)
    f14 = f14_LBP(img)
    f15 = f15_Haar(img)

    feature = np.array([f1, f2, f3], dtype=np.float32)
    feature = np.concatenate([feature, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15])
    feature = feature.reshape(1, -1)

    return feature
# endregion

if __name__ == "__main__":
    pcd = ReadPCD_XYZI(pcd_list[0])
    img = cv2.imread(img_list[0])

    f = getFeatures(pcd, img)
    featureLen = f.shape[0]

    features = np.ndarray((0, featureLen), dtype=np.float32)

    for i in range(dlen):
        pcd = ReadPCD_XYZI(pcd_list[i])
        img = cv2.imread(img_list[i])

        exit()

        feature = getFeatures(pcd, img)
        features = np.concatenate([features, feature])
        print("{0}/{1} OK.".format(i+1, dlen), end='\r')

    np.save(saveFeature, features)
