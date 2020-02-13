from sklearn.cluster import DBSCAN
import numpy as np
import random


def Passthrough(xyz, dim, vMin, vMax):
    xyz = xyz[xyz[:, dim] < vMax,:]
    xyz = xyz[xyz[:, dim] > vMin,:]
    return xyz

def RANSAC_Plane(xyz, thresh=0.1, maxIter=100, zmin = -10, zmax = 0):
    maxLen = -1
    oriZ = xyz[:, 2]
    needZ = 0

    for _ in range(maxIter):
        z = random.uniform(zmin, zmax)
        theZ = oriZ[oriZ < z+thresh]
        theZ = theZ[theZ > z-thresh]

        if (maxLen < theZ.shape[0]):
            maxLen = theZ.shape[0]
            needZ = z

    inliner = np.logical_and((xyz[:, 2] < needZ+thresh), (xyz[:, 2] > needZ-thresh))
    outliner = np.logical_not(inliner)

    return inliner, outliner

def Cluster(xyz, eps=0.3, min_samples=10):
    db = DBSCAN(eps=eps, algorithm='kd_tree', min_samples=min_samples).fit(xyz)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)

    return n_clusters_, labels

def Getlusters(xyz, labels, nLabels, nPoints):
    cluPoints = np.ndarray((0, nPoints, 3))
    for i in range(nLabels):
        cluXYZ = xyz[labels==i]
        vol = cluXYZ.max(axis=0) - cluXYZ.min(axis=0)
        if (vol[0] < 2 and vol[0] > 0.5 and
            vol[1] < 5 and vol[1] > 0.2 and 
            vol[2] < 5 and vol[2] > 0.2):
            choice = np.random.choice(len(cluXYZ), nPoints, replace=True)
            point_set = cluXYZ[choice, :]
            point_set = np.expand_dims(point_set, axis=0)
            cluPoints = np.concatenate([cluPoints, point_set])
    return cluPoints
