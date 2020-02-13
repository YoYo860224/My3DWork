import sys
import vispy
import vispy.scene
import numpy as np
from pypcd import pypcd
from sklearn.cluster import DBSCAN


def ReadPCD_XYZI(filename):
    pcdata = pypcd.PointCloud.from_path(filename)
    pc = np.asarray(pcdata.pc_data[['x', 'y', 'z', 'intensity']].tolist(), dtype=float)
    return pc

def GetPosColor(pc, clipMax=500):
    pcXYZ = pc[:, 0:3]
    pcCol = pc[:, 3:4]
    pcCol = np.clip(pcCol, 0, clipMax)
    pcCol /= clipMax
    pcCol = np.repeat(pcCol, 3, axis=1)
    pcCol[:, 0] = 1
    pcCol[:, 2] = 0

    return pcXYZ, pcCol

def RANSAC_Plane(xyz, thresh=0.3, zmin = -10, zmax = -1):
    import random
    maxLen = -1
    oriZ = xyz[:, 2]
    needZ = 0

    for _ in range(100):
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
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=0.3, min_samples=10).fit(xyz)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)

    return n_clusters_, labels


# generate data
pc = ReadPCD_XYZI("D:\\Downloads\\ntutOutside\\0100.pcd")
pc = pc[pc[:, 0] < 20,:]
pc = pc[pc[:, 0] > -20,:]
pc = pc[pc[:, 1] < 20,:]
pc = pc[pc[:, 1] > -20,:]

pcxyz, pccol = GetPosColor(pc)

inliner, outliner = RANSAC_Plane(pcxyz)

GoodCol = pccol[outliner, :]
GoodXYZ = pcxyz[outliner, :]

n, labels = Cluster(GoodXYZ)

import random
for i in range(n):
    GoodCol[labels==i, 0] = random.uniform(0, 1)
    GoodCol[labels==i, 1] = random.uniform(0, 1)
    GoodCol[labels==i, 2] = random.uniform(0, 1)

pcxyz = GoodXYZ
pccol = GoodCol

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


# Make a canvas and add simple view
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()

# Camera
view.camera = vispy.scene.cameras.TurntableCamera(azimuth=90, elevation=30, distance=30)

# Create visuals objects
# PointCloud
scatter = vispy.scene.Markers()
scatter.set_data(pcxyz, edge_color=None, face_color=pccol, size=3)

# Axis
axis = vispy.scene.XYZAxis()

# Add to view
view.add(axis)
view.add(scatter)

if __name__ == '__main__':

    if sys.flags.interactive != 1:
        vispy.app.run()
