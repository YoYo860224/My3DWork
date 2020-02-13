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

def RANSAC_Plane(xyz, thresh=0.1, zmin = -10, zmax = 0):
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

class MyCube(vispy.scene.Cube):
    def __init__(self, minPos, maxPos, parent=None):
        super().__init__(color=(0, 1, 1, 0.2), parent=parent)
        self.transform = vispy.visuals.transforms.MatrixTransform()
        self.transform.scale((maxPos-minPos))
        self.transform.translate((minPos+maxPos)/2.0)


# generate data
pc = ReadPCD_XYZI("D:\\Downloads\\ntutOutside\\0451.pcd")
# pc = ReadPCD_XYZI("D:\\Downloads\\KITTI\\2011_09_28_drive_0016_extract\\velodyne_points\\pcdb\\0000000000.pcd")
pc = pc[pc[:, 0] < 20,:]
pc = pc[pc[:, 0] > -20,:]
pc = pc[pc[:, 1] < 20,:]
pc = pc[pc[:, 1] > -20,:]

pcxyz, pccol = GetPosColor(pc)

# RANSAC
inliner, outliner = RANSAC_Plane(pcxyz)
GoodXYZ = pcxyz[outliner, :]
GoodCol = pccol[outliner, :]
GoodCol[:, :] = 1

# Cluster
n, labels = Cluster(GoodXYZ)

# Select Cluster
cluPoints = np.ndarray((1, 300, 3))
import random
for i in range(n):
    cluXYZ = GoodXYZ[labels==i]
    vol = cluXYZ.max(axis=0) - cluXYZ.min(axis=0)
    if (vol[0] < 2 and vol[0] > 0.5 and
        vol[1] < 5 and vol[1] > 0.2 and 
        vol[2] < 5 and vol[2] > 0.2):
        choice = np.random.choice(len(cluXYZ), 300, replace=True)
        point_set = cluXYZ[choice, :]
        point_set = np.expand_dims(point_set, axis=0)
        cluPoints = np.concatenate((cluPoints, point_set))

        GoodCol[labels==i, 0] = random.uniform(0, 1)
        GoodCol[labels==i, 1] = random.uniform(0, 1)
        GoodCol[labels==i, 2] = random.uniform(0, 1)

cluPoints = cluPoints[1::]

# PointNet
import torch
from pointnetX.model import PointNetCls, feature_transform_regularizer

# Build
classifier = PointNetCls(k=2, feature_transform=False)
classifier.load_state_dict(torch.load("./pth/cls_model_5000.pth"))
classifier.cuda()

# Data PreProcess
point_set = cluPoints
# To Center
point_set = point_set - np.expand_dims(np.mean(point_set, axis = 1), 1)
# Do Scale
distAll = np.sum(point_set ** 2, axis = -1)
distMax = np.max(distAll, axis = -1)
point_set = point_set / distMax[:, None, None]     
points = torch.tensor(point_set, dtype=torch.float)
points = points.transpose(2, 1)
points = points.cuda()

# Classifier
classifier = classifier.eval()
pred, _, _ = classifier(points)
pred_choice = pred.data.max(1)[1].cpu().numpy()==1
print(pred_choice)

# Get
Cluster_Person = cluPoints[pred_choice]

pcxyz = GoodXYZ
pccol = GoodCol

# Make a canvas and add simple view
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()

# Camera
view.camera = vispy.scene.cameras.TurntableCamera(azimuth=90, elevation=30, distance=30, translate_speed=3.0, fov=20)

# Create visuals objects
# PointCloud
scatter = vispy.scene.Markers()
scatter.set_data(pcxyz, edge_color=None, face_color=pccol, size=3)

# Cubes
cubs = vispy.scene.Node(name="Cubes")
for clu in Cluster_Person:
    cub = MyCube(clu.min(axis=0), clu.max(axis=0), parent=cubs)

# Axis
axis = vispy.scene.XYZAxis()

# Add to view
view.add(axis)
view.add(scatter)
view.add(cubs)

if __name__ == '__main__':

    if sys.flags.interactive != 1:
        vispy.app.run()
