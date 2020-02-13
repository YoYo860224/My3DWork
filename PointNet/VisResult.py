import sys
import vispy
import vispy.scene
import numpy as np
from utils.io import ReadPCD_XYZI
from utils.visuals import GetXYZnRGB
from utils.visuals import MyCube
from utils.tools import Passthrough, RANSAC_Plane, Cluster, Getlusters, DownSampe


# generate data
pc = ReadPCD_XYZI("D:\\Downloads\\ntutOutside\\0451.pcd")
# pc = ReadPCD_XYZI("D:\\Downloads\\KITTI\\2011_09_28_drive_0016_extract\\velodyne_points\\pcdb\\0000000000.pcd")
pc = Passthrough(pc, 0, -20, 20)
pc = Passthrough(pc, 1, -20, 20)
pc = DownSampe(pc)

pcxyz, pccol = GetXYZnRGB(pc, clipMax=500)

# RANSAC
inliner, outliner = RANSAC_Plane(pcxyz, 0.1, 100, -10, -0.5)
GoodXYZ = pcxyz[outliner, :]
GoodCol = pccol[outliner, :]
GoodCol[:, :] = 1
# GoodXYZ = pcxyz
# GoodCol = pccol
# GoodCol[:, :] = 1

# Cluster
nLabels, labels = Cluster(GoodXYZ)
# Select Cluster
cluPoints = Getlusters(GoodXYZ, labels, nLabels, nPoints=300)
colorLabel = np.random.random((nLabels, 3))
for i in range(nLabels):
    GoodCol[labels==i, :] = colorLabel[i]

# PointNet
import torch
from pointnetX.model import PointNetCls, feature_transform_regularizer
from pointnetX.formData import GetTorchInputForPointNet

# Build
classifier = PointNetCls(k=2, feature_transform=False)
classifier.load_state_dict(torch.load("./pth/cls_model_5000.pth"))
classifier.cuda()
classifier = classifier.eval()
point_set = GetTorchInputForPointNet(cluPoints)
point_set = point_set.cuda()

# Classifier
pred, _, _ = classifier(point_set)
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
