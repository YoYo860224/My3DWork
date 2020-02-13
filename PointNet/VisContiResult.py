import os
import sys
import natsort
import vispy
import vispy.scene
import numpy as np
from utils.io import ReadPCD_XYZI
from utils.visuals import GetXYZnRGB
from utils.visuals import MyCube
from utils.tools import Passthrough, RANSAC_Plane, Cluster, Getlusters, DownSampe

# PointNet
import torch
from pointnetX.model import PointNetCls, feature_transform_regularizer
from pointnetX.formData import GetTorchInputForPointNet

# Build
classifier = PointNetCls(k=2, feature_transform=False)
classifier.load_state_dict(torch.load("./pth/cls_model_5000.pth"))
classifier.cuda()
classifier = classifier.eval()

class MyCanvas(vispy.scene.SceneCanvas):
    def __init__(self):
        super().__init__(keys='interactive', show=True, bgcolor='black', size=(800, 600))
        self.unfreeze()
        # Basic View Config
        self.view = self.central_widget.add_view()
        self.view.camera = vispy.scene.cameras.TurntableCamera(azimuth=90, elevation=30, distance=30)
        self.timer = vispy.app.Timer(interval=5, connect=self.timer_go, start=0.05)

        # PC File
        self.fileRoot = ""
        self.filepaths = []
        self.nowID = 0

        # Draw Config
        self.PCscatter = vispy.scene.Markers()
        self.axis = vispy.scene.XYZAxis()
        self.cubesNode = vispy.scene.visuals.Node()
        self.view.add(self.PCscatter)
        self.view.add(self.axis)
        self.freeze()

    def SetFileRoot(self, fileroot):
        self.unfreeze()
        self.fileRoot = fileroot
        self.filepaths = []
        for filename in os.listdir(self.fileRoot):
            self.filepaths += [os.path.join(self.fileRoot, filename)]
        self.filepaths = natsort.natsorted(self.filepaths)
        self.nowID = 0
        self.freeze()

    def on_draw(self, event):
        super().on_draw(event)

    def timer_go(self, event):
        self.nowID += 1
        if self.nowID >= len(self.filepaths):
            self.nowID = 0

        pc = ReadPCD_XYZI(self.filepaths[self.nowID])
        # pc = ReadPCD_XYZI("D:\\Downloads\\ntutOutside\\0451.pcd")
        # pc = ReadPCD_XYZI("D:\\Downloads\\KITTI\\2011_09_28_drive_0016_extract\\velodyne_points\\pcdb\\0000000000.pcd")
        pc = Passthrough(pc, 0, -20, 20)
        pc = Passthrough(pc, 1, -20, 20)
        pc = DownSampe(pc)
        pcxyz, pccol = GetXYZnRGB(pc, clipMax=500)

        # RANSAC
        _, outliner = RANSAC_Plane(pcxyz, 0.1, 100, -10, -0.5)
        GoodXYZ = pcxyz[outliner, :]
        GoodCol = pccol[outliner, :]
        GoodCol[:, :] = 1

        # Cluster
        nLabels, labels = Cluster(GoodXYZ)
        # Select Cluster
        cluPoints = Getlusters(GoodXYZ, labels, nLabels, nPoints=300)

        # Make Data
        point_set = GetTorchInputForPointNet(cluPoints)
        point_set = point_set.cuda()

        # Classifier
        pred, _, _ = classifier(point_set)
        pred_choice = pred.data.max(1)[1].cpu().numpy()==1

        # Get Person Cluster
        Cluster_Person = cluPoints[pred_choice]

        pcxyz = GoodXYZ
        pccol = GoodCol

        self.PCscatter.set_data(pcxyz, edge_color=None, face_color=pccol, size=3)
        self.cubesNode = None
        self.cubesNode = vispy.scene.visuals.Node()
        # Cubes
        for clu in Cluster_Person:
            cub = MyCube(clu.min(axis=0), clu.max(axis=0), parent=self.cubesNode)
            self.cubesNode._add_child(cub)
        self.view.add(self.cubesNode)

if __name__ == '__main__':
    canvas = MyCanvas()
    canvas.SetFileRoot("D:\\Downloads\\ntutOutside\\")

    if sys.flags.interactive != 1:
        vispy.app.run()
