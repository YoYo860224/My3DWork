import os
import math
import vispy
import natsort
import numpy as np
from vispy import app, gloo, scene
from pypcd import pypcd


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


class MyCanvas(vispy.scene.SceneCanvas):
    def __init__(self):
        super().__init__(keys='interactive', show=True, bgcolor='black', size=(800, 600))
        self.unfreeze()
        # Basic View Config
        self.view = self.central_widget.add_view()
        self.view.camera = vispy.scene.cameras.TurntableCamera(azimuth=90, elevation=30, distance=30)
        self.timer = vispy.app.Timer(interval='auto', connect=self.timer_go, start=0.05)

        # PC File
        self.fileRoot = ""
        self.filepaths = []
        self.nowID = 0

        # Draw Config
        self.PCscatter = vispy.scene.Markers()
        self.axis = vispy.scene.XYZAxis()
        self.view.add(self.PCscatter)
        self.view.add(self.axis)
        self.freeze()

    def on_draw(self, event):
        super().on_draw(event)

    def timer_go(self, event):
        self.nowID += 1
        if self.nowID >= len(self.filepaths):
            self.nowID = 0

        pc = ReadPCD_XYZI(self.filepaths[self.nowID])
        pcxyz, pccol = GetPosColor(pc)
        self.PCscatter.set_data(pcxyz, edge_color=None, face_color=pccol, size=3)


    def SetFileRoot(self, fileroot):
        self.unfreeze()
        self.fileRoot = fileroot
        self.filepaths = []
        for filename in os.listdir(self.fileRoot):
            self.filepaths += [os.path.join(self.fileRoot, filename)]
        self.filepaths = natsort.natsorted(self.filepaths)
        self.nowID = 0
        self.freeze()



if __name__ == '__main__':
    canvas = MyCanvas()
    canvas.SetFileRoot("D:\\Downloads\\ntutOutside\\")

    print(canvas.scene.describe_tree(with_transform=True))
    canvas.measure_fps()
    canvas.app.run()
