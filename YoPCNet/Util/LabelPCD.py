import os
import natsort

import pypcd
import numpy as np
import vispy
from vispy import app, gloo, scene


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
    pcCol[:, 1] = 1
    pcCol[:, 2] = 1

    return pcXYZ, pcCol

path_dataS    = "/home/yoyo/桌面/0909lidar1"
path_labelTxt = "/home/yoyo/桌面/0909lidar1.txt"

LABELS = ["car", "moto", "people", "other"]

class MyCanvas(vispy.scene.SceneCanvas):
    def __init__(self):
        super().__init__(keys='interactive', show=True, bgcolor='black', size=(800, 600))
        self.unfreeze()

        self.nowFramePath = ""
        self.fileList = natsort.natsorted(os.listdir(path_dataS))
        self.lenFile = len(self.fileList)
        self.nowIdx = 0
        self.label = [-1] *  self.lenFile

        if os.path.exists(path_labelTxt):
            rf = open(path_labelTxt, "r")
            lines = rf.readlines()
            for i in range(self.lenFile):
                pcdName, label = lines[i][:-1].split(' ')
                self.label[i] = label
            rf.close()

        # Basic View Config
        self.view = self.central_widget.add_view()
        self.view.camera = vispy.scene.cameras.TurntableCamera(azimuth=90, elevation=30, distance=30)

        # Draw Config
        self.PCscatter = vispy.scene.Markers()
        self.PCscatter2 = vispy.scene.Markers()
        self.cube = vispy.scene.visuals.Cube(name='cube1', color=(0, 1, 0, 0.2))
        self.axis = vispy.scene.XYZAxis()
        self.view.add(self.PCscatter)
        self.view.add(self.PCscatter2)
        self.view.add(self.cube)
        self.view.add(self.axis)
        self.freeze()

    def on_draw(self, event):
        super().on_draw(event)

    def on_key_press(self, event):
        # modifiers = [key.name for key in event.modifiers]
        # print('Key pressed - text: %r, key: %s, modifiers: %r' % (
        #         event.text, event.key.name, modifiers))

        if event.key.name == "D":
            print("Default")
            self.label = [-1] * len(self.fileList)
        elif event.key.name == "R":
            print("Read")
            rf = open(path_labelTxt, "r")
            lines = rf.readlines()
            for i in range(self.lenFile):
                pcdName, label = lines[i][:-1].split(' ')
                self.label[i] = label
            rf.close()
        
        if event.key.name == "1":
            print("car")
            self.label[self.nowIdx] = 1
            self.nowIdx += 1
        elif event.key.name == "2":
            print("moto")
            self.label[self.nowIdx] = 2
            self.nowIdx += 1
        elif event.key.name == "3":
            print("people")
            self.label[self.nowIdx] = 3
            self.nowIdx += 1
        elif event.key.name == "0":
            print("other")
            self.label[self.nowIdx] = 0
            self.nowIdx += 1

        if event.key.name == "Right":
            self.nowIdx += 1
        elif event.key.name == "Left":
            self.nowIdx -= 1
        elif event.key.name == "P":
            self.nowIdx += 100
        elif event.key.name == "O":
            self.nowIdx -= 100

        f = self.fileList[self.nowIdx]
        print(os.path.join(path_dataS, f), self.label[self.nowIdx])
        frame, idx = [int(i) for i in f.split(".")[0].split("_")]
        if idx == 0:
            self.nowFramePath = os.path.join(path_dataS, f)
            pc = ReadPCD_XYZI(self.nowFramePath)
            pcxyz, pccol = GetPosColor(pc, 255)
            self.PCscatter.set_data(pcxyz, edge_color=pccol, face_color=pccol, size=3)
            self.label[self.nowIdx] = -1
        else:
            pc = ReadPCD_XYZI(os.path.join(path_dataS, f))
            pcxyz, pccol = GetPosColor(pc, 500)
            xmax, xmin = pcxyz[:, 0].max(), pcxyz[:, 0].min()
            ymax, ymin = pcxyz[:, 1].max(), pcxyz[:, 1].min()
            zmax, zmin = pcxyz[:, 2].max(), pcxyz[:, 2].min()
            x_range, x = xmax-xmin, (xmax+xmin)/2
            y_range, y = ymax-ymin, (ymax+ymin)/2
            z_range, z = zmax-zmin, (zmax+zmin)/2

            cMT = vispy.visuals.transforms.MatrixTransform()
            cMT.scale((x_range,y_range,z_range))
            cMT.translate((x,y,z))
            pccol[:, 0] = 1
            pccol[:, 1] = 0
            pccol[:, 2] = 0
            self.cube.transform = cMT
            self.PCscatter2.set_data(pcxyz, edge_color=None, face_color=pccol, size=3)
        
        wf = open(path_labelTxt, "w")
        for i in range(self.lenFile):
            wf.write("{0} {1}\n".format(self.fileList[i], self.label[i]))
        wf.close()

if __name__ == "__main__":
    canvas = MyCanvas()
    canvas.app.run()
