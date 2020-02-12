import sys
import vispy
import vispy.scene
import numpy as np
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

# generate data
pc = ReadPCD_XYZI("D:\\Downloads\\ntutOutside\\0001.pcd")
pcxyz, pccol = GetPosColor(pc)

# Make a canvas and add simple view
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()

vispy.scene

# Camera
view.camera = vispy.scene.cameras.TurntableCamera(azimuth=90, elevation=30, distance=30)

# Create visuals objects
# PointCloud
scatter = vispy.scene.Markers()
scatter = vispy.scene.visuals.Markers(edge_width=0)
scatter.set_data(pcxyz, edge_color=None, face_color=pccol, size=3)

# Cube
cMT = vispy.visuals.transforms.MatrixTransform()
cube1 = vispy.scene.visuals.Cube(name='cube1', color=(0, 1, 0, 0.2))
cube1.transform = cMT

# Axis
axis = vispy.scene.XYZAxis()

# Add to view
view.add(axis)
view.add(scatter)
view.add(cube1)

def update_here(event):
    cMT.rotate(1, (1, 0, 0))

if __name__ == '__main__':
    timer = vispy.app.Timer(interval='auto', connect=update_here, start=0.05)
    if sys.flags.interactive != 1:
        vispy.app.run()
