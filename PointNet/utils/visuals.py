import os
import vispy
import vispy.scene
import numpy as np
import natsort

def GetXYZnRGB(pc, clipMax=-1):
    pcXYZ = pc[:, 0:3]
    pcCol = pc[:, 3:4]
    if clipMax == -1:
        clipMax = np.max(pcCol)
    else:
        pcCol = np.clip(pcCol, 0, clipMax)
    pcCol /= clipMax
    pcCol = np.repeat(pcCol, 3, axis=1)
    pcCol[:, 0] = 1
    pcCol[:, 2] = 0

    return pcXYZ, pcCol


class MyCube(vispy.scene.Cube):
    def __init__(self, minPos, maxPos, parent=None):
        super().__init__(color=(0, 1, 1, 0.2), parent=parent)
        self.transform = vispy.visuals.transforms.MatrixTransform()
        self.transform.scale((maxPos-minPos))
        self.transform.translate((minPos+maxPos)/2.0)

