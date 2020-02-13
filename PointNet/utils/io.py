import numpy as np
from pypcd import pypcd


def ReadPCD_XYZI(filename):
    pcdata = pypcd.PointCloud.from_path(filename)
    pc = np.asarray(pcdata.pc_data[['x', 'y', 'z', 'intensity']].tolist(), dtype=float)
    return pc

def ReadPCD_XYZ(filename):
    pcdata = pypcd.PointCloud.from_path(filename)
    pc = np.asarray(pcdata.pc_data[['x', 'y', 'z']].tolist(), dtype=float)
    return pc
