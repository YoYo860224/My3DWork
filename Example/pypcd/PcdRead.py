import numpy as np
from pypcd import pypcd

def ReadPCD_XYZI(filename):
    pcdata = pypcd.PointCloud.from_path(filename)
    pc = np.asarray(pcdata.pc_data[['x', 'y', 'z', 'intensity']].tolist(), dtype=float)
    return pc

if __name__ == "__main__":
    pc = ReadPCD_XYZI("D:\\Downloads\\ntutOutside\\0001.pcd")
    print(pc.shape)
