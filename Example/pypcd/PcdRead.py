import numpy as np
from pypcd import pypcd

def ReadPCD_XYZI(filename):
    pcdata = pypcd.PointCloud.from_path(filename)
    pc = np.asarray(pcdata.pc_data[['x', 'y', 'z', 'intensity']].tolist(), dtype=float)
    return pc

if __name__ == "__main__":
    pc = ReadPCD_XYZI("/media/yoyo/harddisk/NTUT_Bagmap/Kitti.pcdb/0000000000.pcd")
    print(pc.shape)
