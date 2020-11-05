import numpy as np
from pypcd import pypcd

def ReadPCD_XYZI(filename):
    pcdata = pypcd.PointCloud.from_path(filename)
    pc = np.asarray(pcdata.pc_data[['x', 'y', 'z', 'intensity']].tolist(), dtype=float)
    return pc

def ReadPCD_XYZI2(filename):
    pcdata = np.fromfile(filename, dtype=float).reshape(-1, 4)
    # pc = np.asarray(pcdata.pc_data[['x', 'y', 'z', 'intensity']].tolist(), dtype=float)
    return pcdata

if __name__ == "__main__":
    # pc = ReadPCD_XYZI("/media/yoyo/harddisk/NTUT_Bagmap/Kitti.pcdb/0000000000.pcd")
    pc = ReadPCD_XYZI2("/home/yoyo/KITTI/object/training/velodyne/000000.bin")
    print(pc.shape)
