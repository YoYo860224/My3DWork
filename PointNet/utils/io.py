import numpy as np
from pypcd import pypcd


def ReadPCD_XYZI(filename):
    pcdata = pypcd.PointCloud.from_path(filename)
    pc = np.asarray(pcdata.pc_data[['x', 'y', 'z', 'intensity']].tolist(), dtype=float)
    return pc


def ReadPCD_XYZI_withoutRing(filename):
    pcdata = pypcd.PointCloud.from_path(filename)
    pc = np.asarray(pcdata.pc_data[['x', 'y', 'z', 'intensity']].tolist(), dtype=float)
    pcxyz = np.asarray(pcdata.pc_data[['x', 'y', 'z']].tolist(), dtype=float)
    pcr = np.asarray(pcdata.pc_data['ring'].tolist(), dtype=int)

    ringsPos = np.ndarray((0), dtype=int)
    for i in range(64):
        ringID = np.where(pcr==i)
        ringPC = pcxyz[ringID]
        ringNorm = np.linalg.norm(ringPC, axis=1)

        dx = np.roll(ringNorm, 1) - ringNorm
        ringPos = np.abs(dx)<0.5

        cnt = 0
        for idx, v in enumerate(ringPos):
            if v == True:
                cnt+=1
            elif (cnt < 50):
                ringPos[idx-cnt:idx] = False
                cnt = 0
            else:
                cnt = 0
        
        ringsPos = np.concatenate([ringsPos, ringID[0][np.where(ringPos==False)]])
    pc = pc[ringsPos]
    return pc

def ReadPCD_XYZ(filename):
    pcdata = pypcd.PointCloud.from_path(filename)
    pc = np.asarray(pcdata.pc_data[['x', 'y', 'z']].tolist(), dtype=float)
    return pc
