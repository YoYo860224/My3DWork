'''
Open3D PointCloud class only supports: x, y, z, r, g, b, normal_x, normal_y, normal_z
'''

import open3d
import numpy as np

def ReadPcd(filename):
    pcloud: open3d.geometry.PointCloud = open3d.io.read_point_cloud(filename)  # pylint: disable=maybe-no-member
    pcloud_np = np.asarray(pcloud.points)
    return pcloud_np

if __name__ == "__main__":
    pc = ReadPcd("D:\\Downloads\\ntutOutside\\0001.pcd")
    print(pc.shape)

