import pcl
import pcl.pcl_visualization

import numpy as np

def IShow(filename):
    # Load point cloud
    cloud = pcl.load_XYZI(filename)

    # Convert to numpy
    print(type(cloud))
    np_cloud = np.empty([cloud.width, 4], dtype=np.float32)
    np_cloud = cloud.to_array()

    # If need centred data
    centred = np_cloud.copy()
    centred[:, 0:3] = (np_cloud - np.mean(np_cloud, axis=0))[:, 0:3]

    # visualize
    visual = pcl.pcl_visualization.CloudViewing()
    while (not visual.WasStopped()):
        visual.ShowGrayCloud(cloud, b'cloud')

def MoreShow(filename):
    # Load point cloud
    cloud = pcl.load(filename)
    
    # Convert to numpy
    np_cloud = np.empty([cloud.width, 3], dtype=np.float32)
    np_cloud = cloud.to_array()

    # If need centred data
    centred = np_cloud.copy()
    centred[:, 0:3] = (np_cloud - np.mean(np_cloud, axis=0))[:, 0:3]

    # visualize
    vis = pcl.pcl_visualization.PCLVisualizering()
    vis.AddPointCloud(cloud)
    vis.AddCoordinateSystem(10.0, 0)
    vis.AddCube(-3, 3, -3, 3, -3, 3, 0.1, 0.1, 0.1, "C")
    while (not vis.WasStopped()):
        vis.SpinOnce(100)

if __name__ == "__main__":
    MoreShow("D:\\Downloads\\ntutOutside\\0001.pcd")