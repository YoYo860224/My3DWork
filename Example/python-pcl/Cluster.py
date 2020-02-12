import numpy as np
import pcl
import pcl.pcl_visualization

oricp = pcl.load("D:/Downloads/ntutOutside/0001.pcd")
cloud = pcl.load("D:/Downloads/ntutOutside/0001.pcd")
nrp = cloud.size

# Pass filter
pf: pcl.PassThroughFilter_PointXYZI = cloud.make_passthrough_filter()
pf.set_filter_field_name("x")
pf.set_filter_limits(-20.0, 20.0)
cloud = pf.filter()

pf: pcl.PassThroughFilter_PointXYZI = cloud.make_passthrough_filter()
pf.set_filter_field_name("y")
pf.set_filter_limits(-20.0, 20.0)
cloud = pf.filter()

pf: pcl.PassThroughFilter_PointXYZI = cloud.make_passthrough_filter()
pf.set_filter_field_name("z")
pf.set_filter_limits(-20.0, 20.0)
cloud = pf.filter()

# downsample
vg: pcl.VoxelGridFilter_PointXYZI = cloud.make_voxel_grid_filter()
vg.set_leaf_size(0.1, 0.1, 0.1)
cloud = vg.filter()

# seg plane
seg: pcl.Segmentation_PointXYZI = cloud.make_segmenter_normals(ksearch=50)
seg = cloud.make_segmenter_normals(ksearch=50)
seg.set_optimize_coefficients(True)
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_max_iterations(1500)
seg.set_distance_threshold(0.3)
seg.set_axis(0, 0, 1)
seg.set_eps_angle(0.26)
inliers, model = seg.segment()
cloud: pcl.PointCloud_PointXYZI = cloud.extract(inliers, True)

# cluster
ce: pcl.EuclideanClusterExtraction = cloud.make_EuclideanClusterExtraction()
ce.set_ClusterTolerance(0.4)
ce.set_MinClusterSize(100)
ce.set_MaxClusterSize(6000)
ce.set_SearchMethod(pcl.KdTree())
clusters = ce.Extract()

vis = pcl.pcl_visualization.PCLVisualizering()

vis.AddCoordinateSystem(2.0, 0)
vis.AddPointCloud(oricp, b'all')
for i in clusters:
    obj1 = cloud.extract(i)
    vis.AddPointCloud_ColorHandler(obj1, pcl.pcl_visualization.PointCloudColorHandleringCustom(obj1,255, 0, 0))
    vis.SpinOnce(200)
    vis.RemovePointCloud(b'cloud', 0)