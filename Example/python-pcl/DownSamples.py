import os
import numpy as np
import pcl

fromPath = "D:\\Downloads\\KITTI\\2011_09_28_drive_0016_extract\\velodyne_points\\pcdb\\"
toPath = "D:\\Downloads\\KITTI\\2011_09_28_drive_0016_extract\\velodyne_points\\pcddb\\"
# fromPath = "D:\\Downloads\\ntutOutside"
# toPath = "D:\\Downloads\\ntutOutsideD"

if not os.path.exists(toPath):
    os.mkdir(toPath)

for filename in os.listdir(fromPath):
    loadfilepath = os.path.join(fromPath, filename)
    savefilepath = os.path.join(toPath, filename)

    cloud = pcl.load_XYZI(loadfilepath)

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

    pcl.save(cloud, savefilepath, binary=True)

    print(loadfilepath, " OK!")
