from laspy.file import File
import numpy as np

inFile = File("./file.las", mode='r')

pointformat = inFile.point_format
for spec in inFile.point_format:
    print(spec.name)
print("==========")

headerformat = inFile.header.header_format
for spec in headerformat:
    print(spec.name)
print("==========")

print(inFile.points[0])
print(inFile.X[0])
print(inFile.Y[0])
print(inFile.Z[0])
print(inFile.intensity[0])
print(inFile.flag_byte[0])
print(inFile.raw_classification[0])
print(inFile.user_data[0])
print(inFile.get_edge_flight_line()[0])
print(inFile.pt_src_id[0])
print(inFile.gps_time[0])

# inFile.visualize()
