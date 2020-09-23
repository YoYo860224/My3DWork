import os
from shutil import copyfile


path_dataS    = "/home/yoyo/桌面/0909lidar1"
path_labelTxt = "/home/yoyo/桌面/0909lidar1.txt"

path_other  = "/home/yoyo/hdl32_data/Other"
path_car    = "/home/yoyo/hdl32_data/Car"
path_moto   = "/home/yoyo/hdl32_data/Moto"
path_people = "/home/yoyo/hdl32_data/People"

keep_label = [0, 0, 0, 0]
path_label = [path_other, path_car, path_moto, path_people]
prefixName = "01"

for i in path_label:
    if not os.path.exists(i):
        print("Make ", i)
        os.makedirs(i)

f = open(path_labelTxt, "r")

for i in f.readlines():
    pcdName, label = i[:-1].split(' ')
    if label != "-1":
        l_int = int(label)
        copyfile(os.path.join(path_dataS, pcdName), os.path.join(path_label[l_int], prefixName + "_" + pcdName))
        keep_label[l_int] += 1

print("other: ",  keep_label[0])
print("car: ",    keep_label[1])
print("moto: ",   keep_label[2])
print("people: ", keep_label[3])
