
import open3d
import os
import natsort

import numpy as np
import torch
import torch.utils.data as data

# pragma pylint: disable=maybe-no-member

def npReadPcd(filename):
    pcloud: open3d.geometry.PointCloud = open3d.io.read_point_cloud(filename)  # pylint: disable=maybe-no-member
    pcloud_np = np.asarray(pcloud.points, dtype=np.float32)
    return pcloud_np


class PCDataset(data.Dataset):
    def __init__(self, root, npoints=50, data_augmentation=True):
        self.root = root
        self.npoints = npoints
        self.pc = []
        self.data_augmentation = data_augmentation

        filepaths = []
        self.labels = []
        path_c = os.path.join(self.root, "C")
        for filename in os.listdir(path_c):
            filepaths += [os.path.join(path_c, filename)]
            self.labels += [0]
        path_p = os.path.join(self.root, "P")
        for filename in os.listdir(path_p):
            filepaths += [os.path.join(path_p, filename)]
            self.labels += [1]

        self.pc = []
        for filepath in filepaths:
            self.pc += [npReadPcd(filepath)]

    def __getitem__(self, index):
        # pc = torch.from_numpy(self.pc[index])
        pc = self.pc[index]
        choice = np.random.choice(len(pc), self.npoints, replace=True)
        point_set = pc[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        pc = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([self.labels[index]]))
        return pc, label

    def __len__(self):
        return len(self.pc)

class PCDataset_Test(data.Dataset):
    def __init__(self, root, npoints=50, data_augmentation=True):
        self.root = root
        self.npoints = npoints
        self.pc = []
        self.data_augmentation = data_augmentation

        filepaths = []
        self.labels = []
        path_c = os.path.join(self.root, "CT")
        for filename in os.listdir(path_c):
            filepaths += [os.path.join(path_c, filename)]
            self.labels += [0]
        path_p = os.path.join(self.root, "PT")
        for filename in os.listdir(path_p):
            filepaths += [os.path.join(path_p, filename)]
            self.labels += [1]

        self.pc = []
        for filepath in filepaths:
            self.pc += [npReadPcd(filepath)]
    def __getitem__(self, index):
        # pc = torch.from_numpy(self.pc[index])
        pc = self.pc[index]
        choice = np.random.choice(len(pc), self.npoints, replace=True)
        point_set = pc[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        pc = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([self.labels[index]]))
        return pc, label

    def __len__(self):
        return len(self.pc)


class PersonDataset(data.Dataset):
    def __init__(self, root, npoints=300, data_augmentation=True):
        self.root = root
        self.npoints = npoints
        self.pc = []
        self.data_augmentation = data_augmentation

        filepaths = []
        self.labels = []
        path_np = os.path.join(self.root, "NP")
        for filename in os.listdir(path_np):
            filepaths += [os.path.join(path_np, filename)]
            self.labels += [0]
        path_p = os.path.join(self.root, "P")
        for filename in os.listdir(path_p):
            filepaths += [os.path.join(path_p, filename)]
            self.labels += [1]

        self.pc = []
        for filepath in filepaths:
            self.pc += [npReadPcd(filepath)]

    def __getitem__(self, index):
        # pc = torch.from_numpy(self.pc[index])
        pc = self.pc[index]
        choice = np.random.choice(len(pc), self.npoints, replace=True)
        point_set = pc[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        pc = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([self.labels[index]], dtype=np.int64))
        return pc, label

    def __len__(self):
        return len(self.pc)

class PersonDataset_Test(data.Dataset):
    def __init__(self, root, npoints=300, data_augmentation=True):
        self.root = root
        self.npoints = npoints
        self.pc = []
        self.data_augmentation = data_augmentation

        filepaths = []
        self.labels = []

        path_t = os.path.join(self.root, "testdata")
        for filename in os.listdir(path_t):
            filepaths += [os.path.join(path_t, filename)]
        filepaths = natsort.natsorted(filepaths)

        with open(os.path.join(self.root, "truth_all.txt"), 'r') as f:
            lines = f.readlines()
            for i in lines:
                if (i) == "-1\n":
                    self.labels += [0]
                elif (i) == "1\n":
                    self.labels += [1]

        self.pc = []
        for filepath in filepaths:
            self.pc += [npReadPcd(filepath)]

    def __getitem__(self, index):
        # pc = torch.from_numpy(self.pc[index])
        pc = self.pc[index]
        choice = np.random.choice(len(pc), self.npoints, replace=True)
        point_set = pc[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        pc = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([self.labels[index]], dtype=np.int64))
        return pc, label

    def __len__(self):
        return 1000


if __name__ == "__main__":
    a = PersonDataset("../data")
    pc, label = a[10]
    print(pc.size(), pc.type())
    print(label.size(), label.type())
    # a = PersonDataset_Test("../data")
    # pc, label = a[1]
    # print(pc.size(), pc.type())
    # print(label.size(), label.type())
    # print(label)