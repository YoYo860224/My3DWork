import os
import sys
import natsort

import numpy as np
import cv2
import torch
import torch.utils.data as data

sys.path.append(sys.path[0] + "/../")
from Util.PcdRead import ReadPCD_XYZI    # pylint: disable=import-error
from Util.ArtFeature import f15_Haar


np.random.seed(10)
NPC_testSize = 50


class NPCDataset(data.Dataset):
    def __init__(self, root, npoints=300, data_augmentation=True, testing=False):
        self.root = root
        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.testing = testing

        self.labels = []
        self.pc = []
        self.pcimg = []
        self.voxels = []

        # region pc and label
        filepaths = []
        s = 0
        path__ = os.path.join(self.root, "O")
        for filename in natsort.natsorted(os.listdir(path__)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path__, filename)]
            self.labels += [0]

        s = 0
        path__ = os.path.join(self.root, "P")
        for filename in natsort.natsorted(os.listdir(path__)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path__, filename)]
            self.labels += [1]
        
        s = 0
        path__ = os.path.join(self.root, "C")
        for filename in natsort.natsorted(os.listdir(path__)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path__, filename)]
            self.labels += [2]

        s = 0
        path__ = os.path.join(self.root, "M")
        for filename in natsort.natsorted(os.listdir(path__)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path__, filename)]
            self.labels += [3]

        for filepath in filepaths:
            self.pc += [ReadPCD_XYZI(filepath)]
        # endregion

        # region img
        filepaths = []
        s = 0
        path__ = os.path.join(self.root, "O_img")
        for filename in natsort.natsorted(os.listdir(path__)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path__, filename)]
        
        s = 0
        path__ = os.path.join(self.root, "P_img")
        for filename in natsort.natsorted(os.listdir(path__)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path__, filename)]

        s = 0
        path__ = os.path.join(self.root, "C_img")
        for filename in natsort.natsorted(os.listdir(path__)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path__, filename)]

        s = 0
        path__ = os.path.join(self.root, "M_img")
        for filename in natsort.natsorted(os.listdir(path__)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path__, filename)]

        for filepath in filepaths:
            self.pcimg += [cv2.imread(filepath)]
        # endregion

    def __getitem__(self, index):
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
        img = torch.from_numpy(self.pcimg[index]).float()
        artF = torch.from_numpy(f15_Haar(self.pcimg[index])).float()

        return pc, label, img, artF

    def __len__(self):
        return len(self.pc)


if __name__ == "__main__":
    a = NPCDataset("/home/yoyo/hdl32_data", testing=True)
    pc, label, img, artF, xd = a[10]
    print(pc.size(), pc.type())
    print(label.size(), label.type())
    print(img.size(), img.type())
    print(artF.size(), artF.type())
