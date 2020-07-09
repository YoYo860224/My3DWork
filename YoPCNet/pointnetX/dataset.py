import os
import sys
import natsort

import numpy as np
import cv2.cv2 as cv2
import torch
import torch.utils.data as data


sys.path.append(sys.path[0] + "/../")
from Util.PcdRead import ReadPCD_XYZI    # pylint: disable=import-error

np.random.seed(10)

NPC_testSize = 100


# pragma pylint: disable=maybe-no-member

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
        path_n = os.path.join(self.root, "N")
        for filename in natsort.natsorted(os.listdir(path_n)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path_n, filename)]
            self.labels += [0]

        s = 0
        path_p = os.path.join(self.root, "P")
        for filename in natsort.natsorted(os.listdir(path_p)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path_p, filename)]
            self.labels += [1]
        
        s = 0
        path_c = os.path.join(self.root, "C")
        for filename in natsort.natsorted(os.listdir(path_c)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path_c, filename)]
            self.labels += [2]

        for filepath in filepaths:
            self.pc += [ReadPCD_XYZI(filepath)]
        # endregion

        # region img
        filepaths = []
        s = 0
        path_n = os.path.join(self.root, "N_img")
        for filename in natsort.natsorted(os.listdir(path_n)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path_n, filename)]
        
        s = 0
        path_p = os.path.join(self.root, "P_img")
        for filename in natsort.natsorted(os.listdir(path_p)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path_p, filename)]

        s = 0
        path_c = os.path.join(self.root, "C_img")
        for filename in natsort.natsorted(os.listdir(path_c)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path_c, filename)]

        for filepath in filepaths:
            self.pcimg += [cv2.imread(filepath)]
        # endregion

        # region ArtFeature
        artF_N = np.load(os.path.join(self.root, "N_feature.npy"))
        artF_P = np.load(os.path.join(self.root, "P_feature.npy"))
        artF_C = np.load(os.path.join(self.root, "C_feature.npy"))
        if testing:
            self.artFeature = np.concatenate([artF_N[0:100, :], artF_P[0:100, :], artF_C[0:100, :]])
        else:
            self.artFeature = np.concatenate([artF_N[100:, :], artF_P[100:, :], artF_C[100:, :]])
        # endregion

        # region Voxel
        filepaths = []
        s = 0
        path_n = os.path.join(self.root, "N_voxel")
        for filename in natsort.natsorted(os.listdir(path_n)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path_n, filename)]

        s = 0
        path_p = os.path.join(self.root, "P_voxel")
        for filename in natsort.natsorted(os.listdir(path_p)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path_p, filename)]
        
        s = 0
        path_c = os.path.join(self.root, "C_voxel")
        for filename in natsort.natsorted(os.listdir(path_c)):
            if self.testing and s >= NPC_testSize:
                break
            if s < NPC_testSize:
                s+=1
                if not self.testing:
                    continue
            filepaths += [os.path.join(path_c, filename)]

        for filepath in filepaths:
            self.voxels += [np.load(filepath)]
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
        artFeature = torch.from_numpy(self.artFeature[index]).float()
        voxel = torch.from_numpy(self.voxels[index]).float()
        return pc, label, img, artFeature, voxel

    def __len__(self):
        return len(self.pc)


if __name__ == "__main__":
    a = NPCDataset("/media/yoyo/harddisk/kitti_npc")
    pc, label, img, artF = a[10]
    print(pc.size(), pc.type())
    print(label.size(), label.type())
    print(img.size(), img.type())
    print(artF.size(), artF.type())
