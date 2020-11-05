import os
import math
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2

from pointnetX.model import PointNetCls, feature_transform_regularizer
from pointnetX.dataset_hdl32 import NPCDataset

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=30)

    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    plt.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default="/home/yoyo/hdl32_data", help='input data root')
    parser.add_argument('--model', type=str, help='epochs')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    args = parser.parse_args()

    # data
    dataset_test = NPCDataset(args.dataroot, testing=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=20,
        shuffle=False,
        num_workers=4)

    # model load
    classifier = PointNetCls(k=4, feature_transform=args.feature_transform)
    classifier.load_state_dict(torch.load(args.model))
    classifier.cuda()

    CM = np.zeros((4, 4), dtype=np.int)
    tlist = []
    failNum = 0
    for data in dataloader_test:
        t = time.time()
        pc, label, img, artF, voxel = data
        label = label[:, 0]
        pc = pc.transpose(2, 1)
        pc, label, img, artF, voxel = pc.cuda(), label.cuda(), img.cuda(), artF.cuda(), voxel.cuda()
        classifier = classifier.eval()
        pred, trans, trans_feat = classifier(pc, img, artF, voxel)
        loss = torch.nn.functional.nll_loss(pred, label)
        if args.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        pred_choice = pred.data.max(1)[1]
        
        pred_List = pred_choice.data.cpu().numpy()
        label_List = label.data.cpu().numpy()
        tlist.append(time.time()-t)
        CM += np.asarray(confusion_matrix(label_List, pred_List, labels=[0, 1, 2, 3]), dtype=np.int)

        failList = (pred_List != label_List)
        for i in range(len(failList)):
            if failList[i]==True:
                failImg = img[i].cpu().numpy()
                cv2.imwrite("{0}__({1}, {2}).png".format(failNum, pred_List[i], label_List[i]), failImg)
                failNum+=1

    print("Spent time per 20 objs.(out first): ", sum(tlist[1:]) / 280.0 * 20.0)
    print("Accu: ", (CM[0, 0]+CM[1, 1]+CM[2, 2]+CM[3, 3])/200.0)
    plot_confusion_matrix(CM, classes=["Others", "Person", "Car", "Moto"], title="Classification")
    plt.show()
