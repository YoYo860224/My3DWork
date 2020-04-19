import os
import math
import argparse
import numpy as np
import open3d
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from pointnetX.model import PointNetCls, feature_transform_regularizer
from pointnetX.dataset import PersonDataset, PersonDataset_Test
from pointnetX.dataset import NPCDataset, NPCDataset_Test

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
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default="./data", help='input data root')
    parser.add_argument('--model', type=str, help='epochs')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    args = parser.parse_args()

    # data
    dataset_test = NPCDataset_Test(args.dataroot)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=20,
        shuffle=False,
        num_workers=4)

    # model load
    classifier = PointNetCls(k=3, feature_transform=args.feature_transform)
    classifier.load_state_dict(torch.load(args.model))
    classifier.cuda()

    CM = np.zeros((3, 3), dtype=np.int)

    for data in dataloader_test:
        pc, label, _ = data
        label = label[:, 0]
        pc = pc.transpose(2, 1)
        pc, label = pc.cuda(), label.cuda()
        classifier = classifier.eval()
        pred, trans, trans_feat = classifier(pc)
        loss = torch.nn.functional.nll_loss(pred, label)
        if args.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        pred_choice = pred.data.max(1)[1]
        
        pred_List = pred_choice.data.cpu().numpy()
        label_List = label.data.cpu().numpy()
        CM += np.asarray(confusion_matrix(label_List, pred_List, labels=[0, 1, 2]), dtype=np.int)

    plot_confusion_matrix(CM, classes=["Others", "Person", "Car"], title="Classification")
    plt.show()

    # tn = CM[0, 0]
    # tp = CM[1, 1]
    # fn = CM[1, 0]
    # fp = CM[0, 1]
    # correct = (tn + tp) / (tn+fn+tp+fp)
    # mAccu = (tn / (tn+fn)+ tp / (tp+fp)) / 2.0
    # recall = tp / (tp+fp)
    # print('loss: %f\naccuracy: %f\nmAccu: %f\nrecall: %f' % (loss.item(), correct, mAccu, recall))
