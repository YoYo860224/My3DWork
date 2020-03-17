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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default="./data", help='input data root')
    parser.add_argument('--model', type=str, help='epochs')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    args = parser.parse_args()

    # data
    dataset_test = PersonDataset_Test(args.dataroot)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=20,
        shuffle=False,
        num_workers=4)

    # model load
    classifier = PointNetCls(k=2, feature_transform=args.feature_transform)
    classifier.load_state_dict(torch.load(args.model))
    classifier.cuda()

    tn, tp, fn, fp = 0, 0, 0, 0

    for data in dataloader_test:
        pc, label, _ = data
        label = label[:, 0]
        pc = pc.transpose(2, 1)
        pc, label = pc.cuda(), label.cuda()
        classifier = classifier.eval()
        pred, trans, trans_feat = classifier(pc, img)
        loss = torch.nn.functional.nll_loss(pred, label)
        if args.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        pred_choice = pred.data.max(1)[1]
        C2= confusion_matrix(pred_choice.data.cpu().numpy(), label.data.cpu().numpy())
        tn += C2[0, 0]
        tp += C2[1, 1]
        fn += C2[1, 0]
        fp += C2[0, 1]

    correct = (tn + tp) / (tn+fn+tp+fp)
    mAccu = (tn / (tn+fn)+ tp / (tp+fp)) / 2.0
    recall = tp / (tp+fp)
    print('loss: %f\naccuracy: %f\nmAccu: %f\nrecall: %f' % (loss.item(), correct, mAccu, recall))
