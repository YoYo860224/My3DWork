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
        batch_size=500,
        shuffle=False,
        num_workers=4)

    # model load
    classifier = PointNetCls(k=2, feature_transform=args.feature_transform)
    classifier.load_state_dict(torch.load(args.model))
    classifier.cuda()

    j, data = next(enumerate(dataloader_test, 0))
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    loss = torch.nn.functional.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    C2= confusion_matrix(pred_choice.data.cpu().numpy(), target.data.cpu().numpy())
    correct = C2[0, 0] + C2[1, 1]
    tn = C2[0, 0]
    tp = C2[1, 1]
    fn = C2[1, 0]
    fp = C2[0, 1]

    correct = (tn + tp) / (tn+fn+tp+fp)
    mAccu = (tn / (tn+fn)+ tp / (tp+fp)) / 2.0
    recall = tp / (tp+fp)
    print('loss: %f\naccuracy: %f\nmAccu: %f\nrecall: %f' % (loss.item(), correct, mAccu, recall))
