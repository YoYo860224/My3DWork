import os
import math
import argparse
import numpy as np
import open3d
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from pointnetX.model import PointNetCls, feature_transform_regularizer
from pointnetX.dataset import PersonDataset, PersonDataset_Test

blue = lambda x: '\033[94m' + x + '\033[0m'
keepTrAcc = []
keepTeAcc = []
keepTeloss = []
keepTrloss = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default="./data", help='input data root')
    parser.add_argument('--batchSize', type=int, default=30, help='input batch size')
    parser.add_argument('--epochs', type=int, default=5000, help='epochs')
    parser.add_argument('--outf', type=str, default="./pth", help='epochs')
    parser.add_argument('--model', type=str, help='epochs')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    args = parser.parse_args()

    # data
    dataset = PersonDataset(args.dataroot)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=4)

    dataset_test = PersonDataset_Test(args.dataroot)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=100,
        shuffle=True,
        num_workers=4)

    # model save
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    # model make
    classifier = PointNetCls(k=2, feature_transform=args.feature_transform)
    if args.model:
        classifier.load_state_dict(torch.load(args.model), feature_transform=args.feature_transform)

    optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    # train
    num_batch = len(dataset) / args.batchSize
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        for i, data in enumerate(dataloader, 0):
            pc, label = data
            label = label[:, 0]
            pc = pc.transpose(2, 1)
            pc, label = pc.cuda(), label.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(pc)
            loss = torch.nn.functional.nll_loss(pred, label)
            if args.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(label.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(args.batchSize)))

            if i == 0:
                keepTrAcc += [correct.item()/float(args.batchSize)]
                keepTrloss += [loss.item()]
                j, data = next(enumerate(dataloader_test, 0))
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                loss = torch.nn.functional.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(100)))
                keepTeAcc += [correct.item()/float(100)]
                keepTeloss += [loss.item()]
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (args.outf, epoch))

    # vis
    plt.ylim(0, 1)
    plt.plot(keepTrAcc, "b")
    plt.plot(keepTeAcc, "r")
    plt.show()
    plt.plot(keepTrloss, "b")
    plt.plot(keepTeloss, "r")
    plt.show()
