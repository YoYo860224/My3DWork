import os
import math
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from pointnetX.model import PointNetCls, feature_transform_regularizer
from pointnetX.dataset_hdl32 import NPCDataset

blue = lambda x: '\033[94m' + x + '\033[0m'
keepTrAccu = []
keepTeAccu = []
keepTeloss = []
keepTrloss = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default="/home/yoyo/hdl32_data", help='input data root')
    parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
    parser.add_argument('--epochs', type=int, default=5000, help='epochs')
    parser.add_argument('--outf', type=str, default="./pth", help='epochs')
    parser.add_argument('--model', type=str, help='epochs')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    args = parser.parse_args()

    print("Prepare Dataset.")
    dataset = NPCDataset(args.dataroot)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=4)

    dataset_test = NPCDataset(args.dataroot, testing=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=25,
        shuffle=True,
        num_workers=4)

    try:
        print("mkdir ", args.outf)
        os.makedirs(args.outf)
    except OSError:
        pass

    print("Make model.")
    classifier = PointNetCls(k=4, feature_transform=args.feature_transform)
    if args.model:
        classifier.load_state_dict(torch.load(args.model), feature_transform=args.feature_transform)

    optimizer = optim.Adam(classifier.parameters(), lr=0.0003, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    print("Training start.")
    # train
    num_batch = len(dataset) // args.batchSize
    for epoch in range(1, args.epochs + 1):
        for i, data in enumerate(dataloader, 0):
            pc, label, img, artF, voxel = data
            tbSize = len(label)
            label = label[:, 0]
            pc = pc.transpose(2, 1)
            pc, label, img, artF, voxel = pc.cuda(), label.cuda(), img.cuda(), artF.cuda(), voxel.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(pc, img, artF, voxel)
            loss = torch.nn.functional.cross_entropy(pred, label)
            # loss = torch.nn.functional.nll_loss(pred, label)
            if args.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(label.data).cpu().sum()
            print('[%5d: %3d/%3d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(tbSize)), end='\r')

        keepTrAccu += [correct.item() / float(tbSize)]
        keepTrloss += [loss.item()]
        
        # Test
        t, data = next(enumerate(dataloader_test, 0))
        pc, label, img, artF, voxel = data
        tbSize = len(label)
        label = label[:, 0]
        pc = pc.transpose(2, 1)
        pc, label, img, artF, voxel = pc.cuda(), label.cuda(), img.cuda(), artF.cuda(), voxel.cuda()
        classifier = classifier.eval()
        pred, trans, trans_feat = classifier(pc, img, artF, voxel)
        loss = torch.nn.functional.nll_loss(pred, label)
        if args.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(label.data).cpu().sum()
        print('[%5d:-------->  %s loss: %f accuracy: %f' % (epoch, blue('test'), loss.item(), correct.item() / float(tbSize)))

        keepTeAccu += [correct.item() / float(tbSize)]
        keepTeloss += [loss.item()]
        
        if epoch > 5:
            if keepTeAccu[epoch-1] >= max(keepTeAccu) or (epoch > 4500 and epoch % 100 == 0):
                torch.save(classifier.state_dict(), '%s/fcePI_cls_model_%d_%f.pth' % (args.outf, epoch, keepTeAccu[epoch-1]))
        scheduler.step()

        # vis
        plt.ylim(0, 1)
        plt.plot(keepTrAccu, "b")
        plt.plot(keepTeAccu, "r")
        plt.savefig("./F1.png")
        plt.cla()
        plt.plot(keepTrloss, "b")
        plt.plot(keepTeloss, "r")
        plt.savefig("./F2.png")
        plt.cla()
