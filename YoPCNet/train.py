import os
import math
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from pointnetX.model import PointNetCls, feature_transform_regularizer
from pointnetX.dataset import NPCDataset
# from pointnetX.dataset_hdl32 import NPCDataset

blue = lambda x: '\033[94m' + x + '\033[0m'
keepAccu_train = []
keeploss_train = []
keepAccu_test = []
keeploss_test = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default="/home/yoyo/hdl32_data", help='Input data root.')
    parser.add_argument('--batchSize', type=int, default=20, help='Input batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--ltype', type=str, default="CE", help='CE: cross-entropy loss, NL: nll loss')
    parser.add_argument('--mtype', type=str, default="P", help='P: pointNet, I: MobileNet, H: Haar-like')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--outd', type=str, default="./pth", help='output dir.')
    parser.add_argument('--model', type=str, help='load model')
    args = parser.parse_args()

    args.dataroot="/media/yoyo/harddisk/kitti_npc"
    class_list=["Others", "Person", "Car"]
    dataName = "v64"

    # args.dataroot="/home/yoyo/hdl32_data"
    # class_list=["Others", "Person", "Car", "Moto"]
    # dataName = "v32"

    # Data
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
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=4)

    # Make Dir
    try:
        print("mkdir ", args.outd)
        os.makedirs(args.outd)
    except OSError:
        pass

    # Model load
    print("Make model.")
    classifier = PointNetCls(mtype=args.mtype, k=len(class_list), feature_transform=args.feature_transform)
    if args.model:
        classifier.load_state_dict(torch.load(args.model), feature_transform=args.feature_transform)

    # Model detail
    if args.ltype=="CE":
        loss_func = torch.nn.functional.cross_entropy
    else:
        loss_func = torch.nn.functional.nll_loss
    optimizer = optim.Adam(classifier.parameters(), lr=0.0003, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()


    # Train
    print("Training start.")
    num_batch = len(dataset) // args.batchSize
    for epoch in range(1, args.epochs + 1):
        # Train
        loss_sum = 0
        corr_sum = 0
        dataNum = float(len(dataset))

        for i, data in enumerate(dataloader, 0):
            # Get Data
            pc, label, img, artF = data
            tbSize = len(label)
            label = label[:, 0]
            pc = pc.transpose(2, 1)
            pc, label, img, artF = pc.cuda(), label.cuda(), img.cuda(), artF.cuda()
            
            # Train
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(pc, img, artF)

            loss = loss_func(pred, label)
            if args.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            
            loss.backward()
            optimizer.step()

            # Predict Result
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(label.data).cpu().sum()
            loss_sum += loss.item()
            corr_sum += correct.item()
            print('[%5d: %3d/%3d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(tbSize)), end='\r')

        loss = loss_sum / dataNum
        accu = corr_sum / dataNum
        print('[%5d: %3d/%3d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss, accu))
        keepAccu_train += [accu]
        keeploss_train += [loss]
        
        # Test
        loss_sum = 0
        corr_sum = 0
        dataNum = len(dataset_test)
        for i, data in enumerate(dataloader_test, 0):
            # Get Data
            pc, label, img, artF = data
            tbSize = len(label)
            label = label[:, 0]
            pc = pc.transpose(2, 1)
            pc, label, img, artF = pc.cuda(), label.cuda(), img.cuda(), artF.cuda()

            # Eval model
            classifier = classifier.eval()
            pred, trans, trans_feat = classifier(pc, img, artF)

            loss = loss_func(pred, label)
            if args.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            
            # Predict Result
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(label.data).cpu().sum()
            loss_sum += loss.item()
            corr_sum += correct.item()

        loss = loss_sum / dataNum
        accu = corr_sum / dataNum
        print(blue(' %5d:-------->  test loss: %f accuracy: %f' % (epoch, loss, accu)))
        keepAccu_test += [accu]
        keeploss_test += [loss]
        
        scheduler.step()

        # Save Model
        if epoch > 5:
            if keepAccu_test[-1] >= max(keepAccu_test) or (epoch > 4500 and epoch % 100 == 0):
                torch.save(classifier.state_dict(), "{0}/{1}/{2}.{3}.{4}_{1}_model_{5:04d}_{6:0.3f}.pth".format(args.outd,
                                                                                                                dataName,
                                                                                                                args.ltype,
                                                                                                                args.mtype,
                                                                                                                "wF" if args.feature_transform else "nF",
                                                                                                                epoch,
                                                                                                                keepAccu_test[-1]))

        # Visualize
        plt.ylim(0, 1)
        plt.plot(keepAccu_train, "b")
        plt.plot(keepAccu_test, "r")
        plt.savefig("{0}/{1}/ImgAccu/{2}.{3}.{4}_{1}_Accu.png".format(args.outd,
                                                                      dataName,
                                                                      args.ltype,
                                                                      args.mtype,
                                                                      "wF" if args.feature_transform else "nF"))
        plt.cla()
        plt.plot(keeploss_train, "b")
        plt.plot(keeploss_test, "r")
        plt.savefig("{0}/{1}/ImgLoss/{2}.{3}.{4}_{1}_Loss.png".format(args.outd,
                                                                     dataName,
                                                                     args.ltype,
                                                                     args.mtype,
                                                                     "wF" if args.feature_transform else "nF"))
        plt.cla()
