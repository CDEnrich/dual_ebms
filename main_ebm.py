import argparse
import logging
import math
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)

CIFAR_ROOT = 'data'

class Net(nn.Module):
    def __init__(self, width=2, n_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 16 * width, kernel_size=3, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=16, stride=16),
                # nn.Conv2d(16 * width, 32 * width, kernel_size=3, stride=1, padding=0, bias=False),
                # nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                # nn.Conv2d(32 * width, 64 * width, kernel_size=3, stride=2, padding=0, bias=False),
                # nn.ReLU(inplace=True),
                )
        #self.classifier = nn.Sequential(nn.Linear(width * 2 * 2, n_classes))

        # preprocessing
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

    def forward(self, x):
        x = (x - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _apply(self, fn):
        super(Net, self)._apply(fn)
        self.mean.data = fn(self.mean.data)
        self.std.data = fn(self.std.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='small data training')
    parser.add_argument('--interactive', action='store_true', help='do not save things models')
    parser.add_argument('--name', default='tmp', help='name of experiment variant')
    parser.add_argument('--data_dir', default=CIFAR_ROOT, help='directory for (cifar) data')

    parser.add_argument('--nsamples', default=256, type=int)
    parser.add_argument('--niter', default=501, type=int)
    parser.add_argument('--eval_delta', default=20, type=int,
                        help='evaluate every delta epochs')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--beta', default=100., type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=256, type=int)
    parser.add_argument('--aug', action='store_true', help='data augmentation')
    parser.add_argument('--width', default=1, type=int)
    args = parser.parse_args()

    torch.random.manual_seed(42)

    # load experiment
    if args.aug:
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

    cifar_root = args.data_dir
    trainset = torchvision.datasets.CIFAR10(root=cifar_root, train=True,
            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
            shuffle=True, num_workers=2)

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net(width=args.width)
    net.to(device)

    # compute "kernel" mean embedding of training set
    logging.info('computing mean embedding of training set')
    mean_map_tr = None
    n = 0
    for ims, labels in trainloader:
        ims = ims.to(device)
        n += ims.shape[0]
        with torch.no_grad():
            enc = net(ims).sum(0)
            if mean_map_tr is None:
                mean_map_tr = enc
            else:
                mean_map_tr += enc
    mean_map_tr /= n

    # initialize walkers
    logging.info('inintializing walkers')
    X = torch.zeros((args.nsamples, 3, 32, 32), device=device)
    idx = 0
    for ims, labels in trainloader:
        ims = ims.to(device)
        sz = min(args.nsamples - idx, ims.shape[0])
        X[idx:idx+sz] = ims[:sz]

    X += 0.00001 * torch.randn(X.shape)
    X.requires_grad_()

    # for saving some images
    writer = SummaryWriter()

    # training loop
    mean_map = torch.zeros(mean_map_tr.shape, device=device)
    for it in range(args.niter):
        logging.info(f'iteration {it}')

        # compute mean embedding of current walkers
        for i in range((args.nsamples - 1) // args.batch_size + 1):
            ims = X[i * args.batch_size:(i + 1) * args.batch_size]
            with torch.no_grad():
                enc = net(ims).sum(0)
                if mean_map is None:
                    mean_map = enc
                else:
                    mean_map += enc
        mean_map /= args.nsamples

        mmd = (mean_map - mean_map_tr).norm()

        # evaluate
        if it % args.eval_delta == 0:
            logging.info(f'mmd = {mmd}')
            # plot samples
            grid = torchvision.utils.make_grid(X[:100])
            writer.add_image('samples', grid, it)
            # writer.close()

        # gradient updates for walkers
        if X.grad is not None:
            X.grad.zero_()

        for i in range((args.nsamples - 1) // args.batch_size + 1):
            ims = X[i * args.batch_size:(i + 1) * args.batch_size]
            encs = net(ims)
            Usum = torch.matmul(encs, mean_map - mean_map_tr).sum()
            Usum.backward()

        X.data.sub_(args.lr * X.grad
                + torch.sqrt(args.lr * mmd / args.beta) * torch.randn(X.shape))

