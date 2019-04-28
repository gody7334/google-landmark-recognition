############ Setup Project ######################
import os
import matplotlib
from utils.project import Global as G

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
matplotlib.use('Agg')

EXPERIMENT="999_dev_pipeline"
DISCRIPTION='develop pipeline'
gpu_id = '0'

G(EXPERIMENT, DISCRIPTION, gpu_id)

mode="DEV"

##################################################

from pathlib import Path
import colored_traceback.always
from matplotlib.pyplot import *

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from utils.lr_finder import LRFinder
from utils.lr_scheduler import TriangularLR
from utils.bot import BaseBot, OneCycle
from dev.gap_bot import *
from dev.data import *
from dev.model import *

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

####################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose(
            [  transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                                download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                               download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                 shuffle=False, num_workers=4)
classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = torch.nn.CrossEntropyLoss()

bot = BaseBot(
    model, trainloader, testloader,
    optimizer=optimizer,
    criterion=criterion,
    echo=True,
    use_tensorboard=True,
    avg_window=25,
    snapshot_policy='last'
)

'stage0'
oc = OneCycle(bot,
        scheduler="Default Triangular",
        unfreeze_layers=[model.fc1,model.fc2,model.fc3],
        n_epoch=5,
        stage='0')
oc.train_one_cycle()

import ipdb; ipdb.set_trace();

'stage1'
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
oc.update_bot(optimizer = optimizer,
        scheduler="Default Triangular",
        unfreeze_all=True,
        n_epoch=10,
        stage='1',
        accu_gradient_step=10)
oc.train_one_cycle()

'stage2'
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
oc.update_bot(optimizer = optimizer,
        scheduler="Default Triangular",
        unfreeze_all=True,
        n_epoch=20,
        stage='2',
        accu_gradient_step=10)
oc.train_one_cycle()

'stage3'
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
oc.update_bot(optimizer = optimizer,
        scheduler="Default Triangular",
        unfreeze_all=True,
        n_epoch=20,
        stage='3',
        accu_gradient_step=10)
oc.train_one_cycle()

