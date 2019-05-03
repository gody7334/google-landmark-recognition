import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torchvision.models import *

class BaseResNet(torch.nn.Module):
    def __init__(self, num_classes=100,
            resnet_type='resnet18'):
        super(BaseResNet, self).__init__()

        self.num_classes = num_classes
        self.resnet_type = resnet_type
        self.block_size = None

        self.init_resnet()

        # copy resnet layer
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.maxpool = self.model.maxpool

        self.layer0 = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,
                self.maxpool)
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.avgpool = self.model.avgpool

        # replace a new fc head
        self.replace_fc()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def init_resnet(self):
        if self.resnet_type=='resnet18':
            self.model = resnet18(pretrained=True)
            self.block_size=1
        elif self.resnet_type=='resnet34':
            self.model = resnet34(pretrained=True)
            self.block_size=1
        elif self.resnet_type=='resnet50':
            self.model = resnet50(pretrained=True)
            self.block_size=4

    def replace_fc(self):
        self.fc = nn.Linear(self.model.fc.in_features, num_classes)


