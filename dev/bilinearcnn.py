import torch
import torch.nn as nn
from torch.nn import Module
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torchvision.models import *
from utils.model import BaseResNet
from dev.compactbilinearpooling import CompactBilinearPooling

class Resnet_Feat(BaseResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def replace_fc(self):
        return


class BCNN(Module):
    def __init__(self, num_classes=100,
            bi_vector_dim= 1024,
            cnn_type='resnet18'):
        super(cnn_model, self).__init__()

        self.feat1 = Resnet_Feat(resnet_type='resnet34')
        self.feat2 = Resnet_Feat(resnet_type='resnet34')
        self.com_bi_pool = CompactBilinearPooling(256*16*16, 256*16*16, bi_vector_dim)
        self.fc = nn.Linear(bi_vector_dim, num_classes)

    def forward(self, x):
        f1 = self.feat1(x)
        f2 = self.feat2(x)
        bi_vector = self.combi_pool(f1, f2)
        fc = self.fc(bi_vector)
        return fc

