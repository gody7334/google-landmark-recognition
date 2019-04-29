import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torchvision.models import *

class cnn_model(torch.nn.Module):
    def __init__(self, num_class=100,
            fc_dim= 256,
            cnn_type='resnet18'):
        super(cnn_model, self).__init__()

        self.model = resnet50(pretrained=True)

        # modify model head
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc = torch.nn.Sequential(
                     torch.nn.Linear(512*4, fc_dim),
                     torch.nn.ReLU(inplace=True),
                     torch.nn.Dropout(0.2),
                     torch.nn.Linear(fc_dim, num_class))

    def forward(self, x):
        x = self.model(x)
        return x

