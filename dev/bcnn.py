import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.modules.batchnorm import *
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torchvision.models import *
from utils.model import BaseResNet
from dev.compactbilinearpooling import CompactBilinearPooling

gpu0 = torch.device("cuda:0")
gpu1 = torch.device("cuda:1")

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        # out = self.gamma * torch.sigmoid(out) * x
        return out

class Resnet_Feat(BaseResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = Self_Attn(256*self.block_size,'relu')

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.attn(x)
        return x

    def replace_fc(self):
        return

class BCNN(Module):
    def __init__(self, num_classes=100,
            bi_vector_dim= 1024,
            cnn_type1='resnet18', cnn_type2='resnet18'):
        super(BCNN, self).__init__()

        self.feat1 = Resnet_Feat(resnet_type=cnn_type1)
        self.feat2 = Resnet_Feat(resnet_type=cnn_type2)
        self.com_bi_pool = CompactBilinearPooling(256*self.feat1.block_size*16*16,
                256*self.feat2.block_size*16*16,
                bi_vector_dim)
        self.bi_vector_bn = nn.Sequential(
                nn.BatchNorm1d(bi_vector_dim),
                nn.Dropout(0.5),
                nn.Linear(bi_vector_dim, 2048),
                nn.ReLU(),
                BatchNorm1d(2048),
            )
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        f1 = self.feat1(x)
        f2 = self.feat2(x)
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        bi_vector = self.com_bi_pool(f1, f2)
        bi_vector = self.bi_vector_bn(bi_vector)
        fc = self.fc(bi_vector)
        return fc, bi_vector

class BCNN_HP(BCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = self.fc.half()

    def forward(self, x):
        f1 = self.feat1(x)
        f2 = self.feat2(x)
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        bi_vector = self.com_bi_pool(f1, f2)
        bi_vector = self.bi_vector_bn(bi_vector)
        fc = self.fc(bi_vector.half()).float()
        return fc, bi_vector


class BCNN_MP(Module):
    def __init__(self, num_classes=100,
            bi_vector_dim= 1024,
            cnn_type1='resnet18', cnn_type2='resnet18'):
        super(BCNN_MP, self).__init__()

        self.feat1 = Resnet_Feat(resnet_type=cnn_type1).to(gpu0)
        self.feat2 = Resnet_Feat(resnet_type=cnn_type2).to(gpu0)
        self.com_bi_pool = CompactBilinearPooling(256*self.feat1.block_size*16*16,
                256*self.feat2.block_size*16*16,
                bi_vector_dim).to(gpu0)
        self.fc = nn.Linear(bi_vector_dim, num_classes).to(gpu1)

    def forward(self, x):
        x = x.to(gpu0)
        f1 = self.feat1(x)
        f2 = self.feat2(x)
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        bi_vector = self.com_bi_pool(f1, f2)
        bi_vector=bi_vector.to(gpu1)
        fc = self.fc(bi_vector)
        return fc.to(gpu0)


class BCNN_CP(BCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def f1_check_point(self):
        def forward(*x):
            f1 = self.feat1(x[0])
            return f1
        return forward

    def f2_check_point(self):
        def forward(*x):
            f2 = self.feat2(x[0])
            return f2
        return forward

    def bi_check_point(self):
        def forward(*x):
            bi_vector = self.com_bi_pool(x[0], x[1])
            bi_vector = self.bi_vector_bn(bi_vector)
            return bi_vector
        return forward

    def check_point(self):
        def custom_forward(*x):
            f1 = self.feat1(x[0])
            f2 = self.feat2(x[0])
            f1 = f1.view(f1.size(0), -1)
            f2 = f2.view(f2.size(0), -1)
            bi_vector = self.com_bi_pool(f1, f2)
            return bi_vector
        return custom_forward

    def forward(self, x):
        # checkpoint bug... need to set input require gradeint
        # then checkpoint will recompute the gradeint fore you...
        x.requires_grad=True
        f1 = checkpoint.checkpoint(self.f1_check_point(), x)
        f2 = checkpoint.checkpoint(self.f2_check_point(), x)
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        bi_vector = checkpoint.checkpoint(self.bi_check_point(), f1, f2)
        fc = self.fc(bi_vector)
        return fc

class BCNN_CP_HP(BCNN_CP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = self.fc.half()

    def forward(self, x):
        # checkpoint bug... need to set input require gradeint
        # then checkpoint will recompute the gradeint fore you...
        x.requires_grad=True
        f1 = checkpoint.checkpoint(self.f1_check_point(), x)
        f2 = checkpoint.checkpoint(self.f2_check_point(), x)
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        bi_vector = checkpoint.checkpoint(self.bi_check_point(), f1, f2)
        fc = self.fc(bi_vector.half()).float()
        return fc, bi_vector
