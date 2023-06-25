from functools import partial
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models,transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import os
import matplotlib.pyplot as plt 
from torch.autograd import Function
from collections import OrderedDict
import torch.nn as nn
import math

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
    
    
class MaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv2d = torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        
        self.encoder_res_block_1 = ResnetBlock(in_channels=64, out_channels=128, conv_shortcut=True, dropout=0.1)
        self.down_sample_1 = Downsample(in_channels=128, with_conv=True)
        self.encoder_res_block_2 = ResnetBlock(in_channels=128, out_channels=256, conv_shortcut=True, dropout=0.1)
        self.down_sample_2 = Downsample(in_channels=256, with_conv=True)
        
        self.res_blocks = nn.ModuleList([ResnetBlock(in_channels=256, out_channels=256, conv_shortcut=True, dropout=0.1) for _ in range(9)])
        
        self.up_sample_1 = Upsample(in_channels=256, with_conv=True)
        self.decoder_res_block_1 = ResnetBlock(in_channels=256, out_channels=128, conv_shortcut=True, dropout=0.1)
        self.up_sample_2 = Upsample(in_channels=128, with_conv=True)
        self.decoder_res_block_2 = ResnetBlock(in_channels=128, out_channels=64, conv_shortcut=True, dropout=0.1)
        
        self.decoder_conv2d = torch.nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.activation = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.encoder_conv2d(x)
        x = self.encoder_res_block_1(x)
        x = self.down_sample_1(x)
        x = self.encoder_res_block_2(x)
        x = self.down_sample_2(x)
        
        for i in range(9):
            x = self.res_blocks[i](x)
        
        x = self.up_sample_1(x)
        x = self.decoder_res_block_1(x)
        x = self.up_sample_2(x)
        x = self.decoder_res_block_2(x)
        x = self.decoder_conv2d(x)
        x = self.activation(x)  
        return x

# x = torch.rand(1, 3, 256, 256)
# net = MaskNet()
# x = net(x)
# print(x.shape)