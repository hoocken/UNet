import torch.nn as nn
import torch

from model.layers import Conv, DownConv, UpConv

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.down_convs = nn.ModuleList([
            DownConv(1, 32), # 256
            DownConv(32, 64), # 128
            DownConv(64, 128), # 64
            DownConv(128, 256), # 32
            DownConv(256, 480), # 16
            DownConv(480, 480), # 8
        ])
        
        self.conv = Conv(480, 480, 2)

        self.up_convs =  nn.ModuleList([
            UpConv(480, 480, 480),
            UpConv(480, 480, 480),
            UpConv(480, 256, 256),
            UpConv(256, 128, 128),
            UpConv(128, 64, 64),
            UpConv(64, 32, 32),
        ])

        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        temp = x
        skip = []

        for conv_layer in self.down_convs:
            temp, conv = conv_layer(temp)
            skip.append(conv)
        
        temp = self.conv(temp)

        for conv_layer in self.up_convs:
            conv = skip.pop()
            temp = conv_layer(temp, conv)

        out = self.out_conv(temp)
        res = self.activation(out)
        return res
