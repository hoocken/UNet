import torch.nn as nn
import torch

from model.layers import Conv, DownConv, UpConv

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.down_convs = nn.ModuleList([
            DownConv(1, 64),
            DownConv(64, 128),
            DownConv(128, 256),
            DownConv(256, 512),
        ])
        
        self.conv = Conv(512, 1024, 2)

        self.up_convs =  nn.ModuleList([
            UpConv(1024, 512),
            UpConv(512, 256),
            UpConv(256, 128),
            UpConv(128, 64),
        ])

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
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
