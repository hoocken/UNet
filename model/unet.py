import torch.nn as nn
import torch

from model.layers import Conv, DownConv, UpConv

class UNet(nn.Module):
    def __init__(self, num_classes=7, num_layers=6, max_channels=512, deep_supervised_layers=3):
        """
        Constructs a U-Net

        Parameters:
            num_classes: Number of output classes
            num_layers: Number of layers for U-Net (including deepest conv block)
            max_channels: Maximum number of channels (limit memory usage)
        """
        super(UNet, self).__init__()

        # Get channel numbers for each layer. Double at every layer if not exceeding max_channels
        channel_numbers = [1] + [min(32 * (2 ** i), max_channels) for i in range(num_layers)]

        self.down_convs = nn.ModuleList([ DownConv(channel_numbers[i], channel_numbers[i + 1]) for i in range(num_layers - 1) ])
        
        self.conv = Conv(channel_numbers[-2], channel_numbers[-1], 2)

        self.up_convs = nn.ModuleList([ UpConv(channel_numbers[i], channel_numbers[i - 1], channel_numbers[i - 1]) for i in range(num_layers, 1, -1)])
        self.deep_supervised_layers = deep_supervised_layers
        
        self.deep_supervision_heads = nn.ModuleList([ 
            nn.Sequential(
                nn.Conv2d(channel_numbers[i - 1], num_classes, kernel_size=1),
                nn.Sigmoid(),
            )
            for i in range(self.deep_supervised_layers + 1, 1, -1) # Deep supervise the top nth layers
        ])
            
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x, deep_supervised=False):
        temp = x
        skip = []
        deep = []

        for conv_layer in self.down_convs:
            temp, conv = conv_layer(temp)
            skip.append(conv)
        
        temp = self.conv(temp)

        deep_supervised_index = 0
        for i in range(len(self.up_convs)):
            conv = skip.pop()
            temp = self.up_convs[i](temp, conv)
            deep_supervised_index = i - len(self.up_convs) + self.deep_supervised_layers
            if deep_supervised and i >= len(self.up_convs) - self.deep_supervised_layers: # i is more than the up_convs_length - deep_supervised_layers
                
                deep.append(self.deep_supervision_heads[deep_supervised_index](temp))

        out = self.out_conv(temp)
        res = self.activation(out)

        if self.deep_supervision_heads:
            return res
        else:
            return res
