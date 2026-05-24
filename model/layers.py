import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers):
        super(Conv, self).__init__()
        conv_layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'), # nnUNet uses same padding
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]

        conv_layers += [
            x for _ in range(num_layers - 1)
            for x in [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
        ]

        self.conv = nn.Sequential(*conv_layers)
        
    def forward(self, x):
        return self.conv(x)
    
class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self,
                 in_channels,
                 bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv = nn.Conv2d(in_channels, in_channels // 2, 2, padding='same')
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)

    def forward(self, x):
        res = self.up(x)
        if self.conv:
            res = self.conv(res)

        return res
    
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.convs = Conv(in_channels, out_channels, 2)
        self.down = Down()

    def forward(self, x):
        """
        Convolves x, then pools it down to a lower resolution.

        Parameters:
            x: Shape of (N, C, H, W)
        
        Returns:
            down: Shape of (N, C, H / 2, W / 2)
            conv: Result of convolved x
        """
        conv = self.convs(x)
        return self.down(conv), conv
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpConv, self).__init__()
        self.convs = Conv(in_channels, out_channels, 2)
        self.up = Up(in_channels, bilinear=bilinear)

    def forward(self, x, skip):
        """
        Up convolves x, concatenates skip, then applies convolution layers.
        Skip is cropped to have the same size as x.

        Parameters:
            x: Shape of (N, C, H, W)
            skip: Shape of (N, C, H', W')
        """
        up = self.up(x)
        _, _, H, W = up.shape
        pad_h = (skip.shape[2] - H) // 2
        pad_w = (skip.shape[3] - W) // 2
        crop = skip[:, :, pad_h:pad_h + H, pad_w:pad_w + W]

        assert crop.shape == up.shape
        concat = torch.concatenate([crop, up], axis=1) # Concat the channels
        return self.convs(concat)