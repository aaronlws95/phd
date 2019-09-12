import torch
from src.backbones.base_backbone import Base_Backbone

def _get_activation(activation):
    if activation == 'leaky':
        return torch.nn.LeakyReLU(0.1)
    elif activation == 'relu':
        return torch.nn.ReLU()

def conv(in_c, out_c, k_size, stride, pad, bias, bn, activation):
    modules = torch.nn.Sequential()
    pad = (k_size - 1)//2 if pad else 0
    modules.add_module('conv', torch.nn.Conv2d(in_channels=in_c,
                                               out_channels=out_c,
                                               kernel_size=k_size,
                                               stride=stride,
                                               padding=pad,
                                               bias=bias))
    if bn:
        modules.add_module('bn', torch.nn.BatchNorm2d(out_c))
    if activation:
        modules.add_module(activation, _get_activation(activation))
    return modules

class Reorg(torch.nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        s = self.stride
        assert(x.dim() == 4)
        B, C, H, W = x.size()
        assert(H%s == 0)
        assert(W%s == 0)
        x = x.view(B, C, H//s, s, W//s, s).transpose(3,4).contiguous()
        x = x.view(B, C, (H//s)*(W//s), s*s).transpose(2,3).contiguous()
        x = x.view(B, C, s*s, H//s, W//s).transpose(1,2).contiguous()
        x = x.view(B, C*s*s, H//s, W//s)
        return x

class YOLOV2(Base_Backbone):
    def __init__(self, cfg):
        super().__init__()

        self.in_channels    = 3
        self.out_filters    = 1024
        self.stride         = 32

        # conv: in_c, out_c, k_size, stride, pad, bias, bn, activation
        self.block0 = torch.nn.Sequential(
            conv(self.in_channels, 32, 3, 1, 1, 0, 1, 'leaky'),
            torch.nn.MaxPool2d(2),
            conv(32, 64, 3, 1, 1, 0, 1, 'leaky'),
            torch.nn.MaxPool2d(2),
            conv(64, 128, 3, 1, 1, 0, 1, 'leaky'),
            conv(128, 64, 1, 1, 1, 0, 1, 'leaky'),
            conv(64, 128, 3, 1, 1, 0, 1, 'leaky'),
            torch.nn.MaxPool2d(2),
            conv(128, 256, 3, 1, 1, 0, 1, 'leaky'),
            conv(256, 128, 1, 1, 1, 0, 1, 'leaky'),
            conv(128, 256, 3, 1, 1, 0, 1, 'leaky'),
            torch.nn.MaxPool2d(2),
            conv(256, 512, 3, 1, 1, 0, 1, 'leaky'),
            conv(512, 256, 1, 1, 1, 0, 1, 'leaky'),
            conv(256, 512, 3, 1, 1, 0, 1, 'leaky'),
            conv(512, 256, 1, 1, 1, 0, 1, 'leaky'),
            conv(256, 512, 3, 1, 1, 0, 1, 'leaky'),
        )

        self.block1_1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            conv(512, 1024, 3, 1, 1, 0, 1, 'leaky'),
            conv(1024, 512, 1, 1, 1, 0, 1, 'leaky'),
            conv(512, 1024, 3, 1, 1, 0, 1, 'leaky'),
            conv(1024, 512, 1, 1, 1, 0, 1, 'leaky'),
            conv(512, 1024, 3, 1, 1, 0, 1, 'leaky'),
            conv(1024, 1024, 3, 1, 1, 0, 1, 'leaky'),
            conv(1024, 1024, 3, 1, 1, 0, 1, 'leaky'),
        )

        self.block1_2 = torch.nn.Sequential(
            conv(512, 64, 1, 1, 1, 0, 1, 'leaky'),
            Reorg(2) # (B, 2048, H, W)
        )

        self.block2 = conv(1280, self.out_filters, 3, 1, 1, 0, 1, 'leaky')

    def forward(self, x):
        out0    = self.block0(x)
        out1_1  = self.block1_1(out0)
        out1_2  = self.block1_2(out0)
        out1    = torch.cat([out1_1, out1_2], 1)
        out2    = self.block2(out1)
        return out2