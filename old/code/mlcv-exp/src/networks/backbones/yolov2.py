import torch

from src.networks.modules import *

class YOLOV2(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # conv: in_c, out_c, k_size, stride, pad, bias, bn, activation
        self.block0 = torch.nn.Sequential(
            conv(3, 32, 3, 1, 1, 0, 1, 'leaky'),
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
        
        self.block2 = conv(1280, 1024, 3, 1, 1, 0, 1, 'leaky')

        self.stride = 32
        
    def forward(self, x):
        out0    = self.block0(x)
        out1_1  = self.block1_1(out0)
        out1_2  = self.block1_2(out0)
        out1    = torch.cat([out1_1, out1_2], 1)
        out2    = self.block2(out1)
        return out2
    
    def get_spatial_attn_map(self, x):
        spatial_attn_map            = []
        spatial_attn_map_block0     = []
        spatial_attn_map_block1_1   = []
        spatial_attn_map_block1_2   = []

        for layer in self.block0:
            x = layer(x)
            spatial_attn_map_block0.append(x) 
        spatial_attn_map.append(spatial_attn_map_block0)

        out1_1 = x.clone()
        for layer in self.block1_1:
            out1_1 = layer(out1_1)
            spatial_attn_map_block1_1.append(out1_1)
        spatial_attn_map.append(spatial_attn_map_block1_1)
        
        out1_2 = x.clone()
        for layer in self.block1_2:
            out1_2 = layer(out1_2)
            spatial_attn_map_block1_2.append(out1_2)
        spatial_attn_map.append(spatial_attn_map_block1_2)
        
        out1 = torch.cat([out1_1, out1_2], 1)
        spatial_attn_map.append([out1])
        out2 = self.block2(out1)
        spatial_attn_map.append([out2])
        
        return spatial_attn_map