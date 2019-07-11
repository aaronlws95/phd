import torch

from src.networks.backbones import get_backbone
from src.networks.modules import *

class YOLOV2_Bbox_Net(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = get_backbone(cfg)
        self.conv_out = conv(1024, 25, 1, 1, 1, 1, 0, 0)

    def forward(self, x):
        # x: B, C, H, W
        out = self.backbone(x)
        out = self.conv_out(out) # B, 25, H/32, W/32
        return out
    
    def get_spatial_attn_map(self, x):
        spatial_attn_map = self.backbone.get_spatial_attn_map(x)
        out = self.backbone(x)
        out = self.conv_out(out)
        spatial_attn_map.append([out])
        
        return spatial_attn_map