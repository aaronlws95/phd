import torch

from src.networks.backbones import get_backbone
from src.networks.modules import *

class HPO_BBOX_AR_Net(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = get_backbone(cfg)
        num_actions = int(cfg['num_actions'])
        num_objects = int(cfg['num_objects'])
        anchors            = [float(i) for i in cfg["anchors"].split(',')]
        num_anchors        = len(anchors)//2
        self.conv_out = conv(1024, (5 + num_actions + num_objects)*num_anchors, 1, 1, 1, 1, 0, 0)

    def forward(self, x):
        # x: B, C, H, W
        out = self.backbone(x)
        out = self.conv_out(out) # B, (NA+NO)*5, H/32, W/32
        return out

    def get_spatial_attn_map(self, x):
        spatial_attn_map = self.backbone.get_spatial_attn_map(x)
        out = self.backbone(x)
        out = self.conv_out(out)
        spatial_attn_map.append([out])

        return spatial_attn_map