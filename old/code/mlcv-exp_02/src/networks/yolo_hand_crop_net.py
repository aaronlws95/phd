import torch

from src.networks.backbones import get_backbone
from src.networks.modules import *

class YOLO_Hand_Crop_Net(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = get_backbone(cfg)
        out_num = 63
        self.conv_out = conv(1024, out_num, 1, 1, 1, 1, 0, 0)

        self.lin_out = self._dense_block(out_num)

    def _dense_block(self, out_num):
        in_features = out_num*13*13
        d0 = torch.nn.Linear(in_features, in_features//2)
        relu0 = torch.nn.ReLU()
        d1  = torch.nn.Linear(in_features//2, in_features//4)
        relu1 = torch.nn.ReLU()
        d2 = torch.nn.Linear(in_features//4, 63)
        return torch.nn.Sequential(d0, relu0, d1, relu1, d2)

    def forward(self, x):
        # x: B, C, H, W
        out = self.backbone(x)
        out = self.conv_out(out) # B, 315, H/32, W/32
        out = out.view(out.shape[0], -1)
        out = self.lin_out(out)
        return out

    def get_spatial_attn_map(self, x):
        spatial_attn_map = self.backbone.get_spatial_attn_map(x)
        out = self.backbone(x)
        out = self.conv_out(out)
        spatial_attn_map.append([out])

        return spatial_attn_map