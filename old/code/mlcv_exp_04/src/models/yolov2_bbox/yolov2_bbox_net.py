import torch

from src.backbones import get_backbone

class YOLOV2_Bbox_Net(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = get_backbone(cfg)
        out_channels = int(cfg['num_anchors'])*5
        self.conv_out = torch.nn.Conv2d(in_channels=self.backbone.out_filters,
                                        out_channels=out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=1)

    def forward(self, x):
        # x: B, C, H, W
        out = self.backbone(x)
        out = self.conv_out(out) # B, 25, H/32, W/32
        return out