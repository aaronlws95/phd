import torch
from src.backbones.base_backbone import Base_Backbone
import torchvision.models as models
from pathlib import Path

from src import ROOT

class ResNet50(Base_Backbone):
    def __init__(self, cfg):
        super().__init__()
        resnet50 = models.resnet50(pretrained=False)
        resnet50.load_state_dict(torch.load(Path(ROOT)/cfg['pytorch_pretrain']))
        resnet50 = list(resnet50.children())[:-2] # remove linear
        self.resnet50 = torch.nn.Sequential(*resnet50)
        self.in_channels    = 3
        self.out_filters    = 2048
        self.stride         = 32

    def forward(self, x):
        out = self.resnet50(x)
        return out