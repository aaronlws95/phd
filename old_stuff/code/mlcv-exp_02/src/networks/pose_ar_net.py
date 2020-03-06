import torch

from src.networks.backbones import get_backbone
from src.networks.modules import *

class Pose_AR_Net(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_segments = int(cfg['num_segments'])
        
        self.block0 = torch.nn.Sequential(
            conv(3, 16, 3, 1, 1, 0, 1, 'leaky'),
            conv(16, 32, 3, 1, 1, 0, 1, 'leaky'),
            conv(32, 64, 3, 1, 1, 0, 1, 'leaky'),
            conv(64, 128, 3, 1, 1, 0, 1, 'leaky'),
        )
        
        num_actions = int(cfg['num_actions'])
        self.lin_out = torch.nn.Linear(128*self.num_segments*21, num_actions)

    def forward(self, x):
        out = self.block0(x)
        out = out.view(out.shape[0], -1)
        out = self.lin_out(out)
        return out