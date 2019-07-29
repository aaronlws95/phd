import torch
import torch.nn.functional as F

from src.networks.backbones import get_backbone
from src.networks.modules import *

class AR_Attn_Net(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_segments = int(cfg['num_segments'])
        self.backbone = get_backbone(cfg)
        if cfg['dropout']:
            self.dropout = torch.nn.Dropout2d(p=float(cfg['dropout']))
        else:
            self.dropout = None
        self.mask_conv = torch.nn.Sequential(
            conv(1024, 1024, 3, 1, 1, 0, 1, 'relu'),
            conv(1024, 512, 3, 1, 1, 0, 1, 'relu'),
            conv(512, 1, 3, 1, 1, 0, 1, 0),
            torch.nn.Sigmoid() 
        )
        hidden_size = int(cfg['hidden_size'])
        mask_size = int(cfg['img_rsz'])//self.backbone.stride
        self.conv_lstm = ConvLSTM(input_size=(mask_size, mask_size),
                                  input_dim=1024,
                                  hidden_dim=[hidden_size],
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=True)
        self.fc_attention = torch.nn.Linear(hidden_size, 1)
        num_actions = int(cfg['num_actions'])
        num_objects = int(cfg['num_objects'])
        self.fc_action = torch.nn.Linear(hidden_size, num_actions)
        self.fc_object = torch.nn.Linear(hidden_size, num_objects)

    def forward(self, x):
        # x: B, S, C, H, W
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        out = self.backbone(x) # B*S, C, H/32, W/32

        if self.dropout:
            out = self.dropout(out)
        
        mask = self.mask_conv(out) # B*S, 1, H/32, W/32
        
        BS, C, H, W = out.shape
        out = out.view(-1, self.num_segments, C, H, W)
        mask = mask.view(-1, self.num_segments, 1, H, W)
        output, hidden = self.conv_lstm(mask*out)
        
        output = output[0]
        output = torch.mean(output, dim=4)
        output = torch.mean(output, dim=3)
        attn_weight = self.fc_attention(output).view(-1, self.num_segments)
        attn_weight = F.softmax(attn_weight, dim=1)
        weighted_output = torch.sum(output*attn_weight.unsqueeze(dim=2), dim =1)
        out_action = self.fc_action(weighted_output)
        out_object = self.fc_object(weighted_output)
        
        return out_action, out_object, attn_weight, mask