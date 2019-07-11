import torch
import time
import math
import time
import torch.nn as nn
import numpy    as np

from src.utils  import FPHA

class HPO_AR_Loss(torch.nn.Module):
    """ HPO loss calculation """
    def __init__(self, cfg):
        super().__init__()
        self.num_action     = int(cfg['num_actions'])
        self.num_obj       = int(cfg['num_objects'])
        self.consensus      = cfg['consensus']
        
    def forward(self, pred, action_gt, obj_gt):
        bs = pred.shape[0]
        H  = pred.shape[2]
        W  = pred.shape[3]
        D  = 5

        pred        = pred.view(bs, self.num_action + self.num_obj, D, H, W)
        pred        = pred.permute(0, 1, 3, 4, 2)
        pred_action = pred[:, :self.num_action, :, :, :]
        pred_action = pred_action.contiguous().view(bs, self.num_action, -1)
        pred_obj   = pred[:,  self.num_action:, :, :, :]
        pred_obj   = pred_obj.contiguous().view(bs, self.num_obj, -1)

        # consensus 
        if self.consensus == 'avg':
            pred_obj   = torch.mean(pred_obj, dim=-1)
            pred_action = torch.mean(pred_action, dim=-1)
        elif self.consensus == 'max':
            pred_obj   = torch.max(pred_obj, dim=-1)
            pred_action = torch.max(pred_action, dim=-1)

        celoss      = nn.CrossEntropyLoss(reduction='sum')
        loss_action = celoss(pred_action, action_gt)/2.0
        loss_obj   = celoss(pred_obj, obj_gt)/2.0
        total_loss  = loss_action + loss_obj

        return total_loss, loss_action, loss_obj
