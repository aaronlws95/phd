import torch
import time
import math
import time
import torch.nn as nn
import numpy    as np

from src.utils  import FPHA

class EK_Action_NounLoss(torch.nn.Module):
    """ HPO loss calculation """
    def __init__(self, cfg):
        super().__init__()
        self.W              = None
        self.H              = None
        self.D              = None
        self.time           = cfg['time']
        self.debug          = cfg['debug']
        self.num_action     = int(cfg['num_action'])
        self.num_noun       = int(cfg['num_noun'])

    def forward(self, pred, action_gt, noun_gt):
        """
        Loss calculation and processing
        Args:
            pred    : (b, (63+1)*D, H, W)
            uvd_gt  : (b, 21, 3)
        Out:
            loss    : Total loss and its components
        """
        FT      = torch.FloatTensor
        self.bs = pred.shape[0]
        self.W  = pred.shape[2]
        self.H  = pred.shape[3]
        self.D  = 5
        
        pred        = pred.view(self.bs, 64 + self.num_action + self.num_noun, self.D, self.H, self.W)
        pred        = pred.permute(0, 1, 3, 4, 2)
        pred_conf   = torch.sigmoid(pred[:, 63, :, :, :])
        pred_conf   = pred_conf.contiguous().view(self.bs, -1)
        top_idx     = torch.topk(pred_conf, 1)[1]
        pred_action = pred[:, 64:(64 + self.num_action), :, :, :]
        pred_action = pred_action.contiguous().view(self.bs, self.num_action, -1)
        pred_noun   = pred[:, (64 + self.num_action):, :, :, :]
        pred_noun   = pred_noun.contiguous().view(self.bs, self.num_noun, -1)
        
        chosen_pred_act = torch.zeros(self.bs, self.num_action).cuda()
        chosen_pred_noun = torch.zeros(self.bs, self.num_noun).cuda()
        for i, idx in enumerate(top_idx):
            chosen_pred_act[i, :] = pred_action[i, :, idx].squeeze()
            chosen_pred_noun[i, :] = pred_noun[i, :, idx].squeeze()
        
        celoss      = nn.CrossEntropyLoss(reduction='sum')
        loss_action = celoss(chosen_pred_act, action_gt)/2.0
        loss_noun   = celoss(chosen_pred_noun, noun_gt)/2.0
        total_loss  = loss_action + loss_noun

        return total_loss, loss_action, loss_noun
