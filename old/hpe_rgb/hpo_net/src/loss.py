import torch
import torch.nn as nn
import numpy as np

class HPOLoss(torch.nn.Module):

    def __init__(self):
        super(HPOLoss,self).__init__()

    def get_true_conf(self, pred_uvd, true_uvd):

        def l2dist(true, pred):
            return torch.mean(torch.sqrt(torch.sum((true-pred)**2, dim=-1) + 1e-8), dim=-1)

        alpha = 2
        d_th = 75
        true_uvd_expand = true_uvd.unsqueeze(1).expand(pred_uvd.shape[0], 845, 21, 3)
        D_T = l2dist(pred_uvd, true_uvd_expand)
        conf = torch.exp(alpha*(1-(D_T/d_th)))
        conf_thresh = conf.clone()
        conf_thresh[D_T >= d_th] = 0
        return conf_thresh

    def uvd_loss_sum(self, true, pred):
        return torch.mean(torch.sum(torch.mean(torch.sum((true-pred)**2, dim=-1), dim=-1), dim=-1))

    def forward(self, pred_uvd_no_offset, uvd_gt_no_offset, pred_uvd, pred_conf, true_uvd, hand_cell_idx):

        true_conf = self.get_true_conf(pred_uvd, true_uvd)
        conf_diff_squared = (pred_conf - true_conf)**2
        conf_diff_squared[hand_cell_idx == 1] = conf_diff_squared[hand_cell_idx == 1]*5
        conf_diff_squared[hand_cell_idx != 1] = conf_diff_squared[hand_cell_idx != 1]*0.1

        uvd_gt_no_offset_expand = uvd_gt_no_offset.unsqueeze(1).expand(pred_uvd.shape[0], 845, 21, 3)

        loss_uvd = self.uvd_loss_sum(uvd_gt_no_offset_expand, pred_uvd_no_offset)
        total_loss = loss_uvd + torch.mean(torch.sum(conf_diff_squared, dim=-1))
        return total_loss
