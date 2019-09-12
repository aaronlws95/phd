import torch
import time
import math
import time
import torch.nn as nn
import numpy    as np

from src.utils  import FPHA

class HPO_Hand_Object_Loss(torch.nn.Module):
    """ HPO loss calculation """
    def __init__(self, cfg):
        super().__init__()
        self.bs             = None
        self.W              = None
        self.H              = None
        self.D              = int(cfg['D'])
        self.hand_scale     = float(cfg["hand_scale"])
        self.no_hand_scale  = float(cfg["no_hand_scale"])
        self.hand_root      = int(cfg["hand_root"])
        self.obj_root       = int(cfg['obj_root'])
        self.sharp          = int(cfg["sharpness"])
        self.d_th           = int(cfg["d_th"])
        self.sil_thresh     = float(cfg["sil_thresh"])
        self.ref_depth      = int(cfg['ref_depth'])
        self.num_joints     = int(cfg['num_joints'])
        self.img_ref_width  = int(cfg['img_ref_width'])
        self.img_ref_height = int(cfg['img_ref_height'])

    def offset_to_uvd(self, x, root):
        """Split prediction into predicted uvd with grid offset and
        predicted conf
        Args:
            x                   : Predicted output (b, (63+1)*D, H, W)
        Out:
            pred_uvd_no_offset  : Predicted uvd without offset
                                  (b, 21, 3, H, W, D)
            pred_uvd            : Predicted uvd with grid offset
                                  (b, 21, 3, H, W, D)
            pred_conf           : Predicted confidence values (b, H, W, D)
        """
        FT      = torch.FloatTensor
        x       = x.permute(0, 1, 3, 4, 2)

        pred_uvd_no_offset  = x[:, :self.num_joints*3, :, :, :].view(self.bs, self.num_joints, 3,
                                                      self.H, self.W, self.D)
        pred_conf           = x[:, self.num_joints*3, :, :, :]
        pred_conf           = torch.sigmoid(pred_conf)

        yv, xv, zv  = torch.meshgrid([torch.arange(self.H),
                                      torch.arange(self.W),
                                      torch.arange(self.D)])
        grid_x      = xv.repeat((self.num_joints, 1, 1, 1)).type(FT).cuda()
        grid_y      = yv.repeat((self.num_joints, 1, 1, 1)).type(FT).cuda()
        grid_z      = zv.repeat((self.num_joints, 1, 1, 1)).type(FT).cuda()

        pred_uvd_no_offset[:, root, :, :, :, :] = \
            torch.sigmoid(pred_uvd_no_offset[:, root, :, :, :, :])

        pred_uvd = pred_uvd_no_offset.clone().detach()
        pred_uvd[:, :, 0, :, :, :] = \
            (pred_uvd[:, :, 0, :, :, :] + grid_x)/self.W
        pred_uvd[:, :, 1, :, :, :] = \
            (pred_uvd[:, :, 1, :, :, :] + grid_y)/self.H
        pred_uvd[:, :, 2, :, :, :] = \
            (pred_uvd[:, :, 2, :, :, :] + grid_z)/self.D

        return pred_uvd_no_offset, pred_uvd, pred_conf

    def calc_conf_grid(self, pred_uvd, uvd_gt):
        """
        Calculate true confidence values in a grid for confidence mask
        Args:
            pred_uvd    : Predicted uvd values with grid offset scaled
                          to (1, 1) (21, 3, H, W, D)
            uvd_gt      : Ground truth uvd repeated in grid (21, 3, H, W, D)
        Out:
            mean_conf   : Mean confidence in a grid (H, W, D)
        """
        dist                = pred_uvd - uvd_gt
        dist[:, 0, :, :, :] = dist[:, 0, :, :, :]*self.img_ref_width
        dist[:, 1, :, :, :] = dist[:, 1, :, :, :]*self.img_ref_height
        dist[:, 2, :, :, :] = dist[:, 2, :, :, :]*self.ref_depth

        eps         = 1e-8
        dist        = torch.sqrt(torch.sum((dist)**2, dim = 1) + eps)
        mask        = (dist < self.d_th).type(torch.FloatTensor)
        conf        = torch.exp(self.sharp*(1 - dist/self.d_th)) - 1
        conf0       = torch.exp(self.sharp*(1 - torch.zeros(conf.shape))) - 1
        conf        = conf/conf0.cuda()
        conf        = mask.cuda()*conf
        mean_conf   = torch.mean(conf, dim=0)
        return mean_conf

    def calc_conf(self, pred_uvd, uvd_gt):
        """
        Calculate true specific target confidence values
        Args:
            pred_uvd    : Predicted uvd at target location (21, 3)
            uvd_gt      : Ground truth uvd (21, 3)
        Out:
            mean_conf   : Mean confidence (1)
        """
        eps         = 1e-8
        dist        = pred_uvd - uvd_gt
        dist[:, 0]  = dist[:, 0]*self.img_ref_width
        dist[:, 1]  = dist[:, 1]*self.img_ref_height
        dist[:, 2]  = dist[:, 2]*self.ref_depth

        dist        = torch.sqrt(torch.sum((dist)**2, dim = 1) + eps)
        mask        = (dist < self.d_th).type(torch.FloatTensor)
        conf        = torch.exp(self.sharp*(1 - dist/self.d_th)) - 1
        conf0       = torch.exp(self.sharp*(1 - torch.zeros(conf.shape))) - 1
        conf        = conf/conf0
        conf        = mask*conf
        mean_conf   = torch.mean(conf, dim=0)
        return mean_conf

    def get_conf_mask(self, pred_uvd, uvd_gt):
        """ Get mask to weigh confidence values
        Args:
            pred_uvd    : Predicted uvd values with grid offset scaled
                          to (1, 1) (21, 3, H, W, D)
            uvd_gt      : Ground truth uvd (21, 3)
        Out:
            conf_mask   : All set to no_hand_scale except those with confidence
                          more than sil_thresh which are set to 0. Later
                          will set target location to hand_scale (b, H, W, D)
        """
        conf_mask = torch.ones(self.bs,
                               self.H,
                               self.W,
                               self.D)*self.no_hand_scale

        for batch in range(self.bs):
            cur_pred_uvd    = pred_uvd[batch]
            cur_uvd_gt      = uvd_gt[batch].repeat(self.H,
                                                   self.W,
                                                   self.D,
                                                   1, 1)
            cur_uvd_gt      = cur_uvd_gt.permute(3, 4, 0, 1, 2)
            cur_conf        = self.calc_conf_grid(cur_pred_uvd, cur_uvd_gt)
            conf_mask[batch][cur_conf > self.sil_thresh] = 0

        return conf_mask

    def get_target(self, uvd_gt, pred_uvd, root):
        """
        Get target boxes and masks
        Args:
            pred_uvd    : Predicted uvd values with grid offset scaled
                          to (1, 1) (21, 3, H, W, D)
            uvd_gt      : Ground truth uvd (21, 3)
        Out:
            Target location refers to the ground truth x, y, z
            conf_mask       : From get_conf_mask. Set target locations to
                              object_scale (b, na, H, W)
            target_conf     : All 0 except at target location it is the conf
                              between the target and predicted bbox
                              (b, H, W, D)
            target_uvd      : Target uvd keypoints. Set all to 0 except at
                              target location where it is scaled to compare
                              with pred_uvd_no_offset (b, 21, 3, H, W, D)
            coord_mask      : All 0 except at target locations it is 1
                              (b, H, W, D)
        """
        FT          = torch.FloatTensor
        target_uvd  = torch.zeros(self.bs, self.num_joints, 3,
                                  self.H, self.W, self.D).type(FT)
        coord_mask  = torch.zeros(self.bs, self.H, self.W, self.D).type(FT)
        target_conf = torch.zeros(self.bs, self.H, self.W, self.D).type(FT)
        conf_mask   = self.get_conf_mask(pred_uvd, uvd_gt)

        pred_uvd    = pred_uvd.cpu()
        uvd_gt      = uvd_gt.cpu()
        for batch in range(self.bs):
            cur_uvd_gt = uvd_gt[batch]
            # get cell where hand root is present
            gi0 = int(cur_uvd_gt[root, 0]*self.W)
            gj0 = int(cur_uvd_gt[root, 1]*self.H)
            gk0 = int(cur_uvd_gt[root, 2]*self.D)
            if gi0 >= self.W:
                gi0 = self.W - 1
            if gj0 >= self.H:
                gj0 = self.H - 1
            if gk0 >= self.D:
                gk0 = self.D - 1

            target_uvd[batch, :, 0, gj0, gi0, gk0]  = \
                cur_uvd_gt[:, 0]*self.W - gi0
            target_uvd[batch, :, 1, gj0, gi0, gk0]  = \
                cur_uvd_gt[:, 1]*self.H - gj0
            target_uvd[batch, :, 2, gj0, gi0, gk0]  = \
                cur_uvd_gt[:, 2]*self.D - gk0
            coord_mask[batch, gj0, gi0, gk0]        = 1

            cur_pred_uvd = pred_uvd[batch, :, :, gj0, gi0, gk0]
            target_conf[batch, gj0, gi0, gk0]   = \
                self.calc_conf(cur_pred_uvd, cur_uvd_gt)
            conf_mask[batch, gj0, gi0, gk0] = self.hand_scale

        target_uvd  = target_uvd.cuda()
        target_conf = target_conf.cuda()
        coord_mask  = coord_mask.cuda()
        conf_mask   = conf_mask.cuda()
        conf_mask   = conf_mask.sqrt()

        return target_uvd, target_conf, coord_mask, conf_mask

    def forward(self, pred, uvd_gt, uvd_gt_obj):
        """
        Loss calculation and processing
        Args:
            pred    : (b, (63+1)*D, H, W)
            uvd_gt  : (b, 21, 3)
        Out:
            loss    : Total loss and its components
        """

        self.bs = pred.shape[0]
        self.W  = pred.shape[2]
        self.H  = pred.shape[3]
        pred    = pred.view(self.bs, 2, self.num_joints*3 + 1, self.D, self.H, self.W)
        pred_hand = pred[:, 0, :]
        pred_obj = pred[:, 1, :]

        pred_uvd_no_offset, pred_uvd, pred_conf = self.offset_to_uvd(pred_hand, self.hand_root)
        targets     = self.get_target(uvd_gt, pred_uvd, self.hand_root)
        target_uvd, target_conf, coord_mask, conf_mask = targets
        coord_mask  = coord_mask.repeat(self.num_joints, 1, 1, 1, 1)
        coord_mask  = coord_mask.permute(1, 0, 2, 3, 4)

        pred_uvd_no_offset_obj, pred_uvd_obj, pred_conf_obj = self.offset_to_uvd(pred_obj, self.obj_root)
        targets_obj = self.get_target(uvd_gt_obj, pred_uvd_obj, self.obj_root)
        target_uvd_obj, target_conf_obj, coord_mask_obj, conf_mask_obj = targets_obj
        coord_mask_obj  = coord_mask_obj.repeat(self.num_joints, 1, 1, 1, 1)
        coord_mask_obj  = coord_mask_obj.permute(1, 0, 2, 3, 4)

        mseloss     = nn.MSELoss(reduction="sum")
        loss_u      = mseloss(coord_mask*pred_uvd_no_offset[:, :, 0, :, :, :],
                              coord_mask*target_uvd[:, :, 0, :, :, :])/2.0
        loss_v      = mseloss(coord_mask*pred_uvd_no_offset[:, :, 1, :, :, :],
                              coord_mask*target_uvd[:, :, 1, :, :, :])/2.0
        loss_d      = mseloss(coord_mask*pred_uvd_no_offset[:, :, 2, :, :, :],
                              coord_mask*target_uvd[:, :, 2, :, :, :])/2.0
        loss_conf   = mseloss(conf_mask*pred_conf, conf_mask*target_conf)/2.0

        loss_u_obj      = mseloss(coord_mask_obj*pred_uvd_no_offset_obj[:, :, 0, :, :, :],
                              coord_mask_obj*target_uvd_obj[:, :, 0, :, :, :])/2.0
        loss_v_obj      = mseloss(coord_mask_obj*pred_uvd_no_offset_obj[:, :, 1, :, :, :],
                              coord_mask_obj*target_uvd_obj[:, :, 1, :, :, :])/2.0
        loss_d_obj      = mseloss(coord_mask_obj*pred_uvd_no_offset_obj[:, :, 2, :, :, :],
                              coord_mask_obj*target_uvd_obj[:, :, 2, :, :, :])/2.0
        loss_conf_obj   = mseloss(conf_mask_obj*pred_conf_obj, conf_mask_obj*target_conf_obj)/2.0

        hand_loss  = loss_u + loss_v + loss_d + loss_conf
        obj_loss   = loss_u_obj + loss_v_obj + loss_d_obj + loss_conf_obj

        total_loss = hand_loss + obj_loss

        return total_loss, hand_loss, obj_loss
