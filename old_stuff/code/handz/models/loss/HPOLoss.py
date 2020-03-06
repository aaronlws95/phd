import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import FPHA_utils as FPHA

class HPOLoss(torch.nn.Module):
    def __init__(self, conf, device):
        super(HPOLoss,self).__init__()
        self.W = None
        self.H = None
        self.D = None
        self.grid_size = None        
        self.hand_scale = conf["hand_scale"]
        self.no_hand_scale = conf["no_hand_scale"]
        self.hand_root = conf["hand_root"]
        self.device = device
        self.sharpness = conf["sharpness"]
        self.d_th = conf["d_th"]
        self.sil_thresh = conf["sil_thresh"]
        
    def offset_to_uvd(self, x):
        """
        split prediction into predicted uvd with grid offset and predicted conf
        """
        self.batch_size = x.shape[0]
        self.W = x.shape[2]
        self.H = x.shape[3]        
        self.D = 5
        self.grid_size = self.W*self.H*self.D
        
        grid_linear = x.reshape(self.batch_size, 64, -1)

        index = torch.from_numpy(np.asarray(np.unravel_index(np.arange(self.grid_size), 
                                                             (self.W, self.H, self.D)))).type(torch.FloatTensor)
        u = index[0, :].unsqueeze(0).expand(21, -1)
        v = index[1, :].unsqueeze(0).expand(21, -1)
        z = index[2, :].unsqueeze(0).expand(21, -1)

        if self.device != "cpu":
            u = u.cuda()
            v = v.cuda()
            z = z.cuda()

        pred_uvd_no_offset = grid_linear[:, :63, :].reshape(self.batch_size, 21, 3, self.grid_size)
        pred_conf = grid_linear[:, 63, :].reshape(self.batch_size, self.W, self.H, self.D)
        pred_conf = torch.sigmoid(pred_conf)
        
        # middle finger root is hand root
        pred_uvd_no_offset[:, self.hand_root, :, :] = torch.sigmoid(pred_uvd_no_offset[:, self.hand_root, :, :])
        
        pred_uvd = pred_uvd_no_offset.clone().detach()
        pred_uvd[:, :, 0, :] = (pred_uvd[:, :, 0, :] + u) / self.W
        pred_uvd[:, :, 1, :] = (pred_uvd[:, :, 1, :] + v) / self.H
        pred_uvd[:, :, 2, :] = (pred_uvd[:, :, 2, :] + z) / self.D
  
        pred_uvd_no_offset = pred_uvd_no_offset.reshape(self.batch_size, 21, 3, self.W, self.H, self.D)
  
        return pred_uvd_no_offset, pred_uvd, pred_conf
    
    def calc_conf(self, pred_uvd, uvd_gt, 
                  im_width=FPHA.ORI_WIDTH, im_height=FPHA.ORI_HEIGHT, ref_depth=FPHA.REF_DEPTH):
        """
        calculate true confidence values 
        input: (21, 3, cells)
        output: (cells)
        """
        num_cells = pred_uvd.shape[-1]
        dist = pred_uvd - uvd_gt
        dist[:, 0, :] = dist[:, 0, :] * im_width
        dist[:, 1, :] = dist[:, 1, :] * im_height
        dist[:, 2, :] = dist[:, 2, :] * ref_depth
        
        eps = 1e-5
        dist = torch.sqrt(torch.sum((dist)**2, dim=1) + eps)
        mask = (dist < self.d_th).type(torch.FloatTensor)
        conf = torch.exp(self.sharpness*(1 - dist/self.d_th)) - 1 
        conf0 = (torch.exp(self.sharpness*(1 - torch.zeros(1, conf.shape[1]))) - 1).expand(21, num_cells)
        
        # if self.device != "cpu":
        #     conf0 = conf0.cuda()
        #     mask = mask.cuda()
        
        # normalize conf
        conf = conf / conf0
        # threshold values above d_th from conf
        conf = mask * conf 
        mean_conf = torch.mean(conf, dim=0)
        return mean_conf        
    
    # def calc_conf_uv_z(self, pred_uvd, uvd_gt, 
    #               im_width=FPHA.ORI_WIDTH, im_height=FPHA.ORI_HEIGHT, ref_depth=FPHA.REF_DEPTH):
    #     """
    #     calculate true confidence values for uv and z separately
    #     input: (21, 3, cells)
    #     output: (cells)
    #     """
    #     num_cells = pred_uvd.shape[-1]
    #     dist = pred_uvd - uvd_gt
    #     dist[:, 0, :] = dist[:, 0, :].clone() * im_width
    #     dist[:, 1, :] = dist[:, 1, :].clone() * im_height
    #     dist[:, 2, :] = dist[:, 2, :].clone() * ref_depth
        
    #     dist_uv = dist[:, :1, :].clone()
    #     dist_z = dist[:, 2, :].clone().unsqueeze(1)

    #     eps = 1e-5
    #     dist_uv = torch.sqrt(torch.sum((dist_uv)**2, dim=1) + eps)
    #     mask_uv = (dist_uv < self.d_th).type(torch.FloatTensor)
    #     conf_uv = torch.exp(self.sharpness*(1 - dist_uv/self.d_th)) - 1 
        
    #     dist_z = torch.sqrt(torch.sum((dist_z)**2, dim=1) + eps)
    #     mask_z = (dist_z < self.d_th).type(torch.FloatTensor)
    #     conf_z = torch.exp(self.sharpness*(1 - dist_z/self.d_th)) - 1 
        
    #     conf0 = (torch.exp(self.sharpness*(1 - torch.zeros(1, conf_uv.shape[1]))) - 1).expand(21, num_cells)

    #     if self.device != "cpu":
    #         conf0 = conf0.cuda()
    #         mask_uv = mask_uv.cuda()
    #         mask_z = mask_z.cuda()
        
    #     # normalize conf
    #     conf_uv = conf_uv / conf0
    #     conf_z = conf_z / conf0
        
    #     # threshold values above d_th from conf
    #     conf_uv = mask_uv * conf_uv 
    #     conf_z = mask_z * conf_z
        
    #     mean_conf_uv = torch.mean(conf_uv, dim=0)
    #     mean_conf_z = torch.mean(conf_z, dim=0)
    #     mean_conf = 0.5*mean_conf_uv + 0.5*mean_conf_z
    #     return mean_conf     
    
    def get_conf_mask(self, pred_uvd, uvd_gt):
        """
        get mask to weigh confidence values
        """
        # conf_mask initialized with no_hand_scale
        conf_mask   = torch.ones(self.batch_size, self.W, self.H, self.D)*self.no_hand_scale 
        for batch in range(self.batch_size):
            cur_pred_uvd = pred_uvd[batch]
            cur_uvd_gt = uvd_gt[batch].unsqueeze(-1).expand(21, 3, self.grid_size)
            cur_conf = self.calc_conf(cur_pred_uvd, cur_uvd_gt).reshape(self.W, self.H, self.D)
            # filter cells where we are not certain if there is no hand
            conf_mask[batch][cur_conf > self.sil_thresh] = 0
        return conf_mask

    def get_target(self, uvd_gt, pred_uvd):
        target_uvd = torch.zeros(self.batch_size, 21, 3, self.W, self.H, self.D).type(torch.FloatTensor)
        coord_mask = torch.zeros(self.batch_size, self.W, self.H, self.D).type(torch.FloatTensor)
        target_conf = torch.zeros(self.batch_size, self.W, self.H, self.D).type(torch.FloatTensor)
        conf_mask = self.get_conf_mask(pred_uvd, uvd_gt)
        
        for batch in range(self.batch_size):
            cur_uvd_gt = uvd_gt[batch]
            # get cell where hand root is present
            gi0 = int(cur_uvd_gt[self.hand_root, 0]*self.W)
            gj0 = int(cur_uvd_gt[self.hand_root, 1]*self.H)
            gk0 = int(cur_uvd_gt[self.hand_root, 2]*self.D)
            if gi0 >= self.W:
                gi0 = self.W - 1
            if gj0 >= self.H:
                gj0 = self.H - 1
            if gk0 >= self.D:
                gk0 = self.D - 1
                 
            target_uvd[batch, :, 0, gi0, gj0, gk0] = cur_uvd_gt[:, 0]*self.W - gi0
            target_uvd[batch, :, 1, gi0, gj0, gk0] = cur_uvd_gt[:, 1]*self.H - gj0
            target_uvd[batch, :, 2, gi0, gj0, gk0] = cur_uvd_gt[:, 2]*self.D - gk0
            coord_mask[batch, gi0, gj0, gk0] = 1
            cur_pred_uvd = pred_uvd[batch, :, :, self.H*self.D*gi0+self.D*gj0+gk0]
            conf = self.calc_conf(cur_pred_uvd.unsqueeze(-1), cur_uvd_gt.unsqueeze(-1))
            # target_conf only non-zero at hand root
            target_conf[batch, gi0, gj0, gk0] = conf
            conf_mask[batch, gi0, gj0, gk0] = self.hand_scale
            
        if self.device != "cpu":
            target_uvd = target_uvd.cuda()
            target_conf = target_conf.cuda()
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()

        conf_mask = conf_mask.sqrt()
        
        return target_uvd, target_conf, coord_mask, conf_mask
    
    def forward(self, pred, uvd_gt, train_out):
        pred_uvd_no_offset, pred_uvd, pred_conf = self.offset_to_uvd(pred)
        
        if self.device != "cpu":
            pred_uvd = pred_uvd.cpu()
            uvd_gt = uvd_gt.cpu()
            
        target_uvd, target_conf, coord_mask, conf_mask = self.get_target(uvd_gt, pred_uvd)
        mseloss = nn.MSELoss(reduction="sum")

        loss_u = 0.0
        for i in range(21):
            loss_u += mseloss(coord_mask*pred_uvd_no_offset[:, i, 0, :, :, :], coord_mask*target_uvd[:, i, 0, :, :, :])/2.0

        loss_v = 0.0
        for i in range(21):
            loss_v += mseloss(coord_mask*pred_uvd_no_offset[:, i, 1, :, :, :], coord_mask*target_uvd[:, i, 1, :, :, :])/2.0
            
        loss_d = 0.0
        for i in range(21):
            loss_d += mseloss(coord_mask*pred_uvd_no_offset[:, i, 2, :, :, :], coord_mask*target_uvd[:, i, 2, :, :, :])/2.0            

        loss_conf = mseloss(conf_mask*pred_conf, conf_mask*target_conf)/2.0

        total_loss = loss_u + loss_v + loss_d + loss_conf
        
        if train_out:
            return total_loss, loss_u, loss_v, loss_d, loss_conf
        else: 
            return total_loss