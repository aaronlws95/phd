import os
import torch
import torch.nn as nn
import numpy as np

from darknet import Darknet
from utils.directory import DATA_DIR, DATASET_DIR
import utils.cam as cam

class HPO_Net(nn.Module):
    def __init__(self, cfgfile='yolov2_hpo.cfg'):
        super(HPO_Net, self).__init__()
        darknet = Darknet(cfgfile)
        self.darknet = darknet

        # index[:, 844] = (12, 12, 4)
        index = torch.from_numpy(np.asarray(np.unravel_index(np.arange(845), (13, 13, 5)))).type(torch.FloatTensor)
        self.u = index[0, :].unsqueeze(1).expand(-1, 21).cuda()
        self.v = index[1, :].unsqueeze(1).expand(-1, 21).cuda()
        self.z = index[2, :].unsqueeze(1).expand(-1, 21).cuda()
        self.X0_COLOR = cam.X0_COLOR
        self.Y0_COLOR = cam.Y0_COLOR
        self.FOCAL_LENGTH_X_COLOR = cam.FOCAL_LENGTH_X_COLOR
        self.FOCAL_LENGTH_Y_COLOR = cam.FOCAL_LENGTH_Y_COLOR

    def offset_to_uvz(self, x):
        batch_size = x.shape[0]
        grid_linear = x.reshape(batch_size, -1, 64)

        pred_uvd = grid_linear[..., :63].reshape(batch_size, -1, 21, 3)
        conf = grid_linear[..., 63]

        pred_uvd[:, :, 0, :] = torch.sigmoid(pred_uvd[:, :, 0, :])

        pred_uvd_no_offset = pred_uvd.clone()

        pred_uvd[..., 0] = 416*pred_uvd[..., 0].clone() + 32*self.u
        pred_uvd[..., 1] = 416*pred_uvd[..., 1].clone() + 32*self.v
        pred_uvd[..., 2] = 1000*pred_uvd[..., 2].clone() + 200*self.z

        return pred_uvd_no_offset, pred_uvd, conf

    def forward(self, x):
        x = self.darknet(x)
        x = self.offset_to_uvz(x)
        return x
