import torch
import os
import time
import torch.nn.functional          as F
import torch.nn                     as nn
import numpy                        as np
from pathlib                        import Path
from tqdm                           import tqdm

from src.models                     import Multireso
from src.utils                      import IMG, FPHA
from src.datasets                   import get_dataset, get_dataloader
from src.components                 import get_optimizer, get_scheduler, Multireso_net

class Multireso_Normuvd(Multireso):
    """ Multi-resolution network keypoint estimation from cropped hand 
    using some fancy normalization technique """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)
        
    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img_list, uvd_gt, handcen_uvd = data_load

        for i in range(len(img_list)):
            img_list[i] = img_list[i].cuda()

        out                             = self.net(img_list)
        pred_uvd                        = out.cpu().numpy()
        mean_u                          = handcen_uvd[0].cpu().numpy()
        mean_v                          = handcen_uvd[1].cpu().numpy()
        mean_z                          = handcen_uvd[2].cpu().numpy()
        handcen_uvd                     = np.asarray([mean_u, mean_v, mean_z])
        handcen_uvd                     = handcen_uvd.T
        uvd_gt                          = uvd_gt.numpy()
        pred_xyz, _                     = FPHA.normuvd2xyzuvd_color(pred_uvd, 
                                                                    handcen_uvd)
        xyz_gt, _                       = FPHA.normuvd2xyzuvd_color(uvd_gt, 
                                                                    handcen_uvd)
        
        for batch in range(uvd_gt.shape[0]):
            cur_xyz_gt      = xyz_gt[batch]
            cur_pred_xyz    = pred_xyz[batch]
            self.val_xyz_21_error.append(np.sqrt(np.sum(np.square(
                cur_xyz_gt-cur_pred_xyz), axis=-1) + 1e-8 ))

    def get_valid_loss(self):
        eps                 = 1e-5
        val_xyz_l2_error    = np.mean(self.val_xyz_21_error)
        val_xyz_squeeze     = np.squeeze(np.asarray(self.val_xyz_21_error))
        pck                 = FPHA.get_pck(val_xyz_squeeze)
        thresholds          = np.arange(0, 85, 5)
        auc                 = FPHA.calc_auc(pck, thresholds)

        val_loss_dict = {
            'xyz_l2_error'  : val_xyz_l2_error,
            'AUC_0_85'      : auc,
        }

        self.val_xyz_21_error = []
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        img_list, uvd_gt, handcen_uvd = data_load

        for i in range(len(img_list)):
            img_list[i]                 = img_list[i].cuda()
        out                             = self.net(img_list)
        pred_uvd                        = out.cpu().numpy()
        mean_u                          = handcen_uvd[0].cpu().numpy()
        mean_v                          = handcen_uvd[1].cpu().numpy()
        mean_z                          = handcen_uvd[2].cpu().numpy()
        handcen_uvd                     = np.asarray([mean_u, mean_v, mean_z])
        handcen_uvd                     = handcen_uvd.T
        _, pred_uvd                     = FPHA.normuvd2xyzuvd_color(pred_uvd, 
                                                                    handcen_uvd)
        
        for batch in range(pred_uvd.shape[0]):
            self.pred_list.append(pred_uvd[batch])
            
    def save_predictions(self, data_split):
        pred_save = 'predict_{}_{}_uvd.txt'.format(self.load_epoch,
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.pred_list, (-1, 63)))

        self.pred_list = []
