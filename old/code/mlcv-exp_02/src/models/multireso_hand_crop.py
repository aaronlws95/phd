import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src import ROOT
from src.models.base_model import Base_Model
from src.datasets import get_dataloader, get_dataset
from src.networks.multireso_hand_crop_net import *
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler
from src.loss.hpo_hand_loss import HPO_Hand_Loss
from src.utils import *
from src.datasets.transforms import *

class Multireso_Hand_Crop(Base_Model):
    def __init__(self, cfg, mode, load_epoch):
        super().__init__(cfg, mode, load_epoch)
        net = cfg['net']

        if net == 'multireso_hand_crop_net':
            self.net        = Multireso_Hand_Crop_Net().cuda()
        elif net == 'multireso_hand_crop_bn_net':
            self.net        = Multireso_Hand_Crop_BN_Net().cuda()

        self.optimizer  = get_optimizer(cfg, self.net)
        self.scheduler  = get_scheduler(cfg, self.optimizer)

        self.train_dataloader   = get_dataloader(cfg, get_dataset(cfg, 'train'))
        self.val_dataloader     = get_dataloader(cfg, get_dataset(cfg, 'val'))

        self.pretrain = cfg['pretrain']
        self.load_weights()

        self.loss       = torch.nn.MSELoss()
        self.ref_depth  = int(cfg['ref_depth'])

        self.val_xyz_21_error       = []

        self.pred_list              = []
        self.gt_list                = []

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img_list, uvd_gt    = data_load

        for i in range(len(img_list)):
            img_list[i] = img_list[i].cuda()

        uvd_gt              = uvd_gt.cuda()
        hand_out            = self.net(img_list)
        uvd_gt              = uvd_gt.view(uvd_gt.shape[0], -1)
        loss                = self.loss(hand_out, uvd_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict = {
            'loss'          : '{:04f}'.format(loss.item()),
        }

        return loss_dict

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img_list, uvd_gt = data_load
        batch_size = uvd_gt.shape[0]
        for i in range(len(img_list)):
            img_list[i] = img_list[i].cuda()

        pred_hand               = self.net(img_list)
        pred_hand               = pred_hand.cpu().numpy()
        pred_hand               = pred_hand.reshape(pred_hand.shape[0], 21, 3)

        # Hand
        uvd_gt[..., 0]          *= FPHA.ORI_WIDTH
        uvd_gt[..., 1]          *= FPHA.ORI_HEIGHT
        uvd_gt[..., 2]          *= self.ref_depth
        xyz_gt                  = FPHA.uvd2xyz_color(uvd_gt)

        pred_hand[..., 0]          *= FPHA.ORI_WIDTH
        pred_hand[..., 1]          *= FPHA.ORI_HEIGHT
        pred_hand[..., 2]          *= self.ref_depth
        pred_xyz                    = FPHA.uvd2xyz_color(pred_hand)

        for batch in range(batch_size):
            cur_xyz_gt      = xyz_gt[batch]
            cur_pred_xyz    = pred_xyz[batch]
            self.val_xyz_21_error.append(np.sqrt(np.sum(np.square(
                cur_xyz_gt-cur_pred_xyz), axis=-1) + 1e-8 ))

    def get_valid_loss(self):
        val_xyz_l2_error    = np.mean(self.val_xyz_21_error)
        val_xyz_squeezed    = np.squeeze(np.asarray(self.val_xyz_21_error))
        pck                 = get_pck(val_xyz_squeezed)
        thresholds          = np.arange(0, 85, 5)
        auc                 = calc_auc(pck, thresholds)

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
        img_list, uvd_gt        = data_load

        for i in range(len(img_list)):
            img_list[i] = img_list[i].cuda()

        out                             = self.net(img_list)
        pred_uvd                        = out.cpu().numpy()
        uvd_gt                          = uvd_gt.numpy()

        for batch in range(pred_uvd.shape[0]):
            self.pred_list.append(pred_uvd[batch])
            self.gt_list.append(uvd_gt[batch])

    def save_predictions(self, data_split):
        pred_save = "predict_{}_{}_uvd.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(ROOT)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.pred_list, (-1, 63)))

        gt_save = "gt_{}_{}_uvd.txt".format(self.load_epoch, data_split)
        gt_file = Path(ROOT)/self.exp_dir/gt_save
        np.savetxt(gt_file, np.reshape(self.gt_list, (-1, 63)))

        self.pred_list              = []
        self.gt_list                = []


    # ========================================================
    # DETECT
    # ========================================================

    def detect(self, img_list):
        with torch.no_grad():
            import matplotlib.pyplot as plt
            pred_uvd = self.net(img_list)
            img = ImgToNumpy()(img_list[0].cpu())[0]
            pred_uvd = pred_uvd.cpu().numpy()
            pred_uvd = pred_uvd.reshape(21, 3)
            pred_uvd[..., 0] *= img.shape[1]
            pred_uvd[..., 1] *= img.shape[0]
            pred_uvd[..., 2] *= self.ref_depth
            fig, ax = plt.subplots()
            ax.imshow(img)
            draw_joints(ax, pred_uvd, 'r')
            plt.axis('off')
            plt.show()

