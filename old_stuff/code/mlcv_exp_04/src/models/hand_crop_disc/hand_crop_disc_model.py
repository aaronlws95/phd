import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from src import ROOT
from src.models.base_model import Base_Model
from src.datasets import get_dataloader, get_dataset
from src.models.hand_crop_disc.hand_crop_disc_net import DCGAN_Discriminator
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler
from src.utils import *
from src.datasets.transforms import *

class Hand_Crop_Disc_Model(Base_Model):
    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg['net'] == 'dcgan':
            self.net = DCGAN_Discriminator(cfg).cuda()
        self.optimizer  = get_optimizer(cfg, self.net)
        self.scheduler  = get_scheduler(cfg, self.optimizer)

        self.train_dataloader   = get_dataloader(cfg, get_dataset(cfg, 'train'))
        self.val_dataloader     = get_dataloader(cfg, get_dataset(cfg, 'val'))

        self.pretrain = cfg['pretrain']
        self.load_weights()

        self.img_rsz = int(cfg['img_rsz'])

        if cfg['loss'] == 'bce':
            self.loss = torch.nn.BCELoss()
        elif cfg['loss'] == 'mse':
            self.loss = torch.nn.MSELoss()

        self.val_loss = []

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, is_hand    = data_load
        img             = img.cuda()
        is_hand         = is_hand.type(torch.FloatTensor).cuda()
        out             = self.net(img)
        loss            = self.loss(out, is_hand)

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
        img, is_hand    = data_load
        img             = img.cuda()
        is_hand         = is_hand.type(torch.FloatTensor).cuda()
        out             = self.net(img)
        loss            = self.loss(out, is_hand)

        self.val_loss.append(loss.item())

    def get_valid_loss(self):
        val_loss = np.mean(self.val_loss)

        val_loss_dict = {
            'val_loss'  : val_loss,
        }

        self.val_loss = []

        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        img                     = data_load[0]
        img                     = img.cuda()
        pred_hand               = self.net(img)
        batch_size              = img.shape[0]
        W                       = pred_hand.shape[3]
        H                       = pred_hand.shape[2]
        D                       = self.D
        pred_hand               = pred_hand.view(batch_size, self.num_joints*3+1, D, H, W)
        pred_hand               = pred_hand.permute(0, 1, 3, 4, 2)

        for batch in range(batch_size):
            # Hand
            cur_pred_hand   = pred_hand[batch]
            pred_uvd        = cur_pred_hand[:self.num_joints*3, :, :, :].view(self.num_joints, 3, H, W, D)
            pred_conf       = torch.sigmoid(cur_pred_hand[self.num_joints*3, :, :, :])

            FT              = torch.FloatTensor
            yv, xv, zv      = torch.meshgrid([torch.arange(H),
                                              torch.arange(W),
                                              torch.arange(D)])
            grid_x          = xv.repeat((self.num_joints, 1, 1, 1)).type(FT).cuda()
            grid_y          = yv.repeat((self.num_joints, 1, 1, 1)).type(FT).cuda()
            grid_z          = zv.repeat((self.num_joints, 1, 1, 1)).type(FT).cuda()

            pred_uvd[self.hand_root, :, :, :, :] = \
                torch.sigmoid(pred_uvd[self.hand_root, :, :, :, :])
            pred_uvd[:, 0, :, :, :] = (pred_uvd[:, 0, :, :, :] + grid_x)/W
            pred_uvd[:, 1, :, :, :] = (pred_uvd[:, 1, :, :, :] + grid_y)/H
            pred_uvd[:, 2, :, :, :] = (pred_uvd[:, 2, :, :, :] + grid_z)/D

            pred_uvd    = pred_uvd.contiguous().view(self.num_joints, 3, -1)
            pred_conf   = pred_conf.contiguous().view(-1)

            top10_pred_uvd = []
            top10_idx = torch.topk(pred_conf, 10)[1]
            for idx in top10_idx:
                top10_pred_uvd.append(pred_uvd[:, :, idx].cpu().numpy())
            self.best_pred_uvd_list.append(top10_pred_uvd[0])
            self.top10_pred_uvd_list.append(top10_pred_uvd)
            self.pred_conf_list.append(pred_conf.cpu().numpy())

    def save_predictions(self, data_split):
        pred_save = "predict_{}_{}_best.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.best_pred_uvd_list, (-1, self.num_joints*3)))

        pred_save = "predict_{}_{}_top10.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.top10_pred_uvd_list, (-1, self.num_joints*3*10)))

        pred_save = "predict_{}_{}_conf.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_conf_list)

        self.pred_list              = []
        self.best_pred_uvd_list     = []
        self.top10_pred_uvd_list    = []
        self.pred_conf_list         = []

    # ========================================================
    # DETECT
    # ========================================================

    def detect(self, img, bbox):
        import matplotlib.pyplot as plt
        import torchvision

        if bbox is not None:
            img_crop = get_img_crop_from_bbox(img, bbox)
            tfrm = []
            tfrm.append(ImgResize((self.img_rsz)))
            tfrm.append(ImgToTorch())
            transform = torchvision.transforms.Compose(tfrm)
            sample      = {'img': img_crop}
            sample      = transform(sample)
            img_crop    = sample['img']
            img_crop = img_crop.unsqueeze(0).cuda()
        else:
            img_crop = img.cuda()

        out = self.net(img_crop)

        img_crop = ImgToNumpy()(img_crop.cpu())[0]

        fig, ax = plt.subplots()
        plt.axis('off')
        ax.imshow(img_crop)
        plt.show()

        print(out.item())

    def detect_out(self, img, bbox):
        import matplotlib.pyplot as plt
        import torchvision

        if bbox is not None:
            img_crop = get_img_crop_from_bbox(img, bbox)
            tfrm = []
            tfrm.append(ImgResize((self.img_rsz)))
            tfrm.append(ImgToTorch())
            transform = torchvision.transforms.Compose(tfrm)
            sample      = {'img': img_crop}
            sample      = transform(sample)
            img_crop    = sample['img']
            img_crop = img_crop.unsqueeze(0).cuda()
        else:
            img_crop = img.cuda()

        out = self.net(img_crop)

        img_crop = ImgToNumpy()(img_crop.cpu())[0]

        return img_crop, out.item()