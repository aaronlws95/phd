import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from src import ROOT
from src.models.base_model import Base_Model
from src.datasets import get_dataloader, get_dataset
from src.models.hpo_hand.hpo_hand_net import HPO_Hand_Net
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler
from src.models.hpo_hand.hpo_hand_loss import HPO_Hand_Loss
from src.utils import *
from src.datasets.transforms import *

class HPO_Hand_Model(Base_Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.net        = HPO_Hand_Net(cfg).cuda()
        self.optimizer  = get_optimizer(cfg, self.net)
        self.scheduler  = get_scheduler(cfg, self.optimizer)

        self.train_dataloader   = get_dataloader(cfg, get_dataset(cfg, 'train'))
        self.val_dataloader     = get_dataloader(cfg, get_dataset(cfg, 'val'))

        self.pretrain = cfg['pretrain']
        self.load_weights()

        self.loss           = HPO_Hand_Loss(cfg)
        self.hand_root      = int(cfg['hand_root'])
        self.ref_depth      = int(cfg['ref_depth'])
        self.img_ref_width  = int(cfg['img_ref_width'])
        self.img_ref_height = int(cfg['img_ref_height'])
        self.D              = int(cfg['D'])
        self.num_joints     = int(cfg['num_joints'])
        self.img_rsz        = int(cfg['img_rsz'])

        self.val_xyz_error       = []
        self.best_pred_uvd_list     = []
        self.top10_pred_uvd_list     = []
        self.pred_conf_list         = []

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, uvd_gt         = data_load
        img                 = img.cuda()
        uvd_gt              = uvd_gt.cuda()
        hand_out            = self.net(img)
        loss, *hand_losses  = self.loss(hand_out, uvd_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_u, loss_v, loss_d, loss_conf = hand_losses
        loss_dict = {
            'loss'          : '{:04f}'.format(loss.item()),
            'loss_u'        : '{:04f}'.format(loss_u.item()),
            'loss_v'        : '{:04f}'.format(loss_v.item()),
            'loss_d'        : '{:04f}'.format(loss_d.item()),
            'loss_conf'     : '{:04f}'.format(loss_conf.item()),
        }

        return loss_dict

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img, uvd_gt             = data_load
        uvd_gt                  = uvd_gt.numpy()
        batch_size              = img.shape[0]
        img                     = img.cuda()

        pred_hand               = self.net(img)
        W                       = pred_hand.shape[3]
        H                       = pred_hand.shape[2]
        D                       = self.D
        pred_hand               = pred_hand.view(batch_size, self.num_joints*3+1, D, H, W)
        pred_hand               = pred_hand.permute(0, 1, 3, 4, 2)

        # Hand
        uvd_gt[..., 0]          *= self.img_ref_width
        uvd_gt[..., 1]          *= self.img_ref_height
        uvd_gt[..., 2]          *= self.ref_depth

        for batch in range(batch_size):
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

            pred_uvd                = pred_uvd.contiguous().view(self.num_joints, 3, -1)
            pred_conf               = pred_conf.contiguous().view(-1)

            top_idx                 = torch.topk(pred_conf, 1)[1]
            best_pred_uvd           = pred_uvd[:, :, top_idx].squeeze().cpu().numpy()
            best_pred_uvd[..., 0]   *= self.img_ref_width
            best_pred_uvd[..., 1]   *= self.img_ref_height
            best_pred_uvd[..., 2]   *= self.ref_depth
            cur_uvd_gt              = uvd_gt[batch]

            if cfg['dataset'] == 'ho3d_hand':
                cur_uvd_gt = HO3D.uvd2ccs(cur_uvd_gt)
                best_pred_uvd = HO3D.uvd2ccs(best_pred_uvd)
            elif cfg['dataset'] == 'fpha_hand':
                cur_uvd_gt = FPHA.uvd2xyz_color(cur_uvd_gt)
                best_pred_uvd = FPHA.uvd2xyz_color(best_pred_uvd)
                
            self.val_xyz_error.append(np.sqrt(np.sum(np.square(
                best_pred_uvd-cur_uvd_gt), axis=-1) + 1e-8 ))

    def get_valid_loss(self):
        val_xyz_l2_error    = np.mean(self.val_xyz_error)
        val_xyz_squeezed    = np.squeeze(np.asarray(self.val_xyz_error))
        pck                 = get_pck(val_xyz_squeezed)
        thresholds          = np.arange(0, 85, 5)
        auc                 = calc_auc(pck, thresholds)

        val_loss_dict = {
            'xyz_l2_error'  : val_xyz_l2_error,
            'AUC_0_85'      : auc,
        }

        self.val_xyz_error = []
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
        self.top10_pred_uvd_list     = []
        self.pred_conf_list         = []

    # ========================================================
    # DETECT
    # ========================================================

    def detect(self, img):
        import matplotlib.pyplot as plt
        best_pred_uvd = self._get_detect(img)
        img = ImgToNumpy()(img.cpu())[0]
        fig, ax = plt.subplots()
        plt.axis('off')
        ax.imshow(img)
        draw_joints(ax, best_pred_uvd)
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        # plt.axis('off')
        draw_3D_joints(ax, best_pred_uvd)
        plt.show()


    def _get_detect(self, img):
        with torch.no_grad():
            FT          = torch.FloatTensor
            pred        = self.net(img)
            W           = pred.shape[3]
            H           = pred.shape[2]
            D           = 5
            batch_size  = 1
            pred_hand   = pred.view(batch_size, 64, D, H, W)
            pred_hand   = pred_hand.permute(0, 1, 3, 4, 2)
            pred_hand   = pred_hand[0]

            pred_uvd    = pred_hand[:63, :, :, :].view(21, 3, H, W, D)
            pred_conf   = torch.sigmoid(pred_hand[63, :, :, :])

            yv, xv, zv  = torch.meshgrid([torch.arange(H),
                                            torch.arange(W),
                                            torch.arange(D)])
            grid_x      = xv.repeat((21, 1, 1, 1)).type(FT).cuda()
            grid_y      = yv.repeat((21, 1, 1, 1)).type(FT).cuda()
            grid_z      = zv.repeat((21, 1, 1, 1)).type(FT).cuda()

            pred_uvd[self.hand_root, :, :, :, :] = \
                torch.sigmoid(pred_uvd[self.hand_root, :, :, :, :])
            pred_uvd[:, 0, :, :, :] = (pred_uvd[:, 0, :, :, :] + grid_x)/W
            pred_uvd[:, 1, :, :, :] = (pred_uvd[:, 1, :, :, :] + grid_y)/H
            pred_uvd[:, 2, :, :, :] = (pred_uvd[:, 2, :, :, :] + grid_z)/D

            pred_uvd        = pred_uvd.contiguous().view(21, 3, -1)
            pred_conf       = pred_conf.contiguous().view(-1)

            top_idx         = torch.topk(pred_conf, 1)[1]
            best_pred_uvd   = pred_uvd[:, :, top_idx].squeeze().cpu().numpy()
            best_pred_uvd[..., 0]   *= img.shape[2]
            best_pred_uvd[..., 1]   *= img.shape[3]
            best_pred_uvd[..., 2]   *= self.ref_depth

        return best_pred_uvd


    def detect_video(self, seq_path, seq_name, fps=12, model_info='', seq_range=None):
        from moviepy.editor import ImageSequenceClip
        from tqdm import tqdm
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        from IPython.display import Image as IPythonImage

        img_size = self.img_rsz

        save_loc = Path(ROOT)/'mlcv-exp/data/saved'/'{}'.format(seq_name.replace('/', '_'))
        if not save_loc.is_dir():
            save_loc.mkdir(parents=True, exist_ok=True)
        print(seq_path)
        with torch.no_grad():
            seq = [x for x in sorted(seq_path.glob('*')) if x.is_file()]
            if seq_range:
                seq = seq[seq_range[0]:seq_range[1]]
                model_info += '_{}_{}'.format(seq_range[0], seq_range[1])

            frames = []
            save_count = 0
            for f in tqdm(seq):
                img = get_img_dataloader(str(f), img_size)
                img = img.unsqueeze(0).cuda()
                best_pred_uvd = self._get_detect(img)
                img = ImgToNumpy()(img.cpu())[0]

                fig, ax = plt.subplots()
                ax = fig.gca()
                ax.axis('off')
                ax.imshow(img)
                draw_joints(ax, best_pred_uvd, 'r')

                fig.canvas.draw()
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(data)

                plt.close()

            segment_clip = ImageSequenceClip(frames, fps=fps)

            name = str(Path(ROOT)/'mlcv-exp/data/saved'/'{}'.format(seq_name.replace('/', '_'))/'detect_{}_{}.gif'.format(seq_name.replace('/', '_'), model_info))
            segment_clip.write_gif(name, fps=fps)

            # with open(name, 'rb') as f:
            #     display(IPythonImage(data=f.read(), format='png'))