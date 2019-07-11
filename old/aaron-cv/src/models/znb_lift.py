import torch
import os
import time
import numpy                        as np
import torch.nn                     as nn
import torch.nn.functional          as F
from pathlib                        import Path
from tqdm                           import tqdm

from src.models                     import Model
from src.utils                      import IMG, RHD
from src.datasets                   import get_dataset, get_dataloader
from src.components                 import get_optimizer, get_scheduler, ZNB_Lift_net

class ZNB_Lift(Model):
    """ ZNB PosePrior and Viewpoint net to lift 2D keypoints to 3D """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)
        # Network
        cur_path            = Path(__file__).resolve().parents[2]
        pp_conv_cfgfile     = cur_path/'net_cfg'/(cfg['pp_conv_cfg'] + '.cfg')
        vp_conv_cfgfile     = cur_path/'net_cfg'/(cfg['vp_conv_cfg'] + '.cfg')
        pp_lin_cfgfile      = cur_path/'net_cfg'/(cfg['pp_lin_cfg'] + '.cfg')
        vp_lin_cfgfile      = cur_path/'net_cfg'/(cfg['vp_lin_cfg'] + '.cfg')
        self.net            = ZNB_Lift_net(pp_conv_cfgfile, pp_lin_cfgfile,
                                           vp_conv_cfgfile, vp_lin_cfgfile)
        self.net            = self.net.cuda()

        if self.training:
            # Optimizer and scheduler
            self.optimizer  = get_optimizer(cfg, self.net)
            self.scheduler  = get_scheduler(cfg, self.optimizer)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        # IMPORTANT TO LOAD WEIGHTS
        self.load_weights(self.load_epoch)
        self.img_size       = int(cfg['img_size'])

        # Training
        if self.training:
            # Dataset
            dataset_kwargs  = {'split_set': cfg['train_set']}
            train_dataset   = get_dataset(cfg, dataset_kwargs)
            self.train_sampler = None
            shuffle = cfg['shuffle']
            kwargs = {'batch_size'  :   int(cfg['batch_size']),
                      'shuffle'     :   shuffle,
                      'num_workers' :   int(cfg['num_workers']),
                      'pin_memory'  :   True}
            self.train_loader = get_dataloader(train_dataset,
                                               self.train_sampler,
                                               kwargs)
            # Validation
            dataset_kwargs          = {'split_set': cfg['val_set']}
            val_dataset             = get_dataset(cfg, dataset_kwargs)
            self.val_loader         = get_dataloader(val_dataset,
                                                     None,
                                                     kwargs)
            self.val_xyz_21_error   = []
            self.vis_list           = []
        # Prediction
        else:
            self.pred_list          = []
            self.pred_xyz_canon     = []
            self.pred_rot_mat       = []
    # ========================================================
    # UTILITY
    # ========================================================

    def get_rot_mat(self, ux, uy, uz):
        """ Get rotation matrix from predicted embedding """
        theta   = torch.sqrt(ux*ux + uy*uy + uz*uz + 1e-8)
        sin     = torch.sin(theta)
        cos     = torch.cos(theta)
        mcos    = 1 - torch.cos(theta)
        norm_ux = ux/theta
        norm_uy = uy/theta
        norm_uz = uz/theta

        row_1   = torch.cat((cos + norm_ux*norm_ux*mcos,
                             norm_ux*norm_uy*mcos - norm_uz*sin,
                             norm_ux*norm_uz*mcos + norm_uy*sin), dim=-1)

        row_2   = torch.cat((norm_uy*norm_ux*mcos + norm_uz*sin,
                             cos + norm_uy*norm_uy*mcos,
                             norm_uy*norm_uz*mcos - norm_ux*sin), dim=-1)

        row_3   = torch.cat((norm_uz*norm_ux*mcos - norm_uy*sin,
                            norm_uz*norm_uy*mcos + norm_ux*sin,
                            cos + norm_uz*norm_uz*mcos), dim=-1)

        # if self.debug:
        #     T = torch.Tensor
        #     row_1_check = torch.cat((T([0]), T([1]), T([2])), dim=-1)
        #     row_2_check = torch.cat((T([3]), T([4]), T([5])), dim=-1)
        #     row_3_check = torch.cat((T([6]), T([7]), T([8])), dim=-1)
        #     print(torch.stack((row_1_check, row_2_check, row_3_check), dim=-1).permute(1, 0))
        
        rot_mat = torch.stack((row_1, row_2, row_3), dim=-1).permute(0, 2, 1)
        
        # if self.debug:
        #     print(rot_mat.requires_grad)
        
        return rot_mat

    # ========================================================
    # LOSS
    # ========================================================

    def loss(self, pred_rot_mat, pred_xyz_canon,
             rot_mat, xyz_gt_canon, vis):
        mseloss         = nn.MSELoss()
        loss_xyz_canon  = mseloss(pred_xyz_canon, xyz_gt_canon)
        loss_rot_mat    = mseloss(pred_rot_mat, rot_mat)
        total_loss      = loss_xyz_canon + loss_rot_mat
        return total_loss, loss_xyz_canon, loss_rot_mat

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        scoremap, xyz_gt_canon, rot_mat, _, _, _, vis, hand_side = data_load

        t0                              = time.time() # start
        scoremap                        = scoremap.cuda() # [b, 21, 256, 256]
        xyz_gt_canon                    = xyz_gt_canon.cuda() # [b, 21, 3]
        rot_mat                         = rot_mat.cuda() # [b, 3, 3]
        hand_side                       = hand_side.cuda()
        vis                             = vis.cuda()
        t1                              = time.time() # CPU to GPU
        pred_xyz_canon, ux, uy, uz      = self.net(scoremap, hand_side)

        # if self.debug:
        #     # 0.1, 0.1, 0.1 output:
        #     #[ 0.9900, -0.0945,  0.1045]
        #     #[ 0.1045,  0.9900, -0.0945]
        #     #[-0.0945,  0.1045,  0.9900]
        #     # 0.1, 0.2, 0.3 output:
        #     # [ 0.9358, -0.2832,  0.2102]
        #     # [ 0.3029,  0.9506, -0.0680]
        #     # [-0.1805,  0.1273,  0.9753]
        #     ux[0, 0] = torch.FloatTensor([0.1])
        #     uy[0, 0] = torch.FloatTensor([0.1])
        #     uz[0, 0] = torch.FloatTensor([0.1])
        #     ux[1, 0] = torch.FloatTensor([0.1])
        #     uy[1, 0] = torch.FloatTensor([0.2])
        #     uz[1, 0] = torch.FloatTensor([0.3])
        #     print(self.get_rot_mat(ux, uy, uz))

        pred_rot_mat                    = self.get_rot_mat(ux, uy, uz)
        t2                              = time.time() # forward
        loss, *other_loss               = self.loss(pred_rot_mat,
                                                    pred_xyz_canon,
                                                    rot_mat,
                                                    xyz_gt_canon,
                                                    vis)
        t3                              = time.time() # loss
        loss_xyz_canon, loss_rot_mat    = other_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        t4 = time.time() # backward

        loss_dict = {
            "loss"              : loss.item(),
            "loss_xyz_canon"    : loss_xyz_canon.item(),
            "loss_rot_mat"      : loss_rot_mat.item(),
        }

        if self.time:
            print('-----Training-----')
            print('        CPU to GPU : %f' % (t1 - t0))
            print('           forward : %f' % (t2 - t1))
            print('              loss : %f' % (t3 - t2))
            print('          backward : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        return loss_dict

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        scoremap, xyz_gt_canon, rot_mat, \
        kpt_scale, K, kp_coord_xyz21, vis, hand_side = data_load

        scoremap                    = scoremap.cuda()
        hand_side                   = hand_side.cuda()
        pred_xyz_canon, ux, uy, uz  = self.net(scoremap, hand_side)
        pred_rot_mat                = self.get_rot_mat(ux, uy, uz)
        pred_xyz_canon              = pred_xyz_canon.cpu().numpy()
        pred_rot_mat                = pred_rot_mat.cpu().numpy()
        cur_kpt                     = kpt_scale.numpy()
        kp_coord_xyz21              = kp_coord_xyz21.numpy()
        hand_side                   = hand_side.cpu().numpy()
        vis                         = vis.cpu().numpy()
        vis                         = vis.astype('uint8')

        for batch in range(scoremap.shape[0]):
            cur_hand_side   = hand_side[batch]
            cur_xyz_can     = pred_xyz_canon[batch]
            cur_rot_mat     = pred_rot_mat[batch]
            cur_kpt         = float(kpt_scale[batch])
            cur_xyz_gt      = kp_coord_xyz21[batch]
            cur_vis         = vis[batch]

            if cur_hand_side == 1:
                cur_xyz_can[:, 2] = -cur_xyz_can[:, 2]

            reform_xyz      = np.matmul(cur_xyz_can, cur_rot_mat)*cur_kpt
            reform_xyz      += cur_xyz_gt[0, :]

            self.val_xyz_21_error.append(np.sqrt(np.sum(np.square(
                cur_xyz_gt[cur_vis]*1000 - reform_xyz[cur_vis]*1000),
                                                        axis=-1) + 1e-8 ))
            self.vis_list.append(cur_vis)

    def get_valid_loss(self):
        eps                 = 1e-5
        val_xyz_l2_error    = np.mean(self.val_xyz_21_error)
        val_xyz_squeeze     = np.squeeze(np.asarray(self.val_xyz_21_error))

        data        = []
        for i in range(21):
            data.append([])

        for i in range(len(val_xyz_squeeze)):
            cur_vis = self.vis_list[i]
            for j in range(21):
                if cur_vis[j]:
                    data[j].append(val_xyz_squeeze[i, j])
        data = np.asarray(data)

        pck                 = RHD.get_pck_with_vis(data)
        thresholds          = np.arange(0, 85, 5)
        auc                 = RHD.calc_auc(pck, thresholds)

        val_loss_dict = {
            'xyz_l2_error'  : val_xyz_l2_error,
            'AUC_0_85'      : auc,
        }

        self.val_xyz_21_error   = []
        self.vis_list           = []
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        scoremap, xyz_gt_canon, rot_mat, \
        kpt_scale, K, kp_coord_xyz21, _, hand_side = data_load

        scoremap                    = scoremap.cuda()
        hand_side                   = hand_side.cuda()
        pred_xyz_canon, ux, uy, uz  = self.net(scoremap, hand_side)
        pred_rot_mat                = self.get_rot_mat(ux, uy, uz)
        pred_xyz_canon              = pred_xyz_canon.cpu().numpy()
        pred_rot_mat                = pred_rot_mat.cpu().numpy()
        cur_kpt                     = kpt_scale.numpy()
        kp_coord_xyz21              = kp_coord_xyz21.numpy()
        hand_side                   = hand_side.cpu().numpy()
        
        for batch in range(scoremap.shape[0]):
            cur_xyz_can     = pred_xyz_canon[batch]
            cur_rot_mat     = pred_rot_mat[batch]
            cur_kpt         = float(kpt_scale[batch])
            cur_xyz_gt      = kp_coord_xyz21[batch]
            cur_hand_side   = hand_side[batch]
            
            self.pred_xyz_canon.append(cur_xyz_can)
            self.pred_rot_mat.append(cur_rot_mat)
            
            if cur_hand_side == 1:
                cur_xyz_can[:, 2] = -cur_xyz_can[:, 2]

            reform_xyz      = np.matmul(cur_xyz_can, cur_rot_mat)*cur_kpt
            reform_xyz      += cur_xyz_gt[0, :]

            self.pred_list.append(reform_xyz)

    def save_predictions(self, data_split):
        pred_save = "predict_{}_{}_xyz.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.pred_list, (-1, 63)))

        pred_save = "predict_{}_{}_canon.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.pred_xyz_canon, (-1, 63)))

        pred_save = "predict_{}_{}_rot.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.pred_rot_mat, (-1, 9)))

        self.pred_list = []
        self.pred_xyz_canon = []
        self.pred_rot_mat = []
