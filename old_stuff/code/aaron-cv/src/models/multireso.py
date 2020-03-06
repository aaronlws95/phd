import torch
import os
import time
import torch.nn                     as nn
import numpy                        as np
import torch.nn.functional          as F
from pathlib                        import Path
from tqdm                           import tqdm

from src.models                     import Model
from src.utils                      import IMG, FPHA
from src.datasets                   import get_dataset, get_dataloader
from src.components                 import get_optimizer, get_scheduler, Multireso_net

class Multireso(Model):
    """ Multi-resolution network keypoint estimation from cropped hand """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)
        # Network
        cur_path            = Path(__file__).resolve().parents[2] 
        net0_cfgfile        = cur_path/'net_cfg'/(cfg['net0_cfg'] + '.cfg')
        net1_cfgfile        = cur_path/'net_cfg'/(cfg['net1_cfg'] + '.cfg')
        net2_cfgfile        = cur_path/'net_cfg'/(cfg['net2_cfg'] + '.cfg')
        dense_cfgfile       = cur_path/'net_cfg'/(cfg['dense_net_cfg'] + '.cfg')
        self.net            = Multireso_net(net0_cfgfile, net1_cfgfile,
                                            net2_cfgfile, dense_cfgfile)
        self.net            = self.net.cuda()
        self.loss           = nn.MSELoss()

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
        
        # Training
        if self.training:         
            # Dataset
            dataset_kwargs = {'split_set': cfg['train_set']}
            train_dataset = get_dataset(cfg, **dataset_kwargs)
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
            val_dataset             = get_dataset(cfg, **dataset_kwargs)
            self.val_loader         = get_dataloader(val_dataset,
                                                     None,
                                                     kwargs)
            self.val_xyz_21_error   = []
        # Prediction
        else:
            self.pred_list          = []
            self.uvd_gt_list        = []
            self.xy_offset_list     = []
            
    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img_list, uvd_gt, _             = data_load
        t0                              = time.time() # start
        
        for i in range(len(img_list)):
            img_list[i] = img_list[i].cuda()
            
        uvd_gt                          = uvd_gt.cuda()
        t1                              = time.time() # CPU to GPU
        out                             = self.net(img_list)
        t2                              = time.time() # forward
        loss                            = self.loss(out, uvd_gt)
        t3                              = time.time() # loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        t4 = time.time() # backward

        loss_dict = {
            'loss': loss.item()
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
        img_list, uvd_gt, _             = data_load

        for i in range(len(img_list)):
            img_list[i] = img_list[i].cuda()
            
        out                             = self.net(img_list)
        pred_uvd                        = out.cpu().numpy()
        pred_uvd                        = IMG.scale_points_WH(pred_uvd, 
                                                              (1, 1), 
                                                              (FPHA.ORI_WIDTH, 
                                                               FPHA.ORI_HEIGHT))
        pred_uvd[..., 2]                *= FPHA.REF_DEPTH
        pred_xyz                        = FPHA.uvd2xyz_color(pred_uvd)
        
        uvd_gt                          = uvd_gt.numpy()
        uvd_gt                          = IMG.scale_points_WH(uvd_gt, 
                                                              (1, 1), 
                                                              (FPHA.ORI_WIDTH, 
                                                               FPHA.ORI_HEIGHT))
        uvd_gt[..., 2]                  *= FPHA.REF_DEPTH
        xyz_gt                          = FPHA.uvd2xyz_color(uvd_gt)
        
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
        img_list, uvd_gt, xy_ofs        = data_load

        for i in range(len(img_list)):
            img_list[i] = img_list[i].cuda()
            
        out                             = self.net(img_list)
        pred_uvd                        = out.cpu().numpy()
        uvd_gt                          = uvd_gt.numpy()
        x_min                           = xy_ofs[0].numpy()
        y_min                           = xy_ofs[1].numpy()
        crop_w                          = xy_ofs[2].numpy()
        crop_h                          = xy_ofs[3].numpy()
        
        for batch in range(pred_uvd.shape[0]):
            self.pred_list.append(pred_uvd[batch])
            self.uvd_gt_list.append(uvd_gt[batch])
            self.xy_offset_list.append((x_min[batch], 
                                        y_min[batch], 
                                        crop_w[batch], 
                                        crop_h[batch]))
            
    def save_predictions(self, data_split):
        pred_save = 'predict_{}_{}_uvd.txt'.format(self.load_epoch,
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.pred_list, (-1, 63)))

        file_name = 'uvd_gt_{}.txt'.format(data_split)
        file_name = Path(self.data_dir)/self.exp_dir/file_name
        np.savetxt(file_name, np.reshape(self.uvd_gt_list, (-1, 63)))
        
        file_name = 'xy_offset_{}.txt'.format(data_split)
        file_name = Path(self.data_dir)/self.exp_dir/file_name
        np.savetxt(file_name, np.reshape(self.xy_offset_list, (-1, 4)))

        self.pred_list      = []
        self.uvd_gt_list    = []
        self.xy_offset_list = []
