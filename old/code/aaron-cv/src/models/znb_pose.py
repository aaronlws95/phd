import torch
import os
import time
import torch.nn.functional          as F
import torch.nn                     as nn
import numpy                        as np
from pathlib                        import Path
from tqdm                           import tqdm

from src.models                     import Model
from src.utils                      import IMG, RHD
from src.datasets                   import get_dataset, get_dataloader

class ZNB_Pose(Model):
    """ ZNB PoseNet 2D keypoint detection by predicting scoremaps from cropped
    hands """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)

        # IMPORTANT TO LOAD WEIGHTS
        self.load_weights(self.load_epoch)
        self.img_size       = int(cfg['img_size'])

        # Training
        if self.training:
            dataset_kwargs  = {'split_set': cfg['train_set']}
            train_dataset   = get_dataset(cfg, dataset_kwargs)
            self.train_sampler  = None
            shuffle = cfg['shuffle']
            kwargs = {'batch_size'  :   int(cfg['batch_size']),
                      'shuffle'     :   shuffle,
                      'num_workers' :   int(cfg['num_workers']),
                      'pin_memory'  :   True}
            self.train_loader       = get_dataloader(train_dataset,
                                                     self.train_sampler,
                                                     kwargs)
            # Validation
            dataset_kwargs          = {'split_set': cfg['val_set']}
            val_dataset             = get_dataset(cfg, dataset_kwargs)
            self.val_loader         = get_dataloader(val_dataset,
                                                     None,
                                                     kwargs)
            self.val_uv_21_error    = []
            self.vis_list           = []
        # Prediction
        else:
            self.pred_list          = []

    # ========================================================
    # LOSS
    # ========================================================

    def loss(self, pred, true, vis):
        def l2dist(pred, true):
            l2_sum = torch.sum(vis*torch.sqrt(torch.mean((true - pred)**2,
                                                         dim=(2,3))))
            return l2_sum/(torch.sum(vis) + 1e-5)

        loss0 = l2dist(pred[0], true)
        loss1 = l2dist(pred[1], true)
        loss2 = l2dist(pred[2], true)

        # if self.debug:
        #     print(loss0.requires_grad,
        #           loss1.requires_grad,
        #           loss2.requires_grad)

        total_loss = loss0 + loss1 + loss2
        return total_loss, loss0, loss1, loss2

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, scoremap, _, keypoint_vis21 = data_load

        t0                  = time.time() # start
        img                 = img.cuda()
        scoremap            = scoremap.cuda() # [b, 21, 256, 256]
        keypoint_vis21      = keypoint_vis21.cuda()
        t1                  = time.time() # CPU to GPU

        out                 = self.net(img) # 3*[b, 21, 32, 32]
        out[0]              = F.interpolate(out[0], 
                                            size=(self.img_size, self.img_size),
                                            mode='bilinear')
        out[1]              = F.interpolate(out[1], 
                                            size=(self.img_size, self.img_size),
                                            mode='bilinear')
        out[2]              = F.interpolate(out[2], 
                                            size=(self.img_size, self.img_size),
                                            mode='bilinear')
        t2                  = time.time() # forward

        loss, *other_loss   = self.loss(out, scoremap, keypoint_vis21)
        t3                  = time.time() # loss
        loss0, loss1, loss2 = other_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        t4 = time.time() # backward

        loss_dict = {
            "loss"  : loss.item(),
            "loss0" : loss0.item(),
            "loss1" : loss1.item(),
            "loss2" : loss2.item(),
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
        img, _, uv_gt, vis  = data_load
        img                 = img.cuda()
        out                 = self.net(img)
        pred_smap           = F.interpolate(out[-1], 
                                            size=(self.img_size, self.img_size),
                                            mode='bilinear')
        pred_smap           = pred_smap.cpu().numpy()
        uv_gt               = uv_gt.cpu().numpy()
        vis                 = vis.cpu().numpy()
        vis                 = vis.astype('uint8')
        
        for batch in range(img.shape[0]):
            cur_pred_smap   = pred_smap[batch]
            cur_pred_smap   = IMG.torchshape2img(cur_pred_smap)
            cur_uv_gt       = uv_gt[batch]
            cur_vis         = vis[batch]
            pred_uv         = RHD.detect_keypoints_from_scoremap(cur_pred_smap)
            if len(pred_uv[cur_vis]) == 0 or len(cur_uv_gt[cur_vis]) == 0:
                self.val_uv_21_error.append(0.0)
                continue
            self.val_uv_21_error.append(np.sqrt(np.sum(np.square(
                cur_uv_gt[cur_vis]-pred_uv[cur_vis]), axis=-1) + 1e-8 ))
            self.vis_list.append(cur_vis)

    def get_valid_loss(self):
        eps             = 1e-5
        val_uv_l2_error = np.mean(self.val_uv_21_error)
        val_uv_squeeze  = np.squeeze(np.asarray(self.val_uv_21_error))
        
        data        = []
        for i in range(21):
            data.append([])

        for i in range(len(val_uv_squeeze)):
            cur_vis = self.vis_list[i]
            for j in range(21):
                if cur_vis[j]:
                    data[j].append(val_uv_squeeze[i, j])
        data = np.asarray(data)
        
        pck             = RHD.get_pck_with_vis(data)
        thresholds      = np.arange(0, 85, 5)
        auc             = RHD.calc_auc(pck, thresholds)

        val_loss_dict = {
            'uv_l2_error'   : val_uv_l2_error,
            'AUC_0_85'      : auc,
        }
        
        self.val_uv_21_error    = []
        self.vis_list           = []
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        img, _, uv_gt, _    = data_load
        img                 = img.cuda()
        out                 = self.net(img)
        pred_smap           = F.interpolate(out[-1], 
                                            size=(self.img_size, self.img_size),
                                            mode='bilinear')
        pred_smap           = pred_smap.cpu().numpy()
        uv_gt               = uv_gt.cpu().numpy()

        for batch in range(img.shape[0]):
            cur_pred_smap   = pred_smap[batch]
            cur_pred_smap   = IMG.torchshape2img(cur_pred_smap)
            pred_uv         = RHD.detect_keypoints_from_scoremap(cur_pred_smap)
            self.pred_list.append(pred_uv)

    def save_predictions(self, data_split):
        pred_save = "predict_{}_{}_uv.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.pred_list, (-1, 42)))
        self.pred_list = []

    # ========================================================
    # DETECTION
    # ========================================================

    def detect(self, img):
        with torch.no_grad():
            img         = img.cuda()
            out         = self.net(img)
            pred_smap   = F.interpolate(out[-1], 
                                        size=(self.img_size, self.img_size),
                                        mode='bilinear')
            pred_smap   = pred_smap.cpu().numpy()
            pred_smap   = IMG.torchshape2img(pred_smap[0])
            pred_uv     = RHD.detect_keypoints_from_scoremap(pred_smap)
            return pred_smap, pred_uv

