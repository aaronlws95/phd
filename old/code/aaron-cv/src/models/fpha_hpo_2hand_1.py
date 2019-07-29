import torch
import random
import torch.nn                     as nn
import numpy                        as np
from pathlib                        import Path
from tqdm                           import tqdm
from PIL                            import Image

from src.models                     import FPHA_HPO
from src.utils                      import YOLO, IMG, FPHA
from src.loss                       import get_loss
from src.datasets                   import get_dataset, get_dataloader

class FPHA_HPO_2Hand_1(FPHA_HPO):
    """ Hand keypoint estimation with HPO method with two hand regression from 
    two separate conv output layers """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)
        # Training
        if self.training:
            dataset_kwargs  = {'split_set': cfg['train_set']}
            train_dataset   = get_dataset(cfg, dataset_kwargs)
            self.train_sampler  = None
            shuffle             = cfg['shuffle']
            kwargs = {'batch_size'  : int(cfg['batch_size']),
                      'shuffle'     : shuffle,
                      'num_workers' : int(cfg['num_workers']),
                      'pin_memory'  : True,
                      'collate_fn'  : self.collate_fn_flip}
            self.train_loader = get_dataloader(train_dataset, 
                                               self.train_sampler,
                                               kwargs)
            # Validation
            dataset_kwargs              = {'split_set': cfg['val_set']}
            val_dataset                 = get_dataset(cfg, dataset_kwargs)
            self.val_loader             = get_dataloader(val_dataset, 
                                                         None,
                                                         kwargs)
        # Prediction
        else:
            self.pred_img_side          = cfg['pred_img_side']
            self.pred_uvd_list_left     = []
            self.pred_conf_left         = []
            self.pred_uvd_list_right    = []
            self.pred_conf_right        = []

    # ========================================================
    # DATA LOADER UTILITIES
    # ========================================================

    def collate_fn_flip(self, batch):
        """
        Flip entire dataset batch to same side
        Args:
            batch   : list of img, bbox_gt, uvd_gt
            img     : [img_1, ..., img_batch]
            bbox_gt : [bbox_gt_1, ..., bbox_gt_batch]
            uvd_gt  : [uvd_gt_1, ..., uvd_gt_batch]
        Out:
            Vertically mirrored inputs
        """
        FT          = torch.FloatTensor
        img, uvd_gt = zip(*batch)
        flip        = random.randint(1, 10000)%2
        # Do flipping
        # 0 = left, 1 = right
        hand_side = 1
        if flip:
            hand_side = 0  

        new_img     = []
        new_uvd     = []
        for i, u in batch:
            if flip:
                i       = i.transpose(Image.FLIP_LEFT_RIGHT)
                u[:, 0] = 0.999 - u[:, 0]
            i = np.asarray(i)
            i = i/255.0
            i = IMG.imgshape2torch(i)
            new_img.append(i)
            new_uvd.append(u)
            
        new_img     = FT(new_img)
        new_uvd     = FT(new_uvd)
        return new_img, new_uvd, hand_side

    def collate_fn_no_flip(self, batch):
        """
        Doesn't flip entire dataset batch to same side
        Args:
            batch   : list of img, bbox_gt, uvd_gt
            img     : [img_1, ..., img_batch]
            bbox_gt : [bbox_gt_1, ..., bbox_gt_batch]
            uvd_gt  : [uvd_gt_1, ..., uvd_gt_batch]
        Out:
            Inputs in correct format for training
        """
        FT          = torch.FloatTensor
        img, uvd_gt = zip(*batch)

        new_img     = []
        new_uvd     = []
        for i, u in batch:
            if self.pred_img_side == 'left':
                i = i.transpose(Image.FLIP_LEFT_RIGHT)
                u[:, 0] = 0.999 - u[:, 0]
            i = np.asarray(i)
            i = i/255.0
            i = IMG.imgshape2torch(i)
            new_img.append(i)
            new_uvd.append(u)
            
        new_img     = FT(new_img)
        new_uvd     = FT(new_uvd)
        return new_img, new_uvd

    # ========================================================
    # LOADING
    # ========================================================

    def load_weights(self, load_epoch):
        if load_epoch == 0:
            if self.pretrain:
                if 'state' in self.pretrain:
                    load_dir = Path(self.data_dir)/self.pretrain
                    ckpt = self.load_ckpt(load_dir)
                    # Load pretrained model with non-exact layers
                    state_dict = ckpt['model_state_dict']
                    cur_dict = self.net.state_dict()
                    # Filter out unnecessary keys
                    pretrained_dict = \
                        {k: v for k, v in state_dict.items() if k in cur_dict 
                         and k != 'module_list.30.conv_30.weight'
                         and k != 'module_list.30.conv_30.bias'}
                    # Overwrite entries in the existing state dict
                    cur_dict.update(pretrained_dict)
                    # Load the new state dict
                    self.net.load_state_dict(cur_dict)
                else:
                    pt_path = str(Path(self.data_dir)/self.pretrain)
                    YOLO.load_darknet_weights(self.net, pt_path)
        else:
            load_dir = Path(self.save_ckpt_dir)/f'model_{load_epoch}.state'
            ckpt = self.load_ckpt(load_dir)
            self.net.load_state_dict(ckpt['model_state_dict'])
            if self.training:
                self.optimizer.load_state_dict(
                    ckpt['optimizer_state_dict'])
                if self.scheduler:
                    self.scheduler.load_state_dict(
                        ckpt['scheduler_state_dict'])

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, uvd_gt, hand_side  = data_load
        batch_size              = img.shape[0]
        img                     = img.cuda()
        uvd_gt                  = uvd_gt.cuda()
        hand_left, hand_right   = self.net(img)
        
        hand_out_to_loss = hand_left if hand_side == 0 else hand_right
        
        loss, *hand_losses      = self.loss(hand_out_to_loss, uvd_gt)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_u, loss_v, loss_d, loss_conf = hand_losses
        hand_side_str = 'left' if hand_side == 0 else 'right'
        loss_dict = {
            'loss'          : loss.item(),
            'loss_u'        : loss_u.item(),
            'loss_v'        : loss_v.item(),
            'loss_d'        : loss_d.item(),
            'loss_conf'     : loss_conf.item(),
            'hand_side'     : hand_side_str
        }

        return loss_dict

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img, uvd_gt, hand_side  = data_load
        uvd_gt                  = uvd_gt.numpy()
        batch_size              = img.shape[0]
        img                     = img.cuda()
        hand_left, hand_right   = self.net(img)
        
        pred_hand = hand_left if hand_side == 0 else hand_right
        
        W                       = pred_hand.shape[2]
        H                       = pred_hand.shape[3]
        D                       = 5
        pred_hand               = pred_hand.view(batch_size, 64, D, H, W)
        pred_hand               = pred_hand.permute(0, 1, 3, 4, 2)

        # Hand
        uvd_gt                  = IMG.scale_points_WH(uvd_gt, (1, 1),
                                                      (FPHA.ORI_WIDTH, 
                                                       FPHA.ORI_HEIGHT))
        uvd_gt[..., 2]          *= FPHA.REF_DEPTH
        xyz_gt                  = FPHA.uvd2xyz_color(uvd_gt)

        for batch in range(batch_size):
            cur_pred_hand   = pred_hand[batch]
            pred_uvd        = cur_pred_hand[:63, :, :, :].view(21, 3, H, W, D)
            pred_conf       = torch.sigmoid(cur_pred_hand[63, :, :, :])

            FT              = torch.FloatTensor
            yv, xv, zv      = torch.meshgrid([torch.arange(H),
                                              torch.arange(W),
                                              torch.arange(D)])
            grid_x          = xv.repeat((21, 1, 1, 1)).type(FT).cuda()
            grid_y          = yv.repeat((21, 1, 1, 1)).type(FT).cuda()
            grid_z          = zv.repeat((21, 1, 1, 1)).type(FT).cuda()

            pred_uvd[self.hand_root, :, :, :, :] = \
                torch.sigmoid(pred_uvd[self.hand_root, :, :, :, :])
            pred_uvd[:, 0, :, :, :] = (pred_uvd[:, 0, :, :, :] + grid_x)/W
            pred_uvd[:, 1, :, :, :] = (pred_uvd[:, 1, :, :, :] + grid_y)/H
            pred_uvd[:, 2, :, :, :] = (pred_uvd[:, 2, :, :, :] + grid_z)/D

            pred_uvd                = pred_uvd.contiguous().view(21, 3, -1)
            pred_conf               = pred_conf.contiguous().view(-1)

            top_idx                 = torch.topk(pred_conf, 1)[1]
            best_pred_uvd           = pred_uvd[:, :, top_idx].squeeze().cpu().numpy()
            best_pred_uvd           = IMG.scale_points_WH(best_pred_uvd, 
                                                          (1, 1),
                                                          (FPHA.ORI_WIDTH, 
                                                           FPHA.ORI_HEIGHT))
            best_pred_uvd[..., 2]   *= FPHA.REF_DEPTH
            best_pred_xyz           = FPHA.uvd2xyz_color(best_pred_uvd)
            cur_xyz_gt              = xyz_gt[batch]

            self.val_xyz_21_error.append(np.sqrt(np.sum(np.square(
                best_pred_xyz-cur_xyz_gt), axis=-1) + 1e-8 ))

    def get_valid_loss(self):
        eps                 = 1e-5
        val_xyz_l2_error    = np.mean(self.val_xyz_21_error)
        val_xyz_squeezed    = np.squeeze(np.asarray(self.val_xyz_21_error))
        pck                 = FPHA.get_pck(val_xyz_squeezed)
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

    def predict(self, cfg, split):
        for data_split in split:
            data_set        = data_split + '_set'
            dataset_kwargs  = {'split_set': cfg[data_set]}
            cfg['aug']      = None
            dataset         = get_dataset(cfg, dataset_kwargs)
            sampler         = None
            kwargs          = {'batch_size'     :   int(cfg['batch_size']),
                               'shuffle'        :   False,
                               'num_workers'    :   int(cfg['num_workers']),
                               'pin_memory'     :   True,
                               'collate_fn'     :   self.collate_fn_no_flip}
            data_loader     = get_dataloader(dataset, sampler, kwargs)        
            
            self.net.eval()
            with torch.no_grad():
                for data_load in tqdm(data_loader):
                    self.predict_step(data_load)
                self.save_predictions(data_split)

    def predict_step(self, data_load):
        img                     = data_load[0]
        img                     = img.cuda()
        hand_left, hand_right   = self.net(img)
        batch_size              = img.shape[0]
        W                       = hand_left.shape[2]
        H                       = hand_left.shape[3]
        D                       = 5
        hand_left               = hand_left.view(batch_size, 64, D, H, W)
        hand_left               = hand_left.permute(0, 1, 3, 4, 2)
        hand_right              = hand_right.view(batch_size, 64, D, H, W)
        hand_right              = hand_right.permute(0, 1, 3, 4, 2)  
        
        for batch in range(batch_size):
            # Left hand
            cur_pred_hand   = hand_left[batch]
            pred_uvd        = cur_pred_hand[:63, :, :, :].view(21, 3, H, W, D)
            pred_conf       = torch.sigmoid(cur_pred_hand[63, :, :, :])

            FT              = torch.FloatTensor
            yv, xv, zv      = torch.meshgrid([torch.arange(H),
                                              torch.arange(W),
                                              torch.arange(D)])
            grid_x          = xv.repeat((21, 1, 1, 1)).type(FT).cuda()
            grid_y          = yv.repeat((21, 1, 1, 1)).type(FT).cuda()
            grid_z          = zv.repeat((21, 1, 1, 1)).type(FT).cuda()

            pred_uvd[self.hand_root, :, :, :, :] = \
                torch.sigmoid(pred_uvd[self.hand_root, :, :, :, :])
            pred_uvd[:, 0, :, :, :] = (pred_uvd[:, 0, :, :, :] + grid_x)/W
            pred_uvd[:, 1, :, :, :] = (pred_uvd[:, 1, :, :, :] + grid_y)/H
            pred_uvd[:, 2, :, :, :] = (pred_uvd[:, 2, :, :, :] + grid_z)/D


            pred_uvd        = pred_uvd.contiguous().view(21, 3, -1)
            pred_conf       = pred_conf.contiguous().view(-1)
            top_idx         = torch.topk(pred_conf, 1)[1]
            best_pred_uvd   = pred_uvd[:, :, top_idx].cpu().numpy()
            
            self.pred_uvd_list_left.append(best_pred_uvd)
            self.pred_conf_left.append(pred_conf.cpu().numpy())
            
            # Right Hand
            cur_pred_hand = hand_right[batch]
            pred_uvd = cur_pred_hand[:63, :, :, :].view(21,
                                                        3,
                                                        H,
                                                        W,
                                                        D)
            pred_conf = torch.sigmoid(cur_pred_hand[63, :, :, :])
            
            FT          = torch.FloatTensor
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
            best_pred_uvd   = pred_uvd[:, :, top_idx].cpu().numpy()
            
            self.pred_uvd_list_right.append(best_pred_uvd)
            self.pred_conf_right.append(pred_conf.cpu().numpy())
            
    def save_predictions(self, data_split):
    
        pred_save = "predict_{}_{}_uvd_left.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.pred_uvd_list_left, (-1, 63)))
        
        pred_save = "predict_{}_{}_conf_left.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_conf_left)
        
        pred_save = "predict_{}_{}_uvd_right.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.pred_uvd_list_right, (-1, 63)))
        
        pred_save = "predict_{}_{}_conf_right.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_conf_right)
        
        self.pred_uvd_list_left     = []
        self.pred_conf_left         = []
        self.pred_uvd_list_right    = []
        self.pred_conf_right        = []
        
    # ========================================================
    # DETECTION
    # ========================================================
    
    def detect(self, img):
        with torch.no_grad():
            FT          = torch.FloatTensor
            img         = np.asarray(img.copy())
            ori_w       = img.shape[1]
            ori_h       = img.shape[0]
            img         = IMG.resize_img(img, (416, 416))
            img         = img/255.0
            img         = IMG.imgshape2torch(img)
            img         = np.expand_dims(img, 0)
            img         = FT(img)
            img         = img.cuda()
            hand_left, hand_right  = self.net(img)
            W           = hand_left.shape[3]
            H           = hand_left.shape[2]
            D           = 5
            batch_size  = 1
            
            # Left
            hand_left   = hand_left.view(batch_size, 64, D, H, W)
            hand_left   = hand_left.permute(0, 1, 3, 4, 2)
            hand_left   = hand_left[0]
            
            pred_uvd    = hand_left[:63, :, :, :].view(21, 3, H, W, D)
            pred_conf   = torch.sigmoid(hand_left[63, :, :, :])

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
            best_pred_left   = pred_uvd[:, :, top_idx].squeeze().cpu().numpy()
            best_pred_left   = IMG.scale_points_WH(best_pred_left, 
                                                (1, 1),
                                                (ori_w, ori_h))

            best_pred_left[..., 2]   *= FPHA.REF_DEPTH
            
            # Right
            hand_right   = hand_right.view(batch_size, 64, D, H, W)
            hand_right   = hand_right.permute(0, 1, 3, 4, 2)
            hand_right   = hand_right[0]
            
            pred_uvd    = hand_right[:63, :, :, :].view(21, 3, H, W, D)
            pred_conf   = torch.sigmoid(hand_right[63, :, :, :])

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
            best_pred_right = pred_uvd[:, :, top_idx].squeeze().cpu().numpy()
            best_pred_right = IMG.scale_points_WH(best_pred_right, 
                                                (1, 1),
                                                (ori_w, ori_h))

            best_pred_right[..., 2]   *= FPHA.REF_DEPTH            
            
            return best_pred_left, best_pred_right     