import torch
import os
import time
import h5py
import torch.nn                     as nn
import numpy                        as np
import torch.nn.functional          as F
from pathlib                        import Path
from tqdm                           import tqdm

from src.models                     import Model
from src.utils                      import IMG, RHD
from src.datasets                   import get_dataset, get_dataloader

class ZNB_Handseg(Model):
    """ ZNB HandSegNet predict hand mask from image """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)

        # IMPORTANT TO LOAD WEIGHTS
        self.load_weights(self.load_epoch)

        self.img_size       = int(cfg['img_size'])
        self.loss           = nn.BCEWithLogitsLoss()

        # Training
        if self.training:
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
            self.iou_list           = []
        # Prediction
        else:
            self.pred_list          = []

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, mask           = data_load

        t0                  = time.time()   # start
        img                 = img.cuda()
        mask                = mask.cuda()   # [b, 2, 256, 256]
        t1                  = time.time()   # CPU to GPU

        out                 = self.net(img)[0] # [b, 2, 32, 32]
        out                 = F.interpolate(out, size=(self.img_size,
                                                       self.img_size),
                                            mode='bilinear')
        t2                  = time.time() # forward

        loss                = self.loss(out, mask)
        t3                  = time.time() # loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        t4 = time.time() # backward

        loss_dict = {
            "loss"  : loss.item()
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
        img, mask           = data_load
        img                 = img.cuda()   
        mask                = mask.numpy()
        out                 = self.net(img)[0] # [b, 2, 32, 32]
        out                 = F.interpolate(out, size=(self.img_size,
                                                       self.img_size),
                                            mode='bilinear')
        out                 = out.cpu().numpy()
        
        for batch in range(len(out)):
            cur_pred                = out[batch][:, :, 1]
            cur_pred[cur_pred > 0]  = 1
            cur_pred[cur_pred <= 0] = 0
            cur_gt                  = mask[batch][:, :, 1]
            cur_gt[cur_gt > 0]      = 1
            cur_gt[cur_gt <= 0]     = 0
            intersection            = np.logical_and(cur_gt, cur_pred)
            union                   = np.logical_or(cur_gt, cur_pred)
            iou_score               = np.sum(intersection)/np.sum(union)
            self.iou_list.append(iou_score)

    def get_valid_loss(self):
        val_loss_dict = {
            'avg_iou': np.mean(self.iou_list)
        }
        
        self.iou_list = []
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        img, _              = data_load
        img                 = img.cuda()    
        out                 = self.net(img)[0] # [b, 2, 32, 32]
        out                 = F.interpolate(out, size=(self.img_size,
                                                       self.img_size),
                                            mode='bilinear')
        out                 = out.permute(0, 2, 3, 1)
        out                 = out.cpu().numpy()
        
        for o in out:
            self.pred_list.append(o)

    def save_predictions(self, data_split):
        pred_save       = "predict_{}_{}_mask.h5".format(self.load_epoch,
                                                         data_split)
        pred_file       = Path(self.data_dir)/self.exp_dir/pred_save
        f               = h5py.File(pred_file, 'w')

        f.create_dataset('mask', data=self.pred_list)
        self.pred_list  = []


