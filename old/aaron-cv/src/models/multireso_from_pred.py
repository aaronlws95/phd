import torch
import os
import time
import torch.nn             as nn
import numpy                as np
import torch.nn.functional  as F
from PIL                    import Image
from pathlib                import Path
from tqdm                   import tqdm

from src.models             import Model
from src.utils              import IMG, FPHA, YOLO
from src.datasets           import get_dataset, get_dataloader
from src.components         import get_optimizer, get_scheduler, Network, Multireso_net

class Multireso_from_pred(Model):
    """ PREDICTION ONLY MODEL. Predict hand keypoints from cropped hand 
    using bbox detection """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)
        assert self.training == False, "Does not support training"
        
        cur_path            = Path(__file__).resolve().parents[2] 
        net0_cfgfile        = cur_path/'net_cfg'/(cfg['net0_cfg'] + '.cfg')
        net1_cfgfile        = cur_path/'net_cfg'/(cfg['net1_cfg'] + '.cfg')
        net2_cfgfile        = cur_path/'net_cfg'/(cfg['net2_cfg'] + '.cfg')
        dense_cfgfile       = cur_path/'net_cfg'/(cfg['dense_net_cfg'] + '.cfg')
        bbox_cfgfile        = cur_path/'net_cfg'/(cfg['bbox_net_cfg'] + '.cfg')
        self.mr_net         = Multireso_net(net0_cfgfile, net1_cfgfile,
                                            net2_cfgfile, dense_cfgfile)
        self.mr_net         = self.mr_net.cuda()
        self.bbox_net       = Network(bbox_cfgfile).cuda()
        self.mr_weights     = cfg['mr_weights']
        self.bbox_weights   = cfg['bbox_weights']
        self.img_rsz        = int(cfg['img_rsz'])
        
        # IMPORTANT TO LOAD WEIGHTS
        self.load_weights(self.load_epoch)
        
        self.pred_list          = []
        self.xy_offset_list     = []
        self.pred_conf_thresh   = float(cfg['pred_conf_thresh'])
        self.pred_nms_thresh    = float(cfg['pred_nms_thresh'])
        self.anchors            = [float(i) for i in cfg["anchors"].split(',')]
        self.num_anchors        = len(self.anchors)//2
        
    # ========================================================
    # LOADING
    # ========================================================
    def load_weights(self, load_epoch):
        load_dir    = Path(self.data_dir)/self.mr_weights
        ckpt        = self.load_ckpt(load_dir)
        self.mr_net.load_state_dict(ckpt['model_state_dict'])
        
        load_dir    = Path(self.data_dir)/self.bbox_weights
        ckpt        = self.load_ckpt(load_dir)
        self.bbox_net.load_state_dict(ckpt['model_state_dict'])
            
            
    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        # Doesn't train
        pass 

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        pass

    def get_valid_loss(self):
        pass

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
                               'pin_memory'     :   True}
            data_loader     = get_dataloader(dataset, sampler, kwargs)        
            
            self.bbox_net.eval()
            self.mr_net.eval()
            with torch.no_grad():
                for data_load in tqdm(data_loader):
                    self.predict_step(data_load)
                self.save_predictions(data_split)

    def predict_step(self, data_load):
        img         = data_load[0]
        img         = img.cuda()
        pred        = self.bbox_net(img)
        batch_size  = img.shape[0]
        pred        = pred[0].cpu()
        max_boxes   = 1 # detect at most 1 box (FPHA only has 1 bbox)
        FT          = torch.FloatTensor
        batch_boxes = YOLO.get_region_boxes(pred,
                                            self.pred_conf_thresh,
                                            0,
                                            self.anchors,
                                            self.num_anchors,
                                            is_cuda = False,
                                            is_time = self.time)
        
        img         = img.cpu().numpy()
        img         = np.swapaxes(img, 2, 3)
        img         = np.swapaxes(img, 1, 3)
        img         = IMG.scale_img_255(img)
        
        img0_batch  = []
        img1_batch  = []
        img2_batch  = []
        for batch in range(batch_size):
            boxes       = batch_boxes[batch]
            boxes       = YOLO.nms_torch(boxes, self.pred_nms_thresh)

            all_boxes   = np.zeros((max_boxes, 5))
            if len(boxes) != 0:

                if len(boxes) > len(all_boxes):
                    fill_range = len(all_boxes)
                else:
                    fill_range = len(boxes)

                for i in range(fill_range):
                    box             = boxes[i]
                    all_boxes[i]    = (float(box[0]), float(box[1]),
                                       float(box[2]), float(box[3]), 
                                       float(box[4]))
                    
            all_boxes       = np.reshape(all_boxes, -1)
            all_boxes[0]    *= img[batch].shape[1]
            all_boxes[1]    *= img[batch].shape[0]
            all_boxes[2]    *= img[batch].shape[1]
            all_boxes[3]    *= img[batch].shape[0]
            
            x_min           = int(all_boxes[0] - all_boxes[2]/2)
            y_min           = int(all_boxes[1] - all_boxes[3]/2)
            crop_w          = all_boxes[2]
            crop_h          = all_boxes[3]
            
            self.xy_offset_list.append((x_min, y_min, crop_w, crop_h))
            
            crop = FPHA.crop_hand_from_bbox(img[batch], all_boxes)

            crop = Image.fromarray(crop)
            crop = crop.resize((self.img_rsz, self.img_rsz))
            img0 = crop
            img1 = crop.resize((self.img_rsz//2, self.img_rsz//2))
            img2 = crop.resize((self.img_rsz//4, self.img_rsz//4))
            img_list = [img0, img1, img2]
            
            for i in range(len(img_list)):
                img_list[i] = np.asarray(img_list[i])
                img_list[i] = (img_list[i]/255.0)
                img_list[i] = IMG.imgshape2torch(img_list[i])
            
            img0_batch.append(img_list[0])
            img1_batch.append(img_list[1])
            img2_batch.append(img_list[2])
        
        img0_batch = FT(img0_batch).cuda()
        img1_batch = FT(img1_batch).cuda()
        img2_batch = FT(img2_batch).cuda()
        
        img_list_batch = [img0_batch, img1_batch, img2_batch]

        out                             = self.mr_net(img_list_batch)
        pred_uvd                        = out.cpu().numpy()

        for batch in range(pred_uvd.shape[0]):
            self.pred_list.append(pred_uvd[batch])
            
    def save_predictions(self, data_split):
        pred_save = 'predict_{}_uvd_from_pred.txt'.format(data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.pred_list, (-1, 63)))

        file_name = 'xy_offset_{}_from_pred.txt'.format(data_split)
        file_name = Path(self.data_dir)/self.exp_dir/file_name
        np.savetxt(file_name, np.reshape(self.xy_offset_list, (-1, 4)))

        self.pred_list = []
        self.xy_offset_list = []
