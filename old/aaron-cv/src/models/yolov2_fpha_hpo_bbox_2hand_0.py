import torch
import time
import random
import torch.nn     as nn
import numpy        as np
from pathlib        import Path
from tqdm           import tqdm
from PIL            import Image

from src.models     import YOLOV2_FPHA_HPO_Bbox
from src.utils      import YOLO, IMG, FPHA 
from src.datasets   import get_dataset, get_dataloader

class YOLOV2_FPHA_HPO_Bbox_2Hand_0(YOLOV2_FPHA_HPO_Bbox):
    """ YOLOv2 bounding box detection and hand regression from HPO method 
    with two hand regression from one conv output layer """
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
            self.pred_uvd_list_left  = []
            self.pred_conf_left      = []
            self.pred_uvd_list_right = []
            self.pred_conf_right     = []

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
        FT                      = torch.FloatTensor
        img, bbox_gt, uvd_gt    = zip(*batch)
        flip                    = random.randint(1, 10000)%2
        # Do flipping
        # 0 = left, 1 = right
        hand_side = 1
        if flip:
            hand_side = 0  

        new_img     = []
        new_bbox    = []
        new_uvd     = []
        for i, b, u in batch:
            if flip:
                i       = i.transpose(Image.FLIP_LEFT_RIGHT)
                b[0]    = 0.999 - b[0]
                u[:, 0] = 0.999 - u[:, 0]
            i = np.asarray(i)
            i = i/255.0
            i = IMG.imgshape2torch(i)
            new_img.append(i)
            new_bbox.append(b)
            new_uvd.append(u)
            
        new_img     = FT(new_img)
        new_bbox    = FT(new_bbox)
        new_uvd     = FT(new_uvd)
        return new_img, new_bbox, new_uvd, hand_side

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
        FT                      = torch.FloatTensor
        img, bbox_gt, uvd_gt    = zip(*batch)

        new_img     = []
        new_bbox    = []
        new_uvd     = []
        for i, b, u in batch:
            i = np.asarray(i)
            i = i/255.0
            i = IMG.imgshape2torch(i)
            new_img.append(i)
            new_bbox.append(b)
            new_uvd.append(u)
            
        new_img     = FT(new_img)
        new_bbox    = FT(new_bbox)
        new_uvd     = FT(new_uvd)
        return new_img, new_bbox, new_uvd

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        t0                                  = time.time() # start
        img, bbox_gt, uvd_gt, hand_side     = data_load
        batch_size                          = img.shape[0]
        img                                 = img.cuda()
        bbox_gt                             = bbox_gt.cuda()
        uvd_gt                              = uvd_gt.cuda()
        t1                                  = time.time() # CPU to GPU
        
        bbox_out, hand_out                  = self.net(img) 
        t2                                  = time.time() # forward
        
        hand_out_left                       = hand_out[:, :320, :, :] 
        hand_out_right                      = hand_out[:, 320:, :, :] 
        
        hand_out_to_loss = hand_out_left if hand_side == 0 else hand_out_right
            
        loss_bbox, *bbox_losses = self.loss(bbox_out, bbox_gt)
        loss_hand, *hand_losses = self.hand_loss(hand_out_to_loss, uvd_gt)
        loss                    = loss_bbox + loss_hand
        t3                      = time.time() # loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        t4 = time.time() #  backward
        
        loss_x, loss_y, loss_w, loss_h, loss_conf   = bbox_losses
        loss_u, loss_v, loss_d, loss_hand_conf      = hand_losses
        hand_side_str = 'left' if hand_side == 0 else 'right'
        loss_dict = {
            'loss'          : loss.item(),
            'loss_x'        : loss_x.item(),
            'loss_y'        : loss_y.item(),
            'loss_w'        : loss_w.item(),
            'loss_h'        : loss_h.item(),
            'loss_conf'     : loss_conf.item(),
            'loss_u'        : loss_u.item(),
            'loss_v'        : loss_v.item(),
            'loss_d'        : loss_d.item(),
            'loss_hand_conf': loss_hand_conf.item(),
            'hand_side'     : hand_side_str
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
        img, bbox_gt, uvd_gt, hand_side     = data_load
        target                              = bbox_gt
        uvd_gt                              = uvd_gt.numpy()
        batch_size                          = img.shape[0]
        img                                 = img.cuda()
        
        pred_bbox, hand_out     = self.net(img)
        hand_out_left           = hand_out[:, :320, :, :] 
        hand_out_right          = hand_out[:, 320:, :, :] 
        
        pred_hand = hand_out_left if hand_side == 0 else hand_out_right
        
        pred_bbox               = pred_bbox.cpu()
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
        
        # Bbox
        all_boxes               = YOLO.get_region_boxes(pred_bbox,
                                                        self.val_conf_thresh,
                                                        0,
                                                        self.anchors,
                                                        self.num_anchors)

        for batch in range(batch_size):
            self.val_total += 1
            # Hand
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
            
            # if self.debug:
            #     check = torch.zeros(pred_uvd.shape)
            #     check[:, :, 12, 12, 4] = 1
            #     check_conf = torch.zeros(pred_conf.shape)
            #     check_conf[12, 12, 4] = 1
            #     check = check.contiguous().view(21, 3, -1)
            #     check_conf = check_conf.contiguous().view(-1)
            #     top_idx = torch.topk(check_conf, 1)[1]
            #     best_check = check[:, :, top_idx]
            #     print(best_check)
                
            pred_uvd                = pred_uvd.contiguous().view(21, 3, -1)
            pred_conf               = pred_conf.contiguous().view(-1)
            
            top_idx                 = torch.topk(pred_conf, 1)[1]
            best_pred_uvd           = pred_uvd[:, :, top_idx]
            best_pred_uvd           = best_pred_uvd.squeeze().cpu().numpy()
            best_pred_uvd           = IMG.scale_points_WH(best_pred_uvd, 
                                                          (1, 1), 
                                                          (FPHA.ORI_WIDTH, 
                                                           FPHA.ORI_HEIGHT))
            best_pred_uvd[..., 2]   *= FPHA.REF_DEPTH
            best_pred_xyz           = FPHA.uvd2xyz_color(best_pred_uvd)
            cur_xyz_gt              = xyz_gt[batch]
                
            self.val_xyz_21_error.append(np.sqrt(np.sum(np.square(
                best_pred_xyz-cur_xyz_gt), axis=-1) + 1e-8))
                  
            # Bbox
            boxes       = all_boxes[batch]
            boxes       = YOLO.nms_torch(boxes, self.val_nms_thresh)
            cur_target  = target[batch]
            
            for i in range(len(boxes)):
                if boxes[i][4] > self.val_conf_thresh:
                    self.val_proposals += 1

            box_gt      = [float(cur_target[0]), float(cur_target[1]), 
                           float(cur_target[2]), float(cur_target[3]), 1.0]
            best_iou    = 0
            best_j      = -1
            for j in range(len(boxes)):
                iou             = YOLO.bbox_iou(box_gt, boxes[j])
                best_iou        = iou
                best_j          = j
                self.avg_iou    += iou
                self.iou_total  += 1
            if best_iou > self.val_iou_thresh:
                self.val_correct += 1
                    
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
        pred_bbox, hand_out     = self.net(img)
        hand_out_left           = hand_out[:, :320, :, :] 
        hand_out_right          = hand_out[:, 320:, :, :] 
        batch_size              = img.shape[0]
        max_boxes               = 1 # detect at most 1 box 
        W                       = hand_out.shape[2]
        H                       = hand_out.shape[3]
        D                       = 5
        hand_out_left           = hand_out_left.view(batch_size, 64, D, H, W)
        hand_out_left           = hand_out_left.permute(0, 1, 3, 4, 2)
        hand_out_right          = hand_out_right.view(batch_size, 64, D, H, W)
        hand_out_right          = hand_out_right.permute(0, 1, 3, 4, 2)                      
        batch_boxes             = YOLO.get_region_boxes(pred_bbox.cpu(), 
                                                        self.pred_conf_thresh, 
                                                        0, 
                                                        self.anchors, 
                                                        self.num_anchors, 
                                                        is_cuda = False,
                                                        is_time = self.time)

        for batch in range(batch_size):
            # Left Hand
            cur_pred_hand = hand_out_left[batch]
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
            
            self.pred_uvd_list_left.append(best_pred_uvd)
            self.pred_conf_left.append(pred_conf.cpu().numpy())

            # Right Hand
            cur_pred_hand = hand_out_right[batch]
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

            # Bbox
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
            all_boxes = np.reshape(all_boxes, -1)
            self.pred_list.append(all_boxes)

    def save_predictions(self, data_split):
        pred_save = "predict_{}_{}_bbox.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_list)
        
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
        
        self.pred_list              = []
        self.pred_uvd_list_left     = []
        self.pred_conf_left         = []
        self.pred_uvd_list_right    = []
        self.pred_conf_right        = []