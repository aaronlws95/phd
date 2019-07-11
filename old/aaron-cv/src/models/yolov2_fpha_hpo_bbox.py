import torch
import time
import torch.nn as nn
import numpy    as np
from pathlib    import Path
from tqdm       import tqdm

from src.models import YOLOV2_FPHA
from src.utils  import YOLO, IMG, FPHA 
from src.loss   import get_loss

class YOLOV2_FPHA_HPO_Bbox(YOLOV2_FPHA):
    """ YOLOv2 bounding box detection and hand regression from HPO method """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)

        self.hand_loss = get_loss(cfg['hand_loss'], cfg)
        self.hand_root = int(cfg['hand_root'])
        
        # Training
        if self.training:
            self.val_xyz_21_error       = []
        # Prediction
        else:
            self.best_pred_uvd_list     = []
            self.topk_pred_uvd_list     = []
            self.pred_conf_list         = []

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        t0                      = time.time() # start
        img, bbox_gt, uvd_gt    = data_load
        batch_size              = img.shape[0]
        img                     = img.cuda()
        bbox_gt                 = bbox_gt.cuda()
        uvd_gt                  = uvd_gt.cuda()
        t1                      = time.time() # CPU to GPU
        
        bbox_out, hand_out      = self.net(img) 
        t2 = time.time() # forward
        loss_bbox, *bbox_losses = self.loss(bbox_out, bbox_gt)
        loss_hand, *hand_losses = self.hand_loss(hand_out, uvd_gt)
        loss                    = loss_bbox + loss_hand
        t3                      = time.time() # loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        t4 = time.time() #  backward
        
        loss_x, loss_y, loss_w, loss_h, loss_conf   = bbox_losses
        loss_u, loss_v, loss_d, loss_hand_conf      = hand_losses
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
        img, bbox_gt, uvd_gt    = data_load
        target                  = bbox_gt
        uvd_gt                  = uvd_gt.numpy()
        batch_size              = img.shape[0]
        img                     = img.cuda()
        
        pred_bbox, pred_hand    = self.net(img)
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

    def get_valid_loss(self):
        eps                 = 1e-5
        precision           = 1.0*self.val_correct/(self.val_proposals + eps)
        recall              = 1.0*self.val_correct/(self.val_total + eps)
        f1score             = 2.0*precision*recall/(precision+recall + eps)
        avg_iou             = self.avg_iou/(self.iou_total + eps)
        
        val_xyz_l2_error    = np.mean(self.val_xyz_21_error)
        val_xyz_squeeze     = np.squeeze(np.asarray(self.val_xyz_21_error))
        pck                 = FPHA.get_pck(val_xyz_squeeze)
        thresholds          = np.arange(0, 85, 5)
        auc                 = FPHA.calc_auc(pck, thresholds)   
   
        val_loss_dict = {
            'precision'     : precision,
            'recall'        : recall,
            'f1score'       : f1score,
            'avg_iou'       : avg_iou,
            'xyz_l2_error'  : val_xyz_l2_error,
            'AUC_0_85'      : auc,
        }        

        # if self.debug:
        #     print(val_loss_dict)

        self.avg_iou            = 0.0
        self.iou_total          = 0.0
        self.val_total          = 0.0
        self.val_proposals      = 0.0
        self.val_correct        = 0.0
        
        self.val_xyz_21_error   = []
        return val_loss_dict
                    
    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        img                     = data_load[0]
        img                     = img.cuda()
        pred_bbox, pred_hand    = self.net(img)
        batch_size              = img.shape[0]
        max_boxes               = 1 # detect at most 1 box 
        W                       = pred_hand.shape[2]
        H                       = pred_hand.shape[3]
        D                       = 5
        pred_hand               = pred_hand.view(batch_size, 64, D, H, W)
        pred_hand               = pred_hand.permute(0, 1, 3, 4, 2)              
        batch_boxes             = YOLO.get_region_boxes(pred_bbox.cpu(), 
                                                        self.pred_conf_thresh, 
                                                        0, 
                                                        self.anchors, 
                                                        self.num_anchors, 
                                                        is_cuda = False,
                                                        is_time = self.time)

        for batch in range(batch_size):
            # Hand
            cur_pred_hand = pred_hand[batch]
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

            pred_uvd = pred_uvd.contiguous().view(21, 3, -1)
            pred_conf = pred_conf.contiguous().view(-1)
            
            topk_pred_uvd = []
            best_pred_uvd = []
            topk_idx = torch.topk(pred_conf, 10)[1]
            for idx in topk_idx:
                topk_pred_uvd.append(pred_uvd[:, :, idx].cpu().numpy())
            self.best_pred_uvd_list.append(topk_pred_uvd[0])
            self.topk_pred_uvd_list.append(topk_pred_uvd)
            self.pred_conf_list.append(pred_conf.cpu().numpy())
            
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
        
        pred_save = "predict_{}_{}_best.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.best_pred_uvd_list, (-1, 63)))
        
        pred_save = "predict_{}_{}_topk.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.topk_pred_uvd_list, (-1, 630)))
        
        pred_save = "predict_{}_{}_conf.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_conf_list)
        
        self.pred_list              = []
        self.best_pred_uvd_list     = []
        self.topk_pred_uvd_list     = []
        self.pred_conf_list         = []