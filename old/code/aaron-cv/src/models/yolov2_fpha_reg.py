import torch.nn     as nn
import numpy        as np
from pathlib        import Path
from tqdm           import tqdm

from src.models     import YOLOV2_FPHA
from src.utils      import YOLO, IMG, FPHA 

class YOLOV2_FPHA_Reg(YOLOV2_FPHA):
    """ YOLOv2 bounding box detection and hand regression from fully 
    connected layers """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)

        self.hand_loss = nn.MSELoss(reduction='sum')
        
        # Training
        if self.training:
            self.val_xyz_21_error       = []
        # Prediction
        else:
            self.pred_uvd_list      = []

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, bbox_gt, uvd_gt        = data_load
        bs                          = img.shape[0]
        
        img                         = img.cuda()
        bbox_gt                     = bbox_gt.cuda()
        uvd_gt                      = uvd_gt.cuda()
        uvd_gt                      = uvd_gt.reshape(bs, -1)
        
        bbox_out, hand_out          = self.net(img) 
        loss_bbox, *bbox_losses     = self.loss(bbox_out, bbox_gt)
        loss_hand                   = self.hand_loss(uvd_gt, hand_out)/2.0
        loss                        = loss_bbox + loss_hand
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_x, loss_y, loss_w, loss_h, loss_conf = bbox_losses
        loss_dict = {
            'loss'      : loss.item(),
            'loss_x'    : loss_x.item(),
            'loss_y'    : loss_y.item(),
            'loss_w'    : loss_w.item(),
            'loss_h'    : loss_h.item(),
            'loss_conf' : loss_conf.item(),
            'loss_hand' : loss_hand.item(),
        }
        
        return loss_dict

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img, bbox_gt, uvd_gt    = data_load
        uvd_gt                  = uvd_gt.numpy()
        bs                      = img.shape[0]
        
        img                     = img.cuda()
        bbox_gt                 = bbox_gt.cuda()
        
        output                  = self.net(img)
        pred_bbox, pred_uvd     = output
        
        pred_uvd                = pred_uvd.view(bs, 21, 3).cpu().numpy()
        pred_uvd                = IMG.scale_points_WH(pred_uvd, (1, 1), 
                                                      (FPHA.ORI_WIDTH, 
                                                       FPHA.ORI_HEIGHT))
        pred_uvd[..., 2]        *= FPHA.REF_DEPTH
        uvd_gt                  = IMG.scale_points_WH(uvd_gt, (1, 1), 
                                                      (FPHA.ORI_WIDTH,
                                                       FPHA.ORI_HEIGHT))
        uvd_gt[..., 2]          *= FPHA.REF_DEPTH
        pred_xyz                = FPHA.uvd2xyz_color(pred_uvd)
        xyz_gt                  = FPHA.uvd2xyz_color(uvd_gt)

        pred_bbox               = pred_bbox.cpu()
        target                  = data_load[1].cpu()
        all_boxes               = YOLO.get_region_boxes(pred_bbox,
                                                        self.val_conf_thresh,
                                                        0,
                                                        self.anchors,
                                                        self.num_anchors)

        for batch in range(bs):
            self.val_total  += 1
            cur_pred_xyz    = pred_xyz[batch]
            cur_xyz_gt      = xyz_gt[batch]
            self.val_xyz_21_error.append(np.sqrt(np.sum(np.square(
                cur_pred_xyz-cur_xyz_gt), axis=-1) + 1e-8 ))            
            
            boxes           = all_boxes[batch]
            boxes           = YOLO.nms_torch(boxes, self.val_nms_thresh)
            cur_target      = target[batch]
            
            for i in range(len(boxes)):
                if boxes[i][4] > self.val_conf_thresh:
                    self.val_proposals += 1

            box_gt = [float(cur_target[0]), float(cur_target[1]), 
                        float(cur_target[2]), float(cur_target[3]), 1.0]
            best_iou = 0
            best_j = -1
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
        img                 = data_load[0]
        img                 = img.cuda()
        pred_bbox, pred_uvd = self.net(img)
        bs                  = img.shape[0]
        max_boxes           = 1 # detect at most 1 box (FPHA only has 1 bbox)
        pred_uvd            = pred_uvd.cpu().numpy()
        for p in pred_uvd:
            self.pred_uvd_list.append(p)
        
        batch_boxes         = YOLO.get_region_boxes(pred_bbox.cpu(), 
                                                    self.pred_conf_thresh, 
                                                    0, 
                                                    self.anchors, 
                                                    self.num_anchors, 
                                                    is_cuda = False,
                                                    is_time = self.time)

        for i in range(bs):
            boxes       = batch_boxes[i]
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
        pred_save = "predict_{}_{}_bbox.txt".format(self.load_epoch, 
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_list)
        
        pred_save = "predict_{}_{}_uvd.txt".format(self.load_epoch, 
                                                   data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_uvd_list)
        
        self.pred_list      = []
        self.pred_uvd_list  = []
        
        