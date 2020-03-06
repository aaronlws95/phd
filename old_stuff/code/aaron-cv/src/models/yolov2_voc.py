import os
import time
import torch.nn                     as nn
import numpy                        as np
from pathlib                        import Path
from tqdm                           import tqdm

from src.models                     import Model
from src.utils                      import YOLO, IMG, FPHA 
from src.datasets                   import get_dataset, get_dataloader
from src.loss                       import get_loss

class YOLOV2_VOC(Model):
    """ YOLOv2 multi-class bounding box detection on PASCAL VOC """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)
        
        self.load_weights(self.load_epoch)
        
        self.loss           = get_loss(cfg['loss'], cfg)
        self.loss.epoch     = self.load_epoch
        self.anchors        = [float(i) for i in cfg["anchors"].split(',')]
        self.num_anchors    = len(self.anchors)//2
        self.num_classes    = int(cfg['classes'])
        
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
            self.val_total          = 0.0
            self.val_proposals      = 0.0
            self.val_correct        = 0.0 
            self.val_conf_thresh    = float(cfg['val_conf_thresh'])
            self.val_nms_thresh     = float(cfg['val_nms_thresh'])
            self.val_iou_thresh     = float(cfg['val_iou_thresh'])
        # Prediction    
        else:
            self.pred_list          = []
            self.load_epoch         = load_epoch
            self.pred_conf_thresh   = float(cfg['pred_conf_thresh'])
            self.pred_nms_thresh    = float(cfg['pred_nms_thresh'])

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        t0                  = time.time() # start
        img, target, _      = data_load
        img                 = img.cuda()
        target              = target.cuda()
        t1                  = time.time() # CPU to GPU
        
        out                 = self.net(img)[0]
        t2                  = time.time() # forward

        loss, *other_losses = self.loss(out, target)
        loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls = other_losses
        t3 = time.time() # loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        t4 = time.time() # backward
        
        loss_dict = {
            "loss"          : loss.item(),
            "loss_x"        : loss_x.item(),
            "loss_y"        : loss_y.item(),
            "loss_w"        : loss_w.item(),
            "loss_h"        : loss_h.item(),
            "loss_conf"     : loss_conf.item(),
            "loss_cls"      : loss_cls.item(),
        }
        
        if self.time:
            print('-----Training-----')
            print('        CPU to GPU : %f' % (t1 - t0))
            print('           forward : %f' % (t2 - t1))
            print('              loss : %f' % (t3 - t2))
            print('          backward : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        return loss_dict

    def post_epoch_process(self, epoch, loss_dict):
        self.loss.epoch += 1
        # validation
        if (epoch+1)%int(self.val_freq) == 0:
            val_loss = self.validate(epoch+1)
        # tensorboard logging
        self.tb_logger.log_dict["loss"] = loss_dict["loss"]
        if (epoch+1)%int(self.val_freq) == 0:
            for key, val in val_loss.items():
                self.tb_logger.log_dict[key] = val
        self.tb_logger.update_scalar_summary(epoch+1)
        # save checkpoint
        if (epoch+1)%int(self.save_freq) == 0:
            self.save_ckpt(epoch+1)

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img, target, _  = data_load
        batch_size      = img.shape[0]
        
        img             = img.cuda()
        
        pred            = self.net(img)
        pred            = pred[0].cpu()
        
        all_boxes   = YOLO.get_region_boxes(pred,
                                            self.val_conf_thresh,
                                            self.num_classes,
                                            self.anchors,
                                            self.num_anchors)

        if self.debug:
            print(all_boxes[0][:10])

        for batch in range(batch_size):
            boxes           = all_boxes[batch]
            boxes           = YOLO.nms_torch(boxes, self.val_nms_thresh)
            cur_target      = target[batch]
            num_gts         = YOLO.get_num_gt(cur_target)
            self.val_total  += num_gts
            
            for i in range(len(boxes)):
                if boxes[i][4] > self.val_conf_thresh:
                    self.val_proposals += 1
            
            for i in range(num_gts):
                box_gt      = [float(cur_target[i][1]), float(cur_target[i][2]), 
                               float(cur_target[i][3]), float(cur_target[i][4]), 
                               1.0, 1.0, float(cur_target[i][0])]
                best_iou    = 0
                best_j      = -1
                for j in range(len(boxes)):
                    iou = YOLO.bbox_iou(box_gt, boxes[j])
                    if iou > best_iou:
                        best_j      = j
                        best_iou    = iou
                if best_iou > self.val_iou_thresh and \
                    boxes[best_j][6] == box_gt[6]:
                    self.val_correct += 1            
            
    def get_valid_loss(self):
        eps         = 1e-5
        precision   = 1.0*self.val_correct/(self.val_proposals+eps)
        recall      = 1.0*self.val_correct/(self.val_total+eps)
        f1score     = 2.0*precision*recall/(precision+recall+eps)
   
        val_loss_dict = {
            "precision"     : precision,
            "recall"        : recall,
            "f1score"       : f1score 
        }        

        if self.debug:
            print(val_loss_dict)

        self.val_total      = 0.0
        self.val_proposals  = 0.0
        self.val_correct    = 0.0
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        img, target, imgpath    = data_load
        img                     = img.cuda()
        target                  = target.cuda()
        
        pred                    = self.net(img)
        batch_size              = img.shape[0]
        pred                    = pred[0].cpu()
        batch_boxes             = YOLO.get_region_boxes(pred, 
                                                        self.pred_conf_thresh, 
                                                        self.num_classes, 
                                                        self.anchors, 
                                                        self.num_anchors, 
                                                        only_objectness = False, 
                                                        is_predict = True,
                                                        is_cuda = False,
                                                        is_time = self.time)

        for i in range(batch_size):
            fileID          = os.path.basename(imgpath[i]).split('.')[0]
            width, height   = YOLO.get_image_size(imgpath[i])
            boxes           = batch_boxes[i]
            boxes           = YOLO.nms_torch(boxes, self.pred_nms_thresh)

            for box in boxes:
                x1          = (box[0] - box[2]/2.0)*width
                y1          = (box[1] - box[3]/2.0)*height
                x2          = (box[0] + box[2]/2.0)*width
                y2          = (box[1] + box[3]/2.0)*height
                det_conf    = box[4]
                
                for j in range((len(box)-5)//2):
                    cls_conf    = box[5+2*j]
                    cls_id      = box[6+2*j]
                    prob        = det_conf*cls_conf
                    self.pred_list.append([cls_id, 
                                           fileID, 
                                           prob, x1, y1, x2, y2])
        
    def save_predictions(self, data_split):
        cls_labels      = YOLO.get_class_labels("VOC")
        fps             = [0]*self.num_classes
        for i in range(self.num_classes):
            pred_args   = (self.load_epoch, data_split, cls_labels[i])
            pred_save   = "predict_{}_{}_{}.txt".format(*pred_args)
            buf         = Path(self.data_dir)/self.exp_dir/pred_save
            fps[i]      = open(buf, 'w')
        
        pred_args = (self.load_epoch, data_split, cls_labels[i])
        pred_save = "predict_{}_{}_all.txt".format(*pred_args)
        pred_save = Path(self.data_dir)/self.exp_dir/pred_save        
        
        with open(pred_save, 'w') as f:
            for pred in tqdm(self.pred_list):
                cls_id, fileID, prob, x1, y1, x2, y2 = pred
                fps[cls_id].write("{} {} {} {} {} {}\n".format(fileID, 
                                                            prob, 
                                                            x1, y1, x2, y2))
                f.write("{} {} {} {} {} {} {}\n".format(fileID, 
                                                     prob, 
                                                     x1, y1, x2, y2, cls_id))
        self.pred_list = []

    # ========================================================
    # SAVING AND LOADING
    # ========================================================

    def load_weights(self, load_epoch):
        if load_epoch == 0:
            if self.pretrain:
                if 'state' in self.pretrain:
                    ckpt = self.load_ckpt(self.pretrain)
                    # Load pretrained model with non-exact layers
                    state_dict = ckpt['model_state_dict']
                    cur_dict = self.net.state_dict()
                    # Filter out unnecessary keys
                    pretrained_dict = \
                        {k: v for k, v in state_dict.items() if k in cur_dict}
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