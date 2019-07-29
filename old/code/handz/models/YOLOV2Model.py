import os
import torch.nn as nn
import torch
import numpy as np
import sys
from tqdm import tqdm

from .BaseModel import BaseModel
from .networks.darknet import Darknet
from .networks.darknet_reg import Darknet_reg
from .loss.RegionLoss import RegionLoss, RegionLoss_1Class, RegionLoss_1Class_reg
from .loss.HPOLoss import HPOLoss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import VOC_utils as VOC
from utils import YOLO_utils as YOLO
from utils import FPHA_utils as FPHA
from utils import HPO_utils as HPO
from utils.eval_utils import *
from utils.image_utils import *

class YOLOV2Model_VOC(BaseModel):
    def __init__(self, conf, device, load_epoch, train_mode, exp_dir, deterministic, logger=None):
        super(YOLOV2Model_VOC, self).__init__(conf, device, train_mode, exp_dir, deterministic, logger)
        net = self.get_net(load_epoch, conf["cfg_file"], conf["pretrain"])
        self.init_network(net, load_epoch)
        
        self.loss                   = RegionLoss(conf, device)
        self.seen                   = self.load_seen(load_epoch)
        self.anchors                = [float(i) for i in conf["anchors"]]
        self.anchor_step            = 2
        self.num_anchors            = len(self.anchors)//self.anchor_step
        self.num_classes            = conf["classes"]
        self.time                   = conf["time"]
        self.init_learning_rate     = conf["optimizer"]["learning_rate"]
        self.batch_size             = conf["optimizer"]["batch_size"]
        self.processed_batches      = self.seen//self.batch_size
        self.scales                 = conf["scales"]
        self.steps                  = conf["steps"]

        # validation
        self.val_total              = 0.0
        self.val_proposals          = 0.0
        self.val_correct            = 0.0
        self.val_conf_thresh        = conf["val_conf_thresh"]
        self.val_nms_thresh         = conf["val_nms_thresh"]        
        self.val_iou_thresh         = conf["val_iou_thresh"]
        
        # prediction
        if not train_mode:
            self.pred_list          = []
            self.load_epoch         = load_epoch
            self.pred_conf_thresh   = conf["pred_conf_thresh"]
            self.pred_nms_thresh    = conf["pred_nms_thresh"]   
                     
    def get_net(self, load_epoch, cfgfile, pretrain):
        if self.deterministic:
            torch.manual_seed(0)        
        net = Darknet(cfgfile)
        if load_epoch == 0:
            if pretrain != "none":
                if pretrain[-5:] == "state":
                    ckpt = self.get_ckpt(pretrain)
                    self.load_ckpt(net, ckpt["model_state_dict"])
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
                else:
                    net.load_weights(pretrain)
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
        return net

    def load_seen(self, load_epoch):
        if load_epoch != 0 :
            load_dir = os.path.join(self.save_ckpt_dir, f"model_{load_epoch}.state")
            ckpt = self.get_ckpt(load_dir)
            seen =  ckpt["seen"]
        else:
            seen = 0
        return seen
    
    def adjust_learning_rate(self):
        """Sets the learning rate"""
        lr = self.init_learning_rate
        for i in range(len(self.steps)):
            scale = self.scales[i] if i < len(self.scales) else 1
            if self.processed_batches >= self.steps[i]:
                lr = lr * scale
                if self.processed_batches == self.steps[i]:
                    break
            else:
                break
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr/self.batch_size
        return lr
    
    def predict(self, data_load):
        img = data_load[0]
        return self.net(img)
   
    def get_loss(self, data_load, out, train_out):
        _, labels = data_load
        return self.loss(out, labels, train_out)

    def train_step(self, data_load):
        self.adjust_learning_rate()
        
        self.processed_batches = self.processed_batches + 1     
        
        out = self.predict(data_load) 
        
        self.seen += out.shape[0]  
        self.loss.seen = self.seen 
        # # DEBUG (need to be deterministic)
        # print('MODEL OUTPUT', torch.sum(out))
        # print('IMG', data_load[0][0, :, 100:200, 100:200], torch.sum(data_load[0]))
        # print('TARGET', data_load[1][8][:2], torch.sum(data_load[1][8]))
        loss, loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls = self.get_loss(data_load, out, True)
        # print('loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), loss.item()))
        # raise ValueError('loss should equal 5211.409180 exactly with YOLO_base_debug.json')
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_dict = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_y": loss_y.item(),
            "loss_w": loss_w.item(),
            "loss_h": loss_h.item(),
            "loss_conf": loss_conf.item(),
            "loss_cls": loss_cls.item(),
            "seen": self.seen,
            "processed_batches": self.processed_batches
        }
        return loss_dict
    
    def valid_step(self, data_load):
        img = data_load[0]
        pred = self.net(img)
        batch_size = img.shape[0]
        pred = pred.cpu()
        target = data_load[1].cpu()
        
        all_boxes = YOLO.get_region_boxes(pred,
                                          self.val_conf_thresh,
                                          self.num_classes,
                                          self.anchors,
                                          self.num_anchors)

        for batch in range(batch_size):
            boxes = all_boxes[batch]
            boxes = YOLO.nms_torch(boxes, self.val_nms_thresh)
            cur_target = target[batch]
            num_gts = YOLO.get_num_gt(cur_target)
            self.val_total += num_gts
            
            for i in range(len(boxes)):
                if boxes[i][4] > self.val_conf_thresh:
                    self.val_proposals += 1
            
            for i in range(num_gts):
                box_gt = [float(cur_target[i][1]), float(cur_target[i][2]), float(cur_target[i][3]), float(cur_target[i][4]), 1.0, 1.0, float(cur_target[i][0])]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = YOLO.bbox_iou(box_gt, boxes[j])
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > self.val_iou_thresh and boxes[best_j][6] == box_gt[6]:
                    self.val_correct += 1            
            
    def get_valid_loss(self):
        eps = 1e-5
        precision = 1.0*self.val_correct/(self.val_proposals+eps)
        recall = 1.0*self.val_correct/(self.val_total+eps)
        f1score = 2.0*precision*recall/(precision+recall+eps)
   
        val_loss_dict = {
            "precision": precision,
            "recall": recall,
            "f1score": f1score 
        }        
        
        if self.logger:
            self.logger.log("VALIDATION LOSS")
            for name, loss in val_loss_dict.items():
                self.logger.log("{}: {}".format(name.upper(), loss))
        
        self.val_total = 0.0
        self.val_proposals = 0.0
        self.val_correct = 0.0
        return val_loss_dict
        
    def predict_step(self, data_load):
        img = data_load[0]
        pred = self.net(img)
        batch_size = img.shape[0]
        pred = pred.cpu() # faster with cpu
        batch_boxes = YOLO.get_region_boxes(pred, 
                                            self.pred_conf_thresh, 
                                            self.num_classes, 
                                            self.anchors, 
                                            self.num_anchors, 
                                            only_objectness = False, 
                                            is_predict = True,
                                            is_cuda = False,
                                            is_time = self.time)
        
        for i in range(batch_size):
            fileID = os.path.basename(data_load[1][i]).split('.')[0]
            width, height = YOLO.get_image_size(data_load[1][i])
      
            boxes = batch_boxes[i]
            boxes = YOLO.nms_torch(boxes, self.pred_nms_thresh)

            for box in boxes:
                x1 = (box[0] - box[2]/2.0) * width
                y1 = (box[1] - box[3]/2.0) * height
                x2 = (box[0] + box[2]/2.0) * width
                y2 = (box[1] + box[3]/2.0) * height

                det_conf = box[4]
                
                for j in range((len(box)-5)//2):
                    cls_conf = box[5+2*j]
                    cls_id = box[6+2*j]
                    prob = det_conf * cls_conf
                    self.pred_list.append([cls_id, fileID, prob, x1, y1, x2, y2])
        
    def save_predictions(self, data_split):
        class_labels = YOLO.get_class_labels("VOC")
        
        fps = [0]*self.num_classes
        for i in range(self.num_classes):
            buf = os.path.join(self.save_dir, "predict_{}_{}_{}.txt".format(self.load_epoch, data_split, class_labels[i]))
            fps[i] = open(buf, 'w')        
        
        if self.logger:
            self.logger.log("WRITING PREDICTIONS")
        for pred in tqdm(self.pred_list):
            cls_id = pred[0]
            fileID = pred[1]
            prob = pred[2]
            x1 = pred[3]
            y1 = pred[4]
            x2 = pred[5]
            y2 = pred[6]
            fps[cls_id].write("{} {} {} {} {} {}\n".format(fileID, prob, x1, y1, x2, y2))
        
        self.pred_list = []
        if self.logger:
            self.logger.log("FINISHING WRITING PREDICTIONS")
        
    def save_ckpt(self, epoch):
        if self.logger:
            self.logger.log(f"SAVING CHECKPOINT EPOCH {epoch}")
        if isinstance(self.net, nn.parallel.DistributedDataParallel):
            network = self.net.module
        else:
            network = self.net
        model_state_dict = network.state_dict()
        for key, param in model_state_dict.items():
            model_state_dict[key] = param.cpu()
        state = {"epoch": epoch, 
                 "model_state_dict": model_state_dict, 
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "seen": self.seen}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.save_ckpt_dir, f"model_{epoch}.state"))

class YOLOV2Model_1Class(BaseModel):
    def __init__(self, conf, device, load_epoch, train_mode, exp_dir, deterministic, logger):
        super(YOLOV2Model_1Class, self).__init__(conf, device, train_mode, exp_dir, deterministic, logger)
        net = self.get_net(load_epoch, conf["cfg_file"], conf["pretrain"])
        self.init_network(net, load_epoch)
        
        self.loss                   = RegionLoss_1Class(conf, device)
        self.seen                   = self.load_seen(load_epoch)
        self.anchors                = [float(i) for i in conf["anchors"]]
        self.anchor_step            = 2
        self.num_anchors            = len(self.anchors)//self.anchor_step
        self.time                   = conf["time"]
        self.init_learning_rate     = conf["optimizer"]["learning_rate"]
        self.batch_size             = conf["optimizer"]["batch_size"]
        self.processed_batches      = self.seen//self.batch_size
        self.scales                 = conf["scales"]
        self.steps                  = conf["steps"]

        # validation    
        self.val_total              = 0.0
        self.val_proposals          = 0.0
        self.val_correct            = 0.0
        self.val_conf_thresh        = conf["val_conf_thresh"]
        self.val_nms_thresh         = conf["val_nms_thresh"]        
        self.val_iou_thresh         = conf["val_iou_thresh"]
        
        # prediction
        if not train_mode:
            self.pred_list          = []
            self.load_epoch         = load_epoch
            self.pred_conf_thresh   = conf["pred_conf_thresh"]
            self.pred_nms_thresh    = conf["pred_nms_thresh"]   
                     
    def get_net(self, load_epoch, cfgfile, pretrain):
        if self.deterministic:
            torch.manual_seed(0)        
        net = Darknet(cfgfile)
        if load_epoch == 0:
            if pretrain != "none":
                if pretrain[-5:] == "state":
                    ckpt = self.get_ckpt(pretrain)
                    self.load_ckpt(net, ckpt["model_state_dict"])
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
                else:
                    net.load_weights(pretrain)
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
        return net

    def load_seen(self, load_epoch):
        if load_epoch != 0 :
            load_dir = os.path.join(self.save_ckpt_dir, f"model_{load_epoch}.state")
            ckpt = self.get_ckpt(load_dir)
            seen =  ckpt["seen"]
        else:
            seen = 0
        return seen
    
    def adjust_learning_rate(self):
        """Sets the learning rate"""
        lr = self.init_learning_rate
        for i in range(len(self.steps)):
            scale = self.scales[i] if i < len(self.scales) else 1
            if self.processed_batches >= self.steps[i]:
                lr = lr * scale
                if self.processed_batches == self.steps[i]:
                    break
            else:
                break
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr/self.batch_size
        return lr
    
    def predict(self, data_load):
        img = data_load[0]
        return self.net(img)
   
    def get_loss(self, data_load, out, train_out):
        _, labels = data_load
        return self.loss(out, labels, train_out)

    def train_step(self, data_load):
        self.adjust_learning_rate()
        
        self.processed_batches = self.processed_batches + 1     

        out = self.predict(data_load) 
        self.seen += out.shape[0]  
        self.loss.seen = self.seen

        loss, loss_x, loss_y, loss_w, loss_h, loss_conf = self.get_loss(data_load, out, True)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_dict = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_y": loss_y.item(),
            "loss_w": loss_w.item(),
            "loss_h": loss_h.item(),
            "loss_conf": loss_conf.item(),
            "seen": self.seen,
            "processed_batches": self.processed_batches
        }
        return loss_dict
    
    def valid_step(self, data_load):
        img = data_load[0]
        pred = self.net(img)
        batch_size = img.shape[0]
        pred = pred.cpu()
        target = data_load[1].cpu()
        
        all_boxes = YOLO.get_region_boxes(pred,
                                          self.val_conf_thresh,
                                          0,
                                          self.anchors,
                                          self.num_anchors)

       
        for batch in range(batch_size):
            boxes = all_boxes[batch]
            boxes = YOLO.nms_torch(boxes, self.val_nms_thresh)
            cur_target = target[batch]
            
            if target.shape[1] == 4:
                num_gts = 1
                cur_target = cur_target.unsqueeze(0)
            else:
                num_gts = YOLO.get_num_gt(cur_target)
                
            self.val_total += 1
            
            for i in range(len(boxes)):
                if boxes[i][4] > self.val_conf_thresh:
                    self.val_proposals += 1
                    
            for i in range(num_gts):
                box_gt = [float(cur_target[i][0]), float(cur_target[i][1]), float(cur_target[i][2]), float(cur_target[i][3]), 1.0]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = YOLO.bbox_iou(box_gt, boxes[j])
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > self.val_iou_thresh:
                    self.val_correct += 1            
            
    def get_valid_loss(self):
        eps = 1e-5
        precision = 1.0*self.val_correct/(self.val_proposals+eps)
        recall = 1.0*self.val_correct/(self.val_total+eps)
        f1score = 2.0*precision*recall/(precision+recall+eps)
   
        val_loss_dict = {
            "precision": precision,
            "recall": recall,
            "f1score": f1score 
        }        
        
        if self.logger:
            self.logger.log("VALIDATION LOSS")
            for name, loss in val_loss_dict.items():
                self.logger.log("{}: {}".format(name.upper(), loss))
        
        self.val_total = 0.0
        self.val_proposals = 0.0
        self.val_correct = 0.0
        return val_loss_dict
        
    def predict_step(self, data_load):
        img = data_load[0]
        pred = self.net(img)
        batch_size = img.shape[0]
        pred = pred.cpu() # faster with cpu
        max_boxes = 10 # detect at most 10 boxes
            
        batch_boxes = YOLO.get_region_boxes(pred, 
                                            self.pred_conf_thresh, 
                                            0, 
                                            self.anchors, 
                                            self.num_anchors, 
                                            is_cuda = False,
                                            is_time = self.time)

        for i in range(batch_size):
            boxes = batch_boxes[i]
            boxes = YOLO.nms_torch(boxes, self.pred_nms_thresh)

            all_boxes = np.zeros((max_boxes, 5))
            if len(boxes) != 0:
                
                if len(boxes) > len(all_boxes):
                    fill_range = len(all_boxes)
                else:
                    fill_range = len(boxes)
                    
                for i in range(fill_range):
                    box = boxes[i]
                    all_boxes[i] = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
            all_boxes = np.reshape(all_boxes, -1)
            self.pred_list.append(all_boxes)
                
    def save_predictions(self, data_split):
        if self.logger:
            self.logger.log("WRITING PREDICTIONS")
        pred_file = os.path.join(self.save_dir, "predict_{}_{}.txt".format(self.load_epoch, data_split))
        np.savetxt(pred_file, self.pred_list)
        self.pred_list = []
        if self.logger:
            self.logger.log("FINISHING WRITING PREDICTIONS")
        
    def save_ckpt(self, epoch):
        if self.logger:
            self.logger.log(f"SAVING CHECKPOINT EPOCH {epoch}")
        if isinstance(self.net, nn.parallel.DistributedDataParallel):
            network = self.net.module
        else:
            network = self.net
        model_state_dict = network.state_dict()
        for key, param in model_state_dict.items():
            model_state_dict[key] = param.cpu()
        state = {"epoch": epoch, 
                 "model_state_dict": model_state_dict, 
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "seen": self.seen}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.save_ckpt_dir, f"model_{epoch}.state"))        

class YOLOV2Model_1Class_reg(BaseModel):
    def __init__(self, conf, device, load_epoch, train_mode, exp_dir, deterministic, logger):
        super(YOLOV2Model_1Class_reg, self).__init__(conf, device, train_mode, exp_dir, deterministic, logger)
        net = self.get_net(load_epoch, conf["cfg_file"], conf["pretrain"])
        self.init_network(net, load_epoch)
        
        self.loss                   = RegionLoss_1Class_reg(conf, device)
        self.seen                   = self.load_seen(load_epoch)
        self.anchors                = [float(i) for i in conf["anchors"]]
        self.anchor_step            = 2
        self.num_anchors            = len(self.anchors)//self.anchor_step
        self.time                   = conf["time"]
        self.init_learning_rate     = conf["optimizer"]["learning_rate"]
        self.batch_size             = conf["optimizer"]["batch_size"]
        self.processed_batches      = self.seen//self.batch_size
        self.scales                 = conf["scales"]
        self.steps                  = conf["steps"]

        # validation    
        self.val_total              = 0.0
        self.val_proposals          = 0.0
        self.val_correct            = 0.0
        self.val_xyz_L2_error       = 0.0
        self.val_xyz_21_error       = []    
        self.val_conf_thresh        = conf["val_conf_thresh"]
        self.val_nms_thresh         = conf["val_nms_thresh"]        
        self.val_iou_thresh         = conf["val_iou_thresh"]

        # prediction
        if not train_mode:
            self.pred_list          = []
            self.pred_uvd_list      = []
            self.load_epoch         = load_epoch
            self.pred_conf_thresh   = conf["pred_conf_thresh"]
            self.pred_nms_thresh    = conf["pred_nms_thresh"]   
            
    def get_net(self, load_epoch, cfgfile, pretrain):
        if self.deterministic:
            torch.manual_seed(0)        
        net = Darknet_reg(cfgfile)
        if load_epoch == 0:
            if pretrain != "none":
                if pretrain[-5:] == "state":
                    ckpt = self.get_ckpt(pretrain)
                    # load pretrained model with non-exact layers
                    state_dict = ckpt["model_state_dict"]
                    cur_dict = net.state_dict()
                    for k, v in cur_dict.items():
                        if k in state_dict:
                            cur_dict[k] = state_dict[k]
                    state_dict.update(cur_dict)
                    self.load_ckpt(net, state_dict)
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
                else:
                    net.load_weights(pretrain)
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
        return net

    def load_seen(self, load_epoch):
        if load_epoch != 0 :
            load_dir = os.path.join(self.save_ckpt_dir, f"model_{load_epoch}.state")
            ckpt = self.get_ckpt(load_dir)
            seen =  ckpt["seen"]
        else:
            seen = 0
        return seen
    
    def adjust_learning_rate(self):
        """Sets the learning rate"""
        lr = self.init_learning_rate
        for i in range(len(self.steps)):
            scale = self.scales[i] if i < len(self.scales) else 1
            if self.processed_batches >= self.steps[i]:
                lr = lr * scale
                if self.processed_batches == self.steps[i]:
                    break
            else:
                break
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr/self.batch_size
        return lr
    
    def predict(self, data_load):
        img = data_load[0]
        return self.net(img)
   
    def get_loss(self, data_load, out, train_out):
        _, labels, uvd_gt = data_load
        return self.loss(out, labels, uvd_gt, train_out)

    def train_step(self, data_load):
        self.adjust_learning_rate()
        
        self.processed_batches = self.processed_batches + 1     

        out = self.predict(data_load) 

        self.seen += out[0].shape[0]  
        self.loss.seen = self.seen

        loss, loss_x, loss_y, loss_w, loss_h, loss_conf, loss_hand = self.get_loss(data_load, out, True)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_dict = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_y": loss_y.item(),
            "loss_w": loss_w.item(),
            "loss_h": loss_h.item(),
            "loss_conf": loss_conf.item(),
            "loss_hand": loss_hand.item(),
            "seen": self.seen,
            "processed_batches": self.processed_batches
        }
        return loss_dict
    
    def valid_step(self, data_load):
        img = data_load[0]
        output = self.net(img)
        batch_size = img.shape[0]
        
        pred_uvd = output[1].reshape(img.shape[0], 21, 3).cpu().numpy()
        pred_uvd = scale_points_WH(pred_uvd, (1, 1), (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT))
        pred_uvd[..., 2] *= FPHA.REF_DEPTH
        uvd_gt = data_load[2].cpu().numpy()
        uvd_gt = scale_points_WH(uvd_gt, (1, 1), (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT))
        uvd_gt[..., 2] *= FPHA.REF_DEPTH
        pred_xyz = FPHA.uvd2xyz_color(pred_uvd)
        xyz_gt = FPHA.uvd2xyz_color(uvd_gt)                
        for batch in range(batch_size):
            cur_pred_xyz = pred_xyz[batch]
            cur_xyz_gt = xyz_gt[batch]
            self.val_xyz_L2_error += mean_L2_error(cur_pred_xyz, cur_xyz_gt)
            self.val_xyz_21_error.append(np.sqrt(np.sum(np.square(cur_pred_xyz-cur_xyz_gt), axis=-1) + 1e-8 ))
        
        pred = output[0]
        pred = pred.cpu()
        target = data_load[1].cpu()
        
        all_boxes = YOLO.get_region_boxes(pred,
                                          self.val_conf_thresh,
                                          0,
                                          self.anchors,
                                          self.num_anchors)

        for batch in range(batch_size):
            boxes = all_boxes[batch]
            boxes = YOLO.nms_torch(boxes, self.val_nms_thresh)
            cur_target = target[batch]
            
            if target.shape[1] == 4:
                num_gts = 1
                cur_target = cur_target.unsqueeze(0)
            else:
                num_gts = YOLO.get_num_gt(cur_target)
                
            self.val_total += 1
            
            for i in range(len(boxes)):
                if boxes[i][4] > self.val_conf_thresh:
                    self.val_proposals += 1
                    
            for i in range(num_gts):
                box_gt = [float(cur_target[i][0]), float(cur_target[i][1]), float(cur_target[i][2]), float(cur_target[i][3]), 1.0]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = YOLO.bbox_iou(box_gt, boxes[j])
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > self.val_iou_thresh:
                    self.val_correct += 1            
            
    def get_valid_loss(self):
        eps = 1e-5
        precision = 1.0*self.val_correct/(self.val_proposals+eps)
        recall = 1.0*self.val_correct/(self.val_total+eps)
        f1score = 2.0*precision*recall/(precision+recall+eps)
   
        val_xyz_l2_error = self.val_xyz_L2_error/(self.val_total+eps)
        pck = get_pck(np.squeeze(np.asarray(self.val_xyz_21_error)))
        thresholds = np.arange(0, 85, 5)
        auc = calc_auc(pck, thresholds)   
   
        val_loss_dict = {
            "precision": precision,
            "recall": recall,
            "f1score": f1score,
            "xyz_l2_error": val_xyz_l2_error,
            "AUC_0_85": auc 
        }        
        
        if self.logger:
            self.logger.log("VALIDATION LOSS")
            for name, loss in val_loss_dict.items():
                self.logger.log("{}: {}".format(name.upper(), loss))
        
        self.val_xyz_l2_error = 0.0
        self.val_total = 0.0
        self.val_proposals = 0.0
        self.val_correct = 0.0
        self.val_xyz_21_error = []
        return val_loss_dict
        
    def predict_step(self, data_load):
        img = data_load[0]
        output = self.net(img)
        
        pred = output[0]
        pred_uvds = output[1].cpu().numpy()
        for p in pred_uvds:    
            self.pred_uvd_list.append(p)
        
        batch_size = img.shape[0]
        pred = pred.cpu() # faster with cpu
        max_boxes = 10 # detect at most 10 boxes
            
        batch_boxes = YOLO.get_region_boxes(pred, 
                                            self.pred_conf_thresh, 
                                            0, 
                                            self.anchors, 
                                            self.num_anchors, 
                                            is_cuda = False,
                                            is_time = self.time)

        for i in range(batch_size):
            boxes = batch_boxes[i]
            boxes = YOLO.nms_torch(boxes, self.pred_nms_thresh)

            all_boxes = np.zeros((max_boxes, 5))
            if len(boxes) != 0:
                
                if len(boxes) > len(all_boxes):
                    fill_range = len(all_boxes)
                else:
                    fill_range = len(boxes)
                    
                for i in range(fill_range):
                    box = boxes[i]
                    all_boxes[i] = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
            all_boxes = np.reshape(all_boxes, -1)
            self.pred_list.append(all_boxes)
                
    def save_predictions(self, data_split):
        if self.logger:
            self.logger.log("WRITING PREDICTIONS")
        pred_file = os.path.join(self.save_dir, "predict_{}_{}.txt".format(self.load_epoch, data_split))
        np.savetxt(pred_file, self.pred_list)
        pred_uvd_file = os.path.join(self.save_dir, "predict_{}_{}_uvd.txt".format(self.load_epoch, data_split))
        np.savetxt(pred_uvd_file, self.pred_uvd_list)
        
        self.pred_uvd_list = []
        self.pred_list = []
        if self.logger:
            self.logger.log("FINISHING WRITING PREDICTIONS")
        
    def save_ckpt(self, epoch):
        if self.logger:
            self.logger.log(f"SAVING CHECKPOINT EPOCH {epoch}")
        if isinstance(self.net, nn.parallel.DistributedDataParallel):
            network = self.net.module
        else:
            network = self.net
        model_state_dict = network.state_dict()
        for key, param in model_state_dict.items():
            model_state_dict[key] = param.cpu()
        state = {"epoch": epoch, 
                 "model_state_dict": model_state_dict, 
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "seen": self.seen}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.save_ckpt_dir, f"model_{epoch}.state"))        

class YOLOV2Model_1Class_HPOreg(BaseModel):
    def __init__(self, conf, device, load_epoch, train_mode, exp_dir, deterministic, logger):
        super(YOLOV2Model_1Class_HPOreg, self).__init__(conf, device, train_mode, exp_dir, deterministic, logger)
        net = self.get_net(load_epoch, conf["cfg_file"], conf["pretrain"])
        self.init_network(net, load_epoch)
        
        self.bbox_loss              = RegionLoss_1Class(conf, device)
        self.hpo_loss               = HPOLoss(conf, device)
        self.seen                   = self.load_seen(load_epoch)
        self.anchors                = [float(i) for i in conf["anchors"]]
        self.anchor_step            = 2
        self.num_anchors            = len(self.anchors)//self.anchor_step
        self.time                   = conf["time"]
        self.init_learning_rate     = conf["optimizer"]["learning_rate"]
        self.batch_size             = conf["optimizer"]["batch_size"]
        self.processed_batches      = self.seen//self.batch_size
        self.scales                 = conf["scales"]
        self.steps                  = conf["steps"]
        self.hand_root              = conf["hand_root"]
        
        # validation    
        self.val_total              = 0.0
        self.val_proposals          = 0.0
        self.val_correct            = 0.0
        self.val_xyz_L2_error       = 0.0
        self.val_xyz_21_error       = []    
        self.val_conf_thresh        = conf["val_conf_thresh"]
        self.val_nms_thresh         = conf["val_nms_thresh"]        
        self.val_iou_thresh         = conf["val_iou_thresh"]

        # prediction
        if not train_mode:
            self.best_pred_uvd_list = []
            self.topk_pred_uvd_list = []
            self.pred_conf_list     = []
            
            self.pred_list          = []
            self.pred_uvd_list      = []
            self.load_epoch         = load_epoch
            self.pred_conf_thresh   = conf["pred_conf_thresh"]
            self.pred_nms_thresh    = conf["pred_nms_thresh"]   
            
    def get_net(self, load_epoch, cfgfile, pretrain):
        if self.deterministic:
            torch.manual_seed(0)        
        net = Darknet_reg(cfgfile)
        if load_epoch == 0:
            if pretrain != "none":
                if pretrain[-5:] == "state":
                    ckpt = self.get_ckpt(pretrain)
                    # load pretrained model with non-exact layers
                    state_dict = ckpt["model_state_dict"]
                    cur_dict = net.state_dict()
                    for k, v in cur_dict.items():
                        if k in state_dict:
                            cur_dict[k] = state_dict[k]
                    state_dict.update(cur_dict)
                    self.load_ckpt(net, state_dict)
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
                else:
                    net.load_weights(pretrain)
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
        return net

    def load_seen(self, load_epoch):
        if load_epoch != 0 :
            load_dir = os.path.join(self.save_ckpt_dir, f"model_{load_epoch}.state")
            ckpt = self.get_ckpt(load_dir)
            seen =  ckpt["seen"]
        else:
            seen = 0
        return seen
    
    def adjust_learning_rate(self):
        """Sets the learning rate"""
        lr = self.init_learning_rate
        for i in range(len(self.steps)):
            scale = self.scales[i] if i < len(self.scales) else 1
            if self.processed_batches >= self.steps[i]:
                lr = lr * scale
                if self.processed_batches == self.steps[i]:
                    break
            else:
                break
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr/self.batch_size
        return lr
    
    def predict(self, data_load):
        img = data_load[0]
        return self.net(img)
   
    def get_loss(self, data_load, out, train_out):
        _, labels, uvd_gt = data_load
        bbox_loss, *bbox_others = self.bbox_loss(out[0], labels, train_out)
        hpo_loss, *hpo_others = self.hpo_loss(out[1], uvd_gt, train_out)
        total_loss = bbox_loss + hpo_loss
        return total_loss, bbox_others, hpo_others
        
    def train_step(self, data_load):
        self.adjust_learning_rate()
        
        self.processed_batches = self.processed_batches + 1     

        out = self.predict(data_load) 
        self.seen += out[0].shape[0]  
        self.bbox_loss.seen = self.seen
        self.hpo_loss.seen = self.seen
        
        loss, bbox_others, hpo_others = self.get_loss(data_load, out, True)

        loss_x, loss_y, loss_w, loss_h, loss_bbox_conf = bbox_others
        loss_u, loss_v, loss_d, loss_hand_conf = hpo_others
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_dict = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_y": loss_y.item(),
            "loss_w": loss_w.item(),
            "loss_h": loss_h.item(),
            "loss_bbox_conf": loss_bbox_conf.item(),
            "loss_u": loss_u.item(),
            "loss_v": loss_v.item(),
            "loss_d": loss_d.item(),
            "loss_hand_conf": loss_hand_conf.item(),        
            "seen": self.seen,
            "processed_batches": self.processed_batches
        }
        return loss_dict
    
    def valid_step(self, data_load):
        img = data_load[0]
        output = self.net(img)
        batch_size = img.shape[0]
        
        output_hpo = output[1]
        W = output_hpo.shape[2]
        H = output_hpo.shape[3]
        D = 5
        grid_size = W*H*D
        uvd_gt = data_load[2].cpu().numpy()
        
        for batch in range(batch_size):
            cur_uvd_gt = uvd_gt[batch]
            cur_output_hpo = output_hpo[batch]
            cur_output_hpo = cur_output_hpo.reshape(64, -1)
            pred_uvd = cur_output_hpo[:63, :].reshape(21, 3, grid_size)
            pred_conf = cur_output_hpo[63, :].reshape(grid_size)
            pred_conf = torch.sigmoid(pred_conf)
            
            index = torch.from_numpy(np.asarray(np.unravel_index(np.arange(grid_size), (W, H, D)))).type(torch.FloatTensor)
            u = index[0, :].unsqueeze(0).expand(21, -1)
            v = index[1, :].unsqueeze(0).expand(21, -1)
            z = index[2, :].unsqueeze(0).expand(21, -1)       
            
            if self.device != "cpu":
                u = u.cuda()
                v = v.cuda()
                z = z.cuda()        
            
            pred_uvd[self.hand_root, :, :] = torch.sigmoid(pred_uvd[self.hand_root, :, :])
            pred_uvd[:, 0, :] = (pred_uvd[:, 0, :] + u) / W
            pred_uvd[:, 1, :] = (pred_uvd[:, 1, :] + v) / H
            pred_uvd[:, 2, :] = (pred_uvd[:, 2, :] + z) / D
            
            top_idx = torch.topk(pred_conf, 1)[1]
            best_pred_uvd = pred_uvd[:, :, top_idx].squeeze().cpu().numpy()
            
            
            cur_uvd_gt = scale_points_WH(cur_uvd_gt, (1, 1), (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT))
            cur_uvd_gt[..., 2] *= FPHA.REF_DEPTH
            best_pred_uvd = scale_points_WH(best_pred_uvd, (1, 1), (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT))
            best_pred_uvd[..., 2] *= FPHA.REF_DEPTH
            best_pred_xyz = FPHA.uvd2xyz_color(best_pred_uvd)
            xyz_gt = FPHA.uvd2xyz_color(cur_uvd_gt)

            self.val_xyz_L2_error += mean_L2_error(best_pred_xyz, xyz_gt)
            self.val_xyz_21_error.append(np.sqrt(np.sum(np.square(best_pred_xyz-xyz_gt), axis=-1) + 1e-8 ))
        
        pred = output[0]
        pred = pred.cpu()
        target = data_load[1].cpu()
        
        all_boxes = YOLO.get_region_boxes(pred,
                                          self.val_conf_thresh,
                                          0,
                                          self.anchors,
                                          self.num_anchors)

        for batch in range(batch_size):
            boxes = all_boxes[batch]
            boxes = YOLO.nms_torch(boxes, self.val_nms_thresh)
            cur_target = target[batch]
            
            if target.shape[1] == 4:
                num_gts = 1
                cur_target = cur_target.unsqueeze(0)
            else:
                num_gts = YOLO.get_num_gt(cur_target)
                
            self.val_total += 1
            
            for i in range(len(boxes)):
                if boxes[i][4] > self.val_conf_thresh:
                    self.val_proposals += 1
                    
            for i in range(num_gts):
                box_gt = [float(cur_target[i][0]), float(cur_target[i][1]), float(cur_target[i][2]), float(cur_target[i][3]), 1.0]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = YOLO.bbox_iou(box_gt, boxes[j])
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > self.val_iou_thresh:
                    self.val_correct += 1            
            
    def get_valid_loss(self):
        eps = 1e-5
        precision = 1.0*self.val_correct/(self.val_proposals+eps)
        recall = 1.0*self.val_correct/(self.val_total+eps)
        f1score = 2.0*precision*recall/(precision+recall+eps)

        val_xyz_l2_error = self.val_xyz_L2_error/(self.val_total+eps)
        pck = get_pck(np.squeeze(np.asarray(self.val_xyz_21_error)))
        thresholds = np.arange(0, 85, 5)
        auc = calc_auc(pck, thresholds)   
   
        val_loss_dict = {
            "precision": precision,
            "recall": recall,
            "f1score": f1score,
            "xyz_l2_error": val_xyz_l2_error,
            "AUC_0_85": auc 
        }        
        
        if self.logger:
            self.logger.log("VALIDATION LOSS")
            for name, loss in val_loss_dict.items():
                self.logger.log("{}: {}".format(name.upper(), loss))
        
        self.val_xyz_l2_error = 0.0
        self.val_total = 0.0
        self.val_proposals = 0.0
        self.val_correct = 0.0
        self.val_xyz_21_error = []
        return val_loss_dict
        
    def predict_step(self, data_load):
        img = data_load[0]
        output = self.net(img)
        batch_size = img.shape[0]
        
        output_hpo = output[1]
        W = output_hpo.shape[2]
        H = output_hpo.shape[3]
        D = 5
        self.grid_size = W*H*D
        
        for batch in range(batch_size):
            cur_output_hpo = output_hpo[batch]
            cur_output_hpo = cur_output_hpo.reshape(64, -1)
            pred_uvd = cur_output_hpo[:63, :].reshape(21, 3, self.grid_size)
            pred_conf = cur_output_hpo[63, :].reshape(self.grid_size)
            pred_conf = torch.sigmoid(pred_conf)
            
            index = torch.from_numpy(np.asarray(np.unravel_index(np.arange(self.grid_size), (W, H, D)))).type(torch.FloatTensor)
            u = index[0, :].unsqueeze(0).expand(21, -1)
            v = index[1, :].unsqueeze(0).expand(21, -1)
            z = index[2, :].unsqueeze(0).expand(21, -1)       
            
            if self.device != "cpu":
                u = u.cuda()
                v = v.cuda()
                z = z.cuda()        
            
            pred_uvd[self.hand_root, :, :] = torch.sigmoid(pred_uvd[self.hand_root, :, :])
            pred_uvd[:, 0, :] = (pred_uvd[:, 0, :] + u) / W
            pred_uvd[:, 1, :] = (pred_uvd[:, 1, :] + v) / H
            pred_uvd[:, 2, :] = (pred_uvd[:, 2, :] + z) / D
            
            topk_pred_uvd = []
            best_pred_uvd = []
            topk_idx = torch.topk(pred_conf, 10)[1]
            for idx in topk_idx:
                topk_pred_uvd.append(pred_uvd[:, :, idx].cpu().numpy())
            self.best_pred_uvd_list.append(topk_pred_uvd[0])
            self.topk_pred_uvd_list.append(topk_pred_uvd)
            self.pred_conf_list.append(pred_conf.cpu().numpy())
                
        pred = output[0]
        pred = pred.cpu() # faster with cpu
        max_boxes = 10 # detect at most 10 boxes
            
        batch_boxes = YOLO.get_region_boxes(pred, 
                                            self.pred_conf_thresh, 
                                            0, 
                                            self.anchors, 
                                            self.num_anchors, 
                                            is_cuda = False,
                                            is_time = self.time)

        for i in range(batch_size):
            boxes = batch_boxes[i]
            boxes = YOLO.nms_torch(boxes, self.pred_nms_thresh)

            all_boxes = np.zeros((max_boxes, 5))
            if len(boxes) != 0:
                
                if len(boxes) > len(all_boxes):
                    fill_range = len(all_boxes)
                else:
                    fill_range = len(boxes)
                    
                for i in range(fill_range):
                    box = boxes[i]
                    all_boxes[i] = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
            all_boxes = np.reshape(all_boxes, -1)
            self.pred_list.append(all_boxes)
                
    def save_predictions(self, data_split):
        if self.logger:
            self.logger.log("WRITING PREDICTIONS")
            
        write_dir = os.path.join(self.save_dir, "predict_{}_{}_best.txt".format(self.load_epoch, data_split))    
        np.savetxt(write_dir, np.asarray(np.reshape(self.best_pred_uvd_list, (-1, 63))))  
        
        write_dir = os.path.join(self.save_dir, "predict_{}_{}_topk.txt".format(self.load_epoch, data_split))      
        np.savetxt(write_dir, np.asarray(np.reshape(self.topk_pred_uvd_list, (-1, 630))))
        
        write_dir = os.path.join(self.save_dir, "predict_{}_{}_conf.txt".format(self.load_epoch, data_split))       
        np.savetxt(write_dir, np.asarray(np.reshape(self.pred_conf_list, (-1, self.grid_size))))

        self.best_pred_uvd_list = []
        self.topk_pred_uvd_list = []
        self.pred_conf_list = []                      
            
        pred_file = os.path.join(self.save_dir, "predict_{}_{}.txt".format(self.load_epoch, data_split))
        np.savetxt(pred_file, self.pred_list)
        pred_uvd_file = os.path.join(self.save_dir, "predict_{}_{}_uvd.txt".format(self.load_epoch, data_split))
        np.savetxt(pred_uvd_file, self.pred_uvd_list)
        
        self.pred_uvd_list = []
        self.pred_list = []
        if self.logger:
            self.logger.log("FINISHING WRITING PREDICTIONS")
        
    def save_ckpt(self, epoch):
        if self.logger:
            self.logger.log(f"SAVING CHECKPOINT EPOCH {epoch}")
        if isinstance(self.net, nn.parallel.DistributedDataParallel):
            network = self.net.module
        else:
            network = self.net
        model_state_dict = network.state_dict()
        for key, param in model_state_dict.items():
            model_state_dict[key] = param.cpu()
        state = {"epoch": epoch, 
                 "model_state_dict": model_state_dict, 
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "seen": self.seen}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.save_ckpt_dir, f"model_{epoch}.state"))    