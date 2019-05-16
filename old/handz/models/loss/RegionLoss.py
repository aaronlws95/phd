import torch
import torch.nn as nn
import numpy as np
import sys
import os
import math
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import YOLO_utils as YOLO

class RegionLoss(torch.nn.Module):
    def __init__(self, conf, device):
        super(RegionLoss,self).__init__()
        self.batch_size = None
        self.W = None
        self.H = None

        self.device = device
        self.object_scale = conf["object_scale"]
        self.no_object_scale = conf["no_object_scale"]        
        self.anchors = [float(i) for i in conf["anchors"]]
        self.anchor_step = 2
        self.num_anchors = len(self.anchors)//self.anchor_step
        self.num_classes = conf["classes"]
        self.class_scale = conf["class_scale"]
        self.coord_scale = conf["coord_scale"]
        self.sil_thresh = conf["sil_thresh"]
        self.seen = 0
        self.time = conf["time"]
        
    def get_conf_mask(self, target, pred_boxes):
        conf_mask   = torch.ones(self.batch_size, self.num_anchors, self.W, self.H)*self.no_object_scale 
        for batch in range(self.batch_size):
            cur_pred_boxes = pred_boxes[batch].permute(1, 0, 2, 3).contiguous().view(4, -1)
            cur_ious = torch.zeros(self.num_anchors*self.W*self.H)
            # interate max_boxes = 50 times
            for label in target[batch]:
                if label[1] == 0:
                    break
                gx = label[1]*self.W
                gy = label[2]*self.H
                gw = label[3]*self.W
                gh = label[4]*self.H
                cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(self.num_anchors*self.W*self.H,1).t()
                
                cur_ious = torch.max(cur_ious, YOLO.multi_bbox_iou(cur_pred_boxes, cur_gt_boxes))
            # set conf_mask location for IOU > sil_thresh to 0
            # we only want locations where we are confident there is an object
            # or there is no object
            conf_mask[batch][cur_ious.view_as(conf_mask[batch])>self.sil_thresh] = 0
        return conf_mask
        
    def get_target(self, target, pred_boxes):
        t0 = time.time()
        target_box = torch.zeros(self.batch_size, self.num_anchors, 4, self.H, self.W).type(torch.FloatTensor)
        coord_mask = torch.zeros(self.batch_size, self.num_anchors, self.H, self.W).type(torch.FloatTensor)
        cls_mask = torch.zeros(self.batch_size, self.num_anchors, self.H, self.W).type(torch.FloatTensor)
        target_conf = torch.zeros(self.batch_size, self.num_anchors, self.H, self.W)
        target_cls = torch.zeros(self.batch_size, self.num_anchors, self.H, self.W).type(torch.FloatTensor)
        conf_mask = self.get_conf_mask(target, pred_boxes)

        # for first few samples calculate loss with all target x,y coordinates
        if self.seen < 12800:
            target_box[:, :, 0, :, :].fill_(0.5)
            target_box[:, :, 1, :, :].fill_(0.5)
            target_box[:, :, 2, :, :].zero_()
            target_box[:, :, 3, :, :].zero_()
            coord_mask.fill_(1)
        t1 = time.time()
        
        for batch in range(self.batch_size):
            # interate max_boxes = 50 times
            for label in target[batch]:
                if label[1] == 0:
                    break
                gx = (label[1]*self.W)
                gy = (label[2]*self.H)
                gw = (label[3]*self.W)
                gh = (label[4]*self.H)
                gi = int(gx)
                gj = int(gy)
                    
                gt_box = [0, 0, gw, gh]
                best_iou = 0.0
                for i in range(self.num_anchors):
                    aw = self.anchors[self.anchor_step*i]
                    ah = self.anchors[self.anchor_step*i + 1]

                    anchor_box = [0, 0, aw, ah] 
                    iou  = YOLO.bbox_iou(anchor_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor_idx = i
                        
                gt_box = [gx.type(torch.FloatTensor), gy.type(torch.FloatTensor), gw.type(torch.FloatTensor), gh.type(torch.FloatTensor)]
                pred_box = pred_boxes[batch, best_anchor_idx, :, gj, gi]
                
                coord_mask[batch, best_anchor_idx, gj, gi] = 1
                cls_mask[batch, best_anchor_idx, gj, gi] = 1
                conf_mask[batch, best_anchor_idx, gj, gi] = self.object_scale
                target_box[batch, best_anchor_idx, 0, gj, gi] = gx - gi
                target_box[batch, best_anchor_idx, 1, gj, gi] = gy - gj
                target_box[batch, best_anchor_idx, 2, gj, gi] = math.log(gw/self.anchors[self.anchor_step*best_anchor_idx])
                target_box[batch, best_anchor_idx, 3, gj, gi] = math.log(gh/self.anchors[self.anchor_step*best_anchor_idx+1])
                iou = YOLO.bbox_iou(gt_box, pred_box)
                target_conf[batch, best_anchor_idx, gj, gi] = iou
                target_cls[batch, best_anchor_idx, gj, gi] = label[0]
                
        t2 = time.time()
        if self.time:
            print('------get_target-----')
            print('        activation : %f' % (t1 - t0))
            print('    get target_box : %f' % (t2 - t1))
            print('             total : %f' % (t2 - t0))        
  
        return coord_mask, conf_mask, cls_mask, target_box, target_conf, target_cls
    
    def forward(self, pred, target, train_out):
        t0 = time.time() # start
        self.batch_size = pred.shape[0]
        self.H = pred.shape[2] 
        self.W = pred.shape[3]

        pred = pred.reshape((self.batch_size, 
                            self.num_anchors,
                            self.num_classes+5,
                            self.W,
                            self.H))
        
        pred_boxes_no_offset = pred[:, :, :4, :, :]
        pred_boxes_no_offset[:, :, :2, :, :] = torch.sigmoid(pred_boxes_no_offset[:, :, :2, :, :]) 
        
        pred_conf = torch.sigmoid(pred[:, :, 4, :, :])
        pred_cls = pred[:, :, 5:, :, :]
        t1 = time.time() # activation
        grid_x = torch.linspace(0, self.W-1, self.W).repeat(self.H,1).repeat(self.batch_size*self.num_anchors, 1, 1).view((self.batch_size, self.num_anchors, self.H, self.W))
        grid_y = torch.linspace(0, self.H-1, self.H).repeat(self.W,1).t().repeat(self.batch_size*self.num_anchors, 1, 1).view((self.batch_size, self.num_anchors, self.H, self.W))
        anchor_w = torch.Tensor(self.anchors).view(self.num_anchors, self.anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(self.anchors).view(self.num_anchors, self.anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(self.batch_size, 1).repeat(1, 1, self.H*self.W).view((self.batch_size, self.num_anchors, self.H, self.W))
        anchor_h = anchor_h.repeat(self.batch_size, 1).repeat(1, 1, self.H*self.W).view((self.batch_size, self.num_anchors, self.H, self.W))

        if self.device != "cpu":
            grid_x = grid_x.cuda()
            grid_y = grid_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        pred_boxes = pred_boxes_no_offset.clone().detach()
        pred_boxes[:, :, 0, :, :] = pred_boxes[:, :, 0, :, :] + grid_x 
        pred_boxes[:, :, 1, :, :] = pred_boxes[:, :, 1, :, :] + grid_y
        pred_boxes[:, :, 2, :, :] = torch.exp(pred_boxes[:, :, 2, :, :]) * anchor_w
        pred_boxes[:, :, 3, :, :] = torch.exp(pred_boxes[:, :, 3, :, :]) * anchor_h   
        t2 = time.time() # create pred_boxes
        
        if self.device != "cpu":
            target = target.cpu()
            pred_boxes = pred_boxes.cpu()
        
        coord_mask, conf_mask, cls_mask, target_box, target_conf, target_cls = self.get_target(target, pred_boxes)
        
        cls_mask = (cls_mask == 1)
        target_cls = target_cls[cls_mask].long()
        cls_mask = cls_mask.view(-1, 1).repeat(1, self.num_classes)

        pred_cls = pred_cls.view(self.batch_size*self.num_anchors, self.num_classes, self.H*self.W).transpose(1,2).contiguous().view(self.batch_size*self.num_anchors*self.H*self.W, self.num_classes)
        
        if self.device != "cpu":
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            cls_mask = cls_mask.cuda()
            target_box = target_box.cuda()
            target_conf = target_conf.cuda()
            target_cls = target_cls.cuda()          
        
        pred_cls = pred_cls[cls_mask].view(-1,  self.num_classes)  
        conf_mask = conf_mask.sqrt()
        t3 = time.time() # get targets
        
        mseloss = nn.MSELoss(reduction="sum")
        loss_x = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 0, :, :]*coord_mask, target_box[:, :, 0, :, :]*coord_mask)/2.0
        loss_y = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 1, :, :]*coord_mask, target_box[:, :, 1, :, :]*coord_mask)/2.0
        loss_w = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 2, :, :]*coord_mask, target_box[:, :, 2, :, :]*coord_mask)/2.0
        loss_h = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 3, :, :]*coord_mask, target_box[:, :, 3, :, :]*coord_mask)/2.0
        loss_conf = mseloss(pred_conf*conf_mask, target_conf*conf_mask)/2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(reduction="sum")(pred_cls, target_cls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if self.time:
            print('-----RegionLoss-----')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('       get targets : %f' % (t3 - t2))
            print('         calc loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        if train_out:
            return loss, loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls
        else:
            return loss
        
class RegionLoss_1Class(torch.nn.Module):
    def __init__(self, conf, device):
        super(RegionLoss_1Class,self).__init__()
        self.batch_size = None
        self.W = None
        self.H = None   

        self.device = device
        self.object_scale = conf["object_scale"]
        self.no_object_scale = conf["no_object_scale"]        
        self.anchors = [float(i) for i in conf["anchors"]]
        self.anchor_step = 2
        self.num_anchors = len(self.anchors)//self.anchor_step
        self.coord_scale = conf["coord_scale"]
        self.sil_thresh = conf["sil_thresh"]
        self.seen = 0
        self.time = conf["time"]
        
    def get_conf_mask(self, target, pred_boxes):
        conf_mask   = torch.ones(self.batch_size, self.num_anchors, self.W, self.H)*self.no_object_scale 
        for batch in range(self.batch_size):
            cur_pred_boxes = pred_boxes[batch].permute(1, 0, 2, 3).contiguous().view(4, -1)
            cur_ious = torch.zeros(self.num_anchors*self.W*self.H)
            gx = target[batch][0]*self.W
            gy = target[batch][1]*self.H
            gw = target[batch][2]*self.W
            gh = target[batch][3]*self.H
            
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(self.num_anchors*self.W*self.H,1).t()
            cur_ious = torch.max(cur_ious, YOLO.multi_bbox_iou(cur_pred_boxes, cur_gt_boxes))
            conf_mask[batch][cur_ious.view_as(conf_mask[batch])>self.sil_thresh] = 0
        return conf_mask
        
    def get_target(self, target, pred_boxes):
        t0 = time.time()
        target_box = torch.zeros(self.batch_size, self.num_anchors, 4, self.H, self.W).type(torch.FloatTensor)
        coord_mask = torch.zeros(self.batch_size, self.num_anchors, self.H, self.W).type(torch.FloatTensor)
        target_conf = torch.zeros(self.batch_size, self.num_anchors, self.H, self.W)
        conf_mask = self.get_conf_mask(target, pred_boxes)
 
        # As the bounding boxes coordinates prediction need to align with our prior information, 
        # a loss term reducing the difference between prior and the predicted is added for few iterations (t<12800)
        if self.seen < 12800:
            target_box[:, :, 0, :, :].fill_(0.5)
            target_box[:, :, 1, :, :].fill_(0.5)
            target_box[:, :, 2, :, :].zero_()
            target_box[:, :, 3, :, :].zero_()
            coord_mask.fill_(1)
        t1 = time.time()
        
        for batch in range(self.batch_size):
            # interate max_boxes = 50 times
  
            gx = (target[batch][0]*self.W)
            gy = (target[batch][1]*self.H)
            gw = (target[batch][2]*self.W)
            gh = (target[batch][3]*self.H)
            gi = int(gx)
            gj = int(gy)
                
            gt_box = [0, 0, gw, gh]
            best_iou = 0.0
            for i in range(self.num_anchors):
                aw = self.anchors[self.anchor_step*i]
                ah = self.anchors[self.anchor_step*i + 1]

                anchor_box = [0, 0, aw, ah] 
                iou  = YOLO.bbox_iou(anchor_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor_idx = i
            
            gt_box = [gx.type(torch.FloatTensor), gy.type(torch.FloatTensor), gw.type(torch.FloatTensor), gh.type(torch.FloatTensor)]
            pred_box = pred_boxes[batch, best_anchor_idx, :, gj, gi]
            
            coord_mask[batch, best_anchor_idx, gj, gi] = 1
            conf_mask[batch, best_anchor_idx, gj, gi] = self.object_scale
            target_box[batch, best_anchor_idx, 0, gj, gi] = gx - gi
            target_box[batch, best_anchor_idx, 1, gj, gi] = gy - gj
            target_box[batch, best_anchor_idx, 2, gj, gi] = math.log(gw/self.anchors[self.anchor_step*best_anchor_idx])
            target_box[batch, best_anchor_idx, 3, gj, gi] = math.log(gh/self.anchors[self.anchor_step*best_anchor_idx+1])
            iou = YOLO.bbox_iou(gt_box, pred_box)
            target_conf[batch, best_anchor_idx, gj, gi] = iou
                
        t2 = time.time()
        if self.time:
            print('------get_target-----')
            print('        activation : %f' % (t1 - t0))
            print('    get target_box : %f' % (t2 - t1))
            print('             total : %f' % (t2 - t0))        
  
        return coord_mask, conf_mask, target_box, target_conf
    
    def forward(self, pred, target, train_out):
        t0 = time.time() # start
        self.batch_size = pred.shape[0]
        self.H = pred.shape[2] 
        self.W = pred.shape[3]

        pred = pred.reshape((self.batch_size, self.num_anchors, 5, self.W, self.H))
        
        pred_boxes_no_offset = pred[:, :, :4, :, :]
        pred_boxes_no_offset[:, :, :2, :, :] = torch.sigmoid(pred_boxes_no_offset[:, :, :2, :, :]) 
        pred_conf = torch.sigmoid(pred[:, :, 4, :, :])
        
        t1 = time.time() # activation
        grid_x = torch.linspace(0, self.W-1, self.W).repeat(self.H,1).repeat(self.batch_size*self.num_anchors, 1, 1).view((self.batch_size, self.num_anchors, self.H, self.W))
        grid_y = torch.linspace(0, self.H-1, self.H).repeat(self.W,1).t().repeat(self.batch_size*self.num_anchors, 1, 1).view((self.batch_size, self.num_anchors, self.H, self.W))
        anchor_w = torch.Tensor(self.anchors).view(self.num_anchors, self.anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(self.anchors).view(self.num_anchors, self.anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(self.batch_size, 1).repeat(1, 1, self.H*self.W).view((self.batch_size, self.num_anchors, self.H, self.W))
        anchor_h = anchor_h.repeat(self.batch_size, 1).repeat(1, 1, self.H*self.W).view((self.batch_size, self.num_anchors, self.H, self.W))

        if self.device != "cpu":
            grid_x = grid_x.cuda()
            grid_y = grid_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        pred_boxes = pred_boxes_no_offset.clone().detach()
        pred_boxes[:, :, 0, :, :] = pred_boxes[:, :, 0, :, :] + grid_x 
        pred_boxes[:, :, 1, :, :] = pred_boxes[:, :, 1, :, :] + grid_y
        pred_boxes[:, :, 2, :, :] = torch.exp(pred_boxes[:, :, 2, :, :]) * anchor_w
        pred_boxes[:, :, 3, :, :] = torch.exp(pred_boxes[:, :, 3, :, :]) * anchor_h   
        t2 = time.time() # create pred_boxes
        
        if self.device != "cpu":
            target = target.cpu()
            pred_boxes = pred_boxes.cpu()
        
        coord_mask, conf_mask, target_box, target_conf = self.get_target(target, pred_boxes)
        
        if self.device != "cpu":
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            target_box = target_box.cuda()
            target_conf = target_conf.cuda()     
        
        conf_mask = conf_mask.sqrt()
        t3 = time.time() # get targets
        
        mseloss = nn.MSELoss(reduction="sum")
        loss_x = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 0, :, :]*coord_mask, target_box[:, :, 0, :, :]*coord_mask)/2.0
        loss_y = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 1, :, :]*coord_mask, target_box[:, :, 1, :, :]*coord_mask)/2.0
        loss_w = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 2, :, :]*coord_mask, target_box[:, :, 2, :, :]*coord_mask)/2.0
        loss_h = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 3, :, :]*coord_mask, target_box[:, :, 3, :, :]*coord_mask)/2.0
        loss_conf = mseloss(pred_conf*conf_mask, target_conf*conf_mask)/2.0
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf
        t4 = time.time()
        if self.time:
            print('-----RegionLoss-----')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('       get targets : %f' % (t3 - t2))
            print('         calc loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        if train_out:
            return loss, loss_x, loss_y, loss_w, loss_h, loss_conf
        else:
            return loss

class RegionLoss_1Class_reg(torch.nn.Module):
    def __init__(self, conf, device):
        super(RegionLoss_1Class_reg,self).__init__()
        self.batch_size = None
        self.W = None
        self.H = None   

        self.device = device
        self.object_scale = conf["object_scale"]
        self.no_object_scale = conf["no_object_scale"]        
        self.anchors = [float(i) for i in conf["anchors"]]
        self.anchor_step = 2
        self.num_anchors = len(self.anchors)//self.anchor_step
        self.coord_scale = conf["coord_scale"]
        self.sil_thresh = conf["sil_thresh"]
        self.seen = 0
        self.time = conf["time"]
        
    def get_conf_mask(self, target, pred_boxes):
        conf_mask   = torch.ones(self.batch_size, self.num_anchors, self.W, self.H)*self.no_object_scale 
        for batch in range(self.batch_size):
            cur_pred_boxes = pred_boxes[batch].permute(1, 0, 2, 3).contiguous().view(4, -1)
            cur_ious = torch.zeros(self.num_anchors*self.W*self.H)
            gx = target[batch][0]*self.W
            gy = target[batch][1]*self.H
            gw = target[batch][2]*self.W
            gh = target[batch][3]*self.H
            
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(self.num_anchors*self.W*self.H,1).t()
            cur_ious = torch.max(cur_ious, YOLO.multi_bbox_iou(cur_pred_boxes, cur_gt_boxes))
            conf_mask[batch][cur_ious.view_as(conf_mask[batch])>self.sil_thresh] = 0
        return conf_mask
        
    def get_target(self, target, pred_boxes):
        t0 = time.time()
        target_box = torch.zeros(self.batch_size, self.num_anchors, 4, self.H, self.W).type(torch.FloatTensor)
        coord_mask = torch.zeros(self.batch_size, self.num_anchors, self.H, self.W).type(torch.FloatTensor)
        target_conf = torch.zeros(self.batch_size, self.num_anchors, self.H, self.W)
        conf_mask = self.get_conf_mask(target, pred_boxes)
 
        # As the bounding boxes coordinates prediction need to align with our prior information, 
        # a loss term reducing the difference between prior and the predicted is added for few iterations (t<12800)
        if self.seen < 12800:
            target_box[:, :, 0, :, :].fill_(0.5)
            target_box[:, :, 1, :, :].fill_(0.5)
            target_box[:, :, 2, :, :].zero_()
            target_box[:, :, 3, :, :].zero_()
            coord_mask.fill_(1)
        t1 = time.time()
        
        for batch in range(self.batch_size):
            # interate max_boxes = 50 times
  
            gx = (target[batch][0]*self.W)
            gy = (target[batch][1]*self.H)
            gw = (target[batch][2]*self.W)
            gh = (target[batch][3]*self.H)
            gi = int(gx)
            gj = int(gy)
                
            gt_box = [0, 0, gw, gh]
            best_iou = 0.0
            for i in range(self.num_anchors):
                aw = self.anchors[self.anchor_step*i]
                ah = self.anchors[self.anchor_step*i + 1]

                anchor_box = [0, 0, aw, ah] 
                iou  = YOLO.bbox_iou(anchor_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor_idx = i
            
            gt_box = [gx.type(torch.FloatTensor), gy.type(torch.FloatTensor), gw.type(torch.FloatTensor), gh.type(torch.FloatTensor)]
            pred_box = pred_boxes[batch, best_anchor_idx, :, gj, gi]
            
            coord_mask[batch, best_anchor_idx, gj, gi] = 1
            conf_mask[batch, best_anchor_idx, gj, gi] = self.object_scale
            target_box[batch, best_anchor_idx, 0, gj, gi] = gx - gi
            target_box[batch, best_anchor_idx, 1, gj, gi] = gy - gj
            target_box[batch, best_anchor_idx, 2, gj, gi] = math.log(gw/self.anchors[self.anchor_step*best_anchor_idx])
            target_box[batch, best_anchor_idx, 3, gj, gi] = math.log(gh/self.anchors[self.anchor_step*best_anchor_idx+1])
            iou = YOLO.bbox_iou(gt_box, pred_box)
            target_conf[batch, best_anchor_idx, gj, gi] = iou
                
        t2 = time.time()
        if self.time:
            print('------get_target-----')
            print('        activation : %f' % (t1 - t0))
            print('    get target_box : %f' % (t2 - t1))
            print('             total : %f' % (t2 - t0))        
  
        return coord_mask, conf_mask, target_box, target_conf
    
    def forward(self, pred_list, target, uvd_gt, train_out):
        t0 = time.time() # start
        
        pred = pred_list[0]        
        pred_uvd = pred_list[1]
        
        self.batch_size = pred.shape[0]
        self.H = pred.shape[2] 
        self.W = pred.shape[3]

        pred = pred.reshape((self.batch_size, self.num_anchors, 5, self.W, self.H))
        
        pred_boxes_no_offset = pred[:, :, :4, :, :]
        pred_boxes_no_offset[:, :, :2, :, :] = torch.sigmoid(pred_boxes_no_offset[:, :, :2, :, :]) 
        pred_conf = torch.sigmoid(pred[:, :, 4, :, :])
        
        t1 = time.time() # activation
        grid_x = torch.linspace(0, self.W-1, self.W).repeat(self.H,1).repeat(self.batch_size*self.num_anchors, 1, 1).view((self.batch_size, self.num_anchors, self.H, self.W))
        grid_y = torch.linspace(0, self.H-1, self.H).repeat(self.W,1).t().repeat(self.batch_size*self.num_anchors, 1, 1).view((self.batch_size, self.num_anchors, self.H, self.W))
        anchor_w = torch.Tensor(self.anchors).view(self.num_anchors, self.anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(self.anchors).view(self.num_anchors, self.anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(self.batch_size, 1).repeat(1, 1, self.H*self.W).view((self.batch_size, self.num_anchors, self.H, self.W))
        anchor_h = anchor_h.repeat(self.batch_size, 1).repeat(1, 1, self.H*self.W).view((self.batch_size, self.num_anchors, self.H, self.W))

        if self.device != "cpu":
            grid_x = grid_x.cuda()
            grid_y = grid_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        pred_boxes = pred_boxes_no_offset.clone().detach()
        pred_boxes[:, :, 0, :, :] = pred_boxes[:, :, 0, :, :] + grid_x 
        pred_boxes[:, :, 1, :, :] = pred_boxes[:, :, 1, :, :] + grid_y
        pred_boxes[:, :, 2, :, :] = torch.exp(pred_boxes[:, :, 2, :, :]) * anchor_w
        pred_boxes[:, :, 3, :, :] = torch.exp(pred_boxes[:, :, 3, :, :]) * anchor_h   
        t2 = time.time() # create pred_boxes
        
        if self.device != "cpu":
            target = target.cpu()
            pred_boxes = pred_boxes.cpu()
        
        coord_mask, conf_mask, target_box, target_conf = self.get_target(target, pred_boxes)
        
        if self.device != "cpu":
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            target_box = target_box.cuda()
            target_conf = target_conf.cuda()     
        
        conf_mask = conf_mask.sqrt()
        t3 = time.time() # get targets
        
        mseloss = nn.MSELoss(reduction="sum")
        loss_x = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 0, :, :]*coord_mask, target_box[:, :, 0, :, :]*coord_mask)/2.0
        loss_y = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 1, :, :]*coord_mask, target_box[:, :, 1, :, :]*coord_mask)/2.0
        loss_w = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 2, :, :]*coord_mask, target_box[:, :, 2, :, :]*coord_mask)/2.0
        loss_h = self.coord_scale * mseloss(pred_boxes_no_offset[:, :, 3, :, :]*coord_mask, target_box[:, :, 3, :, :]*coord_mask)/2.0
        loss_conf = mseloss(pred_conf*conf_mask, target_conf*conf_mask)/2.0
        uvd_gt = uvd_gt.reshape(self.batch_size, -1)
        loss_hand = mseloss(uvd_gt, pred_uvd)/2.0
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_hand
        t4 = time.time()
        if self.time:
            print('-----RegionLoss-----')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('       get targets : %f' % (t3 - t2))
            print('         calc loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        if train_out:
            return loss, loss_x, loss_y, loss_w, loss_h, loss_conf, loss_hand
        else:
            return loss           