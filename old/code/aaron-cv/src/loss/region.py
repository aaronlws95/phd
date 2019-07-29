import torch
import time
import math
import torch.nn as nn

from src.utils import YOLO

class RegionLoss(torch.nn.Module):
    """ YOLOv2 multi-class multi-bbox detection """
    def __init__(self, cfg):
        super().__init__()
        self.bs                 = None
        self.W                  = None
        self.H                  = None   
        self.object_scale       = float(cfg['object_scale'])
        self.no_object_scale    = float(cfg['no_object_scale'])
        self.anc                = [float(i) for i in cfg['anchors'].split(',')]
        self.anc_step           = 2
        self.na                 = len(self.anc)//self.anc_step
        self.cs                 = float(cfg['coord_scale'])
        self.sil_thresh         = float(cfg['sil_thresh'])
        self.epoch              = 0
        self.debug              = cfg['debug']
        self.time               = cfg['time']
        self.nc                 = int(cfg['classes'])
        self.class_scale        = float(cfg['class_scale'])
        
    def get_conf_mask(self, target, pred_boxes_ofs):
        """
        Get confidence mask
        Args:
            target          : Target bbox scaled to grid size (b, [x, y, w, h])
            pred_boxes_ofs  : Predicted bbox with grid offset (b, na, 4, H, W)
        Out:
            conf_mask       : Confidence mask, all set to no_object_scale
                              except for the cases where the iou > sil_thresh
                              then set to 0. Will fill the target
                              location later. For each target iterate over all
                              labels to get the maximum iou for comparison
                              (b, na, H, W)
        """        
        FT          = torch.FloatTensor
        conf_mask   = torch.ones(self.bs, 
                                 self.na, 
                                 self.W, 
                                 self.H)*self.no_object_scale 
        
        for batch in range(self.bs):
            # if self.debug:
            #     if batch == 0:
            #         pred_boxes_ofs = torch.zeros(pred_boxes_ofs.shape)
            #         pred_boxes_ofs[0, 3, :, 11, 11] = 13 
            #         target = torch.ones(target.shape)*100
            #         target[0, :] = 1
                    
            cur_pred_boxes_ofs  = pred_boxes_ofs[batch].permute(1, 0, 2, 3)
            cur_pred_boxes_ofs  = cur_pred_boxes_ofs.contiguous().view(4, -1)
            iou                 = torch.zeros(self.na*self.W*self.H)
            # iterate max_boxes = 50 times
            for label in target[batch]:
                # Check if we have run out of labels
                if label[1] == 0:
                    break
                gx              = label[1]*self.W
                gy              = label[2]*self.H
                gw              = label[3]*self.W
                gh              = label[4]*self.H
                cur_gt_boxes    = FT([gx,gy,gw,gh])
                cur_gt_boxes    = cur_gt_boxes.repeat(self.na*self.W*self.H,
                                                      1).t()

                iou             = torch.max(iou, 
                                            YOLO.multi_bbox_iou(
                                                cur_pred_boxes_ofs, 
                                                cur_gt_boxes))
            #     if self.debug:
            #         if batch == 0:
            #             print(iou)
            #             print(torch.argmax(iou))
            # if self.debug:
            #     if batch  == 0:
            #         print(iou.view_as(conf_mask[batch]))
            
            iou = iou.view_as(conf_mask[batch])
            conf_mask[batch][iou > self.sil_thresh] = 0
        return conf_mask
    
    def get_target(self, target, pred_boxes_ofs):
        """
        Get target boxes and masks
        Args:
            target          : Target bbox scaled to grid size (b, [cls, x, y, w, h])
            pred_boxes_ofs  : Predicted bbox with grid offset (b, na, 4, H, W)    
        Out:
            Target location refers to the ground truth x, y and the best anchor
            with highest iou with ground truth
            coord_mask      : All 0 except at target locations it is 1 
                              (b, na, H, W)
            cls_mask        : All 0 except at target locations it is 1 
                              (b, na, H, W)
            conf_mask       : From get_conf_mask. Set target locations to 
                              object_scale (b, na, H, W)
            target_box      : Target bounding boxes. Set to 0 unless at 
                              target locations. Processed to match initial 
                              predicted bounding box without grid offset
                              (b, na, 4, H, W)
            target_conf     : All 0 except at target location it is the iou 
                              between the target and predicted bbox 
                              (b, na, H, W)
            target_cls      : All 0 except at target location it is the target
                              class (b, na, H, W)
            
        """        
        t0          = time.time() # start
        FT          = torch.FloatTensor
        Z           = torch.zeros
        init_grid   = FT(Z(self.bs, self.na, self.H, self.W))
        target_box  = FT(Z(self.bs, self.na, 
                           4, self.H, self.W))
        coord_mask  = FT(Z(self.bs, self.na, self.H, self.W))
        cls_mask    = FT(Z(self.bs, self.na, self.H, self.W))
        target_conf = FT(Z(self.bs, self.na, self.H, self.W))
        target_cls  = FT(Z(self.bs, self.na, self.H, self.W))

        conf_mask   = self.get_conf_mask(target, pred_boxes_ofs)

        # for first few samples calculate loss with all target x,y coordinates
        if self.epoch < 1:
            target_box[:, :, 0, :, :].fill_(0.5)
            target_box[:, :, 1, :, :].fill_(0.5)
            target_box[:, :, 2, :, :].zero_()
            target_box[:, :, 3, :, :].zero_()
            coord_mask.fill_(1)
        t1 = time.time() # activation
        
        for batch in range(self.bs):
            # interate max_boxes = 50 times
            for label in target[batch]:
                if label[1] == 0:
                    break
                gx          = label[1]*self.W
                gy          = label[2]*self.H
                gw          = label[3]*self.W
                gh          = label[4]*self.H
                gi          = int(gx)
                gj          = int(gy)
                    
                gt_box      = [0, 0, gw, gh]
                best_iou    = 0.0
                for i in range(self.na):
                    aw      = self.anc[self.anc_step*i]
                    ah      = self.anc[self.anc_step*i + 1]
                    anc_box = [0, 0, aw, ah] 
                    iou     = YOLO.bbox_iou(anc_box, gt_box)
                    if iou > best_iou:
                        best_iou        = iou
                        best_anc_idx    = i
                        
                gt_box      = [gx.type(FT), gy.type(FT), 
                               gw.type(FT), gh.type(FT)]
                pred_box    = pred_boxes_ofs[batch, best_anc_idx, :, gj, gi]
                
                coord_mask[batch, best_anc_idx, gj, gi]     = 1
                cls_mask[batch, best_anc_idx, gj, gi]       = 1
                conf_mask[batch, best_anc_idx, gj, gi]      = self.object_scale
                target_box[batch, best_anc_idx, 0, gj, gi]  = gx - gi
                target_box[batch, best_anc_idx, 1, gj, gi]  = gy - gj
                target_box[batch, best_anc_idx, 2, gj, gi]  = \
                    math.log(gw/self.anc[self.anc_step*best_anc_idx])
                target_box[batch, best_anc_idx, 3, gj, gi]  = \
                    math.log(gh/self.anc[self.anc_step*best_anc_idx+1])
                target_conf[batch, best_anc_idx, gj, gi]    = \
                    YOLO.bbox_iou(gt_box, pred_box)
                target_cls[batch, best_anc_idx, gj, gi]     = label[0]
        t2 = time.time() # get target box
        if self.time:
            print('------get_target-----')
            print('        activation : %f' % (t1 - t0))
            print('    get target_box : %f' % (t2 - t1))
            print('             total : %f' % (t2 - t0))        
        
        return (coord_mask, conf_mask, cls_mask, 
                target_box, target_conf, target_cls)
    
    def forward(self, pred, target):
        """
        Loss calculation and processing
        Args:
            pred    : (b, na*(nc + 5), H, W) 
            target  : (b, 50, [x, y, w, h]) empty boxes will be [0, 0, 0, 0]
        Out:
            loss    : Total loss and its components
        """        
        t0                          = time.time() # start
        self.bs                     = pred.shape[0]
        self.H                      = pred.shape[2] 
        self.W                      = pred.shape[3]
        FT                          = torch.FloatTensor
        LT                          = torch.LongTensor
        exp                         = torch.exp
        pred                        = pred.view((self.bs, 
                                                 self.na,
                                                 self.nc + 5,
                                                 self.H,
                                                 self.W))
        
        # if self.debug:
        #     pred = torch.zeros(pred.shape).cuda()
        #     pred[0, 0, 5+15, 11, 11] = 1
        #     pred[0, 0, :4, 11, 11] = 1
        #     pred[0, 0, 4, 11, 11] = 1
        #     pred[0, 0, 5, 12, 12] = 1
        #     pred[0, 0, :4, 12, 12] = 1
        #     pred[0, 0, 4, 12, 12] = 1
        #     target = torch.zeros(target.shape).cuda()
        #     target[0, 0, 0] = 15
        #     target[0, 0, 1] = 11/13
        #     target[0, 0, 2] = 11/13
        #     target[0, 0, 3] = 1/13
        #     target[0, 0, 4] = 1/13
        #     target[0, 1, 0] = 11
        #     target[0, 1, 1] = 12/13
        #     target[0, 1, 2] = 12/13
        #     target[0, 1, 3] = 1/13
        #     target[0, 1, 4] = 1/13
            
        pred_boxes                  = pred[:, :, :4, :, :]
        pred_conf                   = torch.sigmoid(pred[:, :, 4, :, :])
        pred_cls                    = pred[:, :, 5:, :, :]
        pred_boxes[:, :, :2, :, :]  = torch.sigmoid(pred_boxes[:, :, :2, :, :])
  
        t1      = time.time() # activation
        yv, xv  = torch.meshgrid([torch.arange(self.H), torch.arange(self.W)])
        grid_x  = xv.repeat((1, self.na, 1, 1)).type(FT)
        grid_y  = yv.repeat((1, self.na, 1, 1)).type(FT)
        anc     = torch.Tensor(self.anc).view(self.na, self.anc_step)
        anc_w   = anc[:, 0].view(1, self.na, 1, 1).type(FT)
        anc_h   = anc[:, 1].view(1, self.na, 1, 1).type(FT)
        
        grid_x  = grid_x.cuda()
        grid_y  = grid_y.cuda()
        anc_w   = anc_w.cuda()
        anc_h   = anc_h.cuda()

        pred_boxes_ofs = pred_boxes.clone().detach() # B, A, 4, H, W
        pred_boxes_ofs[:, :, 0, :, :] = pred_boxes_ofs[:, :, 0, :, :] + grid_x
        pred_boxes_ofs[:, :, 1, :, :] = pred_boxes_ofs[:, :, 1, :, :] + grid_y
        pred_boxes_ofs[:, :, 2, :, :] = \
            exp(pred_boxes_ofs[:, :, 2, :, :])*anc_w
        pred_boxes_ofs[:, :, 3, :, :] = \
            exp(pred_boxes_ofs[:, :, 3, :, :])*anc_h
        t2 = time.time() # create pred_boxes_ofs

        # CPU better for loops
        target = target.cpu()
        pred_boxes_ofs = pred_boxes_ofs.cpu()
        
        # if self.debug:
        #     print("get_target shouldn't require_grad:", 
        #           target.requires_grad, pred_boxes_ofs.requires_grad)
            
        coord_mask, conf_mask, cls_mask, \
        target_box, target_conf, target_cls = \
            self.get_target(target, pred_boxes_ofs)

        # if self.debug:
        #     print('pred_cls')
        #     print(pred_cls[0, 0, :, 12, 12])
        #     print('target')
        #     print(target[0])
        #     print('cls_mask')
        #     print(cls_mask[0, 0])

        cls_mask    = (cls_mask == 1)
        target_cls  = target_cls[cls_mask].long()
        cls_mask    = cls_mask.view(-1, 1).repeat(1, self.nc)
        pred_cls    = pred_cls.view(self.bs*self.na, self.nc, self.H*self.W)
        pred_cls    = pred_cls.transpose(1, 2).contiguous()
        pred_cls    = pred_cls.view(self.bs*self.na*self.H*self.W, self.nc)
        pred_cls    = pred_cls[cls_mask].view(-1,  self.nc) 
       
        # if self.debug:
        #     print('target_cls')
        #     print(target_cls)
        #     print('pred_cls')
        #     print(pred_cls)

        coord_mask  = coord_mask.cuda()
        conf_mask   = conf_mask.cuda()
        cls_mask    = cls_mask.cuda()
        target_box  = target_box.cuda()
        target_conf = target_conf.cuda()
        target_cls  = target_cls.cuda()
        conf_mask   = conf_mask.sqrt()
        
        t3          = time.time() # get targets
        celoss      = nn.CrossEntropyLoss(reduction='sum')
        mseloss     = nn.MSELoss(reduction='sum')
        loss_x      = self.cs*mseloss(pred_boxes[:, :, 0, :, :]*coord_mask,
                                      target_box[:, :, 0, :, :]*coord_mask)/2.0
        loss_y      = self.cs*mseloss(pred_boxes[:, :, 1, :, :]*coord_mask,
                                      target_box[:, :, 1, :, :]*coord_mask)/2.0
        loss_w      = self.cs*mseloss(pred_boxes[:, :, 2, :, :]*coord_mask,
                                      target_box[:, :, 2, :, :]*coord_mask)/2.0
        loss_h      = self.cs*mseloss(pred_boxes[:, :, 3, :, :]*coord_mask,
                                      target_box[:, :, 3, :, :]*coord_mask)/2.0
        loss_conf   = mseloss(pred_conf*conf_mask, target_conf*conf_mask)/2.0
        loss_cls    = self.class_scale*celoss(pred_cls, target_cls)
        
        # if self.debug:
        #     print('Should require gradient')
        #     print(loss_x.requires_grad, loss_y.requires_grad, 
        #           loss_h.requires_grad, loss_conf.requires_grad, 
        #           loss_cls.requires_grad)
        
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        
        t4 = time.time()
        if self.time:
            print('-----RegionLoss-----')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('       get targets : %f' % (t3 - t2))
            print('         calc loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))

        return loss, loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls
   