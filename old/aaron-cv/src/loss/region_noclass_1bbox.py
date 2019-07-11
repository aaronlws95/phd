import torch
import time
import math
import torch.nn as nn

from src.utils import YOLO

class RegionLoss_NoClass_1Bbox(nn.Module):
    """ YOLOv2 loss calculation with no class and only 1 bounding box """
    def __init__(self, cfg):
        super().__init__()
        self.bs                 = None
        self.H                  = None
        self.W                  = None

        self.object_scale       = float(cfg["object_scale"])
        self.no_object_scale    = float(cfg["no_object_scale"])
        self.anc                = [float(i) for i in cfg["anchors"].split(',')]
        self.anc_step           = 2
        self.epoch              = 0
        self.na                 = len(self.anc)//self.anc_step
        self.cs                 = float(cfg["coord_scale"])
        self.sil_thresh         = float(cfg["sil_thresh"])
        self.time               = cfg["time"]
        self.debug              = cfg['debug']

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
                              location later (b, na, H, W)
        """
        bahw        = self.bs, self.na, self.H, self.W
        conf_mask   = (torch.ones(bahw)*self.no_object_scale).cuda()

        # Be careful with view!!!
        gt_box      = target.repeat(self.na*self.H*self.W, 1, 1)
        gt_box      = gt_box.permute(2, 1, 0)
        gt_box      = gt_box.contiguous().view(4, -1)
        pred_box    = pred_boxes_ofs.permute(0, 2, 1, 3, 4)
        pred_box    = pred_box.contiguous().view(self.bs, 4, -1)
        pred_box    = pred_box.permute(1, 0, 2)
        pred_box    = pred_box.contiguous().view(4, -1)
        iou         = YOLO.multi_bbox_iou(gt_box, pred_box)
        iou         = torch.max(torch.zeros(iou.shape).cuda(), iou)
        iou         = iou.view_as(conf_mask)
        conf_mask[iou > self.sil_thresh] = 0

        # if self.debug:
        #     print('Check if permutation and view dont mess things up')
        #     check_conf_mask = torch.ones(bahw)*self.no_object_scale
        #     batch = 0
        #     anc = 0
        #     check = torch.zeros(pred_boxes_ofs.shape)
        #     check[batch, anc, :, 0, 0] = torch.FloatTensor([100, 100, 50, 50])
        #     target_check = torch.zeros(target.shape)
        #     target_check[batch] = torch.FloatTensor([100, 100, 50, 50])
        #     target_check = target_check.repeat(self.na*self.H*self.W, 1, 1)
        #     target_check = target_check.permute(2, 1, 0)
        #     target_check = target_check.contiguous().view(4, -1)
        #     check = check.permute(0, 2, 1, 3, 4)
        #     check = check.contiguous().view(self.bs, 4, -1).permute(1, 0, 2)
        #     check = check.contiguous().view(4, -1)
        #     iou = YOLO.multi_bbox_iou(target_check, check)
        #     iou = torch.max(torch.zeros(iou.shape), iou)
        #     iou = iou.view_as(conf_mask)
        #     check_conf_mask[iou>self.sil_thresh] = 0
        #     # corresponding point should be 0
        #     print(check_conf_mask[batch])
        
        return conf_mask

    def get_target(self, target, pred_boxes_ofs):
        """
        Get target boxes and masks
        Args:
            target          : Target bbox scaled to grid size (b, [x, y, w, h])
            pred_boxes_ofs  : Predicted bbox with grid offset (b, na, 4, H, W)    
        Out:
            Target location refers to the ground truth x, y and the best anchor
            with highest iou with ground truth
            coord_mask      : All 0 except at target locations it is 1 
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
        """
        t0              = time.time()
        FT              = torch.FloatTensor
        LT              = torch.LongTensor
        TZ              = torch.zeros
        target[:, 0]    = target[:, 0]*self.W
        target[:, 1]    = target[:, 1]*self.H
        target[:, 2]    = target[:, 2]*self.W
        target[:, 3]    = target[:, 3]*self.H
        target_box      = FT(TZ(self.bs, self.na, 4, self.H, self.W)).cuda()
        coord_mask      = FT(TZ(self.bs, self.na, self.H, self.W)).cuda()
        target_conf     = FT(TZ(self.bs, self.na, self.H, self.W)).cuda()
        conf_mask       = self.get_conf_mask(target, pred_boxes_ofs)

        # Warm-up
        if self.epoch < 1:
            target_box[:, :, 0, :, :].fill_(0.5)
            target_box[:, :, 1, :, :].fill_(0.5)
            target_box[:, :, 2, :, :].zero_()
            target_box[:, :, 3, :, :].zero_()
            coord_mask.fill_(1)
        t1 = time.time()

        gx          = target[:, 0]
        gy          = target[:, 1]
        gw          = target[:, 2]
        gh          = target[:, 3]
        gi          = gx.type(torch.IntTensor).type(FT)
        gj          = gy.type(torch.IntTensor).type(FT)

        gt_box      = torch.stack((torch.zeros(gw.shape).cuda(), 
                                   torch.zeros(gw.shape).cuda(),
                                   gw, gh))

        gt_box      = gt_box.repeat(self.na, 1, 1)
        gt_box      = gt_box.permute(1, 2, 0).contiguous().view(4, -1)
        anc         = torch.Tensor(self.anc).view(self.na, self.anc_step)
        anc_box     = [[0, 0, aw, ah] for aw, ah in anc]
        anc_box     = FT(anc_box).t().repeat(1, self.bs).cuda()
        ious        = YOLO.multi_bbox_iou(gt_box, anc_box)
        ious        = ious.view(self.bs, -1)
        best_anc_i  = torch.argmax(ious, dim=1)
        best_anc_i  = best_anc_i.type(LT)

        pred_box    = [pred_boxes_ofs[b, anc, :, j.type(LT), i.type(LT)]
                                    for b, (anc, j, i) in 
                                    enumerate(zip(best_anc_i, gj, gi))]
        pred_box    = torch.stack(pred_box)
        iou         = YOLO.multi_bbox_iou(target.t(), pred_box.t())

        for b in range(self.bs):
            gjb = int(gj[b])
            gib = int(gi[b])
            
            # if self.debug:
            #     print(best_anc_i[b], gjb, gib, gx[b], gy[b], \
            #     gw[b], self.anc[self.anc_step*best_anc_i[b]], \
            #     gh[b], self.anc[self.anc_step*best_anc_i[b]+1], \
            #     math.log(gw[b]/self.anc[self.anc_step*best_anc_i[b]]), \
            #     math.log(gh[b]/self.anc[self.anc_step*best_anc_i[b]+1]), \
            #     iou[b])
            
            coord_mask[b, best_anc_i[b], gjb, gib]     = 1
            conf_mask[b, best_anc_i[b], gjb, gib]      = self.object_scale
            target_box[b, best_anc_i[b], 0, gjb, gib]  = gx[b] - gi[b]
            target_box[b, best_anc_i[b], 1, gjb, gib]  = gy[b] - gj[b]
            target_box[b, best_anc_i[b], 2, gjb, gib]  = \
                math.log(gw[b]/self.anc[self.anc_step*best_anc_i[b]])
            target_box[b, best_anc_i[b], 3, gjb, gib]  = \
                math.log(gh[b]/self.anc[self.anc_step*best_anc_i[b]+1])
            target_conf[b, best_anc_i[b], gjb, gib] = iou[b]
            
            # if self.debug:
            #     if b == 0:
            #         print('conf_mask')
            #         print(conf_mask[0, best_anc_i[b], :, :])
            #         print('coord_mask')
            #         print(coord_mask[0, best_anc_i[b], :, :])
            #         print('target_conf')
            #         print(target_conf[0, best_anc_i[b], :, :])

        t2 = time.time()
        if self.time:
            print('------get_target-----')
            print('        activation : %f' % (t1 - t0))
            print('    get target_box : %f' % (t2 - t1))
            print('             total : %f' % (t2 - t0))


        return coord_mask, conf_mask, target_box, target_conf

    def forward(self, pred, target):
        """
        Loss calculation and processing
        Args:
            pred    : (b, na*5, H, W) 
            target  : (b, [x, y, w, h])
        Out:
            loss    : Total loss and its components
        """
        t0      = time.time()   # start
        self.bs = pred.shape[0]
        self.W  = pred.shape[2] # 13 if imgw=416
        self.H  = pred.shape[3] # 13 if imgh=416
        LT      = torch.LongTensor
        FT      = torch.FloatTensor
        exp     = torch.exp
        bahw    = self.bs, self.na, self.H, self.W

        assert(self.W == self.H)

        pred = pred.view((self.bs, self.na, 5, self.H, self.W)) # B, A, 5, H, W
        
        # if self.debug:
        #     pred = torch.ones(pred.shape).cuda()*100
        #     pred[0, :, :, 12, 12] = 1
        #     target = torch.ones(target.shape).cuda()*0.1
        #     target[0, 0] = 12/13
        #     target[0, 1] = 12/13
        #     target[0, 2] = 2/13
        #     target[0, 3] = 3/13
        
        pred_boxes                  = pred[:, :, :4, :, :]
        pred_boxes[:, :, :2, :, :]  = torch.sigmoid(pred_boxes[:, :, :2, :, :])
        pred_conf                   = torch.sigmoid(pred[:, :, 4, :, :])

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

        # if self.debug:
        #     print('Checking grid')
        #     check = torch.ones(pred_boxes_ofs.shape).cuda()
        #     check[:, :, 0, :, :] = check[:, :, 0, :, :] + grid_x
        #     check[:, :, 1, :, :] = check[:, :, 1, :, :] + grid_y
        #     check[:, :, 2, :, :] = check[:, :, 2, :, :]*anc_w
        #     check[:, :, 3, :, :] = check[:, :, 3, :, :]*anc_h
        #     print(check[0, 0, 0, :, :])
        #     print(check[0, 0, 1, :, :])
        #     print(check[0, 0, 2, :, :])
        #     print(check[0, 0, 3, :, :])

        # target = target.cpu()
        # pred_boxes_ofs = pred_boxes_ofs.cpu()

        # if self.debug:
        #     print("get_target shouldn't require_grad:")
        #     print(target.requires_grad, pred_boxes_ofs.requires_grad)

        coord_mask, conf_mask, target_box, target_conf = \
            self.get_target(target, pred_boxes_ofs)

        coord_mask  = coord_mask.cuda()
        conf_mask   = conf_mask.cuda()
        target_box  = target_box.cuda()
        target_conf = target_conf.cuda()
        conf_mask   = conf_mask.sqrt()
        t3          = time.time() # get targets
        
        # if self.debug:
        #     print('pred_boxes')
        #     print(pred_boxes[0, 0])
        #     print('pred_boxes_ofs')
        #     print(pred_boxes_ofs[0, 0])
        #     print('pred_conf')
        #     print(pred_conf[0, 0])
        #     print('target_conf')
        #     print(target_conf[0, 0])
        #     print('conf_mask')
        #     print(conf_mask[0, 0])
        #     print('coord_mask')
        #     print(coord_mask[0, 0])
        #     print('target_box')
        #     print(target_box[0, 0])            
        
        mseloss     = nn.MSELoss(reduction="sum")
        loss_x      = self.cs*mseloss(pred_boxes[:, :, 0, :, :]*coord_mask,
                                      target_box[:, :, 0, :, :]*coord_mask)/2.0
        loss_y      = self.cs*mseloss(pred_boxes[:, :, 1, :, :]*coord_mask,
                                      target_box[:, :, 1, :, :]*coord_mask)/2.0
        loss_w      = self.cs*mseloss(pred_boxes[:, :, 2, :, :]*coord_mask,
                                      target_box[:, :, 2, :, :]*coord_mask)/2.0
        loss_h      = self.cs*mseloss(pred_boxes[:, :, 3, :, :]*coord_mask,
                                      target_box[:, :, 3, :, :]*coord_mask)/2.0
        loss_conf   = mseloss(pred_conf*conf_mask, target_conf*conf_mask)/2.0
        loss        = loss_x + loss_y + loss_w + loss_h + loss_conf
        t4          = time.time()
        if self.time:
            print('-----RegionLoss-----')
            print('        activation : %f' % (t1 - t0))
            print('    pred_boxes_ofs : %f' % (t2 - t1))
            print('       get targets : %f' % (t3 - t2))
            print('         calc loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))

        return loss, loss_x, loss_y, loss_w, loss_h, loss_conf
