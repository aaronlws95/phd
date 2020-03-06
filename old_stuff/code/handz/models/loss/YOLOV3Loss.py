import numpy as np
import torch
import torch.nn as nn
import sys 
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import YOLO_utils as YOLO

class YOLOV3Loss(torch.nn.Module):
    def __init__(self, conf, device):
        super(YOLOV3Loss,self).__init__()
        self.iou_thresh = conf["loss_iou_thresh"]
        self.loss_mult = conf["loss_mult"]
        self.xy_loss_frac = conf["xy_loss_frac"]
        self.wh_loss_frac = conf["wh_loss_frac"]
        self.cls_loss_frac = conf["cls_loss_frac"]
        self.conf_loss_frac = conf["conf_loss_frac"]
        
    def build_targets(self, net, targets):
        # targets = [image, class, x, y, w, h]
        nt = len(targets)
        txy, twh, tcls, indices = [], [], [], []
        for i in net.yolo_layers:
            layer = net.module_list[i][0]

            # iou of targets-anchors
            t, a = targets, []
            gwh = targets[:, 4:6] * layer.nG
            if nt:
                iou = [YOLO.wh_iou(x, gwh) for x in layer.anchor_vec]
                iou, a = torch.stack(iou, 0).max(0)  # best iou and anchor

                # reject below threshold ious (OPTIONAL, increases P, lowers R)
                reject = True
                if reject:
                    j = iou > self.iou_thresh
                    t, a, gwh = targets[j], a[j], gwh[j]

            # Indices
            b, c = t[:, :2].long().t()  # target image, class
            gxy = t[:, 2:4] * layer.nG
            gi, gj = gxy.long().t()  # grid_i, grid_j
            indices.append((b, a, gj, gi))

            # XY coordinates
            txy.append(gxy - gxy.floor())

            # Width and height
            twh.append(torch.log(gwh / layer.anchor_vec[a]))  # wh yolo method
            # twh.append((gwh / layer.anchor_vec[a]) ** (1 / 3) / 2)  # wh power method

            # Class
            tcls.append(c)
            if c.shape[0]:
                assert c.max() <= layer.nC, 'Target classes exceed model classes'

        return txy, twh, tcls, indices    
    
    def forward(self, p, t, net, train_out):
        targets = self.build_targets(net, t)
        
        FT = torch.cuda.FloatTensor if p[0].is_cuda else torch.FloatTensor
        lxy, lwh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0])
        txy, twh, tcls, indices = targets
        MSE = nn.MSELoss()
        CE = nn.CrossEntropyLoss()
        BCE = nn.BCEWithLogitsLoss()

        # Compute losses
        # gp = [x.numel() for x in tconf]  # grid points
        for i, pi0 in enumerate(p):  # layer i predictions, i
            b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy
            tconf = torch.zeros_like(pi0[..., 0])  # conf
            nt = len(b)  # number of targets

            # Compute losses
            k = self.loss_mult
            if nt:
                pi = pi0[b, a, gj, gi]  # predictions closest to anchors
                tconf[b, a, gj, gi] = 1  # conf

                lxy += (k * self.xy_loss_frac) * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy loss
                lwh += (k * self.wh_loss_frac) * MSE(pi[..., 2:4], twh[i])  # wh yolo loss
                lcls += (k * self.cls_loss_frac) * CE(pi[..., 5:], tcls[i])  # class_conf loss

            lconf += (k * self.conf_loss_frac) * BCE(pi0[..., 4], tconf)  # obj_conf loss
        loss = lxy + lwh + lconf + lcls

        if train_out:
            losses = loss, lxy, lwh, lconf, lcls  
        else: 
            losses = loss

        return losses
    
class YOLOV3Loss_1Class(torch.nn.Module):
    def __init__(self, conf, device):
        super(YOLOV3Loss_1Class,self).__init__()
        self.iou_thresh = conf["loss_iou_thresh"]
        self.loss_mult = conf["loss_mult"]
        self.xy_loss_frac = conf["xy_loss_frac"]
        self.wh_loss_frac = conf["wh_loss_frac"]
        self.conf_loss_frac = conf["conf_loss_frac"]
        
    def build_targets(self, net, targets):
        # targets = [image, class, x, y, w, h]
        nt = len(targets)
        txy, twh, indices = [], [], []
        for i in net.yolo_layers:
            layer = net.module_list[i][0]

            # iou of targets-anchors
            t, a = targets, []
            gwh = targets[:, 3:5] * layer.nG
            if nt:
                iou = [YOLO.wh_iou(x, gwh) for x in layer.anchor_vec]
                iou, a = torch.stack(iou, 0).max(0)  # best iou and anchor

                # reject below threshold ious (OPTIONAL, increases P, lowers R)
                reject = True
                if reject:
                    j = iou > self.iou_thresh
                    t, a, gwh = targets[j], a[j], gwh[j]

            # Indices
            b, c = t[:, :2].long().t()  # target image, class
            gxy = t[:, 2:4] * layer.nG
            gi, gj = gxy.long().t()  # grid_i, grid_j
            indices.append((b, a, gj, gi))

            # XY coordinates
            txy.append(gxy - gxy.floor())

            # Width and height
            twh.append(torch.log(gwh / layer.anchor_vec[a]))  # wh yolo method
            # twh.append((gwh / layer.anchor_vec[a]) ** (1 / 3) / 2)  # wh power method
            
        return txy, twh, indices    
    
    def forward(self, p, t, net, train_out):
        targets = self.build_targets(net, t)
        
        FT = torch.cuda.FloatTensor if p[0].is_cuda else torch.FloatTensor
        lxy, lwh, lconf = FT([0]), FT([0]), FT([0])
        txy, twh, indices = targets
        MSE = nn.MSELoss()
        CE = nn.CrossEntropyLoss()
        BCE = nn.BCEWithLogitsLoss()

        # Compute losses
        # gp = [x.numel() for x in tconf]  # grid points
        for i, pi0 in enumerate(p):  # layer i predictions, i
            b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy
            tconf = torch.zeros_like(pi0[..., 0])  # conf
            nt = len(b)  # number of targets

            # Compute losses
            k = self.loss_mult
            if nt:
                pi = pi0[b, a, gj, gi]  # predictions closest to anchors
                tconf[b, a, gj, gi] = 1  # conf

                lxy += (k * self.xy_loss_frac) * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy loss
                lwh += (k * self.wh_loss_frac) * MSE(pi[..., 2:4], twh[i])  # wh yolo loss

            lconf += (k * self.conf_loss_frac) * BCE(pi0[..., 4], tconf)  # obj_conf loss
        loss = lxy + lwh + lconf

        if train_out:
            losses = loss, lxy, lwh, lconf  
        else: 
            losses = loss

        return losses    