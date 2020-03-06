import os
import torch.nn as nn
import torch
import numpy as np
import sys
from tqdm import tqdm
import torch
import json
from pathlib import Path

from .BaseModel import BaseModel
from .networks.darknetV3 import DarknetV3, load_darknet_weights
from .loss.YOLOV3Loss import YOLOV3Loss, YOLOV3Loss_1Class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import VOC_utils as VOC
from utils import COCO_utils as COCO
from utils import YOLO_utils as YOLO
from utils.eval_utils import *
from utils.image_utils import *

class YOLOV3Model_COCO(BaseModel):
    def __init__(self, conf, device, load_epoch, train_mode, exp_dir, deterministic, logger=None):
        super(YOLOV3Model_COCO, self).__init__(conf, device, train_mode, exp_dir, deterministic, logger)
        net = self.get_net(load_epoch, conf["cfg_file"], conf["pretrain"])
        self.init_network(net, load_epoch)
        
        self.loss                   = YOLOV3Loss(conf, device)
        self.num_classes            = conf["classes"]
        self.time                   = conf["time"]
        self.img_size               = conf["img_size"]
        self.cur_epoch              = 0
        self.cur_step               = 0
        self.batches_per_epoch      = conf["batches_per_epoch"]
        
        if train_mode:
            self.burnin                 = conf["burnin"]
            # validation
            self.init_learning_rate     = conf["optimizer"]["learning_rate"]           
            self.val_loss               = 0.0
            self.val_total              = 0.0
            self.val_stats              = []
            self.val_conf_thresh        = conf["val_conf_thresh"]
            self.val_nms_thresh         = conf["val_nms_thresh"]        
            self.val_iou_thresh         = conf["val_iou_thresh"]
        # prediction
        else:
            self.pred_stats         = []
            self.pred_jdict         = []
            self.pred_img_files     = []
            self.pred_seen          = 0.0
            self.load_epoch         = load_epoch
            self.pred_conf_thresh   = conf["pred_conf_thresh"]
            self.pred_nms_thresh    = conf["pred_nms_thresh"]   
            self.pred_iou_thresh    = conf["pred_iou_thresh"]         
    
    def get_net(self, load_epoch, cfgfile, pretrain):
        if self.deterministic:
            torch.manual_seed(0)        
        net = DarknetV3(cfgfile)
        if load_epoch == 0:
            if pretrain != "none":
                if pretrain[-5:] == "state":
                    ckpt = self.get_ckpt(pretrain)
                    self.load_ckpt(net, ckpt["model_state_dict"])
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
                else:
                    load_darknet_weights(net, pretrain)
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
        return net
    
    def predict(self, data_load):
        img = data_load[0]
        return self.net(img)
   
    def get_loss(self, data_load, out, net, train_out):
        _, labels = data_load
        return self.loss(out, labels, net, train_out)

    def init_epoch(self, epoch):
        self.cur_epoch = epoch
        self.cur_step = 0
        
    def train_step(self, data_load):
        # SGD burn-in
        if self.burnin:
            n_burnin = min(round(self.batches_per_epoch / 5 + 1), 1000)  # burn-in batches
            if self.cur_epoch == 0 and self.cur_step <= n_burnin:
                lr = self.init_learning_rate * (self.cur_step / n_burnin) ** 4
                for x in self.optimizer.param_groups:
                    x['lr'] = lr            
        
        data_load = data_load[:2]
        #2, 3, 13, 13, 85
        #2, 3, 26, 26, 85
        #2, 3, 52, 52, 85         
        out = self.predict(data_load) 

        loss, lxy, lwh, lconf, lcls = self.get_loss(data_load, out, self.net, True)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_dict = {
            "loss": loss.item(),
            "loss_xy": lxy.item(),
            "loss_wh": lwh.item(),
            "loss_conf": lconf.item(),
            "loss_cls": lcls.item(),
        }
        
        self.cur_step += 1
        
        return loss_dict
    
    def valid_step(self, data_load):
        data_load = data_load[:2]
        img = data_load[0]
        targets = data_load[1]
        
        inf_out, train_out = self.net(img)
        loss = self.get_loss(data_load, train_out, self.net, False)
        self.val_loss += loss
        self.val_total += 1
        output = YOLO.non_max_suppression(inf_out, conf_thres=self.val_conf_thresh, nms_thres=self.val_nms_thresh)
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            if pred is None:
                if nl:
                    self.val_stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tbox = YOLO.xywh2xyxy(labels[:, 1:5]) * self.img_size  # target boxes

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    iou, bi = YOLO.bbox_iou_hyperlytics(pbox, tbox).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > self.val_iou_thresh and bi not in detected:
                        correct[i] = 1
                        detected.append(bi)

            # Append statistics (correct, conf, pcls, tcls)
            self.val_stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))        
            
    def get_valid_loss(self):
        # Compute statistics
        stats_np = [np.concatenate(x, 0) for x in list(zip(*self.val_stats))]
        nt = np.bincount(stats_np[3].astype(np.int64), minlength=self.num_classes)  # number of targets per class
        if len(stats_np):
            p, r, ap, f1, ap_class = YOLO.ap_per_class(*stats_np)
            mean_p, mean_r, mean_ap, mean_f1 = p.mean(), r.mean(), ap.mean(), f1.mean()

        self.val_loss = self.val_loss / self.val_total
   
        val_loss_dict = {
            "val_loss":     self.val_loss.item(),
            "precision":    mean_p,
            "recall":       mean_r,
            "f1score":      mean_f1,
            "mAP":          mean_ap 
        }        
        
        if self.logger:
            self.logger.log("VALIDATION LOSS")
            for name, loss in val_loss_dict.items():
                self.logger.log("{}: {}".format(name.upper(), loss))
        
        self.val_stats  = []
        self.val_loss   = 0.0
        self.val_total  = 0.0
        
        return val_loss_dict
        
    def predict_step(self, data_load):
        img = data_load[0]
        targets = data_load[1]
        paths = data_load[2]
        shapes = data_load[3]
        self.pred_img_files.append(paths)
        coco91class = COCO.coco80_to_coco91_class()
        
        inf_out, train_out = self.net(img)
        output = YOLO.non_max_suppression(inf_out, conf_thres=self.pred_conf_thresh, nms_thres=self.pred_nms_thresh)
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            self.pred_seen += 1
            
            if pred is None:
                if nl:
                    self.pred_stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to pycocotools JSON dictionary
            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            image_id = int(Path(paths[si]).stem.split('_')[-1])
            box = pred[:, :4].clone()  # xyxy
            YOLO.scale_coords(self.img_size, box, shapes[si])  # to original shape
            box = YOLO.xyxy2xywh(box)  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for di, d in enumerate(pred):
                self.pred_jdict.append({
                    'image_id': image_id,
                    'category_id': coco91class[int(d[6])],
                    'bbox': [float(x) for x in box[di]],
                    'score': float(d[4])
                })

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tbox = YOLO.xywh2xyxy(labels[:, 1:5]) * self.img_size  # target boxes

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    iou, bi = YOLO.bbox_iou_hyperlytics(pbox, tbox).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > self.pred_iou_thresh and bi not in detected:
                        correct[i] = 1
                        detected.append(bi)

            # Append statistics (correct, conf, pcls, tcls)
            self.pred_stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))   
        
    def save_predictions(self, data_split):
        class_labels = YOLO.get_class_labels("COCO")
        # Compute statistics
        stats_np = [np.concatenate(x, 0) for x in list(zip(*self.pred_stats))]
        nt = np.bincount(stats_np[3].astype(np.int64), minlength=self.num_classes)  # number of targets per class
        if len(stats_np):
            p, r, ap, f1, ap_class = YOLO.ap_per_class(*stats_np)
            mp, mr, mean_ap, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
    
        # Print results
        pf = '%20s' + '%10.3g' * 6  # print format
        print(pf % ('all', self.pred_seen, nt.sum(), mp, mr, mean_ap, mf1), end='\n\n')

        # Print results per class
        if self.num_classes > 1 and len(stats_np):
            for i, c in enumerate(ap_class):
                print(pf % (class_labels[c], self.pred_seen, nt[c], p[i], r[i], ap[i], f1[i]))

        # Save JSON
        results_json = os.path.join(self.save_dir, 'results_{}_{}.json'.format(self.load_epoch, data_split))
        if mean_ap and len(self.pred_jdict):
            imgIds = [int(Path(x[0]).stem.split('_')[-1]) for x in self.pred_img_files]
            np.savetxt(os.path.join(self.save_dir, 'imgIds_{}_{}.txt'.format(self.load_epoch, data_split)), imgIds)
            with open(results_json, 'w') as file:
                json.dump(self.pred_jdict, file)
            
            import pycocotools.coco as pycoco
            import pycocotools.cocoeval as pycocoeval 
            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            coco_split = self.pred_img_files[0][0].split('/')[-2]
            cocoGt = pycoco.COCO(os.path.join(COCO.DIR, 'annotations', 'instances_{}.json'.format(coco_split)))  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(results_json)  # initialize COCO pred api

            cocoEval = pycocoeval.COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            mean_ap = cocoEval.stats[1]  # update mAP to pycocotools mAP        
            print('COCO MeanAP:', mean_ap)
        
        self.pred_jdict     = []
        self.pred_img_files = []
        self.pred_stats     = []
        self.pred_seen      = 0.0
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
                 "optimizer_state_dict": self.optimizer.state_dict()}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.save_ckpt_dir, f"model_{epoch}.state"))  

class YOLOV3Model_VOC(BaseModel):
    def __init__(self, conf, device, load_epoch, train_mode, exp_dir, deterministic, logger=None):
        super(YOLOV3Model_VOC, self).__init__(conf, device, train_mode, exp_dir, deterministic, logger)
        net = self.get_net(load_epoch, conf["cfg_file"], conf["pretrain"])
        self.init_network(net, load_epoch)
        
        self.loss                   = YOLOV3Loss(conf, device)
        self.num_classes            = conf["classes"]
        self.time                   = conf["time"]
        self.init_learning_rate     = conf["optimizer"]["learning_rate"]
        self.img_size               = conf["img_size"]
        self.cur_epoch              = 0
        self.cur_step               = 0
        self.batches_per_epoch      = conf["batches_per_epoch"]
        
        if train_mode:
            self.burnin                 = conf["burnin"]
            # validation
            self.init_learning_rate     = conf["optimizer"]["learning_rate"]           
            self.val_loss               = 0.0
            self.val_total              = 0.0
            self.val_stats              = []
            self.val_conf_thresh        = conf["val_conf_thresh"]
            self.val_nms_thresh         = conf["val_nms_thresh"]        
            self.val_iou_thresh         = conf["val_iou_thresh"]
        
        # prediction
        else:
            self.pred_list          = []
            self.load_epoch         = load_epoch
            self.pred_conf_thresh   = conf["pred_conf_thresh"]
            self.pred_nms_thresh    = conf["pred_nms_thresh"]   
            
    def get_net(self, load_epoch, cfgfile, pretrain):
        if self.deterministic:
            torch.manual_seed(0)        
        net = DarknetV3(cfgfile)
        if load_epoch == 0:
            if pretrain != "none":
                if pretrain[-5:] == "state":
                    ckpt = self.get_ckpt(pretrain)
                    self.load_ckpt(net, ckpt["model_state_dict"])
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
                else:
                    load_darknet_weights(net, pretrain)
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
        return net
    
    def predict(self, data_load):
        img = data_load[0]
        return self.net(img)
   
    def get_loss(self, data_load, out, net, train_out):
        _, labels = data_load
        return self.loss(out, labels, net, train_out)

    def init_epoch(self, epoch):
        self.cur_epoch = epoch
        self.cur_step = 0
        
    def train_step(self, data_load):
        # SGD burn-in
        if self.burnin:
            n_burnin = min(round(self.batches_per_epoch / 5 + 1), 1000)  # burn-in batches
            if self.cur_epoch == 0 and self.cur_step <= n_burnin:
                lr = self.init_learning_rate * (self.cur_step / n_burnin) ** 4
                for x in self.optimizer.param_groups:
                    x['lr'] = lr   
        
        data_load = data_load[:2]
        #2, 3, 13, 13, 85
        #2, 3, 26, 26, 85
        #2, 3, 52, 52, 85         
        out = self.predict(data_load) 

        loss, lxy, lwh, lconf, lcls = self.get_loss(data_load, out, self.net, True)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_dict = {
            "loss": loss.item(),
            "loss_xy": lxy.item(),
            "loss_wh": lwh.item(),
            "loss_conf": lconf.item(),
            "loss_cls": lcls.item(),
        }
        
        self.cur_step += 1
        
        return loss_dict
    
    def valid_step(self, data_load):
        data_load = data_load[:2]
        img = data_load[0]
        targets = data_load[1]
        
        inf_out, train_out = self.net(img)
        loss = self.get_loss(data_load, train_out, self.net, False)
        self.val_loss += loss
        self.val_total += 1
        output = YOLO.non_max_suppression(inf_out, conf_thres=self.val_conf_thresh, nms_thres=self.val_nms_thresh)
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            if pred is None:
                if nl:
                    self.val_stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tbox = YOLO.xywh2xyxy(labels[:, 1:5]) * self.img_size  # target boxes

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    iou, bi = YOLO.bbox_iou_hyperlytics(pbox, tbox).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > self.val_iou_thresh and bi not in detected:
                        correct[i] = 1
                        detected.append(bi)

            # Append statistics (correct, conf, pcls, tcls)
            self.val_stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))        
            
    def get_valid_loss(self):
        # Compute statistics
        stats_np = [np.concatenate(x, 0) for x in list(zip(*self.val_stats))]
        nt = np.bincount(stats_np[3].astype(np.int64), minlength=self.num_classes)  # number of targets per class
        if len(stats_np):
            p, r, ap, f1, ap_class = YOLO.ap_per_class(*stats_np)
            mean_p, mean_r, mean_ap, mean_f1 = p.mean(), r.mean(), ap.mean(), f1.mean()

        self.val_loss = self.val_loss / self.val_total
   
        val_loss_dict = {
            "val_loss":     self.val_loss.item(),
            "precision":    mean_p,
            "recall":       mean_r,
            "f1score":      mean_f1,
            "mAP":          mean_ap 
        }        
        
        if self.logger:
            self.logger.log("VALIDATION LOSS")
            for name, loss in val_loss_dict.items():
                self.logger.log("{}: {}".format(name.upper(), loss))
        
        self.val_stats  = []
        self.val_loss   = 0.0
        self.val_total  = 0.0
        
        return val_loss_dict

    def predict_step(self, data_load):
        img = data_load[0]
        targets = data_load[1]
        paths = data_load[2]
        shapes = data_load[3]
        
        inf_out, train_out = self.net(img)
        output = YOLO.non_max_suppression(inf_out, conf_thres=self.pred_conf_thresh, nms_thres=self.pred_nms_thresh)
        # Statistics per image 
        for si, pred in enumerate(output):
            if pred is None:
                continue
            fileID = Path(paths[si]).stem.split('_')[-1]
            box = pred[:, :4].clone()  # xyxy
            YOLO.scale_coords(self.img_size, box, shapes[si])  # to original shape      
            for sj, p in enumerate(pred):       
                prob = float(p[4] * p[5])
                cls_id = int(p[6])
                x1 = float(box[sj][0])
                y1 = float(box[sj][1])
                x2 = float(box[sj][2])
                y2 = float(box[sj][3])
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
                 "optimizer_state_dict": self.optimizer.state_dict()}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.save_ckpt_dir, f"model_{epoch}.state"))

class YOLOV3Model_1Class(BaseModel):
    def __init__(self, conf, device, load_epoch, train_mode, exp_dir, deterministic, logger=None):
        super(YOLOV3Model_1Class, self).__init__(conf, device, train_mode, exp_dir, deterministic, logger)
        net = self.get_net(load_epoch, conf["cfg_file"], conf["pretrain"])
        self.init_network(net, load_epoch)
        
        self.loss                   = YOLOV3Loss_1Class(conf, device)
        self.time                   = conf["time"]
        self.init_learning_rate     = conf["optimizer"]["learning_rate"]
        self.img_size               = conf["img_size"]
        self.cur_epoch              = 0
        self.cur_step               = 0
        self.batches_per_epoch      = conf["batches_per_epoch"]
        
        if train_mode:
            self.burnin                 = conf["burnin"]
            # validation
            self.init_learning_rate     = conf["optimizer"]["learning_rate"]           
            self.val_loss               = 0.0
            self.val_total              = 0.0
            self.val_iou_total          = 0.0
            self.val_avg_iou            = 0.0
            self.val_conf_thresh        = conf["val_conf_thresh"]
            self.val_nms_thresh         = conf["val_nms_thresh"]        
            self.val_iou_thresh         = conf["val_iou_thresh"]
        
        # prediction
        else:
            self.pred_list          = []
            self.load_epoch         = load_epoch
            self.pred_conf_thresh   = conf["pred_conf_thresh"]
            self.pred_nms_thresh    = conf["pred_nms_thresh"]   
            
    def get_net(self, load_epoch, cfgfile, pretrain):
        if self.deterministic:
            torch.manual_seed(0)        
        net = DarknetV3(cfgfile)
        if load_epoch == 0:
            if pretrain != "none":
                if pretrain[-5:] == "state":
                    ckpt = self.get_ckpt(pretrain)
                    self.load_ckpt(net, ckpt["model_state_dict"])
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
                else:
                    load_darknet_weights(net, pretrain)
                    if self.logger:
                        self.logger.log(f"LOADED {pretrain} WEIGHTS")        
        return net
    
    def predict(self, data_load):
        img = data_load[0]
        return self.net(img)
   
    def get_loss(self, data_load, out, net, train_out):
        _, labels = data_load
        return self.loss(out, labels, net, train_out)

    def init_epoch(self, epoch):
        self.cur_epoch = epoch
        self.cur_step = 0
        
    def train_step(self, data_load):
        # SGD burn-in
        if self.burnin:
            n_burnin = min(round(self.batches_per_epoch / 5 + 1), 1000)  # burn-in batches
            if self.cur_epoch == 0 and self.cur_step <= n_burnin:
                lr = self.init_learning_rate * (self.cur_step / n_burnin) ** 4
                for x in self.optimizer.param_groups:
                    x['lr'] = lr   
                    
        data_load = data_load[:2]
        #2, 3, 13, 13, 5
        #2, 3, 26, 26, 5
        #2, 3, 52, 52, 5         
        out = self.predict(data_load) 
        
        loss, lxy, lwh, lconf = self.get_loss(data_load, out, self.net, True)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_dict = {
            "loss": loss.item(),
            "loss_xy": lxy.item(),
            "loss_wh": lwh.item(),
            "loss_conf": lconf.item(),
        }
        
        self.cur_step += 1
        
        return loss_dict
    
    def valid_step(self, data_load):
        data_load = data_load[:2]
        img = data_load[0]
        targets = data_load[1]
        
        inf_out, train_out = self.net(img)
        loss = self.get_loss(data_load, train_out, self.net, False)
        self.val_loss += loss
        self.val_total += 1
        output = YOLO.non_max_suppression_1class(inf_out, conf_thres=self.val_conf_thresh, nms_thres=self.val_nms_thresh)
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si]
            nl = len(labels)

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tbox = YOLO.xywh2xyxy(labels[:, 1:5]) * self.img_size  # target boxes

                # Search for correct predictions
                for i, (*pbox, pconf) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Best iou, index between pred and targets
                    iou, bi = YOLO.bbox_iou_hyperlytics(pbox, tbox).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > self.val_iou_thresh and bi not in detected:
                        correct[i] = 1
                        self.val_iou_total += 1
                        self.val_avg_iou += iou                        
                        detected.append(bi)  
            
    def get_valid_loss(self):
        eps = 1e-5
        self.val_loss = self.val_loss/(self.val_total+eps)
        self.val_avg_iou = self.val_avg_iou/(self.val_iou_total+eps) 
        
        val_loss_dict = {
            "val_loss":     self.val_loss.item(),
            "avg_iou":      self.val_avg_iou
        }        
        
        if self.logger:
            self.logger.log("VALIDATION LOSS")
            for name, loss in val_loss_dict.items():
                self.logger.log("{}: {}".format(name.upper(), loss))
        
        self.val_loss       = 0.0
        self.val_total      = 0.0
        self.val_avg_iou    = 0.0
        self.val_iou_total  = 0.0
        
        return val_loss_dict

    def predict_step(self, data_load):
        img = data_load[0]
        targets = data_load[1]
        shapes = data_load[2]
        
        inf_out, train_out = self.net(img)
        output = YOLO.non_max_suppression_1class(inf_out, conf_thres=self.pred_conf_thresh, nms_thres=self.pred_nms_thresh)
        # Statistics per image 
        for si, pred in enumerate(output):
            box = pred[:, :4].clone()  # xyxy
            YOLO.scale_coords(self.img_size, box, shapes[si])  # to original shape      
            for sj, p in enumerate(pred):       
                prob = float(p[4])
                x1 = float(box[sj][0])
                y1 = float(box[sj][1])
                x2 = float(box[sj][2])
                y2 = float(box[sj][3])
                self.pred_list.append([prob, x1, y1, x2, y2])

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
                 "optimizer_state_dict": self.optimizer.state_dict()}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.save_ckpt_dir, f"model_{epoch}.state"))