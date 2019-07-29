import time
import torch
import torch.nn                     as nn
import numpy                        as np
from pathlib                        import Path
from tqdm                           import tqdm

from src.models                     import Model
from src.utils                      import YOLO, IMG, FPHA
from src.datasets                   import get_dataset, get_dataloader
from src.loss                       import get_loss

class YOLOV2_FPHA(Model):
    """ YOLOv2 single bounding box detection on FPHA dataset """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)

        self.load_weights(self.load_epoch)

        self.loss           = get_loss(cfg['loss'], cfg)
        self.anchors        = [float(i) for i in cfg["anchors"].split(',')]
        self.num_anchors = len(self.anchors)//2
        
        # Training
        if self.training:
            dataset_kwargs  = {'split_set': cfg['train_set']}
            train_dataset   = get_dataset(cfg, dataset_kwargs)
            self.train_sampler  = None
            shuffle             = cfg['shuffle']
            kwargs = {'batch_size'  :   int(cfg['batch_size']),
                      'shuffle'     :   shuffle,
                      'num_workers' :   int(cfg['num_workers']),
                      'pin_memory'  :   True}
            self.train_loader = get_dataloader(train_dataset, 
                                               self.train_sampler,
                                               kwargs)
            # Validation
            dataset_kwargs              = {'split_set': cfg['val_set']}
            val_dataset                 = get_dataset(cfg, dataset_kwargs)
            self.val_loader             = get_dataloader(val_dataset, 
                                                         None,
                                                         kwargs)
            self.val_total              = 0.0
            self.val_proposals          = 0.0
            self.val_correct            = 0.0
            self.avg_iou                = 0.0
            self.iou_total              = 0.0
            self.val_conf_thresh        = float(cfg['val_conf_thresh'])
            self.val_nms_thresh         = float(cfg['val_nms_thresh'])
            self.val_iou_thresh         = float(cfg['val_iou_thresh'])
        # Prediction
        else:
            self.pred_list              = []
            self.load_epoch             = load_epoch
            self.pred_conf_thresh       = float(cfg['pred_conf_thresh'])
            self.pred_nms_thresh        = float(cfg['pred_nms_thresh'])

    # ========================================================
    # LOADING
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
            load_dir = Path(self.save_ckpt_dir) / f'model_{load_epoch}.state'
            ckpt = self.load_ckpt(load_dir)
            self.net.load_state_dict(ckpt['model_state_dict'])
            if self.training:
                self.optimizer.load_state_dict(
                    ckpt['optimizer_state_dict'])
                if self.scheduler:
                    self.scheduler.load_state_dict(
                        ckpt['scheduler_state_dict'])

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        t0              = time.time()               # start

        img, bbox_gt    = data_load
        batch_size      = img.shape[0]
        img             = img.cuda()                # (B, C, H, W)
        bbox_gt         = bbox_gt.cuda()            # (B, [x, y, w, h])
        t1              = time.time()               # CPU to GPU

        out             = self.net(img)[0]          # (B, A*5, W/32, H/32)
        t2              = time.time()               # forward

        loss, *losses   = self.loss(out, bbox_gt)
        t3              = time.time()               # loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        t4 = time.time()                            # backward
        
        loss_x, loss_y, loss_w, loss_h, loss_conf = losses
        loss_dict = {
            'loss'      : loss.item(),
            'loss_x'    : loss_x.item(),
            'loss_y'    : loss_y.item(),
            'loss_w'    : loss_w.item(),
            'loss_h'    : loss_h.item(),
            'loss_conf' : loss_conf.item(),
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
        img, _      = data_load
        batch_size  = img.shape[0]

        img         = img.cuda()

        output      = self.net(img)
        pred_bbox   = output[0]

        pred_bbox   = pred_bbox.cpu()
        target      = data_load[1].cpu()
        all_boxes   = YOLO.get_region_boxes(pred_bbox,
                                            self.val_conf_thresh,
                                            0,
                                            self.anchors,
                                            self.num_anchors)
        
        if self.debug:
            print(all_boxes[0][:10])
            
        for batch in range(batch_size):
            self.val_total  += 1
            boxes           = all_boxes[batch]
            boxes           = YOLO.nms_torch(boxes, self.val_nms_thresh)
            cur_target      = target[batch]

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
        eps         = 1e-5
        precision   = 1.0*self.val_correct/(self.val_proposals + eps)
        recall      = 1.0*self.val_correct/(self.val_total + eps)
        f1score     = 2.0*precision*recall/(precision+recall + eps)
        avg_iou     = self.avg_iou/(self.iou_total + eps)

        val_loss_dict = {
            'precision'     : precision,
            'recall'        : recall,
            'f1score'       : f1score,
            'avg_iou'       : avg_iou
        }

        if self.debug:
            print(val_loss_dict)
        
        self.iou_total      = 0.0
        self.avg_iou        = 0.0
        self.val_total      = 0.0
        self.val_proposals  = 0.0
        self.val_correct    = 0.0
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        img         = data_load[0]
        img         = img.cuda()
        pred        = self.net(img)
        batch_size  = img.shape[0]
        pred        = pred[0].cpu()
        max_boxes   = 1 # detect at most 1 box (FPHA only has 1 bbox)

        batch_boxes = YOLO.get_region_boxes(pred,
                                            self.pred_conf_thresh,
                                            0,
                                            self.anchors,
                                            self.num_anchors,
                                            is_cuda = False,
                                            is_time = self.time)

        for i in range(batch_size):
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
        pred_save = "predict_{}_{}_bbox.txt".format(self.load_epoch, data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_list)
        self.pred_list = []
        
    # ========================================================
    # DETECTION
    # ========================================================
    
    def detect(self, img):
        with torch.no_grad():
            FT          = torch.FloatTensor
            img         = np.asarray(img.copy())
            ori_w       = img.shape[1]
            ori_h       = img.shape[0]
            img         = IMG.resize_img(img, (416, 416))
            img         = img/255.0
            img         = IMG.imgshape2torch(img)
            img         = np.expand_dims(img, 0)
            img         = FT(img)
            img         = img.cuda()
            pred        = self.net(img)[0].cpu()
            max_boxes   = 2
            box         = YOLO.get_region_boxes(pred,
                                                self.pred_conf_thresh,
                                                0,
                                                self.anchors,
                                                self.num_anchors,
                                                is_cuda = False,
                                                is_time = self.time)[0]
            boxes       = YOLO.nms_torch(box, self.pred_nms_thresh)

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
            return all_boxes
        