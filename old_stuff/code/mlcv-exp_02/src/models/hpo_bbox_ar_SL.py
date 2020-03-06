import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src import ROOT
from src.models.base_model import Base_Model
from src.datasets import get_dataloader, get_dataset
from src.networks.hpo_bbox_ar_SL_net import HPO_BBOX_AR_SL_Net
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler
from src.loss.hpo_ar_loss import HPO_AR_Loss
from src.loss.yolov2_bbox_loss import YOLOV2_Bbox_Loss
from src.datasets.transforms import *
from src.utils import *

class HPO_Bbox_AR_SL(Base_Model):
    def __init__(self, cfg, mode, load_epoch):
        super().__init__(cfg, mode, load_epoch)
        self.net        = HPO_BBOX_AR_SL_Net(cfg).cuda()
        self.optimizer  = get_optimizer(cfg, self.net)
        self.scheduler  = get_scheduler(cfg, self.optimizer)

        if mode == 'train':
            self.train_dataloader   = get_dataloader(cfg, get_dataset(cfg, 'train'))
            self.val_dataloader     = get_dataloader(cfg, get_dataset(cfg, 'val'))

        self.pretrain = cfg['pretrain']
        self.load_weights()

        self.ar_loss            = HPO_AR_Loss(cfg)
        self.bbox_loss          = YOLOV2_Bbox_Loss(cfg)
        self.consensus          = cfg['consensus']
        self.num_action         = int(cfg['num_actions'])
        self.num_obj            = int(cfg['num_objects'])
        self.anchors            = [float(i) for i in cfg["anchors"].split(',')]
        self.num_anchors        = len(self.anchors)//2

        self.val_top1_action    = Average_Meter()
        self.val_action         = Average_Meter()
        self.val_top1_obj       = Average_Meter()
        self.val_obj            = Average_Meter()
        self.avg_iou            = 0.0
        self.iou_total          = 0.0
        self.val_conf_thresh    = float(cfg['val_conf_thresh'])
        self.val_nms_thresh     = float(cfg['val_nms_thresh'])
        self.val_iou_thresh     = float(cfg['val_iou_thresh'])
        self.best_val           = 0.0

        self.pred_action        = []
        self.pred_obj           = []
        self.action_class_dist  = []
        self.obj_class_dist     = []
        self.pred_bbox          = []
        self.pred_conf_thresh   = float(cfg['pred_conf_thresh'])
        self.pred_nms_thresh    = float(cfg['pred_nms_thresh'])

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, bbox_gt, action_gt, obj_gt, _  = data_load
        img                     = img.cuda()
        action_gt               = action_gt.cuda()
        obj_gt                  = obj_gt.cuda()
        bbox_gt                 = bbox_gt.cuda()
        out                     = self.net(img)
        pred_bbox               = out[:, :(5*self.num_anchors), :, :]
        pred_ar                 = out[:, (5*self.num_anchors):, :, :]
        ar_loss, *ar_other_losses     = self.ar_loss(pred_ar, action_gt, obj_gt)
        bbox_loss, *bbox_other_losses = self.bbox_loss(pred_bbox, bbox_gt)

        loss = ar_loss + bbox_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_action, loss_obj = ar_other_losses
        loss_x, loss_y, loss_w, loss_h, loss_conf = bbox_other_losses
        loss_dict = {
            'loss'          : '{:04f}'.format(loss.item()),
            'loss_x'        : '{:04f}'.format(loss_x.item()),
            'loss_y'        : '{:04f}'.format(loss_y.item()),
            'loss_w'        : '{:04f}'.format(loss_w.item()),
            'loss_h'        : '{:04f}'.format(loss_h.item()),
            'loss_conf'     : '{:04f}'.format(loss_conf.item()),
            'loss_action'   : '{:04f}'.format(loss_action.item()),
            'loss_obj'      : '{:04f}'.format(loss_obj.item()),
        }

        return loss_dict

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img, bbox_gt, action_gt, obj_gt, _ = data_load

        img         = img.cuda()
        action_gt   = action_gt.cuda()
        obj_gt      = obj_gt.cuda()
        bbox_gt     = bbox_gt.cpu()

        pred        = self.net(img)

        bs          = pred.shape[0]
        H           = pred.shape[2]
        W           = pred.shape[3]
        D           = 5
        pred_bbox               = pred[:, :(5*self.num_anchors), :, :].cpu()
        pred_ar                 = pred[:, (5*self.num_anchors):, :, :]

        pred_ar        = pred_ar.view(bs, self.num_action + self.num_obj, D, H, W)
        pred_ar        = pred_ar.permute(0, 1, 3, 4, 2)

        all_boxes   = get_region_boxes(pred_bbox,
                                       self.val_conf_thresh,
                                       0,
                                       self.anchors,
                                       self.num_anchors)

        for batch in range(bs):
            cur_pred        = pred_ar[batch]
            cur_obj_gt      = obj_gt[batch].unsqueeze(0)
            cur_action_gt   = action_gt[batch].unsqueeze(0)

            pred_action = cur_pred[:self.num_action, :, :, :]
            pred_action = pred_action.contiguous().view(self.num_action, -1)
            pred_obj    = cur_pred[self.num_action:, :, :, :]
            pred_obj    = pred_obj.contiguous().view(self.num_obj, -1)

            # consensus
            if self.consensus == 'avg':
                pred_obj    = torch.mean(pred_obj, dim=-1)
                pred_action = torch.mean(pred_action, dim=-1)
            elif self.consensus == 'max':
                pred_obj    = torch.max(pred_obj, dim=-1)
                pred_action = torch.max(pred_action, dim=-1)

            pred_action = pred_action.unsqueeze(0)
            pred_obj    = pred_obj.unsqueeze(0)
            prec1_act, prec5_act    = topk_accuracy(pred_action, cur_action_gt, topk=(1,5))
            prec1_obj, prec5_obj    = topk_accuracy(pred_obj, cur_obj_gt, topk=(1,5))

            self.val_top1_action.update(prec1_act.item(), 1)
            self.val_action.update(prec5_act.item(), 1)
            self.val_top1_obj.update(prec1_obj.item(), 1)
            self.val_obj.update(prec5_obj.item(), 1)

            # BBOX

            boxes           = all_boxes[batch]
            boxes           = nms_torch(boxes, self.val_nms_thresh)
            cur_target      = bbox_gt[batch]

            box_gt      = [float(cur_target[0]), float(cur_target[1]),
                           float(cur_target[2]), float(cur_target[3]), 1.0]
            best_iou    = 0
            best_j      = -1
            for j in range(len(boxes)):
                iou             = bbox_iou(box_gt, boxes[j])
                best_iou        = iou
                best_j          = j
                self.avg_iou    += iou
                self.iou_total  += 1

    def get_valid_loss(self):
        eps         = 1e-5
        avg_iou     = self.avg_iou/(self.iou_total + eps)

        # if avg_iou > self.best_val:
        #     state = {'epoch': self.cur_epoch,
        #             'model_state_dict': self.net.state_dict(),
        #             'optimizer_state_dict': self.optimizer.state_dict(),
        #             'scheduler_state_dict': self.scheduler.state_dict()}
        #     torch.save(state, self.save_ckpt_dir/'model_best.state')
        #     self.best_val = avg_iou

        val_loss_dict = {
            'top1_act'      : self.val_top1_action.avg,
            'top5_act'      : self.val_action.avg,
            'top1_obj'      : self.val_top1_obj.avg,
            'top5_obj'      : self.val_obj.avg,
            'avg_iou'       : avg_iou
        }

        self.iou_total      = 0.0
        self.avg_iou        = 0.0

        self.val_top1_action.reset()
        self.val_action.reset()
        self.val_top1_obj.reset()
        self.val_obj.reset()
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        img         = data_load[0]
        img         = img.cuda()
        pred        = self.net(img)
        pred_bbox               = pred[:, :(5*self.num_anchors), :, :].cpu()
        pred_ar                 = pred[:, (5*self.num_anchors):, :, :]
        bs          = img.shape[0]
        W           = pred.shape[3]
        H           = pred.shape[2]
        D           = 5
        pred_ar        = pred_ar.view(bs, self.num_action + self.num_obj, D, H, W)
        pred_ar        = pred_ar.permute(0, 1, 3, 4, 2)

        max_boxes   = 1

        batch_boxes = get_region_boxes(pred_bbox,
                                       self.pred_conf_thresh,
                                       0,
                                       self.anchors,
                                       self.num_anchors,
                                       is_cuda=False)

        for batch in range(bs):
            cur_pred    = pred_ar[batch]
            pred_action = cur_pred[:self.num_action, :, :, :]
            pred_action = pred_action.contiguous().view(self.num_action, -1)
            pred_obj    = cur_pred[self.num_action:, :, :, :]
            pred_obj    = pred_obj.contiguous().view(self.num_obj, -1)

            # consensus
            if self.consensus == 'avg':
                pred_obj    = torch.mean(pred_obj, dim=-1)
                pred_action = torch.mean(pred_action, dim=-1)
            elif self.consensus == 'max':
                pred_obj    = torch.max(pred_obj, dim=-1)
                pred_action = torch.max(pred_action, dim=-1)

            top_action = torch.topk(pred_action, 1)[1].cpu().numpy()
            top_obj    = torch.topk(pred_obj, 1)[1].cpu().numpy()

            self.action_class_dist.append(pred_action.cpu().numpy())
            self.obj_class_dist.append(pred_obj.cpu().numpy())
            self.pred_action.append(top_action)
            self.pred_obj.append(top_obj)

            boxes       = batch_boxes[batch]
            boxes       = nms_torch(boxes, self.pred_nms_thresh)

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
            self.pred_bbox.append(all_boxes)

    def save_predictions(self, data_split):
        pred_save = "predict_{}_{}_bbox.txt".format(self.load_epoch, data_split)
        pred_file = Path(ROOT)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_bbox)

        pred_save = "predict_{}_{}_action.txt".format(self.load_epoch, data_split)
        pred_file = Path(ROOT)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_action)

        pred_save = "predict_{}_{}_obj.txt".format(self.load_epoch, data_split)
        pred_file = Path(ROOT)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_obj)

        pred_save = "predict_{}_{}_action_dist.txt".format(self.load_epoch, data_split)
        pred_file = Path(ROOT)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.action_class_dist, (-1, self.num_action)))

        pred_save = "predict_{}_{}_obj_dist.txt".format(self.load_epoch, data_split)
        pred_file = Path(ROOT)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.obj_class_dist, (-1, self.num_obj)))

        self.pred_bbox          = []
        self.action_class_dist  = []
        self.obj_class_dist     = []
        self.pred_action        = []
        self.pred_obj           = []

    # ========================================================
    # DETECT
    # ========================================================

    def _get_detect(self, img):
        pred        = self.net(img)
        img_copy    = ImgToNumpy()(img.cpu())[0]
        bs          = img.shape[0]
        W           = pred.shape[3]
        H           = pred.shape[2]
        pred_bbox   = pred[:, :(5*self.num_anchors), :, :].cpu()
        pred_ar     = pred[:, (5*self.num_anchors):, :, :]
        bs          = img.shape[0]
        W           = pred.shape[3]
        H           = pred.shape[2]
        D           = 5
        pred_ar        = pred_ar.view(bs, self.num_action + self.num_obj, D, H, W)
        pred_ar        = pred_ar.permute(0, 1, 3, 4, 2)

        max_boxes   = 1

        batch_boxes = get_region_boxes(pred_bbox,
                                       self.pred_conf_thresh,
                                       0,
                                       self.anchors,
                                       self.num_anchors,
                                       is_cuda=False)

        pred_conf   = pred_bbox.view((bs, self.num_anchors, 5, H, W))
        pred_conf   = torch.sigmoid(pred_conf[:, :, 4, :, :])
        pred_conf   = pred_conf.contiguous().view(bs, -1)

        max_boxes = 1

        # BBOX
        boxes       = batch_boxes[0]
        boxes       = nms_torch(boxes, self.pred_nms_thresh)

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

        cur_pred    = pred_ar[0]
        pred_action = cur_pred[:self.num_action, :, :, :]
        pred_action = pred_action.contiguous().view(self.num_action, -1)
        pred_obj    = cur_pred[self.num_action:, :, :, :]
        pred_obj    = pred_obj.contiguous().view(self.num_obj, -1)

        # consensus
        if self.consensus == 'avg':
            pred_obj    = torch.mean(pred_obj, dim=-1)
            pred_action = torch.mean(pred_action, dim=-1)
        elif self.consensus == 'max':
            pred_obj    = torch.max(pred_obj, dim=-1)
            pred_action = torch.max(pred_action, dim=-1)

        top_action = torch.topk(pred_action, 1)[1].cpu().numpy()
        top_obj    = torch.topk(pred_obj, 1)[1].cpu().numpy()

        all_boxes[0]  = all_boxes[0]*img_copy.shape[1]
        all_boxes[1]  = all_boxes[1]*img_copy.shape[0]
        all_boxes[2]  = all_boxes[2]*img_copy.shape[1]
        all_boxes[3]  = all_boxes[3]*img_copy.shape[0]

        return all_boxes, top_action[0], top_obj[0]

    def detect(self, img):
        import matplotlib.pyplot as plt
        bbox_pred, top_action, top_obj = self._get_detect(img)
        img = ImgToNumpy()(img.cpu())[0]
        action_dict    = FPHA.get_action_dict()
        obj_dict       = FPHA.get_obj_dict()
        print(action_dict[top_action], obj_dict[top_obj])

        fig, ax = plt.subplots()
        ax.imshow(img)
        print(bbox_pred)
        draw_bbox(ax, bbox_pred, 'r')
        plt.show()

    def detect_video(self, seq_path, seq_name, fps=12, img_size=416,  model_info=''):
        from moviepy.editor import ImageSequenceClip
        from tqdm import tqdm
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        from IPython.display import Image as IPythonImage

        with torch.no_grad():
            seq = [x for x in sorted(seq_path.glob('*')) if x.is_file()]

            frames = []
            pred_action_list = []
            pred_obj_list = []
            for f in tqdm(seq):
                img = get_img_dataloader(str(f), img_size)
                img = img.unsqueeze(0).cuda()
                bbox_pred, pred_action, pred_obj = self._get_detect(img)
                img = ImgToNumpy()(img.cpu())[0]

                pred_action_list.append(pred_action)
                pred_obj_list.append(pred_obj)

                fig, ax = plt.subplots()
                ax = fig.gca()
                ax.axis('off')
                ax.imshow(img)
                draw_bbox(ax, bbox_pred, 'r')

                fig.canvas.draw()
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(data)

                plt.close()

            pred_action = np.mean(pred_action_list, axis=0)
            pred_obj = np.mean(pred_obj_list, axis=0)
            top_action = np.argmax(pred_action)
            top_obj    = np.argmax(pred_obj)

            action_dict    = FPHA.get_action_dict()
            obj_dict       = FPHA.get_obj_dict()

            print(action_dict[top_action], obj_dict[top_obj])

            segment_clip = ImageSequenceClip(frames, fps=fps)
            name = str(Path(ROOT)/'mlcv-exp/data/saved'/'detect_{}_{}.gif'.format(seq_name.replace('/', '_'), model_info))
            segment_clip.write_gif(name, fps=fps)

            with open(name, 'rb') as f:
                display(IPythonImage(data=f.read(), format='png'))