import torch
import torch.nn                     as nn
import numpy                        as np
from pathlib                        import Path
from tqdm                           import tqdm

from src import ROOT
from src.models.base_model import Base_Model
from src.datasets import get_dataloader, get_dataset
from src.models.yolov2_bbox.yolov2_bbox_net import YOLOV2_Bbox_Net
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler
from src.models.yolov2_bbox.yolov2_bbox_loss import YOLOV2_Bbox_Loss
from src.datasets.transforms import *
from src.utils import *
from src.utils.yolo import *

class YOLOV2_Bbox_Model(Base_Model):
    """ YOLOv2 single bounding box detection """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.net        = YOLOV2_Bbox_Net(cfg).cuda()
        self.optimizer  = get_optimizer(cfg, self.net)
        self.scheduler  = get_scheduler(cfg, self.optimizer)

        self.train_dataloader   = get_dataloader(cfg, get_dataset(cfg, 'train'))
        self.val_dataloader     = get_dataloader(cfg, get_dataset(cfg, 'val'))

        self.pretrain = cfg['pretrain']
        self.load_weights()

        self.loss           = YOLOV2_Bbox_Loss(cfg)
        self.anchors        = [float(i) for i in cfg["anchors"].split(',')]
        self.num_anchors    = len(self.anchors)//2

        self.val_total              = 0.0
        self.val_proposals          = 0.0
        self.val_correct            = 0.0
        self.avg_iou                = 0.0
        self.iou_total              = 0.0
        self.val_conf_thresh        = float(cfg['val_conf_thresh'])
        self.val_nms_thresh         = float(cfg['val_nms_thresh'])
        self.val_iou_thresh         = float(cfg['val_iou_thresh'])
        self.best_val               = 0.0

        self.pred_list              = []
        self.pred_conf_thresh       = float(cfg['pred_conf_thresh'])
        self.pred_nms_thresh        = float(cfg['pred_nms_thresh'])

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, bbox_gt    = data_load
        img             = img.cuda()        # (B, C, H, W)
        bbox_gt         = bbox_gt.cuda()    # (B, [x, y, w, h])

        out             = self.net(img)     # (B, A*5, W/32, H/32)

        loss, *losses   = self.loss(out, bbox_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_x, loss_y, loss_w, loss_h, loss_conf = losses
        loss_dict = {
            'loss'      : '{:04f}'.format(loss.item()),
            'loss_x'    : '{:04f}'.format(loss_x.item()),
            'loss_y'    : '{:04f}'.format(loss_y.item()),
            'loss_w'    : '{:04f}'.format(loss_w.item()),
            'loss_h'    : '{:04f}'.format(loss_h.item()),
            'loss_conf' : '{:04f}'.format(loss_conf.item()),
        }
        return loss_dict

    def post_epoch(self):
        self.loss.epoch += 1

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img, target      = data_load
        batch_size  = img.shape[0]

        img         = img.cuda()

        pred_bbox   = self.net(img).cpu()
        target      = target.cpu()

        all_boxes   = get_region_boxes(pred_bbox,
                                       self.val_conf_thresh,
                                       0,
                                       self.anchors,
                                       self.num_anchors)

        for batch in range(batch_size):
            self.val_total  += 1
            boxes           = all_boxes[batch]
            boxes           = nms_torch(boxes, self.val_nms_thresh)
            cur_target      = target[batch]

            for i in range(len(boxes)):
                if boxes[i][4] > self.val_conf_thresh:
                    self.val_proposals += 1

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
            if best_iou > self.val_iou_thresh:
                self.val_correct += 1

    def get_valid_loss(self):
        eps         = 1e-5
        precision   = 1.0*self.val_correct/(self.val_proposals + eps)
        recall      = 1.0*self.val_correct/(self.val_total + eps)
        f1score     = 2.0*precision*recall/(precision+recall + eps)
        avg_iou     = self.avg_iou/(self.iou_total + eps)

        if avg_iou > self.best_val:
            state = {'epoch': self.cur_epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()}
            torch.save(state, self.save_ckpt_dir/'model_best.state')
            self.best_val = avg_iou

        val_loss_dict = {
            'precision'     : precision,
            'recall'        : recall,
            'f1score'       : f1score,
            'avg_iou'       : avg_iou
        }

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
        pred        = self.net(img).cpu()
        batch_size  = img.shape[0]
        max_boxes   = 2

        batch_boxes = get_region_boxes(pred,
                                       self.pred_conf_thresh,
                                       0,
                                       self.anchors,
                                       self.num_anchors,
                                       is_cuda=False)

        for i in range(batch_size):
            boxes       = batch_boxes[i]
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
            self.pred_list.append(all_boxes)

    def save_predictions(self, data_split):
        pred_save = "predict_{}_{}_bbox.txt".format(self.load_epoch, data_split)
        pred_file = Path(ROOT)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_list)
        self.pred_list = []

    # ========================================================
    # DETECT
    # ========================================================

    def _get_detect(self, img):
        img_copy    = img.clone()
        pred        = self.net(img).cpu()
        img_copy    = ImgToNumpy()(img_copy.cpu())[0]
        max_boxes   = 2

        box         = get_region_boxes(pred,
                                    self.pred_conf_thresh,
                                    0,
                                    self.anchors,
                                    self.num_anchors,
                                    is_cuda=False)[0]

        boxes       = nms_torch(box, self.pred_nms_thresh)

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

        bbox_pred_1 = all_boxes[0]
        bbox_pred_2 = all_boxes[1]

        bbox_pred_1     = bbox_pred_1.copy()
        bbox_pred_1[0]  = bbox_pred_1[0]*img_copy.shape[1]
        bbox_pred_1[1]  = bbox_pred_1[1]*img_copy.shape[0]
        bbox_pred_1[2]  = bbox_pred_1[2]*img_copy.shape[1]
        bbox_pred_1[3]  = bbox_pred_1[3]*img_copy.shape[0]

        bbox_pred_2     = bbox_pred_2.copy()
        bbox_pred_2[0]  = bbox_pred_2[0]*img_copy.shape[1]
        bbox_pred_2[1]  = bbox_pred_2[1]*img_copy.shape[0]
        bbox_pred_2[2]  = bbox_pred_2[2]*img_copy.shape[1]
        bbox_pred_2[3]  = bbox_pred_2[3]*img_copy.shape[0]

        return bbox_pred_1, bbox_pred_2

    def detect(self, img):
        import matplotlib.pyplot as plt
        bbox_pred_1, bbox_pred_2 = self._get_detect(img)
        img = ImgToNumpy()(img.cpu())[0]

        fig, ax = plt.subplots()
        ax.imshow(img)
        print(bbox_pred_1[4])
        draw_bbox(ax, bbox_pred_1, 'r')
        # draw_bbox(ax, bbox_pred_2, 'r')
        plt.show()

    def detect_video(self, seq_path, fps=12, img_size=416,  model_info=''):
        from moviepy.editor import ImageSequenceClip
        from tqdm import tqdm
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        from IPython.display import Image as IPythonImage
        seq = [x for x in sorted(seq_path.glob('*')) if x.is_file()]

        frames = []
        for f in tqdm(seq):
            img = get_img_dataloader(str(f), img_size)
            img = img.unsqueeze(0).cuda()
            bbox_pred_1, bbox_pred_2 = self._get_detect(img)
            img = ImgToNumpy()(img.cpu())[0]

            fig, ax = plt.subplots()
            ax = fig.gca()
            ax.axis('off')
            ax.imshow(img)
            draw_bbox(ax, bbox_pred_1, 'r')

            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(data)

            plt.close()

        segment_clip = ImageSequenceClip(frames, fps=fps)
        seq_name = str(seq_path).split('/')[-1]
        name = str(Path(ROOT)/'mlcv-exp/data/saved'/'{}_{}.gif'.format(seq_name, model_info))
        segment_clip.write_gif(name, fps=fps)

        with open(name, 'rb') as f:
            display(IPythonImage(data=f.read(), format='png'))