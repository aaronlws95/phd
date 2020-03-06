import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src import ROOT
from src.eval_helper.base_eval_helper import Base_Eval_Helper
from src.utils import *

class YOLOV2_Bbox_Helper(Base_Eval_Helper):
    def __init__(self, cfg, epoch, split):
        super().__init__(cfg, epoch, split)
        if cfg['dataset'] != 'concatdata_bbox':
            self.img_root = Path(ROOT)/cfg['img_root']
            self.img_ref_width = int(cfg['img_ref_width'])
            self.img_ref_height = int(cfg['img_ref_height'])
        self.img_rsz = int(cfg['img_rsz'])
        self.img_paths, self.bbox_gt = self.gt
        if self.epoch is not None:
            self.bbox_pred = self.get_pred()

    def get_len(self):
        return len(self.img_paths)

    def get_pred(self):
        pred_file = Path(ROOT)/self.exp_dir/'predict_{}_{}_bbox.txt'.format(self.epoch, self.split)
        bbox_pred = np.loadtxt(pred_file)
        return bbox_pred

    def visualize_one_prediction(self, idx):
        if self.cfg['dataset'] != 'concatdata_bbox':
            img = cv2.imread(str(self.img_root/self.img_paths[idx]))[:, :, ::-1]
        else:
            img = cv2.imread(str(self.img_paths[idx]))[:, :, ::-1]
        img = cv2.resize(img, (self.img_rsz, self.img_rsz))
        fig, ax = plt.subplots()
        print(self.img_paths[idx])
        bbox_pred   = self.bbox_pred[idx].copy()
        bbox_pred[0] = bbox_pred[0]*self.img_rsz
        bbox_pred[1] = bbox_pred[1]*self.img_rsz
        bbox_pred[2] = bbox_pred[2]*self.img_rsz
        bbox_pred[3] = bbox_pred[3]*self.img_rsz

        bbox_gt     = self.bbox_gt[idx].copy()
        if self.cfg['dataset'] != 'concatdata_bbox':
            bbox_gt[0] = (bbox_gt[0]/self.img_ref_width)*self.img_rsz
            bbox_gt[1] = (bbox_gt[1]/self.img_ref_height)*self.img_rsz
            bbox_gt[2] = (bbox_gt[2]/self.img_ref_width)*self.img_rsz
            bbox_gt[3] = (bbox_gt[3]/self.img_ref_height)*self.img_rsz
        else:
            bbox_gt[0] = bbox_gt[0]*self.img_rsz
            bbox_gt[1] = bbox_gt[1]*self.img_rsz
            bbox_gt[2] = bbox_gt[2]*self.img_rsz
            bbox_gt[3] = bbox_gt[3]*self.img_rsz

        print(bbox_iou(bbox_pred, bbox_gt))

        ax.imshow(img)
        draw_bbox(ax, bbox_pred, c='r')
        draw_bbox(ax, bbox_gt, c='b')
        plt.show()

    def eval(self):
        if self.cfg['dataset'] != 'concatdata_bbox':
            bbox_pred   = self.bbox_pred.copy()
            bbox_pred[..., 0] = bbox_pred[..., 0]*self.img_ref_width
            bbox_pred[..., 1] = bbox_pred[..., 1]*self.img_ref_height
            bbox_pred[..., 2] = bbox_pred[..., 2]*self.img_ref_width
            bbox_pred[..., 3] = bbox_pred[..., 3]*self.img_ref_height
        else:
            bbox_pred   = self.bbox_pred.copy()
            # bbox_pred[..., 0] = bbox_pred[..., 0]
            # bbox_pred[..., 1] = bbox_pred[..., 1]
            # bbox_pred[..., 2] = bbox_pred[..., 2]
            # bbox_pred[..., 3] = bbox_pred[..., 3]

        iou_thresh      = 0.5
        correct         = 0
        avg_iou         = 0
        worst_iou       = 10
        worst_iou_idx   = 0
        iou_list        = []
        for i in range(len(bbox_pred)):
            iou = bbox_iou(bbox_pred[i], self.bbox_gt[i])
            avg_iou += iou
            iou_list.append(iou)
            if iou < worst_iou:
                worst_iou = iou
                worst_iou_idx = i
            if iou > iou_thresh:
                correct += 1

        recall = correct/len(bbox_pred)
        avg_iou = avg_iou/len(bbox_pred)
        print('Recall:', recall)
        print('Avg_IOU:', avg_iou)
        print('Worst Bbox id:', worst_iou_idx)
        # for idx in np.argsort(iou_list):
        #     print(idx)