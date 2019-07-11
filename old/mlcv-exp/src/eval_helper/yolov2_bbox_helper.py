import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src import ROOT
from src.eval_helper.base_eval_helper import Base_Eval_Helper
from src.networks.backbones import get_stride
from src.utils import *

class YOLOV2_Bbox_Helper(Base_Eval_Helper):
    def __init__(self, cfg, epoch, split):
        super().__init__(cfg, epoch, split)
        self.img_root = Path(ROOT)/cfg['img_root']
        
        self.img_paths, self.bbox_gt = self.gt
        if self.epoch is not None:
            self.bbox_pred_1, self.bbox_pred_2 = self.get_pred()
        
    def get_len(self):
        return len(self.img_paths)
    
    def get_pred(self):
        pred_file = Path(ROOT)/self.exp_dir/'predict_{}_{}_bbox.txt'.format(self.epoch, self.split)
        bbox_pred = np.loadtxt(pred_file)
        bbox_pred_1 = np.asarray([bbox[:5] for bbox in bbox_pred]) 
        bbox_pred_2 = np.asarray([bbox[5:] for bbox in bbox_pred])
        return bbox_pred_1, bbox_pred_2
        
    def visualize_one_prediction(self, idx):
        img = cv2.imread(str(self.img_root/self.img_paths[idx]))[:, :, ::-1]
        fig, ax = plt.subplots()
        print(self.img_paths[idx])
        bbox_pred_1   = self.bbox_pred_1[idx].copy()
        bbox_pred_1[0] = bbox_pred_1[0]*img.shape[0]
        bbox_pred_1[1] = bbox_pred_1[1]*img.shape[1]
        bbox_pred_1[2] = bbox_pred_1[2]*img.shape[0]
        bbox_pred_1[3] = bbox_pred_1[3]*img.shape[1]
        
        bbox_pred_2   = self.bbox_pred_2[idx].copy()
        bbox_pred_2[0] = bbox_pred_2[0]*img.shape[0]
        bbox_pred_2[1] = bbox_pred_2[1]*img.shape[1]
        bbox_pred_2[2] = bbox_pred_2[2]*img.shape[0]
        bbox_pred_2[3] = bbox_pred_2[3]*img.shape[1]
    
        bbox_gt     = self.bbox_gt[idx].copy()
        bbox_gt[0] = bbox_gt[0]*img.shape[0]
        bbox_gt[1] = bbox_gt[1]*img.shape[1]
        bbox_gt[2] = bbox_gt[2]*img.shape[0]
        bbox_gt[3] = bbox_gt[3]*img.shape[1]
        
        ax.imshow(img)
        draw_bbox(ax, bbox_pred_1, 'r')
        # draw_bbox(ax, bbox_pred_2, 'r')
        draw_bbox(ax, bbox_gt, 'b')
        plt.show()

    def eval(self):
        iou_thresh      = 0.5
        correct         = 0
        avg_iou         = 0
        worst_iou       = 10
        worst_iou_idx   = 0
        iou_list        = []
        for i in range(len(self.bbox_pred_1)):
            iou = bbox_iou(self.bbox_pred_1[i], self.bbox_gt[i])
            avg_iou += iou
            iou_list.append(iou)
            if iou < worst_iou:
                worst_iou = iou
                worst_iou_idx = i
            if iou > iou_thresh:
                correct += 1

        recall = correct/len(self.bbox_pred_1)
        avg_iou = avg_iou/len(self.bbox_pred_1)
        print('Recall:', recall)
        print('Avg_IOU:', avg_iou)
        print('Worst Bbox id:', worst_iou_idx)
        # for idx in np.argsort(iou_list):
        #     print(idx)