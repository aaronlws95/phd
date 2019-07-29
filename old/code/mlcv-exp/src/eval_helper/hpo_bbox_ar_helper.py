import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

from src import ROOT
from src.eval_helper.base_eval_helper import Base_Eval_Helper
from src.networks.backbones import get_stride
from src.utils import *

class HPO_Bbox_AR_Helper(Base_Eval_Helper):
    def __init__(self, cfg, epoch, split):
        super().__init__(cfg, epoch, split)
        self.img_root = Path(ROOT)/cfg['img_root']

        split_set = Path(ROOT)/(cfg['{}_set'.format(split)]).replace('bbox', 'ar_seq')

        with open(split_set, 'r') as f:
            img_labels = f.read().splitlines()
        self.seq_info = [(int(i.split(' ')[1]), int(i.split(' ')[2]), int(i.split(' ')[3])) for i in img_labels]

        self.num_actions = int(cfg['num_actions'])
        self.num_objects = int(cfg['num_objects'])

        self.action_dict    = FPHA.get_action_dict()
        self.obj_dict       = FPHA.get_obj_dict()

        self.img_paths, self.bbox_gt, self.action_gt, self.obj_gt = self.gt
        if self.epoch is not None:
            self.bbox_pred, self.action_pred, self.obj_pred, self.action_dist, self.obj_dist = self.get_pred()

    def get_len(self):
        return len(self.img_paths)

    def get_pred(self):
        pred_file = Path(ROOT)/self.exp_dir/'predict_{}_{}_bbox.txt'.format(self.epoch, self.split)
        bbox_pred = np.loadtxt(pred_file)

        pred_file   = Path(ROOT)/self.exp_dir/'predict_{}_{}_action.txt'.format(self.epoch, self.split)
        action_pred = np.loadtxt(pred_file)
        pred_file   = Path(ROOT)/self.exp_dir/'predict_{}_{}_obj.txt'.format(self.epoch, self.split)
        obj_pred    = np.loadtxt(pred_file)
        pred_file   = Path(ROOT)/self.exp_dir/'predict_{}_{}_action_dist.txt'.format(self.epoch, self.split)
        action_dist = np.loadtxt(pred_file)
        pred_file   = Path(ROOT)/self.exp_dir/'predict_{}_{}_obj_dist.txt'.format(self.epoch, self.split)
        obj_dist    = np.loadtxt(pred_file)

        return bbox_pred, action_pred, obj_pred, action_dist, obj_dist

    def visualize_one_prediction(self, idx):
        img = cv2.imread(str(self.img_root/self.img_paths[idx]))[:, :, ::-1]
        fig, ax = plt.subplots()
        print(self.img_paths[idx])
        bbox_pred   = self.bbox_pred[idx].copy()
        bbox_pred[0] = bbox_pred[0]*img.shape[0]
        bbox_pred[1] = bbox_pred[1]*img.shape[1]
        bbox_pred[2] = bbox_pred[2]*img.shape[0]
        bbox_pred[3] = bbox_pred[3]*img.shape[1]

        bbox_gt     = self.bbox_gt[idx].copy()
        bbox_gt[0] = bbox_gt[0]*img.shape[0]
        bbox_gt[1] = bbox_gt[1]*img.shape[1]
        bbox_gt[2] = bbox_gt[2]*img.shape[0]
        bbox_gt[3] = bbox_gt[3]*img.shape[1]
        print(bbox_pred)
        print('Ground-truth action + object:', self.action_dict[self.action_gt[idx]], '+', self.obj_dict[self.obj_gt[idx]])
        print('Predicted action + object:', self.action_dict[self.action_pred[idx]], '+', self.obj_dict[self.obj_pred[idx]])

        ax.imshow(img)
        draw_bbox(ax, bbox_pred, 'r')
        draw_bbox(ax, bbox_gt, 'b')
        plt.show()

    def eval(self):
        act_acc, obj_acc, both_acc = mean_accuracy_2(self.action_dist, self.obj_dist, self.action_gt, self.obj_gt)
        print('Top1 Action Accuracy', act_acc)
        print('Top1 Object Accuracy', obj_acc)
        print('Top1 Both Accuracy', both_acc)
        print('------------------------------')
        act_acc, obj_acc, both_acc = mean_topk_2(self.action_dist, self.obj_dist, self.action_gt, self.obj_gt, 5)
        print('Top5 Action Accuracy', act_acc)
        print('Top5 Object Accuracy', obj_acc)
        print('Top5 Both Accuracy', both_acc)
        print('------------------------------')
        act_acc, obj_acc, both_acc = mean_class_accuracy_2(self.action_dist, self.obj_dist, self.action_gt, self.obj_gt, self.num_actions, self.num_objects)
        print('Top1 Mean-Class Action Accuracy', act_acc)
        print('Top1 Mean-Class Object Accuracy', obj_acc)
        print('Top1 Mean-Class Both Accuracy', both_acc)
        print('------------------------------')
        act_acc, obj_acc, both_acc = mean_class_topk_2(self.action_dist, self.obj_dist, self.action_gt, self.obj_gt, 5, self.num_actions, self.num_objects)
        print('Top5 Mean-Class Action Accuracy', act_acc)
        print('Top5 Mean-Class Object Accuracy', obj_acc)
        print('Top5 Mean-Class Both Accuracy', both_acc)
        print('------------------------------')

        # Doesn't work cause of invalid img path removal
        # act_acc, obj_acc, both_acc = seq_mean_class_accuracy_2(self.seq_info, self.action_dist, self.obj_dist, self.action_gt, self.obj_gt, 'mean_dist', self.num_actions, self.num_objects)
        # print('Top1 Seq Mean_dist Mean-Class Action Accuracy', act_acc)
        # print('Top1 Seq Mean_dist Mean-Class Object Accuracy', obj_acc)
        # print('Top1 Seq Mean_dist Mean-Class Both Accuracy', both_acc)
        # print('------------------------------')
        # act_acc, obj_acc, both_acc = seq_mean_class_accuracy_2(self.seq_info, self.action_dist, self.obj_dist, self.action_gt, self.obj_gt, 'pred_freq', self.num_actions, self.num_objects)
        # print('Top1 Seq Pred_freq Mean-Class Action Accuracy', act_acc)
        # print('Top1 Seq Pred_freq Mean-Class Object Accuracy', obj_acc)
        # print('Top1 Seq Pred_freq Mean-Class Both Accuracy', both_acc)
        # print('------------------------------')
        # num_segments = 5
        # act_acc, obj_acc, both_acc = seq_seg_mean_class_accuracy_2(self.seq_info, self.action_dist, self.obj_dist, self.action_gt, self.obj_gt, 'mean_dist', num_segments, self.num_actions, self.num_objects)
        # print(f'Top1 Seq Seg{num_segments} Mean_dist Mean-Class Action Accuracy', act_acc)
        # print(f'Top1 Seq Seg{num_segments} Mean_dist Mean-Class Object Accuracy', obj_acc)
        # print(f'Top1 Seq Seg{num_segments} Mean_dist Mean-Class Both Accuracy', both_acc)
        # print('------------------------------')

        iou_thresh      = 0.5
        correct         = 0
        avg_iou         = 0
        worst_iou       = 10
        worst_iou_idx   = 0
        iou_list        = []
        for i in range(len(self.bbox_pred)):
            iou = bbox_iou(self.bbox_pred[i], self.bbox_gt[i])
            avg_iou += iou
            iou_list.append(iou)
            if iou < worst_iou:
                worst_iou = iou
                worst_iou_idx = i
            if iou > iou_thresh:
                correct += 1

        recall = correct/len(self.bbox_pred)
        avg_iou = avg_iou/len(self.bbox_pred)
        print('Recall:', recall)
        print('Avg_IOU:', avg_iou)
        print('Worst Bbox id:', worst_iou_idx)
        # for idx in np.argsort(iou_list):
        #     print(idx)