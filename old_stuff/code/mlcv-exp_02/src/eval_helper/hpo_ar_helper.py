import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src import ROOT
from src.eval_helper.base_eval_helper import Base_Eval_Helper
from src.networks.backbones import get_stride
from src.utils import *

class HPO_AR_Helper(Base_Eval_Helper):
    def __init__(self, cfg, epoch, split):
        super().__init__(cfg, epoch, split)
        self.img_root = Path(ROOT)/cfg['img_root']

        split_set = Path(ROOT)/cfg['{}_set'.format(split)]

        with open(split_set, 'r') as f:
            img_labels = f.read().splitlines()
        self.seq_info = [(int(i.split(' ')[1]), int(i.split(' ')[2]), int(i.split(' ')[3])) for i in img_labels]

        self.num_actions = int(cfg['num_actions'])
        self.num_objects = int(cfg['num_objects'])

        if 'fpha' in cfg['dataset']:
            self.action_dict    = FPHA.get_action_dict()
            self.obj_dict       = FPHA.get_obj_dict()
        elif 'ek' in cfg['dataset']:
            self.action_dict    = EK.get_verb_dict()
            self.obj_dict       = EK.get_noun_dict()

        self.img_paths, self.action_gt, self.obj_gt = self.gt

        if self.epoch is not None:
            self.action_pred, self.obj_pred, self.action_dist, self.obj_dist = self.get_pred()

    def get_len(self):
        return len(self.img_paths)

    def get_pred(self):
        pred_file   = Path(ROOT)/self.exp_dir/'predict_{}_{}_action.txt'.format(self.epoch, self.split)
        action_pred = np.loadtxt(pred_file)
        pred_file   = Path(ROOT)/self.exp_dir/'predict_{}_{}_obj.txt'.format(self.epoch, self.split)
        obj_pred    = np.loadtxt(pred_file)
        pred_file   = Path(ROOT)/self.exp_dir/'predict_{}_{}_action_dist.txt'.format(self.epoch, self.split)
        action_dist = np.loadtxt(pred_file)
        pred_file   = Path(ROOT)/self.exp_dir/'predict_{}_{}_obj_dist.txt'.format(self.epoch, self.split)
        obj_dist    = np.loadtxt(pred_file)
        return action_pred, obj_pred, action_dist, obj_dist

    def visualize_one_prediction(self, idx):
        img = cv2.imread(str(self.img_root/self.img_paths[idx]))[:, :, ::-1]
        print(self.img_paths[idx])
        print('Ground-truth action + object:', self.action_dict[self.action_gt[idx]], '+', self.obj_dict[self.obj_gt[idx]])
        print('Predicted action + object:', self.action_dict[self.action_pred[idx]], '+', self.obj_dict[self.obj_pred[idx]])
        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.show()

    def eval(self):
        # self.print_wrong_id()
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
        act_acc, obj_acc, both_acc = seq_mean_class_accuracy_2(self.seq_info, self.action_dist, self.obj_dist, self.action_gt, self.obj_gt, 'mean_dist', self.num_actions, self.num_objects)
        print('Top1 Seq Mean_dist Mean-Class Action Accuracy', act_acc)
        print('Top1 Seq Mean_dist Mean-Class Object Accuracy', obj_acc)
        print('Top1 Seq Mean_dist Mean-Class Both Accuracy', both_acc)
        print('------------------------------')
        act_acc, obj_acc, both_acc = seq_mean_class_accuracy_2(self.seq_info, self.action_dist, self.obj_dist, self.action_gt, self.obj_gt, 'pred_freq', self.num_actions, self.num_objects)
        print('Top1 Seq Pred_freq Mean-Class Action Accuracy', act_acc)
        print('Top1 Seq Pred_freq Mean-Class Object Accuracy', obj_acc)
        print('Top1 Seq Pred_freq Mean-Class Both Accuracy', both_acc)
        print('------------------------------')
        num_segments = 5
        act_acc, obj_acc, both_acc = seq_seg_mean_class_accuracy_2(self.seq_info, self.action_dist, self.obj_dist, self.action_gt, self.obj_gt, 'mean_dist', num_segments, self.num_actions, self.num_objects)
        print(f'Top1 Seq Seg{num_segments} Mean_dist Mean-Class Action Accuracy', act_acc)
        print(f'Top1 Seq Seg{num_segments} Mean_dist Mean-Class Object Accuracy', obj_acc)
        print(f'Top1 Seq Seg{num_segments} Mean_dist Mean-Class Both Accuracy', both_acc)
        print('------------------------------')

    def print_wrong_id(self):
        wrong_id = [i for i, (pred, gt) in enumerate(zip(self.action_pred, self.action_gt))
                    if pred != gt]
        print(wrong_id)
