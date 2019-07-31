import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src import ROOT
from src.eval_helper.base_eval_helper import Base_Eval_Helper
from src.networks.backbones import get_stride
from src.utils import *

class Multireso_Hand_Crop_Normuvd_Helper(Base_Eval_Helper):
    def __init__(self, cfg, epoch, split):
        super().__init__(cfg, epoch, split)
        self.img_root = Path(ROOT)/cfg['img_root']
        self.ref_depth = int(cfg['ref_depth'])
        self.img_rsz = int(cfg['img_rsz'])

        self.img_paths, uvd_gt_ori = self.gt
        if self.epoch is not None:
            self.pred_uvd, self.uvd_gt = self.get_pred()

        self.uvd_gt_ori = uvd_gt_ori.copy()
        self.xyz_gt = FPHA.uvd2xyz_color(self.uvd_gt_ori)

    def get_len(self):
        return len(self.img_paths)

    def get_pred(self):
        pred_file = Path(ROOT)/self.exp_dir/'predict_{}_{}_uvd.txt'.format(self.epoch, self.split)
        pred_uvd = np.reshape(np.loadtxt(pred_file), (-1, 21, 3))

        gt_file = Path(ROOT)/self.exp_dir/'gt_{}_{}_uvd.txt'.format(self.epoch, self.split)
        gt_uvd = np.reshape(np.loadtxt(gt_file), (-1, 21, 3))

        return pred_uvd, gt_uvd

    def visualize_one_prediction(self, idx):
        img = cv2.imread(str(self.img_root/self.img_paths[idx]))[:, :, ::-1]

        uvd_gt = self.uvd_gt_ori[idx].copy()
        img_ori = img.copy()
        img, uvd_gt, x_min, y_min = FPHA.crop_hand(img, uvd_gt)
        # uvd_gt[..., 0] *= img.shape[1]/FPHA.ORI_WIDTH
        # uvd_gt[..., 1] *= img.shape[0]/FPHA.ORI_HEIGHT
        uvd_gt[..., 0] *= self.img_rsz/img.shape[1]
        uvd_gt[..., 1] *= self.img_rsz/img.shape[0]
        pred_uvd = self.pred_uvd[idx].copy()
        pred_uvd[..., 0] *= img_ori.shape[1]/FPHA.ORI_WIDTH
        pred_uvd[..., 1] *= img_ori.shape[0]/FPHA.ORI_HEIGHT
        pred_uvd[..., 0] -= x_min
        pred_uvd[..., 1] -= y_min
        pred_uvd[..., 0] *= self.img_rsz/img.shape[1]
        pred_uvd[..., 1] *= self.img_rsz/img.shape[0]

        img = cv2.resize(img, (self.img_rsz, self.img_rsz))

        print(self.img_paths[idx])
        fig, ax = plt.subplots()
        ax.imshow(img)
        draw_joints(ax, pred_uvd, c='r')
        draw_joints(ax, uvd_gt, c='b')
        plt.show()

    def eval(self):

        print('UVD mean_l2_error: ', mean_L2_error(self.uvd_gt[:len(self.pred_uvd)], self.pred_uvd))
        # print('XYZ mean_l2_error: ', mean_L2_error(self.xyz_gt[:len(self.pred_uvd)], self.pred_xyz))
        error = []
        for i, (pred, uvd) in enumerate(zip(self.pred_uvd, self.uvd_gt)):
            # print(mean_L2_error(uvd, pred))
            error.append(mean_L2_error(uvd, pred))
        error = np.asarray(error)
        min_error_idx = np.argmin(error)
        max_error_idx = np.argmax(error)
        print('Best Pose id:', min_error_idx, 'uvd_error:', error[min_error_idx])
        print('Worst Pose id:', max_error_idx, 'uvd_error:', error[max_error_idx])
        # for idx in np.argsort(error):
        #     print(idx)

        pck = percentage_frames_within_error_curve(self.uvd_gt[:len(self.pred_uvd)], self.pred_uvd)
        pck_str = '['
        for p in pck:
            pck_str += str(p) + ', '
        pck_str += ']'
        print(pck_str)
        thresholds = np.arange(0, 85, 5)
        print('AUC:', calc_auc(pck, thresholds))