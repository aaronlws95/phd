import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src import ROOT
from src.eval_helper.base_eval_helper import Base_Eval_Helper
from src.networks.backbones import get_stride
from src.utils import *

class Multireso_Hand_Crop_Helper(Base_Eval_Helper):
    def __init__(self, cfg, epoch, split):
        super().__init__(cfg, epoch, split)
        self.img_root = Path(ROOT)/cfg['img_root']
        self.ref_depth = int(cfg['ref_depth'])
        self.img_rsz = int(cfg['img_rsz'])
        self.pad = int(cfg['pad'])
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
        img, _, _, _ = FPHA.crop_hand(img, uvd_gt, pad=self.pad)
        # uvd_gt[..., 0] *= self.img_rsz/img.shape[1]
        # uvd_gt[..., 1] *= self.img_rsz/img.shape[0]

        img = cv2.resize(img, (self.img_rsz, self.img_rsz))

        print(self.img_paths[idx])

        pred_uvd = self.pred_uvd[idx].copy()
        pred_uvd[..., 0] *= self.img_rsz
        pred_uvd[..., 1] *= self.img_rsz
        pred_uvd[..., 2] *= self.ref_depth

        gt_uvd = self.uvd_gt[idx].copy()
        gt_uvd[..., 0] *= self.img_rsz
        gt_uvd[..., 1] *= self.img_rsz
        gt_uvd[..., 2] *= self.ref_depth

        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.axis('off')
        draw_joints(ax, pred_uvd, c='r')
        draw_joints(ax, gt_uvd, c='b')
        plt.show()

    def eval(self):

        pred_uvd = self.pred_uvd.copy()
        pred_uvd[..., 0] *= 96
        pred_uvd[..., 1] *= 96
        pred_uvd[..., 2] *= self.ref_depth

        uvd_gt = self.uvd_gt.copy()
        uvd_gt[..., 0] *= 96
        uvd_gt[..., 1] *= 96
        uvd_gt[..., 2] *= self.ref_depth

        print('UVD mean_l2_error: ', mean_L2_error(uvd_gt[:len(pred_uvd)], pred_uvd))
        # print('XYZ mean_l2_error: ', mean_L2_error(self.xyz_gt[:len(self.pred_uvd)], self.pred_xyz))
        error = []
        for i, (pred, uvd) in enumerate(zip(pred_uvd, uvd_gt)):
            # print(mean_L2_error(uvd, pred))
            error.append(mean_L2_error(uvd, pred))
        error = np.asarray(error)
        min_error_idx = np.argmin(error)
        max_error_idx = np.argmax(error)
        print('Best Pose id:', min_error_idx, 'uvd_error:', error[min_error_idx])
        print('Worst Pose id:', max_error_idx, 'uvd_error:', error[max_error_idx])
        # for idx in np.argsort(error):
        #     print(idx)

        pck = percentage_frames_within_error_curve(uvd_gt[:len(pred_uvd)], pred_uvd)
        pck_str = '['
        for p in pck:
            pck_str += str(p) + ', '
        pck_str += ']'
        print(pck_str)
        thresholds = np.arange(0, 85, 5)
        print('AUC:', calc_auc(pck, thresholds))

        print('UVD 2D mean_l2_error: ', mean_L2_error(uvd_gt[:len(pred_uvd)][..., :2], pred_uvd[..., :2]))

        pck = percentage_frames_within_error_curve(uvd_gt[:len(pred_uvd)][..., :2], pred_uvd[..., :2])
        pck_str = '['
        for p in pck:
            pck_str += str(p) + ', '
        pck_str += ']'
        print(pck_str)
        thresholds = np.arange(0, 85, 5)
        print('AUC:', calc_auc(pck, thresholds))

        print('X', np.mean(np.sqrt(np.square(uvd_gt[..., 0] - pred_uvd[..., 0]) + 1e-8)))
        print('Y', np.mean(np.sqrt(np.square(uvd_gt[..., 1] - pred_uvd[..., 1]) + 1e-8)))
        print('Z', np.mean(np.sqrt(np.square(uvd_gt[..., 2] - pred_uvd[..., 2]) + 1e-8)))

        # pck = percentage_frames_within_error_curve(self.xyz_gt[:len(self.pred_uvd)], self.pred_xyz)
        # pck_str = '['
        # for p in pck:
        #     pck_str += str(p) + ', '
        # pck_str += ']'
        # print(pck_str)
        # thresholds = np.arange(0, 85, 5)
        # print('AUC:', calc_auc(pck, thresholds))