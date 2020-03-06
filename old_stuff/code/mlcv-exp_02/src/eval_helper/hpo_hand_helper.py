import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src import ROOT
from src.eval_helper.base_eval_helper import Base_Eval_Helper
from src.networks.backbones import get_stride
from src.utils import *

class HPO_Hand_Helper(Base_Eval_Helper):
    def __init__(self, cfg, epoch, split):
        super().__init__(cfg, epoch, split)
        self.img_root = Path(ROOT)/cfg['img_root']
        self.ref_depth = int(cfg['ref_depth'])
        self.img_rsz = int(cfg['img_rsz'])
        self.dataset_name = cfg['dataset']

        if self.dataset_name == 'fpha_hand':
            self.img_paths, uvd_gt_ori = self.gt
            if self.epoch is not None:
                self.pred_uvd, self.pred_xyz, self.pred_conf = self.get_pred()

            self.uvd_gt = uvd_gt_ori.copy()
            self.xyz_gt = FPHA.uvd2xyz_color(self.uvd_gt)

            self.uvd_gt[..., 0] *= self.img_rsz/FPHA.ORI_WIDTH
            self.uvd_gt[..., 1] *= self.img_rsz/FPHA.ORI_HEIGHT

        elif self.dataset_name == 'fpha_hand_crop':
            self.pad = int(cfg['pad'])
            self.img_paths, uvd_gt_ori = self.gt
            if self.epoch is not None:
                self.pred_uvd, self.pred_xyz, self.pred_conf, self.uvd_gt = self.get_pred()

            self.uvd_gt_ori = uvd_gt_ori.copy()
            self.xyz_gt = FPHA.uvd2xyz_color(self.uvd_gt)

    def get_len(self):
        return len(self.img_paths)

    def get_pred(self):
        pred_file = Path(ROOT)/self.exp_dir/'predict_{}_{}_best.txt'.format(self.epoch, self.split)
        pred_uvd = np.reshape(np.loadtxt(pred_file), (-1, 21, 3))

        pred_xyz = pred_uvd.copy()
        pred_xyz[..., 0] *= FPHA.ORI_WIDTH
        pred_xyz[..., 1] *= FPHA.ORI_HEIGHT
        pred_xyz[..., 2] *= self.ref_depth
        pred_xyz = FPHA.uvd2xyz_color(pred_xyz)

        pred_uvd[..., 0] *= self.img_rsz
        pred_uvd[..., 1] *= self.img_rsz
        pred_uvd[..., 2] *= self.ref_depth

        pred_file = Path(ROOT)/self.exp_dir/'predict_{}_{}_conf.txt'.format(self.epoch, self.split)
        pred_conf = np.loadtxt(pred_file)

        if self.dataset_name =='fpha_hand':
            return pred_uvd, pred_xyz, pred_conf
        elif self.dataset_name == 'fpha_hand_crop':
            uvd_file = Path(ROOT)/self.exp_dir/'uvd_gt_{}_{}.txt'.format(self.epoch, self.split)
            uvd_gt = np.reshape(np.loadtxt(uvd_file), (-1, 21, 3))
            uvd_gt[..., 0] *= self.img_rsz
            uvd_gt[..., 1] *= self.img_rsz
            uvd_gt[..., 2] *= self.ref_depth

            return pred_uvd, pred_xyz, pred_conf, uvd_gt

    def visualize_one_prediction(self, idx):

        if self.dataset_name == 'fpha_hand':
            img = cv2.imread(str(self.img_root/self.img_paths[idx]))[:, :, ::-1]
            print(self.img_paths[idx])
            print('conf: {}'.format(np.max(self.pred_conf[idx])))
            img = cv2.resize(img, (self.img_rsz, self.img_rsz))
            fig, ax = plt.subplots()
            ax.imshow(img)
            plt.axis('off')
            draw_joints(ax, self.pred_uvd[idx], c='r')
            draw_joints(ax, self.uvd_gt[idx], c='b')
            plt.show()
        elif self.dataset_name == 'fpha_hand_crop':
            img = cv2.imread(str(self.img_root/self.img_paths[idx]))[:, :, ::-1]

            uvd_gt = self.uvd_gt_ori[idx].copy()
            img, _, _, _ = FPHA.crop_hand(img, uvd_gt, pad=self.pad)
            # uvd_gt[..., 0] *= self.img_rsz/img.shape[1]
            # uvd_gt[..., 1] *= self.img_rsz/img.shape[0]

            img = cv2.resize(img, (self.img_rsz, self.img_rsz))

            print(self.img_paths[idx])

            pred_uvd = self.pred_uvd[idx].copy()
            # pred_uvd[..., 0] *= self.img_rsz
            # pred_uvd[..., 1] *= self.img_rsz
            # pred_uvd[..., 2] *= self.ref_depth

            gt_uvd = self.uvd_gt[idx].copy()
            # gt_uvd[..., 0] *= self.img_rsz
            # gt_uvd[..., 1] *= self.img_rsz
            # gt_uvd[..., 2] *= self.ref_depth

            fig, ax = plt.subplots()
            ax.imshow(img)
            plt.axis('off')
            draw_joints(ax, pred_uvd, c='r')
            draw_joints(ax, gt_uvd, c='b')
            plt.show()

    def eval(self):
        print('UVD mean_l2_error: ', mean_L2_error(self.uvd_gt[:len(self.pred_uvd)], self.pred_uvd))
        print('XYZ mean_l2_error: ', mean_L2_error(self.xyz_gt[:len(self.pred_uvd)], self.pred_xyz))
        error = []
        for i, (pred, uvd) in enumerate(zip(self.pred_uvd, self.uvd_gt)):
            error.append(mean_L2_error(uvd, pred))
        error = np.asarray(error)
        min_error_idx = np.argmin(error)
        max_error_idx = np.argmax(error)
        print('Best Pose id:', min_error_idx, 'uvd_error:', error[min_error_idx])
        print('Worst Pose id:', max_error_idx, 'uvd_error:', error[max_error_idx])
        # for idx in np.argsort(error):
        #     print(idx)

        if self.dataset_name == 'fpha_hand':
            pck = percentage_frames_within_error_curve(self.xyz_gt[:len(self.pred_uvd)], self.pred_xyz)
            pck_str = '['
            for p in pck:
                pck_str += str(p) + ', '
            pck_str += ']'
            print(pck_str)
            thresholds = np.arange(0, 85, 5)
            print('AUC:', calc_auc(pck, thresholds))
        elif self.dataset_name == 'fpha_hand_crop':
            pck = percentage_frames_within_error_curve(self.uvd_gt[:len(self.pred_uvd)], self.pred_uvd)
            pck_str = '['
            for p in pck:
                pck_str += str(p) + ', '
            pck_str += ']'
            print(pck_str)
            thresholds = np.arange(0, 85, 5)
            print('AUC:', calc_auc(pck, thresholds))

        # Z Normalized

        z_norm_pred_xyz = []
        for gt, pred in zip(self.xyz_gt, self.pred_xyz):
            new_pred = pred.copy()
            new_pred[:, 2] = new_pred[:, 2] - np.mean(pred[:, 2]) + np.mean(gt[:, 2])
            z_norm_pred_xyz.append(new_pred)
        z_norm_pred_uvd = []
        for gt, pred in zip(self.uvd_gt, self.pred_uvd):
            new_pred = pred.copy()
            new_pred[:, 2] = new_pred[:, 2] - np.mean(pred[:, 2]) + np.mean(gt[:, 2])
            z_norm_pred_uvd.append(new_pred)

        print('\nZ-normed UVD mean_l2_error: ', mean_L2_error(self.uvd_gt[:len(z_norm_pred_uvd)], z_norm_pred_uvd))
        print('Z-normed XYZ mean_l2_error: ', mean_L2_error(self.xyz_gt[:len(z_norm_pred_uvd)], z_norm_pred_xyz))
        error = []
        for i, (pred, uvd) in enumerate(zip(z_norm_pred_uvd, self.uvd_gt)):
            error.append(mean_L2_error(uvd, pred))
        error = np.asarray(error)
        min_error_idx = np.argmin(error)
        max_error_idx = np.argmax(error)
        print('Z-normed Best Pose id:', min_error_idx, 'uvd_error:', error[min_error_idx])
        print('Z-normed Worst Pose id:', max_error_idx, 'uvd_error:', error[max_error_idx])
        # for idx in np.argsort(error):
        #     print(idx)

        pck = percentage_frames_within_error_curve(self.xyz_gt[:len(z_norm_pred_uvd)], z_norm_pred_xyz)
        pck_str = '['
        for p in pck:
            pck_str += str(p) + ', '
        pck_str += ']'
        print(pck_str)
        thresholds = np.arange(0, 85, 5)
        print('AUC:', calc_auc(pck, thresholds))

        pred_uvd = self.pred_uvd.copy()
        pred_uvd[..., 0] *= 96/self.img_rsz
        pred_uvd[..., 1] *= 96/self.img_rsz
        # pred_uvd[..., 2] *= self.ref_depth

        uvd_gt = self.uvd_gt.copy()
        uvd_gt[..., 0] *= 96/self.img_rsz
        uvd_gt[..., 1] *= 96/self.img_rsz
        # uvd_gt[..., 2] *= self.ref_depth

        print('UVD mean_l2_error (96x96): ', mean_L2_error(uvd_gt[:len(pred_uvd)], pred_uvd))
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