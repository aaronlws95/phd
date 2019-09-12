import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src import ROOT
from src.eval_helper.base_eval_helper import Base_Eval_Helper
from src.utils import *

class HPO_Hand_Object_Helper(Base_Eval_Helper):
    def __init__(self, cfg, epoch, split):
        super().__init__(cfg, epoch, split)
        self.img_root = Path(ROOT)/cfg['img_root']
        self.ref_depth = int(cfg['ref_depth'])
        self.img_rsz = int(cfg['img_rsz'])
        self.img_ref_width = int(cfg['img_ref_width'])
        self.img_ref_height = int(cfg['img_ref_height'])

        self.img_paths, self.uvd_gt, self.obj_gt, self.ccs_gt = self.gt
        if self.epoch is not None:
            self.pred_uvd, self.pred_conf, self.pred_uvd_obj, self.pred_conf_obj = self.get_pred()

            self.ccs_gt = np.reshape(self.ccs_gt, (-1, 21, 3))

            pred_ccs = self.pred_uvd.copy()
            pred_ccs[..., 0] *= self.img_ref_width
            pred_ccs[..., 1] *= self.img_ref_height
            pred_ccs[..., 2] *= self.ref_depth

            pred_ccs[:, :, 2] = -pred_ccs[:, :, 2]
            pred_ccs = HO3D.uvd2ccs(pred_ccs)
            pred_ccs = HO3D.from_opengl_coord(pred_ccs)
            self.pred_ccs = pred_ccs

            pred_ccs_obj = self.pred_uvd_obj.copy()
            pred_ccs_obj[..., 0] *= self.img_ref_width
            pred_ccs_obj[..., 1] *= self.img_ref_height
            pred_ccs_obj[..., 2] *= self.ref_depth

            pred_ccs_obj[:, :, 2] = -pred_ccs_obj[:, :, 2]
            pred_ccs_obj = HO3D.uvd2ccs(pred_ccs_obj)
            pred_ccs_obj = HO3D.from_opengl_coord(pred_ccs_obj)
            self.pred_ccs_obj = pred_ccs_obj

        ccs_gt_obj = self.obj_gt.copy()

        ccs_gt_obj[:, :, 2] = -ccs_gt_obj[:, :, 2]
        ccs_gt_obj = HO3D.uvd2ccs(ccs_gt_obj)
        ccs_gt_obj = HO3D.from_opengl_coord(ccs_gt_obj)
        self.ccs_gt_obj = ccs_gt_obj

    def get_len(self):
        return len(self.img_paths)

    def get_pred(self):
        pred_file = Path(ROOT)/self.exp_dir/'predict_{}_{}_best.txt'.format(self.epoch, self.split)
        pred_uvd = np.reshape(np.loadtxt(pred_file), (-1, 21, 3))
        pred_file = Path(ROOT)/self.exp_dir/'predict_{}_{}_conf.txt'.format(self.epoch, self.split)
        pred_conf = np.loadtxt(pred_file)
        pred_file = Path(ROOT)/self.exp_dir/'predict_{}_{}_best_obj.txt'.format(self.epoch, self.split)
        pred_uvd_obj = np.reshape(np.loadtxt(pred_file), (-1, 21, 3))
        pred_file = Path(ROOT)/self.exp_dir/'predict_{}_{}_conf_obj.txt'.format(self.epoch, self.split)
        pred_conf_obj = np.loadtxt(pred_file)
        return pred_uvd, pred_conf, pred_uvd_obj, pred_conf_obj

    def visualize_one_prediction(self, idx):
        img = cv2.imread(str(self.img_root/self.img_paths[idx]))[:, :, ::-1]
        img = cv2.resize(img, (self.img_rsz, self.img_rsz))
        print(self.img_paths[idx])
        print('hand conf: {}'.format(np.max(self.pred_conf[idx])))
        print('object conf: {}'.format(np.max(self.pred_conf_obj[idx])))

        uvd_gt = self.uvd_gt[idx].copy()
        uvd_gt[..., 0] *= self.img_rsz/self.img_ref_width
        uvd_gt[..., 1] *= self.img_rsz/self.img_ref_height

        obj_gt = self.obj_gt[idx].copy()
        obj_gt[..., 0] *= self.img_rsz/self.img_ref_width
        obj_gt[..., 1] *= self.img_rsz/self.img_ref_height

        pred_uvd = self.pred_uvd[idx].copy()
        pred_uvd[..., 0] *= self.img_rsz
        pred_uvd[..., 1] *= self.img_rsz
        pred_uvd[..., 2] *= self.ref_depth

        pred_uvd_obj = self.pred_uvd_obj[idx].copy()
        pred_uvd_obj[..., 0] *= self.img_rsz
        pred_uvd_obj[..., 1] *= self.img_rsz
        pred_uvd_obj[..., 2] *= self.ref_depth

        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.axis('off')
        draw_joints(ax, pred_uvd, c='r')
        draw_joints(ax, uvd_gt, c='b')
        draw_obj_joints(ax, obj_gt, c='b')
        draw_obj_joints(ax, pred_uvd_obj, c='r')
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        # plt.axis('off')
        draw_3D_joints(ax, pred_uvd, c='r')
        draw_3D_joints(ax, uvd_gt, c='b')
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        # plt.axis('off')
        draw_obj_3D_joints(ax, pred_uvd_obj, c='r')
        # draw_3D_joints(ax, pred_uvd, c='r')
        draw_obj_3D_joints(ax, obj_gt, c='b')
        # draw_3D_joints(ax, uvd_gt, c='b')
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        # plt.axis('off')
        draw_obj_3D_joints(ax, pred_uvd_obj, c='r')
        draw_3D_joints(ax, pred_uvd, c='r')
        draw_obj_3D_joints(ax, obj_gt, c='b')
        draw_3D_joints(ax, uvd_gt, c='b')
        plt.show()

    def eval_hand(self, error_dict):
        pred_uvd = self.pred_uvd.copy()
        pred_uvd[..., 0] *= self.img_ref_width
        pred_uvd[..., 1] *= self.img_ref_height
        pred_uvd[..., 2] *= self.ref_depth

        uvd_mean_l2_error = mean_L2_error(self.uvd_gt[:len(pred_uvd)], pred_uvd)
        # error_dict['UVD mean_l2_error'] = uvd_mean_l2_error

        error = []
        for i, (pred, uvd) in enumerate(zip(pred_uvd, self.uvd_gt)):
            error.append(mean_L2_error(uvd, pred))
        error = np.asarray(error)
        min_error_idx = np.argmin(error)
        max_error_idx = np.argmax(error)
        print('Best Pose id:', min_error_idx, 'uvd_error:', error[min_error_idx])
        print('Worst Pose id:', max_error_idx, 'uvd_error:', error[max_error_idx])
        # for idx in np.argsort(error):
        #     print(idx)

        uvd_mean_l2_error_2D = mean_L2_error(self.uvd_gt[:len(pred_uvd)][..., :2], pred_uvd[..., :2])
        x_error = mean_L2_error(self.uvd_gt[:len(pred_uvd)][..., 0], pred_uvd[..., 0])
        y_error = mean_L2_error(self.uvd_gt[:len(pred_uvd)][..., 1], pred_uvd[..., 1])
        z_error = mean_L2_error(self.uvd_gt[:len(pred_uvd)][..., 2], pred_uvd[..., 2])

        # error_dict['2D UVD mean_l2_error'] = uvd_mean_l2_error_2D
        # error_dict['x_error_uvd'] = x_error
        # error_dict['y_error_uvd'] = y_error
        # error_dict['z_error_uvd'] = z_error

        # CCS
        ccs_mean_l2_error_2D = mean_L2_error(self.ccs_gt[:len(self.pred_ccs)], self.pred_ccs)
        error_dict['CCS mean_l2_error'] = ccs_mean_l2_error_2D

        pck = percentage_frames_within_error_curve(self.ccs_gt[:len(self.pred_ccs)], self.pred_ccs)
        pck_str = '['
        for p in pck:
            pck_str += str(p) + ', '
        pck_str += ']'
        # print(pck_str)
        thresholds = np.arange(0, 85, 5)
        auc = calc_auc(pck, thresholds)
        error_dict['CCS AUC'] = auc

        pck = percentage_frames_within_error_curve(self.ccs_gt[:len(self.pred_ccs)][..., :2], self.pred_ccs[..., :2], plot=False)
        pck_str = '['
        for p in pck:
            pck_str += str(p) + ', '
        pck_str += ']'
        # print(pck_str)
        thresholds = np.arange(0, 85, 5)
        auc_2D = calc_auc(pck, thresholds)
        error_dict['2D CCS AUC'] = auc_2D

        x_error = mean_L2_error(self.ccs_gt[:len(pred_uvd)][..., 0], self.pred_ccs[..., 0])
        y_error = mean_L2_error(self.ccs_gt[:len(pred_uvd)][..., 1], self.pred_ccs[..., 1])
        z_error = mean_L2_error(self.ccs_gt[:len(pred_uvd)][..., 2], self.pred_ccs[..., 2])

        error_dict['x_error_ccs'] = x_error
        error_dict['y_error_ccs'] = y_error
        error_dict['z_error_ccs'] = z_error

    def eval_object(self, error_dict):
        pred_uvd = self.pred_uvd_obj.copy()
        pred_uvd[..., 0] *= self.img_ref_width
        pred_uvd[..., 1] *= self.img_ref_height
        pred_uvd[..., 2] *= self.ref_depth

        uvd_mean_l2_error = mean_L2_error(self.obj_gt[:len(pred_uvd)], pred_uvd)
        # error_dict['UVD mean_l2_error'] = uvd_mean_l2_error

        error = []
        for i, (pred, uvd) in enumerate(zip(pred_uvd, self.obj_gt)):
            error.append(mean_L2_error(uvd, pred))
        error = np.asarray(error)
        min_error_idx = np.argmin(error)
        max_error_idx = np.argmax(error)
        print('Best Pose id OBJ:', min_error_idx, 'uvd_error:', error[min_error_idx])
        print('Worst Pose id OBJ:', max_error_idx, 'uvd_error:', error[max_error_idx])
        # for idx in np.argsort(error):
        #     print(idx)

        uvd_mean_l2_error_2D = mean_L2_error(self.obj_gt[:len(pred_uvd)][..., :2], pred_uvd[..., :2])
        x_error = mean_L2_error(self.obj_gt[:len(pred_uvd)][..., 0], pred_uvd[..., 0])
        y_error = mean_L2_error(self.obj_gt[:len(pred_uvd)][..., 1], pred_uvd[..., 1])
        z_error = mean_L2_error(self.obj_gt[:len(pred_uvd)][..., 2], pred_uvd[..., 2])

        # error_dict['2D UVD mean_l2_error'] = uvd_mean_l2_error_2D
        # error_dict['x_error_uvd'] = x_error
        # error_dict['y_error_uvd'] = y_error
        # error_dict['z_error_uvd'] = z_error

        # CCS
        ccs_mean_l2_error_2D = mean_L2_error(self.ccs_gt_obj[:len(self.pred_ccs_obj)], self.pred_ccs_obj)
        error_dict['CCS mean_l2_error OBJ'] = ccs_mean_l2_error_2D

        pck = percentage_frames_within_error_curve(self.ccs_gt_obj[:len(self.pred_ccs_obj)], self.pred_ccs_obj)
        pck_str = '['
        for p in pck:
            pck_str += str(p) + ', '
        pck_str += ']'
        # print(pck_str)
        thresholds = np.arange(0, 85, 5)
        auc = calc_auc(pck, thresholds)
        error_dict['CCS AUC OBJ'] = auc

        pck = percentage_frames_within_error_curve(self.ccs_gt_obj[:len(self.pred_ccs_obj)][..., :2], self.pred_ccs_obj[..., :2], plot=False)
        pck_str = '['
        for p in pck:
            pck_str += str(p) + ', '
        pck_str += ']'
        # print(pck_str)
        thresholds = np.arange(0, 85, 5)
        auc_2D = calc_auc(pck, thresholds)
        error_dict['2D CCS AUC OBJ'] = auc_2D

        x_error = mean_L2_error(self.ccs_gt_obj[:len(pred_uvd)][..., 0], self.pred_ccs_obj[..., 0])
        y_error = mean_L2_error(self.ccs_gt_obj[:len(pred_uvd)][..., 1], self.pred_ccs_obj[..., 1])
        z_error = mean_L2_error(self.ccs_gt_obj[:len(pred_uvd)][..., 2], self.pred_ccs_obj[..., 2])

        error_dict['x_error_ccs OBJ'] = x_error
        error_dict['y_error_ccs OBJ'] = y_error
        error_dict['z_error_ccs OBJ'] = z_error

    def eval(self):
        error_dict = {}
        self.eval_hand(error_dict)
        self.eval_object(error_dict)
        eval_txt = self.exp_dir/'eval_{}.txt'.format(self.split)
        for i, v in error_dict.items():
            print(i, ':', v)
        with open(eval_txt, 'w') as f:
            for i, v in error_dict.items():
                f.write('{} {}\n'.format(i, v))
