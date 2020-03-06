import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

from src import ROOT
from src.datasets.transforms import *
from src.utils import *

class HO3D_Hand_Object_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        split_set = Path(ROOT)/cfg['{}_set'.format(split)]
        ccs_gt = np.loadtxt(split_set)

        split_set = Path(ROOT)/(cfg['{}_set'.format(split)]).replace('ccs', 'obj6D_ccs')
        obj_ccs_gt = np.loadtxt(split_set)

        split_set = Path(ROOT)/(cfg['{}_set'.format(split)]).replace('ccs', 'corner_ccs')
        corner_gt = np.reshape(np.loadtxt(split_set), (-1, 8, 3))

        split_set = Path(ROOT)/(cfg['{}_set'.format(split)]).replace('ccs', 'img')
        with open(split_set, 'r') as f:
            self.img_paths = f.read().splitlines()

        corner_gt_trans = []
        for p, c in zip(obj_ccs_gt, corner_gt):
            obj_rot = p[:3]
            obj_trans = p[3:]
            corner_gt_trans.append(np.matmul(c, cv2.Rodrigues(obj_rot)[0].T) + obj_trans)

        self.ccs_gt = ccs_gt.copy()
        ccs_gt = HO3D.from_opengl_coord(np.reshape(ccs_gt, (-1, 21, 3)))
        self.uvd_gt = HO3D.ccs2uvd(ccs_gt)
        self.uvd_gt[:, :, 2] = -self.uvd_gt[:, :, 2]

        corner_gt_trans = np.asarray(corner_gt_trans)*1000
        corner_gt_trans = HO3D.from_opengl_coord(np.asarray(corner_gt_trans))
        obj_gt = HO3D.ccs2uvd(corner_gt_trans)
        obj_gt[:, :, 2] = -obj_gt[:, :, 2]

        self.obj_gt = []
        for o in obj_gt:
            obj_extend_gt = []
            for corners in o:
                obj_extend_gt.append(corners)
            obj_pt_pairs = [[0, 4], [1, 5], [3, 7], [2, 6], [0, 1], [4, 5], [2, 3], [6, 7],
                            [1, 3], [5, 7], [4, 6], [0, 2]]
            for p in obj_pt_pairs:
                obj_extend_gt.append(midpoint(o[p[0]], o[p[1]]))
            obj_extend_gt.append(centroid(np.asarray(obj_extend_gt)))
            self.obj_gt.append(obj_extend_gt)

        self.obj_gt = np.asarray(self.obj_gt)

        self.img_root = Path(ROOT)/cfg['img_root']
        self.img_rsz = int(cfg['img_rsz'])
        self.img_ref_width = int(cfg['img_ref_width'])
        self.img_ref_height = int(cfg['img_ref_height'])
        self.ref_depth = int(cfg['ref_depth'])

        if cfg['len'] == 'max':
            self.num_data = len(self.img_paths)
        else:
            self.num_data = int(cfg['len'])

        tfrm = []

        if cfg['aug']:
            if cfg['jitter']:
                jitter = float(cfg['jitter'])
                tfrm.append(ImgPtsTranslate(jitter))
            if cfg['flip']:
                tfrm.append(ImgPtsHFlip())
            if cfg['hsv']:
                hue = float(cfg['hue'])
                sat = float(cfg['sat'])
                val = float(cfg['val'])
                tfrm.append(ImgDistortHSV(hue, sat, val))
            tfrm.append(ImgResize((self.img_rsz)))
            tfrm.append(PtsResize((self.img_ref_width, self.img_ref_height), (self.img_rsz, self.img_rsz)))
            if cfg['rot']:
                rot = float(cfg['rot'])
                tfrm.append(ImgPtsRotate(rot))
            tfrm.append(PtsResize((self.img_rsz, self.img_rsz), (1, 1)))
        else:
            tfrm.append(ImgResize((self.img_rsz)))
            tfrm.append(PtsResize((self.img_ref_width, self.img_ref_height), (1, 1)))

        tfrm.append(ImgToTorch())

        self.transform = torchvision.transforms.Compose(tfrm)

    def __getitem__(self, index):
        img         = cv2.imread(str(self.img_root/self.img_paths[index]))[:, :, ::-1]
        obj_gt      = self.obj_gt[index]
        uvd_gt      = self.uvd_gt[index]
        pts_gt      = np.concatenate((obj_gt, uvd_gt))
        pts_gt[:, 2] /= self.ref_depth
        sample          = {'img': img, 'pts': pts_gt[:, :2].copy()}
        sample          = self.transform(sample)
        img             = sample['img']
        pts_gt[:, :2]   = sample['pts']

        obj_gt = pts_gt[:21]
        uvd_gt = pts_gt[21:]

        return img, uvd_gt.astype('float32'), obj_gt.astype('float32')

    def __len__(self):
        return self.num_data

    def visualize(self, data_load, idx, figsize=(5, 5)):
        import matplotlib.pyplot as plt
        img, uvd_gt, obj_gt = data_load
        img = ImgToNumpy()(img)
        img = img[idx].copy()
        uvd_gt = uvd_gt[idx].numpy().copy()
        uvd_gt[:, 0] *= img.shape[1]
        uvd_gt[:, 1] *= img.shape[0]
        uvd_gt[:, 2] *= self.ref_depth
        obj_gt = obj_gt[idx].numpy().copy()
        obj_gt[:, 0] *= img.shape[1]
        obj_gt[:, 1] *= img.shape[0]
        obj_gt[:, 2] *= self.ref_depth
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        draw_joints(ax, uvd_gt)
        draw_obj_joints(ax, obj_gt)
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        # plt.axis('off')
        draw_obj_3D_joints(ax, obj_gt, c='b')
        draw_3D_joints(ax, uvd_gt, c='b')
        plt.show()

    def visualize_multi(self, data_load, figsize=(15, 15)):
        import matplotlib.pyplot as plt
        img, uvd_gt, obj_gt = data_load
        img = ImgToNumpy()(img)
        fig, ax = plt.subplots(4, 4, figsize=figsize)
        idx = 0
        for i in range(4):
            for j in range(4):
                if idx >= img.shape[0]:
                    break
                cur_img = img[idx].copy()
                cur_uvd_gt = uvd_gt[idx].numpy().copy()
                cur_uvd_gt[:, 0] *= cur_img.shape[1]
                cur_uvd_gt[:, 1] *= cur_img.shape[0]
                cur_obj_gt = obj_gt[idx].numpy().copy()
                cur_obj_gt[:, 0] *= cur_img.shape[1]
                cur_obj_gt[:, 1] *= cur_img.shape[0]
                ax[i, j].imshow(cur_img)
                draw_joints(ax[i, j], cur_uvd_gt)
                draw_obj_joints(ax[i, j], cur_obj_gt)
                idx += 1
        plt.show()

    def get_gt(self):
        return self.img_paths, self.uvd_gt, self.obj_gt, self.ccs_gt