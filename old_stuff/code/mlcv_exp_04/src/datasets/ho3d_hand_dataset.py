import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path

from src import ROOT
from src.datasets.transforms import *
from src.utils import *

class HO3D_Hand_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        split_set = Path(ROOT)/cfg['{}_set'.format(split)]
        self.ccs_gt = np.loadtxt(split_set)

        split_set = Path(ROOT)/(cfg['{}_set'.format(split)]).replace('ccs', 'img')
        with open(split_set, 'r') as f:
            self.img_paths = f.read().splitlines()

        self.uvd_gt = HO3D.from_opengl_coord(np.reshape(self.ccs_gt, (-1, 21, 3)))
        self.uvd_gt = HO3D.ccs2uvd(self.uvd_gt)
        self.uvd_gt[:, :, 2] = -self.uvd_gt[:, :, 2]

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
        uvd_gt      = self.uvd_gt[index]
        uvd_gt[:, 2] /= self.ref_depth

        sample          = {'img': img, 'pts': uvd_gt[:, :2].copy()}
        sample          = self.transform(sample)
        img             = sample['img']
        uvd_gt[:, :2]   = sample['pts']

        return img, uvd_gt.astype('float32')

    def __len__(self):
        return self.num_data

    def visualize(self, data_load, idx, figsize=(5, 5)):
        import matplotlib.pyplot as plt
        img, uvd_gt = data_load
        img = ImgToNumpy()(img)
        img = img[idx].copy()
        uvd_gt = uvd_gt[idx].numpy().copy()
        uvd_gt[:, 0] *= img.shape[1]
        uvd_gt[:, 1] *= img.shape[0]
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        draw_joints(ax, uvd_gt)
        plt.show()

    def visualize_multi(self, data_load, figsize=(15, 15)):
        import matplotlib.pyplot as plt
        img, uvd_gt = data_load
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
                ax[i, j].imshow(cur_img)
                draw_joints(ax[i, j], cur_uvd_gt)
                idx += 1
        plt.show()

    def get_gt(self):
        return self.img_paths, self.uvd_gt, self.ccs_gt