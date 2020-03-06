import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path

from src import ROOT
from src.datasets.transforms import *
from src.utils import *

class FPHA_Hand_Crop_Class_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        split_set = Path(ROOT)/cfg['{}_set'.format(split)].replace('img', 'xyz')
        xyz = np.loadtxt(split_set)

        split_set = Path(ROOT)/(cfg['{}_set'.format(split)])
        with open(split_set, 'r') as f:
            img_paths = f.read().splitlines()

        self.img_paths = []
        xyz_gt = []

        for i in range(len(img_paths)):
            path = img_paths[i]
            self.img_paths.append(path)
            xyz_gt.append(xyz[i])

        self.uvd_gt = FPHA.xyz2uvd_color(np.reshape(xyz_gt, (-1, 21, 3)))

        self.uvd_gt[..., 0] /= int(cfg['img_ref_width'])
        self.uvd_gt[..., 1] /= int(cfg['img_ref_height'])

        self.img_root = Path(ROOT)/cfg['img_root']
        self.img_rsz = int(cfg['img_rsz'])

        if cfg['len'] == 'max':
            self.num_data = len(self.img_paths)
        else:
            self.num_data = int(cfg['len'])

        self.iou_thresh = float(cfg['iou_thresh'])
        self.pad = int(cfg['pad'])
        self.percent_true = float(cfg['percent_true'])
        self.crop_jitter = float(cfg['crop_jitter'])
        self.is_iou = cfg['is_iou']
        tfrm = []

        if cfg['aug']:
            if cfg['jitter']:
                jitter = float(cfg['jitter'])
                tfrm.append(ImgTranslate(jitter))
            if cfg['flip']:
                tfrm.append(ImgHFlip())
            if cfg['hsv']:
                hue = float(cfg['hue'])
                sat = float(cfg['sat'])
                val = float(cfg['val'])
                tfrm.append(ImgDistortHSV(hue, sat, val))
            if cfg['zoom']:
                zoom_factor = int(cfg['zoom'])
                tfrm.append(ImgZoom(zoom_factor))
            tfrm.append(ImgResize((self.img_rsz)))
            if cfg['rot']:
                rot = float(cfg['rot'])
                tfrm.append(ImgRotate(rot))
        else:
            tfrm.append(ImgResize((self.img_rsz)))

        tfrm.append(ImgToTorch())

        self.transform = torchvision.transforms.Compose(tfrm)

    def __getitem__(self, index):
        img         = cv2.imread(str(self.img_root/self.img_paths[index]))[:, :, ::-1]
        uvd_gt      = self.uvd_gt[index].copy()
        uvd_gt[..., 0] *= img.shape[1]
        uvd_gt[..., 1] *= img.shape[0]

        img_crop, is_hand, iou = get_img_crop(uvd_gt, img, self.iou_thresh, pad=self.pad, percent_true=self.percent_true, jitter=self.crop_jitter)
        sample          = {'img': img_crop}
        sample          = self.transform(sample)
        img_crop        = sample['img']

        if self.is_iou:
            return img_crop, iou
        else:
            return img_crop, is_hand

    def __len__(self):
        return self.num_data

    def visualize(self, data_load, idx, figsize=(5, 5)):
        import matplotlib.pyplot as plt
        img, is_hand = data_load
        img = ImgToNumpy()(img)
        img = img[idx].copy()
        is_hand = is_hand[idx]
        print(is_hand.item())
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        plt.show()

    def visualize_multi(self, data_load, figsize=(15, 15)):
        import matplotlib.pyplot as plt
        img, is_hand = data_load
        img = ImgToNumpy()(img)
        fig, ax = plt.subplots(4, 4, figsize=figsize)
        idx = 0
        for i in range(4):
            for j in range(4):
                if idx >= img.shape[0]:
                    break
                cur_img = img[idx].copy()
                cur_is_hand = is_hand[idx]
                ax[i, j].imshow(cur_img)
                ax[i, j].text(0, 0, '{:04f}'.format(cur_is_hand.item()), c='r', fontsize=20)
                idx += 1
        plt.show()

    def get_gt(self):
        return self.img_paths