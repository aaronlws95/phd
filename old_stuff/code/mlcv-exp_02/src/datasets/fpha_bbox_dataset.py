import torch
import torchvision
import cv2 
import numpy as np
from pathlib import Path

from src import ROOT
from src.datasets.transforms import *
from src.utils import *

class FPHA_Bbox_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        split_set = Path(ROOT)/cfg['{}_set'.format(split)]
        bbox = np.loadtxt(split_set)
        
        split_set = Path(ROOT)/(cfg['{}_set'.format(split)]).replace('bbox', 'img')
        with open(split_set, 'r') as f:
            img_paths = f.read().splitlines()
        
        self.img_paths = []
        self.bbox_gt = []
        if cfg['invalid_set']:
            with open(Path(ROOT)/cfg['invalid_set'], 'r') as f:
                invalid_paths = f.readlines()
            invalid_paths = [i.rstrip() for i in invalid_paths]
        else:
            invalid_paths = []

        for i in range(len(img_paths)):
            path = img_paths[i]
            if path not in invalid_paths:
                self.img_paths.append(path)
                self.bbox_gt.append(bbox[i])
            
        self.img_root = Path(ROOT)/cfg['img_root']
        self.img_rsz = int(cfg['img_rsz'])
        
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
        tfrm.append(ImgToTorch())
        self.transform = torchvision.transforms.Compose(tfrm)
        
    def __getitem__(self, index):
        img         = cv2.imread(str(self.img_root/self.img_paths[index]))[:, :, ::-1]
        H, W        = img.shape[:2]
        bbox        = self.bbox_gt[index]
        img_ori     = img.copy()
        bbox_ori    = bbox.copy()
        bbox[0]     *= W
        bbox[1]     *= H
        bbox[2]     *= W
        bbox[3]     *= H
        bbox        = xywh2xyxy(bbox).reshape(-1, 2)
        sample      = {'img': img, 'pts': bbox}
        sample      = self.transform(sample)
        img         = sample['img']
        bbox        = sample['pts'].reshape(-1)
        bbox[0]     = max(0, min(bbox[0], W))
        bbox[1]     = max(0, min(bbox[1], H))
        bbox[2]     = max(0, min(bbox[2], W))
        bbox[3]     = max(0, min(bbox[3], H))
        new_bbox    = np.zeros(len(bbox))
        new_bbox[0] = min(bbox[0], bbox[2])
        new_bbox[1] = min(bbox[1], bbox[3])
        new_bbox[2] = max(bbox[0], bbox[2])
        new_bbox[3] = max(bbox[1], bbox[3])
        bbox        = xyxy2xywh(new_bbox)
        bbox[0]     /= W
        bbox[1]     /= H
        bbox[2]     /= W
        bbox[3]     /= H
        
        if bbox[2] < 0.001 or bbox[3] < 0.001:
            img = img_ori
            sample = {'img': img}
            sample = ImgResize((self.img_rsz))(sample)
            sample = ImgToTorch()(sample)
            img = sample['img']
            bbox = bbox_ori
        
        return img, bbox.astype('float32')

    def __len__(self):
        return self.num_data
    
    def visualize(self, data_load, idx, figsize=(5, 5)):
        import matplotlib.pyplot as plt
        img, bbox = data_load
        img = ImgToNumpy()(img)
        img = img[idx].copy()
        bbox = bbox[idx].numpy().copy()
        bbox[0] *= img.shape[1]
        bbox[1] *= img.shape[0]
        bbox[2] *= img.shape[1]
        bbox[3] *= img.shape[0]
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        draw_bbox(ax, bbox)
        plt.show()
        
    def visualize_multi(self, data_load, figsize=(15, 15)):
        import matplotlib.pyplot as plt
        img, bbox = data_load
        img = ImgToNumpy()(img)
        fig, ax = plt.subplots(4, 4, figsize=figsize)
        idx = 0
        for i in range(4):
            for j in range(4):
                if idx >= img.shape[0]:
                    break
                cur_img = img[idx].copy()
                cur_bbox = bbox[idx].numpy().copy()
                cur_bbox[0] *= cur_img.shape[1]
                cur_bbox[1] *= cur_img.shape[0]
                cur_bbox[2] *= cur_img.shape[1]
                cur_bbox[3] *= cur_img.shape[0]
                ax[i, j].imshow(cur_img)
                draw_bbox(ax[i, j], cur_bbox)
                idx += 1
        plt.show()

    def get_gt(self, split_set):
        return self.img_paths, self.bbox_gt