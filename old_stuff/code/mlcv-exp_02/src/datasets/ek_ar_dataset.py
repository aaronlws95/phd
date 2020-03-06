import torch
import torchvision
import cv2 
import numpy as np
from pathlib import Path

from src import ROOT
from src.datasets.transforms import *
from src.utils import *

class EK_AR_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        split_set = Path(ROOT)/cfg['{}_set'.format(split)]
        
        with open(split_set, 'r') as f:
            img_labels = f.read().splitlines()
        img_paths   = [i.split(' ')[0] for i in img_labels]
        path_length = [int(i.split(' ')[1]) for i in img_labels]
        action_cls  = [int(i.split(' ')[2]) for i in img_labels]
        obj_cls     = [int(i.split(' ')[3]) for i in img_labels]
        
        self.img_root = Path(ROOT)/cfg['img_root']
        
        self.obj_cls    = []
        self.action_cls = []
        self.img_paths  = []
        for p, l, v, n in zip(img_paths, path_length, action_cls, obj_cls):
            for i in range(l):
                self.img_paths.append(self.img_root/Path(p)/cfg['img_tmpl'].format(i))
                self.action_cls.append(v)
                self.obj_cls.append(n)

        self.img_rsz = int(cfg['img_rsz'])
        
        if cfg['len'] == 'max':
            self.num_data = len(self.img_paths)
        else:
            self.num_data = int(cfg['len'])

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
        sample      = {'img': img}
        sample      = self.transform(sample)
        img         = sample['img']
        action_cls  = self.action_cls[index]
        obj_cls     = self.obj_cls[index]
        
        return img, action_cls, obj_cls

    def __len__(self):
        return self.num_data
    
    def visualize(self, data_load, idx, figsize=(5, 5)):
        import matplotlib.pyplot as plt
        img, action, obj = data_load
        img = ImgToNumpy()(img)
        img = img[idx].copy()
        action = action[idx].item()
        obj = obj[idx].item()
        action_dict = EK.get_verb_dict()
        obj_dict = EK.get_noun_dict()
        print('Action:', str(action_dict[action]) + ' ' + str(obj_dict[obj]))
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        plt.show()
        
    def visualize_multi(self, data_load, figsize=(15, 15)):
        import matplotlib.pyplot as plt
        img, _, _ = data_load
        img = ImgToNumpy()(img)
        fig, ax = plt.subplots(4, 4, figsize=figsize)
        idx = 0
        for i in range(4):
            for j in range(4):
                if idx >= img.shape[0]:
                    break
                cur_img = img[idx].copy()
                ax[i, j].imshow(cur_img)
                idx += 1
        plt.show()

    def get_gt(self, split_set):
        return self.img_paths, self.action_cls, self.obj_cls