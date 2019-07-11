import torch
import torchvision
import cv2 
import numpy as np
from pathlib import Path

from src import ROOT
from src.datasets.transforms import *

class _Labels(object):
    def __init__(self, label_info, num_segments, rand_frame):
        super().__init__()
        l = label_info.split(' ')
        self.dir    = Path(l[0]) 
        self.action = int(l[2])
        self.object = int(l[3])
        num_frames  = int(l[1])
        avg_duration = num_frames//num_segments
        if rand_frame:
            offset = np.random.randint(avg_duration, size=num_segments)
        else:
            offset = 0
        self.frames = np.multiply(list(range(num_segments)), avg_duration) \
                    + offset

class FPHA_AR_Seq_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        split_set = Path(ROOT)/cfg['{}_set'.format(split)]
        with open(split_set, 'r') as f:
            labels = f.read().splitlines()
            
        self.num_segments = int(cfg['num_segments'])
        self.labels = []
        for l in labels:
            self.labels.append(_Labels(l, self.num_segments, cfg['rand_frame']))

        self.img_root = Path(ROOT)/cfg['img_root']
        self.img_tmpl = cfg['img_tmpl']
        self.img_rsz = int(cfg['img_rsz'])
        
        if cfg['len'] == 'max':
            self.num_data = len(self.labels)
        else:
            self.num_data = int(cfg['len'])

        tfrm = []
        if cfg['aug']:
            if cfg['jitter']:
                jitter = float(cfg['jitter'])
                tfrm.append(GroupImgTranslate(jitter))
            if cfg['flip']:
                tfrm.append(GroupImgHFlip())
            if cfg['hsv']:
                hue = float(cfg['hue'])
                sat = float(cfg['sat'])
                val = float(cfg['val'])
                tfrm.append(GroupImgDistortHSV(hue, sat, val))
        
        tfrm.append(GroupImgResize((self.img_rsz)))
        tfrm.append(GroupImgToTorch())
        self.transform = torchvision.transforms.Compose(tfrm)

    def __getitem__(self, index):
        # out: (B, Seg, C, H, W), B, B
        label       = self.labels[index]
        img_paths   = [str(label.dir/self.img_tmpl.format(i)) 
                       for i in label.frames]
        img_list    = [cv2.imread(str(self.img_root/p))[:, :, ::-1] for p in img_paths]
        sample      = {'img': img_list}
        img_list    = self.transform(sample)['img']
        img_list    = torch.stack(img_list)
        return img_list, label.action, label.object, img_paths

    def __len__(self):
        return self.num_data
    
    def visualize(self, data_load, idx, figsize=(20, 20)):
        import matplotlib.pyplot as plt
        img_list, action_id, object_id, _ = data_load
        img_list = np.asarray(GroupImgToNumpy()(img_list))
        fig, ax = plt.subplots(1, self.num_segments, figsize=figsize)
        for i in range(len(img_list[idx])):
            ax[i].imshow(img_list[idx][i])
        plt.show()

    def visualize_multi(self, data_load, figsize=(20, 20)):
        import matplotlib.pyplot as plt
        img_list, action_id, object_id, _ = data_load
        img_list = np.asarray(GroupImgToNumpy()(img_list))
        fig, ax = plt.subplots(5, self.num_segments, figsize=figsize)
        for i in range(5):
            if i >= len(img_list):
                break
            for j in range(len(img_list[i])):
    
                ax[i, j].imshow(img_list[i][j])
        plt.show()

    def get_gt(self, split_set):
        split_set = Path(ROOT)/split_set
        with open(split_set, 'r') as f:
            labels = f.read().splitlines()
        action_gt = [int(i.split(' ')[2]) for i in labels]
        obj_gt = [int(i.split(' ')[3]) for i in labels]
        return action_gt, obj_gt