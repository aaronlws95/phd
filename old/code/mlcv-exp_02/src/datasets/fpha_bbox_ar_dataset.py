import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src import ROOT
from src.datasets.transforms import *
from src.utils import *

class FPHA_Bbox_AR_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        bbox_set = cfg['bbox_set']
        split_set = Path(ROOT)/cfg['{}_set'.format(split)].replace('ar_seq', bbox_set)
        bbox = np.loadtxt(split_set)

        split_set = Path(ROOT)/(cfg['{}_set'.format(split)])
        with open(split_set, 'r') as f:
            img_labels = f.read().splitlines()
        seq_paths   = [i.split(' ')[0] for i in img_labels]
        path_length = [int(i.split(' ')[1]) for i in img_labels]
        action_cls  = [int(i.split(' ')[2]) for i in img_labels]
        obj_cls     = [int(i.split(' ')[3]) for i in img_labels]

        self.img_paths  = []
        self.bbox_gt    = []
        self.obj_cls    = []
        self.action_cls = []
        self.img_root   = Path(ROOT)/cfg['img_root']
        self.img_rsz    = int(cfg['img_rsz'])

        if cfg['invalid_set']:
            with open(Path(ROOT)/cfg['invalid_set'], 'r') as f:
                invalid_paths = f.readlines()
            invalid_paths = [i.rstrip() for i in invalid_paths]
        else:
            invalid_paths = []

        idx = 0
        for i in tqdm(range(len(seq_paths)), desc='{} dataset'.format(split)):
            for j in range(path_length[i]):
                path = self.img_root/Path(seq_paths[i])/cfg['img_tmpl'].format(j)
                if str(Path(seq_paths[i])/cfg['img_tmpl'].format(j)) not in invalid_paths:
                    self.action_cls.append(action_cls[i])
                    self.obj_cls.append(obj_cls[i])
                    self.img_paths.append(path)
                    self.bbox_gt.append(bbox[idx])
                idx += 1

        # labels = [(p,b,a,o) for p,b,a,o in zip(self.img_paths, self.bbox_gt, self.action_cls, self.obj_cls) if p not in invalid_paths]
        # labels = list(zip(*labels))
        # self.img_paths, self.bbox_gt, self.action_cls, self.obj_cls = labels

        if cfg['len'] == 'max':
            self.num_data = len(self.img_paths)
        else:
            self.num_data = int(cfg['len'])

        tfrm = []
        if cfg['aug']:
            if cfg['jitter']:
                jitter = float(cfg['jitter'])
                tfrm.append(ImgPtsTranslate(jitter))
            if cfg['scale']:
                scale = float(cfg['scale'])
                tfrm.append(ImgPtsScale(scale))
            if cfg['flip']:
                tfrm.append(ImgPtsHFlip())
            # if cfg['hsv']:
            #     hue = float(cfg['hue'])
            #     sat = float(cfg['sat'])
            #     val = float(cfg['val'])
            #     tfrm.append(ImgDistortHSV(hue, sat, val))
            if cfg['rot']:
                rot = float(cfg['rot'])
                tfrm.append(ImgBboxRotate(rot))

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

        action_cls  = self.action_cls[index]
        obj_cls     = self.obj_cls[index]

        return img, bbox.astype('float32'), action_cls, obj_cls, str(self.img_paths[index])

    def __len__(self):
        return self.num_data

    def visualize(self, data_load, idx, figsize=(5, 5)):
        import matplotlib.pyplot as plt
        img, bbox, action, obj, path = data_load
        print(path[idx])
        img = ImgToNumpy()(img)
        img = img[idx].copy()
        action = action[idx].item()
        obj = obj[idx].item()
        action_dict = FPHA.get_action_dict()
        obj_dict = FPHA.get_obj_dict()
        print('Action:', str(action_dict[action]) + ' ' + str(obj_dict[obj]))
        bbox = bbox[idx].numpy().copy()
        print(bbox)
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
        img, bbox, _, _, _ = data_load
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
        return self.img_paths, self.bbox_gt, self.action_cls, self.obj_cls