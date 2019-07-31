import torch
import torchvision
import cv2 
import numpy as np
from pathlib import Path

from src import ROOT
from src.datasets.transforms import *
from src.utils import *

class FPHA_Pose_AR_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        split_set = Path(ROOT)/cfg['{}_set'.format(split)]
        xyz_gt = np.loadtxt(split_set)
        uvd_gt = FPHA.xyz2uvd_color(np.reshape(xyz_gt, (-1, 21, 3)))
        
        split_set = Path(ROOT)/(cfg['{}_set'.format(split)]).replace('xyz', 'ar_seq')
        with open(split_set) as f:
            img_list = f.read().splitlines()
        path_length = [int(i.split(' ')[1]) for i in img_list]
        action_cls  = [int(i.split(' ')[2]) for i in img_list]
        
        self.rand_frame = cfg['rand_frame']
        
        self.num_segments = int(cfg['num_segments'])
        track = 0
        self.seq_uvd_gt = []
        self.action_cls = []
        for num_frames, a_cls in zip(path_length, action_cls):
            if num_frames < self.num_segments:
                continue
            self.seq_uvd_gt.append(uvd_gt[track:track+num_frames])
            self.action_cls.append(a_cls)
            track += num_frames
        
        if cfg['len'] == 'max':
            self.num_data = len(self.action_cls)
        else:
            self.num_data = int(cfg['len'])
        
        tfrm = []
        tfrm.append(ImgToTorch())
        self.transform = torchvision.transforms.Compose(tfrm)
        
    def __getitem__(self, index):
        uvd = self.seq_uvd_gt[index].copy()
        action_id = self.action_cls[index]
        
        uvd[..., 0] = (uvd[..., 0] - np.mean(uvd[..., 0]))/FPHA.ORI_WIDTH
        uvd[..., 1] = (uvd[..., 1] - np.mean(uvd[..., 1]))/FPHA.ORI_HEIGHT
        uvd[..., 2] = (uvd[..., 2] - np.mean(uvd[..., 2]))/750
        
        num_frames = len(uvd)
        avg_duration = num_frames//self.num_segments
        if self.rand_frame:
            offset = np.random.randint(avg_duration, size=self.num_segments)
        else:
            offset = 0
        frames = list(np.multiply(list(range(self.num_segments)), avg_duration) + offset)
        uvd_first = uvd[0].reshape(-1)
        uvd_out = np.stack([uvd[i] for i in frames])
        
        sample      = {'img': uvd_out}
        sample      = self.transform(sample)
        uvd_out        = sample['img']
        return uvd_out, action_id
    
    def __len__(self):
            return self.num_data
        
    def visualize(self, data_load, idx):
        import matplotlib.pyplot as plt
        uvd_gt, action_id = data_load
        uvd_gt = uvd_gt.permute(0, 2, 3, 1).numpy()
        uvd_gt = uvd_gt[idx].copy()
        for i in range(len(uvd_gt)):
            fig, ax = plt.subplots()
            draw_joints(ax, uvd_gt[i])
            plt.show()

    def visualize_multi(self, data_load):
        uvd_gt, action_id = data_load
        uvd_gt = uvd_gt.permute(0, 2, 3, 1).numpy()

        fig, ax = plt.subplots(5, 5)
        idx = 0
        for i in range(5):
            if i >= uvd_gt.shape[0]:
                    break
            for j in range(5):
                draw_joints(ax[i, j], uvd_gt[i, j])
        plt.show()

    def get_gt(self, split_set):
        return self.seq_uvd_gt, self.action_cls