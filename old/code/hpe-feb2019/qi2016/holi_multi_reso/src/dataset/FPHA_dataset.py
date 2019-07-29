import numpy as np
import torch
import torch.utils.data as data

class FPHA_dataset(data.Dataset):
    def __init__(self, img0, img1, img2, uvd_norm_gt):
        super(FPHA_dataset, self).__init__()
        self.img0 = img0
        self.img1 = img1
        self.img2 = img2
        self.uvd_norm_gt = uvd_norm_gt

    def __getitem__(self, index):
        img0 = self.img0[index]
        img1 = self.img1[index]
        img2 = self.img2[index]
        uvd_norm_gt = self.uvd_norm_gt[index]
        return [img0, img1, img2], uvd_norm_gt

    def __len__(self):
        return len(self.uvd_norm_gt)
