import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from skimage.transform import resize
import os
import sys
import lmdb
import os
import pickle
import imgaug as ia
from imgaug import augmenters as iaa

from utils.directory import DATA_DIR, DATASET_DIR
import utils.prepare_data as pd
import utils.xyzuvd as xyzuvd

class FPHA(data.Dataset):
    def __init__(self, save_prefix, aug=False):
        super(FPHA, self).__init__()

        self.uvd_gt_env = None
        self.aug = aug
        self.save_prefix = save_prefix
        keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
        keys = pickle.load(open(keys_cache_file, "rb"))
        self.keys = keys

    def __init_db(self):
        dataroot_uvd_gt = os.path.join(DATA_DIR, self.save_prefix + '_uvd_gt_resize.lmdb')
        self.uvd_gt_env = lmdb.open(dataroot_uvd_gt, readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        # import time
        # start = time.time()

        if self.uvd_gt_env is None:
            self.__init_db()

        key = self.keys[index]

        img = np.asarray(Image.open(os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files_416', key)))
        uvd_gt = pd.read_lmdb(key, self.uvd_gt_env, np.float32, (21, 3))

        if self.aug:
            skel_kps = []
            for kps in uvd_gt:
                skel_kps.append(ia.Keypoint(x=kps[0],y=kps[1]))
            skel_kpsoi = ia.KeypointsOnImage(skel_kps, shape=img.shape)

            seq = iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(0, iaa.Add((0, 90))), #hue
                iaa.WithChannels(1, iaa.Add((0, 128))), #saturation
                iaa.WithChannels(2, iaa.Add((0, 128))), #exposure
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
                iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
            ])

            seq_det = seq.to_deterministic()
            img_aug = seq_det.augment_images([img])[0]
            kps_aug = seq_det.augment_keypoints([skel_kpsoi])[0]
            kps_aug = kps_aug.get_coords_array()
            uvd_gt = np.concatenate((kps_aug, np.expand_dims((uvd_gt[:, 2]), -1)), -1)
            img = img_aug

        img = (img / 255.0) - 0.5
        # img = (img - np.mean(img))/np.std(img)
        img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1])).astype('float32')

        uvd_gt_no_offset = uvd_gt.copy()
        uvd_gt_no_offset[:, 0] = (uvd_gt[:, 0] - (uvd_gt[0, 0]//32)*32) / 416
        uvd_gt_no_offset[:, 1] = (uvd_gt[:, 1] - (uvd_gt[0, 1]//32)*32) / 416
        uvd_gt_no_offset[:, 2] = (uvd_gt[:, 2] - (uvd_gt[0, 2]//200)*200) / 1000

        hand_cell_idx = pd.get_hand_cell_idx(uvd_gt)
        # end = time.time()
        # print(end-start)
        return img, uvd_gt_no_offset, uvd_gt, hand_cell_idx

    def __len__(self):
        return len(self.keys)
