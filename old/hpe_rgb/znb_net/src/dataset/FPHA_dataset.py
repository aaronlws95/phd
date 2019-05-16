import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from skimage.transform import resize
import torchvision.transforms as transforms
import sys
import lmdb
import os
import pickle
import imgaug as ia
from imgaug import augmenters as iaa
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.prepare_data as pd
from utils.directory import DATA_DIR, DATASET_DIR

class FPHA_pose_net_dataset(data.Dataset):
    def __init__(self, save_prefix, aug=False):
        super(FPHA_pose_net_dataset, self).__init__()

        self.uvd_gt_scaled_env = None
        self.save_prefix = save_prefix
        keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
        keys = pickle.load(open(keys_cache_file, "rb"))
        self.keys = keys
        self.aug = aug

    def __init_db(self):
        dataroot_uvd_gt_scaled = os.path.join(DATA_DIR, self.save_prefix + '_uvd_gt_scaled.lmdb')
        self.uvd_gt_scaled_env = lmdb.open(dataroot_uvd_gt_scaled, readonly=True, lock=False, readahead=False, meminit=False)


    def __getitem__(self, index):
        # import time
        # start = time.time()

        if self.uvd_gt_scaled_env is None:
            self.__init_db()

        key = self.keys[index]

        img = np.asarray(Image.open(os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files_256_crop', key)))
        uvd_gt_scaled = pd.read_lmdb(key, self.uvd_gt_scaled_env, np.float32, (21, 3))

        if self.aug:
            skel_kps = []
            for kps in uvd_gt_scaled:
                skel_kps.append(ia.Keypoint(x=kps[0],y=kps[1]))
            skel_kpsoi = ia.KeypointsOnImage(skel_kps, shape=img.shape)

            seq = iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(0, iaa.Add((-18, 18))), #hue
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
                iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
            ])

            seq_det = seq.to_deterministic()
            img_aug = seq_det.augment_images([img])[0]
            kps_aug = seq_det.augment_keypoints([skel_kpsoi])[0]
            kps_aug = kps_aug.get_coords_array()
            uvd_gt_scaled = np.concatenate((kps_aug, np.expand_dims((uvd_gt_scaled[:, 2]), -1)), -1)
            img = img_aug

        img = (img / 255.0) - 0.5
        img = pd.move_channel_dim_2_to_0(img).astype('float32')

        scoremap_gt = pd.create_multiple_gaussian_map(uvd_gt_scaled, (256, 256))
        scoremap_gt = resize(scoremap_gt, (32, 32), order=3, preserve_range=True)
        scoremap_gt = pd.move_channel_dim_2_to_0(scoremap_gt).astype('float32')

        # end = time.time()
        # print(end-start)
        return img, scoremap_gt

    def __len__(self):
        return len(self.keys)

class FPHA_lifting_net_dataset(data.Dataset):
    def __init__(self, save_prefix):
        super(FPHA_lifting_net_dataset, self).__init__()

        self.uvd_gt_scaled_env = None
        self.xyz_gt_canon_env = None
        self.rot_mat_env = None
        self.save_prefix = save_prefix
        keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
        keys = pickle.load(open(keys_cache_file, "rb"))
        self.keys = keys

    def __init_db(self):
        dataroot_uvd_gt_scaled = os.path.join(DATA_DIR, self.save_prefix + '_uvd_gt_scaled.lmdb')
        self.uvd_gt_scaled_env = lmdb.open(dataroot_uvd_gt_scaled, readonly=True, lock=False, readahead=False, meminit=False)

        dataroot_xyz_gt_canon = os.path.join(DATA_DIR, self.save_prefix + '_xyz_gt_canon.lmdb')
        self.xyz_gt_canon_env = lmdb.open(dataroot_xyz_gt_canon, readonly=True, lock=False, readahead=False, meminit=False)

        dataroot_rot_mat_env = os.path.join(DATA_DIR, self.save_prefix + '_rot_mat.lmdb')
        self.rot_mat_env = lmdb.open(dataroot_rot_mat_env, readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.uvd_gt_scaled_env is None or \
        self.xyz_gt_canon_env is None or \
        self.rot_mat_env is None:
            self.__init_db()

        key = self.keys[index]
        rot_mat = pd.read_lmdb(key, self.rot_mat_env, np.float32, (3, 3))

        xyz_gt_canon = pd.read_lmdb(key, self.xyz_gt_canon_env, np.float32, (21, 3))

        uvd_gt_scaled = pd.read_lmdb(key, self.uvd_gt_scaled_env, np.float32, (21, 3))
        scoremap_gt = pd.create_multiple_gaussian_map(uvd_gt_scaled, (256, 256))
        scoremap_gt = pd.move_channel_dim_2_to_0(scoremap_gt).astype('float32')

        return scoremap_gt, xyz_gt_canon, rot_mat

    def __len__(self):
        return len(self.keys)

class FPHA_pred_smap_dataset(data.Dataset):
    def __init__(self, exp, epoch, data_split):
        super(FPHA_pred_smap_dataset, self).__init__()

        f= h5py.File(os.path.join(DATA_DIR, exp, 'scoremap_%s_%s.h5' %(epoch, data_split)))
        self.scoremap = f['scoremap'][...]
        f.close()

    def __getitem__(self, index):

        smap = self.scoremap[index]
        smap = pd.sk_resize(smap, (256, 256))
        smap = np.reshape(smap, (smap.shape[2], smap.shape[0], smap.shape[1])).astype(np.float32)

        return smap

    def __len__(self):
        return len(self.scoremap)
