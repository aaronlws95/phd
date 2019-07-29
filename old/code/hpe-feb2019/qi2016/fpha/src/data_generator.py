import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import keras
from PIL import Image

import constants
import utils
import data_augmentation as aug
from prepare_data import prepare_data, read_prepare_data_h5py

class DataGenerator_dtip(keras.utils.Sequence):
    def __init__(self, data_file_h5py, jnt_idx, aug=False, shuffle=False):
        # jnt_idx = [7,8] [10,11] [13,14] [16,17] [19. 20]
        # ([[0,1,2,3,4]*3 + 7]
        self.jnt_idx = [jnt_idx*3 + 7, jnt_idx*3 + 8]
        self.prev_jnt_idx = jnt_idx*3 + 6
        self.current_idx = 0
        self.aug = aug
        self.batch_size = constants.BATCH_SIZE
        self.shuffle = shuffle
        img0, _, _, uvd_norm_gt, _, _, _, _, = read_prepare_data_h5py(data_file_h5py)

        self.img0 = np.expand_dims(img0, -1)
        self.uvd_norm_cur_jnt_gt = uvd_norm_gt[:, jnt_idx, :]
        self.uvd_norm_prev_jnt_gt = uvd_norm_gt[:, prev_jnt_idx, :]
        self.uvd_norm_palm_gt = uvd_norm_gt[:, :6, :]

        self.on_epoch_end()

    def __len__(self):
        #number of batches per epoch
        return len(self.uvd_norm_gt)//self.batch_size

    def __getitem__(self, index):
        #generate one batch of data
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        x, y = self.__generate(indices)
        return x, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.uvd_norm_cur_jnt_gt))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __generate(self, indices):
        #generate data
        img0 = self.img0[indices]
        uvd_norm_cur_jnt_gt = self.uvd_norm_cur_jnt_gt[indices]
        uvd_norm_prev_jnt_gt = self.uvd_norm_prev_jnt_gt[indices]
        uvd_norm_palm_gt = self.uvd_norm_palm_gt[indices]

        crop0, crop1, offset_uvd_gt = aug.get_crop_for_finger_part_s0(img0, uvd_norm_cur_jnt_gt, uvd_norm_palm_gt, uvd_norm_prev_jnt_gt, if_aug=self.aug, aug_trans=0.03, aug_rot=15)

        return [crop0, crop1], offset_uvd_gt

class DataGenerator_pip(keras.utils.Sequence):
    def __init__(self, data_file_h5py, jnt_idx, aug=False, shuffle=False):
        # jnt_idx = [6,9,12,15,18] choose one
        # ([0,1,2,3,4] + 2)*3
        self.jnt_idx = jnt_idx*3 + 6
        self.prev_jnt_idx = jnt_idx + 1
        self.current_idx = 0
        self.aug = aug
        self.batch_size = constants.BATCH_SIZE
        self.shuffle = shuffle
        img0, _, _, uvd_norm_gt, _, _, _, _, = read_prepare_data_h5py(data_file_h5py)

        self.img0 = np.expand_dims(img0, -1)
        self.uvd_norm_cur_jnt_gt = uvd_norm_gt[:, jnt_idx, :]
        self.uvd_norm_prev_jnt_gt = uvd_norm_gt[:, prev_jnt_idx, :]
        self.uvd_norm_palm_gt = uvd_norm_gt[:, :6, :]

        self.on_epoch_end()

    def __len__(self):
        #number of batches per epoch
        return len(self.uvd_norm_gt)//self.batch_size

    def __getitem__(self, index):
        #generate one batch of data
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        x, y = self.__generate(indices)
        return x, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.uvd_norm_cur_jnt_gt))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __generate(self, indices):
        #generate data
        img0 = self.img0[indices]
        uvd_norm_cur_jnt_gt = self.uvd_norm_cur_jnt_gt[indices]
        uvd_norm_prev_jnt_gt = self.uvd_norm_prev_jnt_gt[indices]
        uvd_norm_palm_gt = self.uvd_norm_palm_gt[indices]

        crop0, crop1, offset_uvd_gt = aug.get_crop_for_finger_part_s0(img0, uvd_norm_cur_jnt_gt, uvd_norm_palm_gt, uvd_norm_prev_jnt_gt, if_aug=self.aug, aug_trans=0.02, aug_rot=15)

        return [crop0, crop1], offset_uvd_gt

class DataGenerator_palm(keras.utils.Sequence):
    def __init__(self, data_file_h5py, aug=False, shuffle=False):
        self.current_idx = 0
        self.aug = aug
        self.batch_size = constants.BATCH_SIZE
        self.shuffle = shuffle
        img0, img1, img2, uvd_norm_gt, _, _, _, _, = read_prepare_data_h5py(data_file_h5py)

        self.img0 = np.expand_dims(img0, -1)
        self.img1 = np.expand_dims(img1, -1)
        self.img2 = np.expand_dims(img2, -1)
        self.uvd_norm_gt = np.reshape(uvd_norm_gt[:, :6, :], (-1, 18))

        self.on_epoch_end()

    def __len__(self):
        #number of batches per epoch
        return len(self.uvd_norm_gt)//self.batch_size

    def __getitem__(self, index):
        #generate one batch of data
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        x, y = self.__generate(indices)
        return x, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.uvd_norm_gt))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __generate(self, indices):
        #generate data
        img0 = self.img0[indices]
        img1 = self.img1[indices]
        img2 = self.img2[indices]
        uvd_norm_gt = self.uvd_norm_gt[indices]

        if self.aug:
            img0, img1, img2, uvd_norm_gt = aug.augment_data_3d_rot_scale(img0, img1, img2, uvd_norm_gt)

        return [img0, img1, img2], uvd_norm_gt

class DataGenerator_h5py(keras.utils.Sequence):
    def __init__(self, data_file_h5py, shuffle=False):
        self.current_idx = 0
        self.batch_size = constants.BATCH_SIZE
        self.shuffle = shuffle
        img0, img1, img2, uvd_norm_gt, _, _, _, _, = read_prepare_data_h5py(data_file_h5py)

        self.img0 = np.expand_dims(img0, -1)
        self.img1 = np.expand_dims(img1, -1)
        self.img2 = np.expand_dims(img2, -1)
        self.uvd_norm_gt = np.reshape(uvd_norm_gt, (-1, 63))

        self.on_epoch_end()

    def __len__(self):
        #number of batches per epoch
        return len(self.uvd_norm_gt)//self.batch_size

    def __getitem__(self, index):
        #generate one batch of data
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        x, y = self.__generate(indices)
        return x, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.uvd_norm_gt))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __generate(self, indices):
        #generate data
        img0 = self.img0[indices]
        img1 = self.img1[indices]
        img2 = self.img2[indices]
        uvd_norm_gt = self.uvd_norm_gt[indices]

        return [img0, img1, img2], uvd_norm_gt

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_list, shuffle=False):
        self.current_idx = 0
        self.batch_size = constants.BATCH_SIZE
        self.data_list = data_list
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        #number of batches per epoch
        return len(self.data_list)//self.batch_size

    def __getitem__(self, index):
        #generate one batch of data
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        x, y = self.__generate(indices)
        return x, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __generate(self, indices):
        #generate data
        file_name =[self.data_list[i][0] for i in indices]
        xyz_gt = np.asarray([self.data_list[i][1] for i in indices])

        img0, img1, img2, uvd_norm_gt, _, _, _, _ = prepare_data(file_name, xyz_gt)

        img0 = np.expand_dims(img0, -1)
        img1 = np.expand_dims(img1, -1)
        img2 = np.expand_dims(img2, -1)

        uvd_norm_gt = np.reshape(uvd_norm_gt, (-1, 63))

        return [img0, img1, img2], uvd_norm_gt
