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
        cur_batch_list = [self.data_list[k] for k in indices]
        x, y = self.__generate(cur_batch_list)
        return x, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __generate(self, cur_batch_list):
        #generate data

        img = []
        skel_kpsoi = []
        for cur_batch in cur_batch_list:
            img.append(np.asarray(Image.open(cur_batch[0])))
            skel_xyz = np.reshape(cur_batch[1], (21, 3))
            skel_uvd = utils.xyz2uvd(skel_xyz)

            skel_kps = []
            for kps in skel_uvd:
                skel_kps.append(ia.Keypoint(x=kps[0],y=kps[1]))
            skel_kpsoi.append(ia.KeypointsOnImage(skel_kps, shape=img[0].shape))

        seq = iaa.Sequential([
            iaa.size.Resize({"height": constants.RSZ_HEIGHT, "width": constants.RSZ_WIDTH}),
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((0, 30))), #hue
            iaa.WithChannels(1, iaa.Add((0, 30))), #saturation
            iaa.WithChannels(2, iaa.Add((0, 60))), #exposure
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
        ])

        batch = ia.Batch(images=img, keypoints=skel_kpsoi)
        batch_aug = list(seq.augment_batches([batch]))[0]
        x = np.asarray(batch_aug.images_aug)
        skel_uvd_aug = batch_aug.keypoints_aug

        y = []
        for skel in skel_uvd_aug:
            skel_uvd_proj = []
            for kps in skel.keypoints:
                skel_uvd_proj.append([kps.x_int, kps.y_int])
            skel_uvd_proj = np.asarray(skel_uvd_proj)
            skel_uvd_rsz = np.concatenate([skel_uvd_proj, np.reshape(skel_uvd[:, 2], (21, 1))], 1)
            skel_gt = skel_uvd_rsz
            # skel_gt = utils.uvd2camcoord(skel_uvd_rsz)

            y.append(np.asarray(np.reshape(skel_gt, (-1))))

        y = np.broadcast_to(np.reshape(np.asarray(y), (self.batch_size, 1, -1)), (self.batch_size, 845, 63))
        y = np.concatenate([y, np.zeros((self.batch_size, 845, 1))], axis=-1)

        return x, y

class DataGenerator_noconf(keras.utils.Sequence):
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
        cur_batch_list = [self.data_list[k] for k in indices]
        x, y = self.__generate(cur_batch_list)
        return x, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __generate(self, cur_batch_list):
        #generate data

        img = []
        skel_kpsoi = []
        for i, idx in enumerate(cur_batch_list):
            img.append(np.asarray(Image.open(self.data_list[i][0])))
            skel_xyz = np.reshape(self.data_list[i][1], (21, 3))
            skel_uvd = utils.xyz2uvd(skel_xyz)

            skel_kps = []
            for kps in skel_uvd:
                skel_kps.append(ia.Keypoint(x=kps[0],y=kps[1]))
            skel_kpsoi.append(ia.KeypointsOnImage(skel_kps, shape=img[0].shape))

        seq = iaa.Sequential([
            iaa.size.Resize({"height": constants.RSZ_HEIGHT, "width": constants.RSZ_WIDTH}),
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((0, 30))), #hue
            iaa.WithChannels(1, iaa.Add((0, 30))), #saturation
            iaa.WithChannels(2, iaa.Add((0, 60))), #exposure
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
        ])

        batch = ia.Batch(images=img, keypoints=skel_kpsoi)
        batch_aug = list(seq.augment_batches([batch]))[0]
        x = np.asarray(batch_aug.images_aug)
        skel_uvd_aug = batch_aug.keypoints_aug

        y = []
        for skel in skel_uvd_aug:
            skel_uvd_proj = []
            for kps in skel.keypoints:
                skel_uvd_proj.append([kps.x_int, kps.y_int])
            skel_uvd_proj = np.asarray(skel_uvd_proj)
            skel_uvd_rsz = np.concatenate([skel_uvd_proj, np.reshape(skel_uvd[:, 2], (21, 1))], 1)
            skel_gt = skel_uvd_rsz
            # skel_gt = utils.uvd2camcoord(skel_uvd_rsz)

            y.append(np.asarray(np.reshape(skel_gt, (-1))))

        y = np.broadcast_to(np.reshape(np.asarray(y), (self.batch_size, 1, -1)), (self.batch_size, 845, 63))

        return x, y
