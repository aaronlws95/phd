import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.ndimage.interpolation  as interplt
from PIL import Image
import h5py

import utils
import constants

train_pairs, test_pairs = utils.get_data_list('color')
file_name = [i for i,j in train_pairs]
xyz_gt = [j for i,j in train_pairs]
uvd_gt = utils.xyz2uvd_batch_color(np.reshape(xyz_gt, (-1,21,3)))
indices = np.arange(len(file_name))
np.random.shuffle(indices)
for idx in indices:
    cur_frame = file_name[idx]
    color = Image.open(cur_frame)
    color = np.asarray(color, dtype='uint32')

    # imgcopy=depth.copy()
    # min = imgcopy.min()
    # max = imgcopy.max()
    # #scale to 0 - 255
    # imgcopy = (imgcopy - min) / (max - min) * 255.
    # imgcopy = imgcopy.astype('uint8')
    # imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2BGR)

    print(cur_frame)

    fig, ax = plt.subplots()
    ax.imshow(color)
    utils.visualize_joints_2d(ax, uvd_gt[idx][constants.REORDER], joint_idxs=False)
    plt.show()
