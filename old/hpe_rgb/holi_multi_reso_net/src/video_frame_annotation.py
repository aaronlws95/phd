import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2
from tqdm import tqdm
import imageio

from utils.directory import DATASET_DIR, DATA_DIR
import utils.prepare_data as pd
import utils.error as error

REORDER = [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]

def visualize_joints_2d(ax, joints, colors, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, colors[0])

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, colors, links, alpha=alpha)

def _draw2djoints(ax, annots, colors, links, alpha=1):
    """Draw segments, one color per link"""
    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)

def _draw2dseg(ax, annot, idx1, idx2, c, alpha=1):
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha)

# epoch = 215
# exp = 'base_RGB_batchnorm'
# epoch = 50
# exp = 'base_RGB'

data_split = 'test'
data_file_h5py = os.path.join(DATA_DIR, '%s_fpha_RGB.h5' %data_split)
_, _, _, _, _, _, _, file_name = pd.read_data_h5py(data_file_h5py)
# file_name = [i.replace('media/aaron/SHAREDDATA', '4TB/aaron') for i in file_name]

# vid_list = []
# save_file_comp = []
# for i, file in enumerate(file_name):
#     if file[-9:] == '0000.jpeg':
#         file_comp = file.split('/')
#         if i != 0:
#             name = (save_file_comp[6] + '_' + save_file_comp[7] + '_' + save_file_comp[8] + '_' + save_file_comp[9]).lower()
#             directory = os.path.join(save_file_comp[6],
#                                      save_file_comp[7],
#                                      save_file_comp[8],
#                                      save_file_comp[9])
#             vid_list.append((start_i, i, name, directory))
#         start_i = i
#         save_file_comp = file_comp

# with open('video_frame_ranges.txt', "w") as f:
#     for i, vid in enumerate(vid_list):
#         f.write(str(i) + ' '
#                 + str(vid[0]) + ' '
#                 + str(vid[1]) + ' '
#                 + vid[2] + ' '
#                 + vid[3] +'\n')

with open('video_frame_ranges.txt', "r") as f:
    lines = f.readlines()

vid_list = []
for l in lines:
    comp = l.split()
    vid_list.append((int(comp[1]), int(comp[2]), comp[3]))

experiment_numbers = [232]
methods = [(215, 'base_RGB_batchnorm'),
(50, 'base_RGB')]

for idx in experiment_numbers:
    vid_range = range(vid_list[idx][0], vid_list[idx][1])
    vid = vid_list[idx][2]
    for epoch, exp in methods:
        print(idx, epoch, exp, vid_list[idx][0], vid_list[idx][1], vid_list[idx][2])
        # train_xyz_gt, train_pred_xyz, train_uvd_gt, train_pred_uvd, train_uvd_norm_gt, train_pred_normuvd = error.get_pred_gt(epoch, 'train', exp)

        test_xyz_gt, test_pred_xyz, test_uvd_gt, test_pred_uvd, test_uvd_norm_gt, test_pred_normuvd = error.get_pred_gt(epoch, 'test', exp)

        test_pred_uvd = np.reshape(test_pred_uvd, (-1, 21, 3))
        test_uvd_gt = np.reshape(test_uvd_gt, (-1, 21, 3))

        SAVE_DIR = os.path.join(DATA_DIR, exp, 'img_annot_2d_epoch_%i' %epoch)
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        file_paths = []
        for i in tqdm(vid_range):
            file_name_i = file_name[i]
            img = Image.open(file_name_i)
            img = np.asarray(img, dtype='uint32')
            fig, ax = plt.subplots()
            plt.axis('off')
            ax.imshow(img)

            colors = ['r', 'r', 'r', 'r', 'r']
            visualize_joints_2d(ax, test_pred_uvd[i][REORDER], colors,  joint_idxs=False)
            colors = ['b', 'b', 'b', 'b', 'b']
            visualize_joints_2d(ax, test_uvd_gt[i][REORDER], colors, joint_idxs=False)
            files = os.path.join(SAVE_DIR, 'img_annot_%i' %i)
            file_paths.append(files + '.png')
            plt.savefig(files)
            plt.close()

        images = []
        for filename in tqdm(file_paths):
            images.append(imageio.imread(filename))

        imageio.mimsave(os.path.join(SAVE_DIR, '%s_%i_%s.gif' %(exp, epoch, vid)), images)
