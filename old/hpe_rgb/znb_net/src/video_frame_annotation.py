import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2
from tqdm import tqdm
import imageio
import pickle
import lmdb

from utils.directory import DATASET_DIR, DATA_DIR
import utils.prepare_data as pd
import utils.error as error
import utils.convert_xyz_uvd as xyzuvd

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

data_split = 'test'
save_prefix = 'test_fpha'
keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
test_keys = pickle.load(open(keys_cache_file, "rb"))

# vid_list = []
# save_file_comp = []
# for i, file in enumerate(test_keys):
#     if file[-9:] == '0000.jpeg':
#         file_comp = file.split('/')
#         if i != 0:
#             name = (save_file_comp[0] + '_' + save_file_comp[1] + '_' + save_file_comp[2] + '_' + save_file_comp[3]).lower()
#             vid_list.append((start_i, i, name))
#         start_i = i
#         save_file_comp = file_comp

# with open('video_frame_ranges.txt', "w") as f:
#     for i, vid in enumerate(vid_list):
#         f.write(str(i) + ' '
#                 + str(vid[0]) + ' '
#                 + str(vid[1]) + ' '
#                 + vid[2] +'\n')

with open('video_frame_ranges.txt', "r") as f:
    lines = f.readlines()

vid_list = []
for l in lines:
    comp = l.split()
    vid_list.append((int(comp[1]), int(comp[2]), comp[3], comp[4]))

experiment_numbers = [232]
methods = [(340, 'base_lift_net', 150, 'pose_net_aug')]



for idx in experiment_numbers:
    vid_range = range(vid_list[idx][0], vid_list[idx][1])
    vid = vid_list[idx][2]
    for epoch, exp, epoch_pose, exp_pose in methods:
        print(idx, epoch, exp, epoch_pose, exp_pose, vid_list[idx][0], vid_list[idx][1], vid_list[idx][2])

        dataroot_xyz_gt_saved = os.path.join(DATA_DIR, save_prefix + '_xyz_gt.lmdb')
        test_xyz_gt = pd.read_all_lmdb_from_file(test_keys, dataroot_xyz_gt_saved, np.float32, (21, 3))
        test_xyz_gt = np.asarray(test_xyz_gt)
        test_uvd_gt = xyzuvd.xyz2uvd_color(test_xyz_gt)
        pred_file = os.path.join(DATA_DIR, exp, 'xyz_canon_%s_%s_smap_%s_%s.txt' %(epoch, data_split, epoch_pose, exp_pose))
        test_pred_xyz_canon = pd.get_pred_xyz_canon(pred_file, save_prefix)
        pred_file = os.path.join(DATA_DIR, exp, 'rot_mat_%s_%s_smap_%s_%s.txt' %(epoch, data_split, epoch_pose, exp_pose))
        test_rot_mat_pred = pd.get_pred_rot_mat(pred_file, save_prefix)
        keypoint_scale = pd.get_keypoint_scale(test_xyz_gt)

        test_pred_xyz = []
        for canon, rot, scale in zip(test_pred_xyz_canon, test_rot_mat_pred, keypoint_scale):
            test_pred_xyz.append(np.matmul(canon, rot)*scale)
        test_pred_xyz += np.expand_dims(test_xyz_gt[:, 0, :], axis=1)
        test_pred_uvd = xyzuvd.xyz2uvd_color(np.asarray(test_pred_xyz))

        SAVE_DIR = os.path.join(DATA_DIR, exp, 'img_annot_2d_epoch_%i' %epoch)
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        file_paths = []
        for num_frame, i in tqdm(enumerate(vid_range)):
            file_name_i = os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files', vid_list[idx][3], 'color_%04d.jpeg' %num_frame)
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

    for epoch, exp, _, _ in methods:
        print(idx, epoch, exp, vid_list[idx][0], vid_list[idx][1], vid_list[idx][2], 'GT')

        dataroot_xyz_gt_saved = os.path.join(DATA_DIR, save_prefix + '_xyz_gt.lmdb')
        test_xyz_gt = pd.read_all_lmdb_from_file(test_keys, dataroot_xyz_gt_saved, np.float32, (21, 3))
        test_xyz_gt = np.asarray(test_xyz_gt)
        test_uvd_gt = xyzuvd.xyz2uvd_color(test_xyz_gt)
        pred_file = os.path.join(DATA_DIR, exp, 'xyz_canon_%s_%s.txt' %(epoch, data_split))
        test_pred_xyz_canon = pd.get_pred_xyz_canon(pred_file, save_prefix)
        pred_file = os.path.join(DATA_DIR, exp, 'rot_mat_%s_%s.txt' %(epoch, data_split))
        test_rot_mat_pred = pd.get_pred_rot_mat(pred_file, save_prefix)
        keypoint_scale = pd.get_keypoint_scale(test_xyz_gt)

        test_pred_xyz = []
        for canon, rot, scale in zip(test_pred_xyz_canon, test_rot_mat_pred, keypoint_scale):
            test_pred_xyz.append(np.matmul(canon, rot)*scale)
        test_pred_xyz += np.expand_dims(test_xyz_gt[:, 0, :], axis=1)
        test_pred_uvd = xyzuvd.xyz2uvd_color(np.asarray(test_pred_xyz))

        SAVE_DIR = os.path.join(DATA_DIR, exp, 'img_annot_2d_epoch_%i_GT' %epoch)
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        file_paths = []
        for num_frame, i in tqdm(enumerate(vid_range)):
            file_name_i = os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files', vid_list[idx][3], 'color_%04d.jpeg' %num_frame)
            img = Image.open(file_name_i)
            img = np.asarray(img, dtype='uint32')
            fig, ax = plt.subplots()
            plt.axis('off')
            ax.imshow(img)

            colors = ['r', 'r', 'r', 'r', 'r']
            visualize_joints_2d(ax, test_pred_uvd[i][REORDER], colors,  joint_idxs=False)
            colors = ['b', 'b', 'b', 'b', 'b']
            visualize_joints_2d(ax, test_uvd_gt[i][REORDER], colors, joint_idxs=False)
            files = os.path.join(SAVE_DIR, 'img_annot_%i_GT' %i)
            file_paths.append(files + '.png')
            plt.savefig(files)
            plt.close()

        images = []
        for filename in tqdm(file_paths):
            images.append(imageio.imread(filename))

        imageio.mimsave(os.path.join(SAVE_DIR, '%s_%i_%s_GT.gif' %(exp, epoch, vid)), images)
