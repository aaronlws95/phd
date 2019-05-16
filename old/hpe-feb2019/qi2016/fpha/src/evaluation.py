import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from PIL import Image

import constants
import utils
import error
from prepare_data import prepare_data, read_prepare_data_h5py

epoch = 40
data_split = 'train'
if data_split == 'train':
    data_file_h5py = os.path.join(constants.DATA_DIR, 'train_fpha.h5')
else:
    data_file_h5py = os.path.join(constants.DATA_DIR, 'test_fpha.h5')

exp = ''
pred_file = os.path.join(constants.DATA_DIR, exp, 'predict_%s_%s.txt' %(epoch, data_split))
img0, img1, img2, uvd_norm_gt, uvd_gt, xyz_gt, hand_center_uvd, file_name = read_prepare_data_h5py(data_file_h5py)
pred_xyz, pred_uvd, pred_normuvd = error.get_pred_xyzuvd_from_normuvd(pred_file, hand_center_uvd)
xyz_gt = np.reshape(xyz_gt[:pred_uvd.shape[0]], (-1, 63))
uvd_gt = np.reshape(uvd_gt[:pred_uvd.shape[0]], (-1, 63))
uvd_norm_gt = np.reshape(uvd_norm_gt[:pred_uvd.shape[0]], (-1, 63))
pred_xyz = np.reshape(pred_xyz, (-1, 63))
pred_uvd = np.reshape(pred_uvd, (-1, 63))
pred_normuvd = np.reshape(pred_normuvd, (-1, 63))

#normuvd
pred = pred_normuvd
true = uvd_norm_gt
#print('normuvd mean_jnt_error:', error.mean_jnt_error(true, pred))
#print('normuvd mean_dim_error:', error.mean_dim_error(true, pred))
#print('normuvd mean_overall_error:', error.mean_overall_error(true, pred))
print('normuvd mean_squared_error:', np.mean(error.mean_squared_error(true, pred)))

#xyz
pred = pred_xyz
true = xyz_gt
#print('xyz mean_jnt_error:', error.mean_jnt_error(true, pred))
#print('xyz mean_dim_error:', error.mean_dim_error(true, pred))
print('xyz mean_overall_error:', error.mean_overall_error(true, pred))

#uvd
pred = pred_uvd
true = uvd_gt
#print('uvd mean_jnt_error:', error.mean_jnt_error(true, pred))
#print('uvd mean_dim_error:', error.mean_dim_error(true, pred))
#print('uvd mean_overall_error:', error.mean_overall_error(true, pred))

# Percentage of correct key points
pred = np.reshape(pred_xyz, (-1, 63))
true = np.asarray(xyz_gt)
#error.percentage_frames_within_error_curve(true, pred)

i = 50000

depth = Image.open(file_name[i])
depth = np.asarray(depth, dtype='uint16')
fig, ax = plt.subplots(1,2, figsize=(18, 10))
ax[0].imshow(depth)
ax[1].imshow(depth)
ax[0].set_title('pred')
ax[1].set_title('true')
utils.visualize_joints_2d(ax[0], np.reshape(pred_uvd, (-1,21,3))[i][constants.REORDER], joint_idxs=False)
utils.visualize_joints_2d(ax[1], np.reshape(uvd_gt, (-1,21,3))[i][constants.REORDER], joint_idxs=False)
#plt.show()

fig = plt.figure(figsize=(12, 5))
ax_1 = fig.add_subplot(1, 2, 1, projection='3d')
ax_1.set_title('pred')
ax_2 = fig.add_subplot(1, 2, 2, projection='3d')
ax_2.set_title('true')

for ax in [ax_1, ax_2]:
    ax.view_init(elev=30, azim=45)

utils.visualize_joints_3d(ax_1, np.reshape(pred_uvd, (-1,21,3))[i][constants.REORDER], joint_idxs=False)
utils.visualize_joints_3d(ax_2, np.reshape(uvd_gt, (-1,21,3))[i][constants.REORDER], joint_idxs=False)
#plt.show()

