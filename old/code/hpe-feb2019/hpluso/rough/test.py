import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import keras
from matplotlib import pyplot as plt

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import constants
import utils
import data_generator

# dataset_dir = '/media/aaron/DATA/ubuntu/fpha-dataset/'
dataset_dir = '/media/aaron/DATADRIVE1/First_Person_Action_Benchmark'
image_dir = os.path.join(dataset_dir, 'Video_files')
annot_dir = os.path.join(dataset_dir, 'Hand_pose_annotation_v1')

img_details = ['Subject_1', 'put_salt', '1']

sample = {
    'subject': img_details[0],
    'action_name': img_details[1],
    'seq_idx': img_details[2],
    'frame_idx': 0,
    'object': None
}

img_color_path = os.path.join(image_dir,
                   sample['subject'],
                   sample['action_name'],
                   sample['seq_idx'],
                   'color',
                   'color_%04d.jpeg' %sample['frame_idx'])
img = cv2.imread(img_color_path)[:,:,::-1]

fig, ax = plt.subplots()
ax.imshow(img)

reorder_idx = np.array([
    0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19,
    20
])

skel_xyz = utils.get_skeleton(sample, annot_dir)[reorder_idx]
skel_uvd = utils.xyz2uvd(skel_xyz)
skel_new_xyz = utils.uvd2xyz(skel_uvd)
utils.visualize_joints_2d(ax, skel_uvd, joint_idxs=False)

