import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2
from tqdm import tqdm
import imageio

DATA_DIR1 = '/media/aaron/SHAREDDATA/data_holi'
DATA_DIR2 = '/media/aaron/SHAREDDATA/data_znb'

# epoch1 = 215
# exp1 = 'base_RGB_batchnorm'

# epoch2 = 50
# exp2 = 'base_RGB'

# SAVE_DIR1 = os.path.join(DATA_DIR, exp1, 'img_annot_2d_epoch_%i' %epoch1)

# SAVE_DIR2 = os.path.join(DATA_DIR, exp2, 'img_annot_2d_epoch_%i' %epoch2)

# with open('video_frame_ranges.txt', "r") as f:
#     lines = f.readlines()

# vid_list = []
# for l in lines:
#     comp = l.split()
#     vid_list.append((int(comp[1]), int(comp[2]), comp[3]))

# experiment_numbers = [20, 165, 272]

# for idx in experiment_numbers:
#     images = []
#     for i in tqdm(range(vid_list[idx][0], vid_list[idx][1])):
#         img1 = np.asarray(Image.open(os.path.join(SAVE_DIR1, 'img_annot_%i.png' %i)))
#         img2 = np.asarray(Image.open(os.path.join(SAVE_DIR2, 'img_annot_%i.png' %i)))
#         newimg = np.concatenate((img1, img2), axis=1)
#         images.append(newimg)

#     imageio.mimsave(os.path.join(DATA_DIR, 'saved', '%s_%i_%s_%i_%s_combined.gif' %(exp1, epoch1, exp2, epoch2, vid_list[idx][2])), images)
#     print('saved to %s' %os.path.join(DATA_DIR, 'saved', '%s_%i_%s_%i_%s_combined.gif' %(exp1, epoch1, exp2, epoch2, vid_list[idx][2])))

# epoch1 = 340
# exp1 = 'base_lift_net'

# SAVE_DIR1 = os.path.join(DATA_DIR2, exp1, 'img_annot_2d_epoch_%i' %epoch1)

# SAVE_DIR2 = os.path.join(DATA_DIR2, exp1, 'img_annot_2d_epoch_%i_GT' %epoch1)

# with open('video_frame_ranges.txt', "r") as f:
#     lines = f.readlines()

# vid_list = []
# for l in lines:
#     comp = l.split()
#     vid_list.append((int(comp[1]), int(comp[2]), comp[3]))

# experiment_numbers = [20, 165, 272]

# for idx in experiment_numbers:
#     images = []
#     for i in tqdm(range(vid_list[idx][0], vid_list[idx][1])):
#         img1 = np.asarray(Image.open(os.path.join(SAVE_DIR1, 'img_annot_%i.png' %i)))
#         img2 = np.asarray(Image.open(os.path.join(SAVE_DIR2, 'img_annot_%i_GT.png' %i)))
#         newimg = np.concatenate((img1, img2), axis=1)
#         images.append(newimg)

#     imageio.mimsave(os.path.join(DATA_DIR2, 'saved', '%s_%i_%s_%i_%s_GT_combined.gif' %(exp1, epoch1, exp1, epoch1, vid_list[idx][2])), images)
#     print('saved to %s' %os.path.join(DATA_DIR2, 'saved', '%s_%i_%s_%i_%s_GT_combined.gif' %(exp1, epoch1, exp1, epoch1, vid_list[idx][2])))

epoch2 = 340
exp2 = 'base_lift_net'

epoch1 = 215
exp1 = 'base_RGB_batchnorm'

SAVE_DIR1 = os.path.join(DATA_DIR1, exp1, 'img_annot_2d_epoch_%i' %epoch1)

SAVE_DIR2 = os.path.join(DATA_DIR2, exp2, 'img_annot_2d_epoch_%i' %epoch2)

with open('video_frame_ranges.txt', "r") as f:
    lines = f.readlines()

vid_list = []
for l in lines:
    comp = l.split()
    vid_list.append((int(comp[1]), int(comp[2]), comp[3]))

experiment_numbers = [232]

for idx in experiment_numbers:
    images = []
    for i in tqdm(range(vid_list[idx][0], vid_list[idx][1])):
        img1 = np.asarray(Image.open(os.path.join(SAVE_DIR1, 'img_annot_%i.png' %i)))
        img2 = np.asarray(Image.open(os.path.join(SAVE_DIR2, 'img_annot_%i.png' %i)))
        newimg = np.concatenate((img1, img2), axis=1)
        images.append(newimg)

    imageio.mimsave(os.path.join(DATA_DIR2, 'saved', '%s_%i_%s_%i_%s_combined.gif' %(exp1, epoch1, exp1, epoch1, vid_list[idx][2])), images)
    print('saved to %s' %os.path.join(DATA_DIR2, 'saved', '%s_%i_%s_%i_%s_combined.gif' %(exp1, epoch1, exp1, epoch1, vid_list[idx][2])))
