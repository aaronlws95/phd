import argparse
import os
import numpy as np
import trimesh
from matplotlib import pyplot as plt
from PIL import Image

import constants

# Loading utilities
def load_objects(obj_root):
    object_names = ['juice_bottle', 'liquid_soap', 'milk', 'salt']
    all_models = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                '{}_model.ply'.format(obj_name))
        mesh = trimesh.load(obj_path)
        all_models[obj_name] = {
            'verts': np.array(mesh.vertices),
            'faces': np.array(mesh.faces)
        }
    return all_models


def get_skeleton(sample, skel_root):
    skeleton_path = os.path.join(skel_root, sample['subject'],
                                 sample['action_name'], sample['seq_idx'],
                                 'skeleton.txt')
    print('Loading skeleton from {}'.format(skeleton_path))
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                            -1)[sample['frame_idx']]
    return skeleton


def get_obj_transform(sample, obj_root):
    seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                            sample['seq_idx'], 'object_pose.txt')
    with open(seq_path, 'r') as seq_f:
        raw_lines = seq_f.readlines()
    raw_line = raw_lines[sample['frame_idx']]
    line = raw_line.strip().split(' ')
    trans_matrix = np.array(line[1:]).astype(np.float32)
    trans_matrix = trans_matrix.reshape(4, 4).transpose()
    print('Loading obj transform from {}'.format(seq_path))
    return trans_matrix


# Display utilities
def visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha)


def _draw2djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']
    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)


def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha)

# takes (,3) shape arrays
def xyz2camcoord(skel_xyz):
    skel = np.copy(skel_xyz)
    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
    skel_camcoords = np.array(constants.CAM_EXTR).dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
    return skel_camcoords

def xyz2uvd(skel_xyz, camcoord=False):
    skel = np.copy(skel_xyz)
    if not camcoord:
        skel = xyz2camcoord(skel)
    skel_uvd = np.array(constants.CAM_INTR_COLOR).dot(skel.transpose()).transpose()
    skel_uvd[:, :2] =  skel_uvd[:, :2]/skel_uvd[:, 2:]
    return skel_uvd

def uvd2camcoord(skel_uvd):
    skel = np.copy(skel_uvd)
    skel[:, :2] = skel[:, :2]*skel[:, 2:]
    inv_cam_intr = np.linalg.inv(constants.CAM_INTR_COLOR)
    skel_camcoords = np.array(inv_cam_intr).dot(skel.transpose()).transpose()
    return skel_camcoords

def uvd2xyz(skel_uvd, camcoord=False):
    skel = np.copy(skel_uvd)
    if not camcoord:
        skel = uvd2camcoord(skel)
    skel = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
    inv_cam_extr = np.linalg.inv(constants.CAM_EXTR)
    skel_xyz = inv_cam_extr.dot(skel.transpose()).transpose()
    return skel_xyz[:, :3]

def get_data_list():
    train_pairs = []
    test_pairs = []
    with open(os.path.join(constants.DATASET_DIR, 'data_split_action_recognition.txt')) as f:
        cur_split = 'Training'
        lines = f.readlines()
        for l in lines:
            words = l.split()
            if(words[0] == 'Training' or words[0] == 'Test'):
                cur_split = words[0]
            else:
                path = l.split()[0]
                full_path = os.path.join(constants.IMG_DIR, path, 'color')
                len_frame_idx = len([x for x in os.listdir(full_path)
                                    if os.path.join(full_path, x)])
                skeleton_path = os.path.join(constants.SKEL_DIR, path, 'skeleton.txt')
                skeleton_vals = np.loadtxt(skeleton_path)
                for i in range(len_frame_idx):
                    img_path = os.path.join(constants.IMG_DIR, path, 'color', 'color_%04d.jpeg' %i)
                    skel_xyz = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], -1)[i]
                    data_pair = (img_path, skel_xyz)
                    if cur_split == 'Training':
                        train_pairs.append(data_pair)
                    else:
                        test_pairs.append(data_pair)
    return train_pairs, test_pairs
