import argparse
import os
import numpy as np
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import constants

def get_gt_xyzuvd_depth(data_split):
    train_pairs, test_pairs = get_data_list('depth')
    if data_split == 'test':
        file_name = [i for i,j in test_pairs]
        xyz_gt = [j for i,j in test_pairs]
    else:
        file_name = [i for i,j in train_pairs]
        xyz_gt = [j for i,j in train_pairs]
    uvd_gt = xyz2uvd_batch_depth(np.reshape(xyz_gt, (-1, 21, 3)))
    return np.asarray(xyz_gt), np.reshape(uvd_gt, (-1, 63)), file_name

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

def visualize_joints_3d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    z = joints[:, 2]
    ax.scatter(x, y, z, c='red')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw3djoints(ax, joints, links, alpha=alpha)

def _draw3djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']
    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw3dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)

def _draw3dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color"""

    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]], [annot[idx1, 2], annot[idx2, 2]],
        c=c,
        alpha=alpha)

def xyz2uvd_batch_color(skel_xyz):
    skel = np.copy(skel_xyz)
    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], skel.shape[1], 1])], 2)
    skel_camcoords = np.transpose(np.matmul(constants.CAM_EXTR, np.transpose(skel_hom, (0, 2, 1))), (0, 2, 1))[..., :3]
    skel_uvd = np.transpose(np.matmul(constants.CAM_INTR_COLOR, np.transpose(skel_camcoords, (0, 2, 1))), (0, 2, 1))
    skel_uvd[..., :2] =  skel_uvd[..., :2]/skel_uvd[..., 2:]
    return skel_uvd

def uvd2xyz_batch_color(skel_uvd):
    skel = np.copy(skel_uvd)
    skel[..., :2] = skel[..., :2]*skel[..., 2:]
    inv_cam_intr = np.linalg.inv(constants.CAM_INTR_COLOR)
    skel_camcoords = np.transpose(np.matmul(inv_cam_intr, np.transpose(skel, (0, 2, 1))), (0, 2, 1))
    skel_hom = np.concatenate([skel_camcoords, np.ones([skel_camcoords.shape[0], skel_camcoords.shape[1], 1])], 2)
    inv_cam_extr = np.linalg.inv(constants.CAM_EXTR)
    skel_xyz = np.transpose(np.matmul(inv_cam_extr, np.transpose(skel_hom, (0, 2, 1))), (0, 2, 1))
    return skel_xyz[..., :3]

# takes (,3) shape arrays
def xyz2camcoord_color(skel_xyz):
    skel = np.copy(skel_xyz)
    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
    skel_camcoords = np.array(constants.CAM_EXTR).dot(skel_hom.transpose()).transpose()[:, :3]
    return skel_camcoords

def xyz2uvd_color(skel_xyz, camcoord=False):
    skel = np.copy(skel_xyz)
    if not camcoord:
        skel = xyz2camcoord(skel)
    skel_uvd = np.array(constants.CAM_INTR_COLOR).dot(skel.transpose()).transpose()
    skel_uvd[:, :2] =  skel_uvd[:, :2]/skel_uvd[:, 2:]
    return skel_uvd

def uvd2camcoord_color(skel_uvd):
    skel = np.copy(skel_uvd)
    skel[:, :2] = skel[:, :2]*skel[:, 2:]
    inv_cam_intr = np.linalg.inv(constants.CAM_INTR_COLOR)
    skel_camcoords = np.array(inv_cam_intr).dot(skel.transpose()).transpose()
    return skel_camcoords

def uvd2xyz_color(skel_uvd, camcoord=False):
    skel = np.copy(skel_uvd)
    if not camcoord:
        skel = uvd2camcoord(skel)
    skel = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
    inv_cam_extr = np.linalg.inv(constants.CAM_EXTR)
    skel_xyz = inv_cam_extr.dot(skel.transpose()).transpose()
    return skel_xyz[:, :3]

def xyz2uvd_batch_depth(skel_xyz):
    skel = np.copy(skel_xyz)
    skel_uvd = np.transpose(np.matmul(constants.CAM_INTR_DEPTH, np.transpose(skel, (0, 2, 1))), (0, 2, 1))
    skel_uvd[..., :2] =  skel_uvd[..., :2]/skel_uvd[..., 2:]
    return skel_uvd.astype('float32')

def uvd2xyz_batch_depth(skel_uvd):
    #works for (batch_size, -1, 3)
    skel = np.copy(skel_uvd)
    skel[..., :2] = skel[..., :2]*skel[..., 2:]
    inv_cam_intr = np.linalg.inv(constants.CAM_INTR_DEPTH)
    skel_xyz = np.transpose(np.matmul(inv_cam_intr, np.transpose(skel, (0, 2, 1))), (0, 2, 1))
    return skel_xyz.astype('float32')

def xyz2uvd_depth(skel_xyz):
    #works for (-1, 3)
    skel = np.copy(skel_xyz)
    skel_uvd = np.array(constants.CAM_INTR_DEPTH).dot(skel.transpose()).transpose()
    skel_uvd[..., :2] =  skel_uvd[..., :2]/skel_uvd[..., 2:]
    return skel_uvd.astype('float32')

def uvd2xyz_depth(skel_uvd):
    skel = np.copy(skel_uvd)
    skel[..., :2] = skel[..., :2]*skel[..., 2:]
    inv_cam_intr = np.linalg.inv(constants.CAM_INTR_DEPTH)
    skel_xyz = np.array(inv_cam_intr).dot(skel.transpose()).transpose()
    return skel_xyz.astype('float32')

def uvd2xyz_indiv_depth(uvd):
    focal_length_x = constants.FOCAL_LENGTH_X_DEPTH
    focal_length_y = constants.FOCAL_LENGTH_Y_DEPTH
    u0= constants.X0_DEPTH
    v0= constants.Y0_DEPTH

    if len(uvd.shape)==3:
        xyz = np.empty((uvd.shape[0],uvd.shape[1],uvd.shape[2]),dtype='float32')
        xyz[:,:,2] = uvd[:,:,2]
        xyz[:,:,0] = ( uvd[:,:,0] - u0)/focal_length_x*xyz[:,:,2]
        xyz[:,:,1] = ( uvd[:,:,1]- v0)/focal_length_y*xyz[:,:,2]
    else:
        xyz = np.empty((uvd.shape[0],uvd.shape[1]),dtype='float32')
        z =  uvd[:,2]
        xyz[:,2] = z
        xyz[:,0] = ( uvd[:,0]- u0)/focal_length_x*z
        xyz[:,1] = ( uvd[:,1]- v0)/focal_length_y*z
    return xyz.astype('float32')

def xyz2uvd_indiv_depth(xyz):
    focal_length_x = constants.FOCAL_LENGTH_X_DEPTH
    focal_length_y = constants.FOCAL_LENGTH_Y_DEPTH
    u0= constants.X0_DEPTH
    v0= constants.Y0_DEPTH

    uvd = np.empty_like(xyz)
    if len(uvd.shape)==3:
        trans_x= xyz[:,:,0]
        trans_y= xyz[:,:,1]
        trans_z = xyz[:,:,2]
        uvd[:,:,0] = u0 + focal_length_x * ( trans_x / trans_z )
        uvd[:,:,1] = v0 +  focal_length_y * ( trans_y / trans_z )
        uvd[:,:,2] = trans_z #convert m to mm
    else:
        trans_x= xyz[:,0]
        trans_y= xyz[:,1]
        trans_z = xyz[:,2]
        uvd[:,0] = u0 +  focal_length_x * ( trans_x / trans_z )
        uvd[:,1] = v0 +  focal_length_y * ( trans_y / trans_z )
        uvd[:,2] = trans_z #convert m to mm
    return uvd.astype('float32')

def normuvd2xyzuvd_batch_depth(norm_uvd,hand_center_uvd):
    #works for (batch_size, -1, 3)
    u0 = constants.X0_DEPTH
    v0 = constants.Y0_DEPTH
    bbsize = constants.BBSIZE
    mean_u = np.expand_dims(hand_center_uvd[:, 0], axis=-1)
    mean_v = np.expand_dims(hand_center_uvd[:, 1], axis=-1)
    mean_z = np.expand_dims(hand_center_uvd[:, 2], axis=-1)
    ref_z = constants.REF_Z

    _, bbox_uvd = get_bbox(bbsize, ref_z, u0, v0)

    uvd_hand = np.empty_like(norm_uvd)
    uvd_hand[:, :, 0] = norm_uvd[:, :, 0]*bbox_uvd[0, 0] + mean_u - bbox_uvd[0,0]/2
    uvd_hand[:, :, 1] = norm_uvd[:, :, 1]*bbox_uvd[0, 1] + mean_v - bbox_uvd[0,1]/2
    uvd_hand[:, :, 2] = norm_uvd[:, :, 2]*bbsize + ref_z - bbsize/2
    xyz = uvd2xyz_batch_depth(uvd_hand)
    xyz[:, :,2] = xyz[:, :,2] - ref_z + mean_z
    uvd = xyz2uvd_batch_depth(xyz)
    return xyz, uvd

def normuvd2xyzuvd_depth(norm_uvd,hand_center_uvd):
    #works for (-1, 3)
    u0 = constants.X0_DEPTH
    v0 = constants.Y0_DEPTH
    bbsize = constants.BBSIZE
    mean_u = hand_center_uvd[0]
    mean_v = hand_center_uvd[1]
    mean_z = hand_center_uvd[2]
    ref_z = constants.REF_Z

    _, bbox_uvd = get_bbox(bbsize, ref_z, u0, v0)

    uvd = np.empty_like(norm_uvd)
    uvd[:, 0] = norm_uvd[:, 0]*bbox_uvd[0, 0] + mean_u - bbox_uvd[0,0]/2
    uvd[:, 1] = norm_uvd[:, 1]*bbox_uvd[0, 1] + mean_v - bbox_uvd[0,1]/2
    uvd[:, 2] = norm_uvd[:, 2]*bbsize + ref_z - bbsize/2
    xyz = uvd2xyz_depth(uvd)
    xyz[:,2] = xyz[:,2] - ref_z + mean_z
    uvd = xyz2uvd_depth(xyz)
    return xyz, uvd


def normuvd2xyzuvd_batch_depth_hier(norm_uvd,hand_center_uvd):
    #works for (batch_size, -1, 3)
    u0 = constants.X0_DEPTH
    v0 = constants.Y0_DEPTH
    bbsize = constants.BBSIZE
    mean_u = np.expand_dims(hand_center_uvd[:, 0], axis=-1)
    mean_v = np.expand_dims(hand_center_uvd[:, 1], axis=-1)
    mean_z = np.expand_dims(hand_center_uvd[:, 2], axis=-1)
    ref_z = constants.REF_Z

    _, bbox_uvd = get_bbox(bbsize, ref_z, u0, v0)

    uvd_hand = np.empty_like(norm_uvd)
    uvd_hand[:, :, 0] = norm_uvd[:, :, 0]*bbox_uvd[0, 0] + mean_u
    uvd_hand[:, :, 1] = norm_uvd[:, :, 1]*bbox_uvd[0, 1] + mean_v
    uvd_hand[:, :, 2] = norm_uvd[:, :, 2]*bbsize + ref_z
    xyz = uvd2xyz_batch_depth(uvd_hand)
    xyz[:, :,2] = xyz[:, :,2] - ref_z + mean_z
    uvd = xyz2uvd_batch_depth(xyz)
    return xyz, uvd

def normuvd2xyzuvd_depth_hier(norm_uvd,hand_center_uvd):
    #works for (-1, 3)
    u0 = constants.X0_DEPTH
    v0 = constants.Y0_DEPTH
    bbsize = constants.BBSIZE
    mean_u = hand_center_uvd[0]
    mean_v = hand_center_uvd[1]
    mean_z = hand_center_uvd[2]
    ref_z = constants.REF_Z

    _, bbox_uvd = get_bbox(bbsize, ref_z, u0, v0)

    uvd = np.empty_like(norm_uvd)
    uvd[:, 0] = norm_uvd[:, 0]*bbox_uvd[0, 0] + mean_u
    uvd[:, 1] = norm_uvd[:, 1]*bbox_uvd[0, 1] + mean_v
    uvd[:, 2] = norm_uvd[:, 2]*bbsize + ref_z
    xyz = uvd2xyz_depth(uvd)
    xyz[:,2] = xyz[:,2] - ref_z + mean_z
    uvd = xyz2uvd_depth(xyz)
    return xyz, uvd

def get_bbox(bbsize,ref_z,u0,v0):
    bbox_xyz = np.array([(bbsize,bbsize,ref_z)])
    bbox_uvd = xyz2uvd_depth(bbox_xyz)
    bbox_uvd[0,0] = np.ceil(bbox_uvd[0,0] - u0)
    bbox_uvd[0,1] = np.ceil(bbox_uvd[0,1] - v0)
    return bbox_xyz, bbox_uvd

def get_data_list(modality):
    train_pairs = []
    test_pairs = []
    if modality == 'depth':
        img_type = 'png'
    else:
        img_type = 'jpeg'
    with open(os.path.join(constants.DATASET_DIR, 'data_split_action_recognition.txt')) as f:
        cur_split = 'Training'
        lines = f.readlines()
        for l in lines:
            words = l.split()
            if(words[0] == 'Training' or words[0] == 'Test'):
                cur_split = words[0]
            else:
                path = l.split()[0]
                full_path = os.path.join(constants.IMG_DIR, path, modality)
                len_frame_idx = len([x for x in os.listdir(full_path)
                                    if os.path.join(full_path, x)])
                skeleton_path = os.path.join(constants.SKEL_DIR, path, 'skeleton.txt')
                skeleton_vals = np.loadtxt(skeleton_path)
                for i in range(len_frame_idx):
                    img_path = os.path.join(constants.IMG_DIR, path, modality, '%s_%04d.%s' %(modality, i, img_type))
                    skel_xyz = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], -1)[i]
                    data_pair = (img_path, skel_xyz)
                    if cur_split == 'Training':
                        train_pairs.append(data_pair)
                    else:
                        test_pairs.append(data_pair)
    return train_pairs, test_pairs

def get_data_list_cross_subject(modality):
    train_pairs = []
    test_pairs = []
    with open(os.path.join(constants.DATASET_DIR, 'data_split_action_recognition.txt')) as f:
        lines = f.readlines()
        for l in lines:
            words = l.split()
            if(words[0] != 'Training' and words[0] != 'Test'):
                path = l.split()[0]
                full_path = os.path.join(constants.IMG_DIR, path, modality)
                len_frame_idx = len([x for x in os.listdir(full_path)
                                    if os.path.join(full_path, x)])
                skeleton_path = os.path.join(constants.SKEL_DIR, path, 'skeleton.txt')
                skeleton_vals = np.loadtxt(skeleton_path)
                for i in range(len_frame_idx):
                    img_path = os.path.join(constants.IMG_DIR, path, modality, '%s_%04d.png' %(modality, i))
                    skel_xyz = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], -1)[i]
                    data_pair = (img_path, skel_xyz)
                    if path[8] in ['1', '3', '4']:
                        train_pairs.append(data_pair)
                    else:
                        test_pairs.append(data_pair)
    return train_pairs, test_pairs

def depth_to_uvd(depth):
    #convert depth to uv (2d coordinate values for depth points) and d(depth)
    #output: H x W x 3
    v, u = np.meshgrid(range(0, depth.shape[0], 1), range(0, depth.shape[1], 1), indexing= 'ij')
    v = np.asarray(v, 'uint16')[:, :, np.newaxis]
    u = np.asarray(u, 'uint16')[:, :, np.newaxis]
    depth = depth[:, :, np.newaxis]
    uvd = np.concatenate((u, v, depth), axis=2)
    return uvd.astype('float32')
