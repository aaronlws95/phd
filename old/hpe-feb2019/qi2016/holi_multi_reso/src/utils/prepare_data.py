import sys
import os
import numpy as np
import scipy.ndimage.interpolation  as interplt
from PIL import Image
import h5py
from tqdm import tqdm

import utils.convert_xyz_uvd as xyzuvd
import utils.camera_info as cam

def _depth_to_uvd(depth):
    #convert depth to uv (2d coordinate values for depth points) and d(depth)
    #output: H x W x 3
    v, u = np.meshgrid(range(0, depth.shape[0], 1), range(0, depth.shape[1], 1), indexing= 'ij')
    v = np.asarray(v, 'uint16')[:, :, np.newaxis]
    u = np.asarray(u, 'uint16')[:, :, np.newaxis]
    depth = depth[:, :, np.newaxis]
    uvd = np.concatenate((u, v, depth), axis=2)
    return uvd.astype('float32')

def _get_pts_in_aabbox(depth, xyz_hand_gt, bound_loose=20):
    xyz_hand_gt_bounds = np.array([np.min(xyz_hand_gt[:, 0]), np.max(xyz_hand_gt[:, 0]),
                               np.min(xyz_hand_gt[:, 1]), np.max(xyz_hand_gt[:, 1]),
                               np.min(xyz_hand_gt[:, 2]), np.max(xyz_hand_gt[:, 2])])

    uvd = _depth_to_uvd(depth)
    uvd = uvd.reshape(uvd.shape[0]*uvd.shape[1], 3)
    all_points = xyzuvd.uvd2xyz_depth(uvd)

    all_points[:, 2] = all_points[:, 2] #m to mm

    all_points_bounds = np.array([np.min(all_points[:, 0]), np.max(all_points[:, 0]),
                               np.min(all_points[:, 1]), np.max(all_points[:, 1]),
                               np.min(all_points[:, 2]), np.max(all_points[:, 2])])
    # how much looser we want to make the bbox
    mask = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
    xyz_hand_gt_bounds[mask] -= bound_loose
    xyz_hand_gt_bounds[~mask] += bound_loose

    mask_x = (all_points[:, 0] >= xyz_hand_gt_bounds[0]) & (all_points[:, 0] <= xyz_hand_gt_bounds[1]) #x
    mask_y = (all_points[:, 1] >= xyz_hand_gt_bounds[2]) & (all_points[:, 1] <= xyz_hand_gt_bounds[3]) #y
    mask_z = (all_points[:, 2] >= xyz_hand_gt_bounds[4]) & (all_points[:, 2] <= xyz_hand_gt_bounds[5]) #z
    return all_points[mask_x & mask_y & mask_z] # points within the bbox

def get_fpha_data_list(modality, dataset_dir):
    train_pairs = []
    test_pairs = []
    img_dir = os.path.join(dataset_dir, 'Video_files')
    skel_dir = os.path.join(dataset_dir, 'Hand_pose_annotation_v1')
    if modality == 'depth':
        img_type = 'png'
    else:
        img_type = 'jpeg'
    with open(os.path.join(dataset_dir, 'data_split_action_recognition.txt')) as f:
        cur_split = 'Training'
        lines = f.readlines()
        for l in lines:
            words = l.split()
            if(words[0] == 'Training' or words[0] == 'Test'):
                cur_split = words[0]
            else:
                path = l.split()[0]
                full_path = os.path.join(img_dir, path, modality)
                len_frame_idx = len([x for x in os.listdir(full_path)
                                    if os.path.join(full_path, x)])
                skeleton_path = os.path.join(skel_dir, path, 'skeleton.txt')
                skeleton_vals = np.loadtxt(skeleton_path)
                for i in range(len_frame_idx):
                    img_path = os.path.join(img_dir, path, modality, '%s_%04d.%s' %(modality, i, img_type))
                    skel_xyz = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], -1)[i]
                    data_pair = (img_path, skel_xyz)
                    if cur_split == 'Training':
                        train_pairs.append(data_pair)
                    else:
                        test_pairs.append(data_pair)
    return train_pairs, test_pairs

def write_data_h5py(file_name, xyz_gt, save_dir):
    # xyz_gt: (-1, 63)

    center_joint_idx = 3
    ref_z = 1000
    bbsize = 260 #124x124 bbox in uvd
    img_size = 96
    u0 = cam.X0_DEPTH
    v0 = cam.Y0_DEPTH

    #to save
    new_file_name=[]
    uvd_norm_gt=[]
    hand_center_uvd=[]
    r0=[]
    r1=[]
    r2=[]

    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyzuvd.xyz2uvd_depth(xyz_gt)

    for i in tqdm(range(uvd_gt.shape[0])):
        depth = Image.open(file_name[i])
        depth = np.asarray(depth, dtype='uint16')

        #collect the points within the axis-aligned BBOX of the hand
        xyz_hand_gt = xyz_gt[i].copy()

        hand_points_xyz = _get_pts_in_aabbox(depth, xyz_hand_gt)
        _, bbox_uvd = xyzuvd.get_bbox(bbsize, ref_z, u0, v0)

        #mean calculation
        mean_z = xyz_hand_gt[center_joint_idx, 2]
        hand_points_xyz[:,2] += ref_z - mean_z
        xyz_hand_gt[:,2] += ref_z - mean_z
        uvd_hand_gt = xyzuvd.xyz2uvd_depth(xyz_hand_gt)
        mean_u = uvd_hand_gt[center_joint_idx, 0]
        mean_v = uvd_hand_gt[center_joint_idx, 1]

        hand_points_uvd = xyzuvd.xyz2uvd_depth(hand_points_xyz)

        #U
        hand_points_uvd[:,0] = hand_points_uvd[:,0] - mean_u + bbox_uvd[0,0]/2
        hand_points_uvd[np.where(hand_points_uvd[:,0]>=bbox_uvd[0,0]),0]=bbox_uvd[0,0]-1
        hand_points_uvd[np.where(hand_points_uvd[:,0]<0),0]=0

        uvd_hand_gt[:,0] = (uvd_hand_gt[:,0] - mean_u + bbox_uvd[0,0]/2 ) / bbox_uvd[0,0]
        uvd_hand_gt[np.where(uvd_hand_gt[:,0]>1),0]=1
        uvd_hand_gt[np.where(uvd_hand_gt[:,0]<0),0]=0

        #V
        hand_points_uvd[:,1] = hand_points_uvd[:,1] - mean_v + bbox_uvd[0,1]/2
        hand_points_uvd[ np.where(hand_points_uvd[:,1]>=bbox_uvd[0,1]),1]=bbox_uvd[0,1]-1
        hand_points_uvd[ np.where(hand_points_uvd[:,1]<0),1]=0

        uvd_hand_gt[:,1] =(uvd_hand_gt[:,1] - mean_v+bbox_uvd[0,1]/2) / bbox_uvd[0,1]
        uvd_hand_gt[ np.where(uvd_hand_gt[:,1]>1),1]=1
        uvd_hand_gt[ np.where(uvd_hand_gt[:,1]<0),1]=0

        # Z
        hand_points_uvd[:,2] = (hand_points_uvd[:,2] - ref_z + bbsize/2)/bbsize
        uvd_hand_gt[:,2] = (uvd_hand_gt[:,2] - ref_z + bbsize/2)/bbsize

        #get cropped hand
        crop_hand = np.ones((int(bbox_uvd[0,1]),int(bbox_uvd[0,0])),dtype='float32')
        crop_hand[np.asarray(np.floor(hand_points_uvd[:,1]),dtype='int16'),
                 np.asarray(np.floor(hand_points_uvd[:,0]),dtype='int16')] = hand_points_uvd[:,2]

        r0_i = interplt.zoom(crop_hand, img_size/bbox_uvd[0,0],order=1, mode='nearest',prefilter=True)
        r1_i = interplt.zoom(crop_hand, img_size/bbox_uvd[0,0]/2,order=1, mode='nearest',prefilter=True)
        r2_i = interplt.zoom(crop_hand, img_size/bbox_uvd[0,0]/4,order=1, mode='nearest',prefilter=True)

        r0.append(r0_i)
        r1.append(r1_i)
        r2.append(r2_i)
        uvd_norm_gt.append(uvd_hand_gt)
        hand_center_uvd.append([mean_u, mean_v, mean_z])

    f = h5py.File(save_dir, 'w')
    f.create_dataset('img0', data=r0)
    f.create_dataset('img1', data=r1)
    f.create_dataset('img2', data=r2)
    f.create_dataset('uvd_gt', data=uvd_gt)
    f.create_dataset('xyz_gt', data=xyz_gt)
    f.create_dataset('uvd_norm_gt', data=uvd_norm_gt)
    f.create_dataset('hand_center_uvd', data=hand_center_uvd)
    #encode the strings in a format h5py handles
    file_name_h5 = np.array(file_name, dtype=h5py.special_dtype(vlen=str))
    f.create_dataset('file_name', data = file_name_h5)
    f.close()

def read_data_h5py(load_file):
    f = h5py.File(load_file, 'r')
    # print('keys=', list(f.keys()))
    img0 = f['img0'][...]
    img1 = f['img1'][...]
    img2 = f['img2'][...]
    uvd_norm_gt = f['uvd_norm_gt'][...]
    uvd_gt = f['uvd_gt'][...]
    xyz_gt = f['xyz_gt'][...]
    hand_center_uvd = f['hand_center_uvd'][...]
    file_name = f['file_name'][...]
    f.close()

    return img0, img1, img2, uvd_norm_gt, uvd_gt, xyz_gt, hand_center_uvd, file_name

def move_channel_dim_3_to_1(img):
    '''
    input: (batch_size, h, w, channel) or (batch_size, h, w)
    output: (batch_size, channel, h, w)
    '''
    if len(img.shape) == 3:
        img = np.expand_dims(img, -1)
    return np.reshape(img, (img.shape[0], img.shape[3], img.shape[1], img.shape[2]))

def _read_predict(pred_file):
    pred_normuvd = []
    with open(pred_file, "r") as f:
        pred_line = f.readlines()
        for lines in pred_line:
            pred_normuvd.append([float(i) for i in lines.strip().split()])
    return pred_normuvd

def get_pred_xyzuvd_from_normuvd(pred_file, hand_center_uvd):
    pred_normuvd = np.reshape(_read_predict(pred_file), (-1, 21, 3))
    pred_xyz, pred_uvd = xyzuvd.normuvd2xyzuvd_depth(pred_normuvd, hand_center_uvd[:pred_normuvd.shape[0]])
    return np.reshape(pred_xyz, (-1, 63)), np.reshape(pred_uvd, (-1, 63)), np.reshape(pred_normuvd, (-1, 63))
