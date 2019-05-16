import sys
import os
import numpy as np
import scipy.ndimage.interpolation  as interplt
from PIL import Image
import h5py
from tqdm import tqdm

ROOT_DIR = os.path.abspath("../src")
sys.path.append(ROOT_DIR)
import utils
import constants

def _get_pts_in_aabbox(depth, xyz_hand_gt, bound_loose=20):
    xyz_hand_gt_bounds = np.array([np.min(xyz_hand_gt[:, 0]), np.max(xyz_hand_gt[:, 0]),
                               np.min(xyz_hand_gt[:, 1]), np.max(xyz_hand_gt[:, 1]),
                               np.min(xyz_hand_gt[:, 2]), np.max(xyz_hand_gt[:, 2])])

    uvd = utils.depth_to_uvd(depth)
    uvd = uvd.reshape(uvd.shape[0]*uvd.shape[1], 3)
    all_points = utils.uvd2xyz_depth(uvd)

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

def prepare_data(file_name, xyz_gt):
    # xyz_gt: (-1, 63)
    # output: (-1, 21, 3)
    center_joint_idx = constants.CENTRE_JNT_IDX
    ref_z = constants.REF_Z
    bbsize = constants.BBSIZE #124x124 bbox in uvd
    img_size = constants.IMG_RSZ
    u0 = constants.X0_DEPTH
    v0 = constants.Y0_DEPTH

    #to save
    new_file_name=[]
    uvd_norm_gt=[]
    hand_center_uvd=[]
    r0=[]
    r1=[]
    r2=[]

    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = utils.xyz2uvd_batch_depth(xyz_gt)

    for i in range(uvd_gt.shape[0]):
        depth = Image.open(file_name[i])
        depth = np.asarray(depth, dtype='uint16')

        #collect the points within the axis-aligned BBOX of the hand
        xyz_hand_gt = xyz_gt[i].copy()

        hand_points_xyz = _get_pts_in_aabbox(depth, xyz_hand_gt)
        _, bbox_uvd = utils.get_bbox(bbsize, ref_z, u0, v0)

        #mean calculation
        mean_z = xyz_hand_gt[center_joint_idx, 2]
        hand_points_xyz[:,2] += ref_z - mean_z
        xyz_hand_gt[:,2] += ref_z - mean_z
        uvd_hand_gt = utils.xyz2uvd_depth(xyz_hand_gt)
        mean_u = uvd_hand_gt[center_joint_idx, 0]
        mean_v = uvd_hand_gt[center_joint_idx, 1]

        hand_points_uvd = utils.xyz2uvd_depth(hand_points_xyz)

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

    return r0, r1, r2, uvd_norm_gt, uvd_gt, xyz_gt, hand_center_uvd, file_name

def prepare_data_h5py_hier(file_name, xyz_gt, save_dir):
    # xyz_gt: (-1, 63)
    from skimage.transform import resize
    center_joint_idx = constants.CENTRE_JNT_IDX
    img_size = constants.IMG_RSZ
    u0 = constants.X0_DEPTH
    ref_z = constants.REF_Z
    v0 = constants.Y0_DEPTH
    bbsize = constants.BBSIZE
    #to save
    new_file_name=[]
    uvd_norm_gt=[]
    hand_center_uvd=[]
    r0=[]
    r1=[]
    r2=[]

    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = utils.xyz2uvd_batch_depth(xyz_gt)

    for i in tqdm(range(uvd_gt.shape[0])):
        depth = Image.open(file_name[i])
        depth = np.asarray(depth, dtype='uint16')

        #collect the points within the axis-aligned BBOX of the hand
        xyz_hand_gt = xyz_gt[i].copy()

        hand_points_xyz = _get_pts_in_aabbox(depth, xyz_hand_gt)
        _, bbox_uvd = utils.get_bbox(bbsize, ref_z, u0, v0)

        #mean calculation
        mean_z = xyz_hand_gt[center_joint_idx, 2]
        hand_points_xyz[:,2] += ref_z - mean_z
        xyz_hand_gt[:,2] += ref_z - mean_z
        uvd_hand_gt = utils.xyz2uvd_depth(xyz_hand_gt)
        mean_u = uvd_hand_gt[center_joint_idx, 0]
        mean_v = uvd_hand_gt[center_joint_idx, 1]

        hand_points_uvd = utils.xyz2uvd_depth(hand_points_xyz)

        #U
        hand_points_uvd[:,0] = hand_points_uvd[:,0] - mean_u + bbox_uvd[0,0]/2
        hand_points_uvd[np.where(hand_points_uvd[:,0]>=bbox_uvd[0,0]),0]=bbox_uvd[0,0]-1
        hand_points_uvd[np.where(hand_points_uvd[:,0]<0),0]=0

        #V
        hand_points_uvd[:,1] = hand_points_uvd[:,1] - mean_v + bbox_uvd[0,1]/2
        hand_points_uvd[ np.where(hand_points_uvd[:,1]>=bbox_uvd[0,1]),1]=bbox_uvd[0,1]-1
        hand_points_uvd[ np.where(hand_points_uvd[:,1]<0),1]=0

        # Z
        hand_points_uvd[:,2] = (hand_points_uvd[:,2] - ref_z)/bbsize

        #get cropped hand
        crop_hand = np.ones((int(bbox_uvd[0,1]),int(bbox_uvd[0,0])),dtype='float32')
        crop_hand[np.asarray(np.floor(hand_points_uvd[:,1]),dtype='int16'),
                 np.asarray(np.floor(hand_points_uvd[:,0]),dtype='int16')] = hand_points_uvd[:,2]

        r0_i = resize(crop_hand, (img_size, img_size), order=3,preserve_range=True)
        r1_i = resize(crop_hand, (img_size/2, img_size/2), order=3,preserve_range=True)
        r2_i = resize(crop_hand, (img_size/4, img_size/4), order=3,preserve_range=True)

        uvd_hand_gt[:, 0] = (uvd_hand_gt[:,0] - mean_u)/bbox_uvd[0,0]
        uvd_hand_gt[:, 1] = (uvd_hand_gt[:,1] - mean_v)/bbox_uvd[0,1]
        uvd_hand_gt[:, 2] = (uvd_hand_gt[:,2] - ref_z)/bbsize

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

def prepare_data_h5py(file_name, xyz_gt, save_dir):
    # xyz_gt: (-1, 63)

    center_joint_idx = 3
    ref_z = 1000
    bbsize = 260 #124x124 bbox in uvd
    img_size = 96
    u0 = constants.X0_DEPTH
    v0 = constants.Y0_DEPTH

    #to save
    new_file_name=[]
    uvd_norm_gt=[]
    hand_center_uvd=[]
    r0=[]
    r1=[]
    r2=[]

    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyz2uvd.xyz2uvd_indiv_depth(xyz_gt)

    for i in tqdm(range(uvd_gt.shape[0])):
        depth = Image.open(file_name[i])
        depth = np.asarray(depth, dtype='uint16')

        #collect the points within the axis-aligned BBOX of the hand
        xyz_hand_gt = xyz_gt[i].copy()

        hand_points_xyz = _get_pts_in_aabbox(depth, xyz_hand_gt)
        _, bbox_uvd = _get_bbox(bbsize, ref_z, u0, v0)

        #mean calculation
        mean_z = xyz_hand_gt[center_joint_idx, 2]
        hand_points_xyz[:,2] += ref_z - mean_z
        xyz_hand_gt[:,2] += ref_z - mean_z
        uvd_hand_gt = xyz2uvd.xyz2uvd_indiv_depth(xyz_hand_gt)
        mean_u = uvd_hand_gt[center_joint_idx, 0]
        mean_v = uvd_hand_gt[center_joint_idx, 1]

        hand_points_uvd = xyz2uvd.xyz2uvd_indiv_depth(hand_points_xyz)

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

def prepare_data_h5py_shanxin(file_name, xyz_gt, save_dir):
    # xyz_gt: (-1, 63)

    center_joint_idx = constants.CENTRE_JNT_IDX
    ref_z = constants.REF_Z
    bbsize = constants.BBSIZE #124x124 bbox in uvd
    img_size = constants.IMG_RSZ
    u0 = constants.X0_DEPTH
    v0 = constants.Y0_DEPTH

    #to save
    new_file_name=[]
    uvd_norm_gt=[]
    hand_center_uvd=[]
    r0=[]
    r1=[]
    r2=[]

    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = utils.xyz2uvd_batch_depth(xyz_gt)

    for i in tqdm(range(uvd_gt.shape[0])):
        depth = Image.open(file_name[i])
        depth = np.asarray(depth, dtype='uint16')

        #collect the points within the axis-aligned BBOX of the hand
        xyz_hand_gt = xyz_gt[i].copy()

        hand_points_xyz = _get_pts_in_aabbox(depth, xyz_hand_gt)
        _, bbox_uvd = utils.get_bbox(bbsize, ref_z, u0, v0)

        #mean calculation
        mean_z = xyz_hand_gt[center_joint_idx, 2]
        hand_points_xyz[:,2] += ref_z - mean_z
        xyz_hand_gt[:,2] += ref_z - mean_z
        uvd_hand_gt = utils.xyz2uvd_depth(xyz_hand_gt)
        mean_u = uvd_hand_gt[center_joint_idx, 0]
        mean_v = uvd_hand_gt[center_joint_idx, 1]

        hand_points_uvd = utils.xyz2uvd_depth(hand_points_xyz)

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
    f.create_dataset('r0', data=r0)
    f.create_dataset('r1', data=r1)
    f.create_dataset('r2', data=r2)
    f.create_dataset('uvd_jnt_gt', data=uvd_gt)
    f.create_dataset('xyz_jnt_gt', data=xyz_gt)
    f.create_dataset('uvd_jnt_gt_norm', data=uvd_norm_gt)
    f.create_dataset('hand_center_uvd', data=hand_center_uvd)
    resx = constants.ORI_X
    resy = constants.ORI_Y
    f.create_dataset('resxy', data=[resx,resy])
    f.create_dataset('ref_z', data=ref_z)
    f.create_dataset('bbsize', data=bbsize)
    #encode the strings in a format h5py handles
    file_name_h5 = np.array(file_name, dtype=h5py.special_dtype(vlen=str))
    f.create_dataset('file_name', data = file_name_h5)
    f.close()

def read_prepare_data_h5py(load_file):
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

def get_hand_center_uvd(file_name, xyz_gt):
    # xyz_gt: (-1, 63)

    center_joint_idx = constants.CENTRE_JNT_IDX
    ref_z = constants.REF_Z

    #to save
    hand_center_uvd=[]
    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = utils.xyz2uvd_batch_depth(xyz_gt)

    for i in tqdm(range(uvd_gt.shape[0])):
        depth = Image.open(file_name[i])
        depth = np.asarray(depth, dtype='uint16')

        #collect the points within the axis-aligned BBOX of the hand
        xyz_hand_gt = xyz_gt[i].copy()

        hand_points_xyz = _get_pts_in_aabbox(depth, xyz_hand_gt)

        #mean calculation
        mean_z = xyz_hand_gt[center_joint_idx, 2]
        hand_points_xyz[:,2] += ref_z - mean_z
        xyz_hand_gt[:,2] += ref_z - mean_z
        uvd_hand_gt = utils.xyz2uvd_depth(xyz_hand_gt)
        mean_u = uvd_hand_gt[center_joint_idx, 0]
        mean_v = uvd_hand_gt[center_joint_idx, 1]

        hand_center_uvd.append([mean_u, mean_v, mean_z])

    return hand_center_uvd

def read_hand_center_uvd(hand_center_file):
    hand_center_uvd = []
    with open(hand_center_file, "r") as f:
        mean_uvd_line = f.readlines()
        for lines in mean_uvd_line:
            hand_center_uvd.append([float(i) for i in lines.strip().split()])
    return hand_center_uvd

def write_hand_center_uvd(file_name, xyz_gt, save_dir):
    # xyz_gt: (-1, 63)

    center_joint_idx = constants.CENTRE_JNT_IDX
    ref_z = constants.REF_Z

    #to save
    hand_center_uvd=[]
    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = utils.xyz2uvd_batch_depth(xyz_gt)

    for i in tqdm(range(uvd_gt.shape[0])):
        depth = Image.open(file_name[i])
        depth = np.asarray(depth, dtype='uint16')

        #collect the points within the axis-aligned BBOX of the hand
        xyz_hand_gt = xyz_gt[i].copy()

        hand_points_xyz = _get_pts_in_aabbox(depth, xyz_hand_gt)

        #mean calculation
        mean_z = xyz_hand_gt[center_joint_idx, 2]
        hand_points_xyz[:,2] += ref_z - mean_z
        xyz_hand_gt[:,2] += ref_z - mean_z
        uvd_hand_gt = utils.xyz2uvd_depth(xyz_hand_gt)
        mean_u = uvd_hand_gt[center_joint_idx, 0]
        mean_v = uvd_hand_gt[center_joint_idx, 1]

        hand_center_uvd.append([mean_u, mean_v, mean_z])

    with open(save_dir, "w") as f:
        for mean_uvd in hand_center_uvd:
            for uvd in mean_uvd:
                f.write(str(uvd) + ' ')
            f.write('\n')

if __name__ == '__main__':
    data_split = 'test'
    train_pairs, test_pairs = utils.get_data_list('depth')
    if data_split == 'train':
        file_name = [i for i,j in train_pairs]
        xyz_gt = [j for i,j in train_pairs]
    else:
        file_name = [i for i,j in test_pairs]
        xyz_gt = [j for i,j in test_pairs]

    prepare_data_h5py(file_name[:10], xyz_gt[:10], os.path.join(constants.DATA_DIR, 'train_fpha_blabla.h5'))
    # prepare_data_h5py(file_name, xyz_gt, os.path.join(constants.DATA_DIR, 'test_fpha_cross_subject.h5'))
    # prepare_data_h5py(file_name, xyz_gt, os.path.join(constants.DATA_DIR, 'test_fpha.h5'))

    # write_hand_center_uvd(file_name, xyz_gt, os.path.join(constants.DATA_DIR, 'hand_center_uvd_train.txt'))
