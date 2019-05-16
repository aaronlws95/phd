import sys
import os
import numpy as np
from skimage.transform import resize
from PIL import Image
import h5py
from tqdm import tqdm
import lmdb
import pickle
from multiprocessing import Pool
import time
from shutil import get_terminal_size

import utils.convert_xyz_uvd as xyzuvd
import utils.camera_info as cam
from utils.directory import DATASET_DIR, DATA_DIR

def sk_resize(data, size):
    return resize(data, size, order=3, preserve_range=True)

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

def get_fpha_data_list_general(modality, dataset_dir):
    train_pairs = []
    test_pairs = []
    img_dir = os.path.join(dataset_dir, 'First_Person_Action_Benchmark', 'Video_files')
    skel_dir = os.path.join(dataset_dir, 'First_Person_Action_Benchmark', 'Hand_pose_annotation_v1')
    if modality == 'depth':
        img_type = 'png'
    else:
        img_type = 'jpeg'
    with open(os.path.join(dataset_dir, 'First_Person_Action_Benchmark', 'data_split_action_recognition.txt')) as f:
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
                    img_path = os.path.join(path, modality, '%s_%04d.%s' %(modality, i, img_type))
                    skel_xyz = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], -1)[i]
                    data_pair = (img_path, skel_xyz)
                    if cur_split == 'Training':
                        train_pairs.append(data_pair)
                    else:
                        test_pairs.append(data_pair)
    return train_pairs, test_pairs

def get_fpha_data_list_general_val(modality, dataset_dir):
    train_pairs = []
    test_pairs = []
    val_pairs = []
    img_dir = os.path.join(dataset_dir, 'First_Person_Action_Benchmark', 'Video_files')
    skel_dir = os.path.join(dataset_dir, 'First_Person_Action_Benchmark', 'Hand_pose_annotation_v1')
    if modality == 'depth':
        img_type = 'png'
    else:
        img_type = 'jpeg'
    with open(os.path.join(dataset_dir, 'First_Person_Action_Benchmark', 'data_split_action_recognition_with_val_random_80_20.txt')) as f:
        cur_split = 'Training'
        lines = f.readlines()
        for l in lines:
            words = l.split()
            if(words[0] == 'Training' or words[0] == 'Test' or words[0] == 'Validation'):
                cur_split = words[0]
            else:
                path = l.split()[0]
                full_path = os.path.join(img_dir, path, modality)
                len_frame_idx = len([x for x in os.listdir(full_path)
                                    if os.path.join(full_path, x)])
                skeleton_path = os.path.join(skel_dir, path, 'skeleton.txt')
                skeleton_vals = np.loadtxt(skeleton_path)
                for i in range(len_frame_idx):
                    img_path = os.path.join(path, modality, '%s_%04d.%s' %(modality, i, img_type))
                    skel_xyz = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], -1)[i]
                    data_pair = (img_path, skel_xyz)
                    if cur_split == 'Training':
                        train_pairs.append(data_pair)
                    elif cur_split == 'Validation':
                        val_pairs.append(data_pair)
                    else:
                        test_pairs.append(data_pair)
    return train_pairs, test_pairs, val_pairs

def create_multiple_gaussian_map(uvd_gt, output_size, sigma=25.0):
    coords_uv = np.stack([uvd_gt[:, 1], uvd_gt[:, 0]], -1)

    coords_uv = coords_uv.astype(np.int32)

    cond_1_in = np.logical_and(np.less(coords_uv[:, 0], output_size[0]-1), np.greater(coords_uv[:, 0], 0))
    cond_2_in = np.logical_and(np.less(coords_uv[:, 1], output_size[1]-1), np.greater(coords_uv[:, 1], 0))
    cond = np.logical_and(cond_1_in, cond_2_in)

    coords_uv = coords_uv.astype(np.float32)

    # create meshgrid
    x_range = np.expand_dims(np.arange(output_size[0]), 1)
    y_range = np.expand_dims(np.arange(output_size[1]), 0)

    X = np.tile(x_range, [1, output_size[1]]).astype(np.float32)
    Y = np.tile(y_range, [output_size[0], 1]).astype(np.float32)

    X = np.expand_dims(X, -1)
    Y = np.expand_dims(Y, -1)

    X_b = np.tile(X, [1, 1, coords_uv.shape[0]])
    Y_b = np.tile(Y, [1, 1, coords_uv.shape[0]])

    X_b -= coords_uv[:, 0]
    Y_b -= coords_uv[:, 1]

    dist = np.square(X_b) + np.square(Y_b)

    scoremap = np.exp(-dist / np.square(sigma)) * cond.astype(np.float32)

    return scoremap

def left_handed_rot_mat(dim, angle):
    c, s = np.cos(angle), np.sin(angle)
    if dim == 'x':
        rot_mat = [[1., 0, 0],
                   [0, c, s],
                   [0, -s, c]]
    elif dim == 'y':
        rot_mat = [[c, 0, -s],
                  [0, 1, 0],
                  [s, 0, c]]
    elif dim == 'z':
        rot_mat = [[c, s, 0],
                   [-s, c, 0],
                   [0, 0, 1]]
    else:
        raise ValueError('dim needs to be x, y or z')

    return rot_mat

def norm_keypoint(coords):
    ROOT_NODE_ID = 0
    translate = coords[:, ROOT_NODE_ID, :]
    coords_norm = coords - translate
    root_bone_length = np.sqrt(np.sum(np.square(coords_norm[:, 12, :] - coords_norm[:, 3, :])))
    return coords_norm / root_bone_length, root_bone_length

def get_keypoint_scale(coords):
    if len(coords.shape) == 2:
        coords_proc = np.expand_dims(coords, axis=0)
    else:
        coords_proc = coords.copy()
    return np.sqrt(np.sum(np.square(coords_proc[:, 12, :] - coords_proc[:, 3, :]), axis=-1))

def canonical_transform(coords):
    ROOT_NODE_ID = 0
    ALIGN_NODE_ID = 3 # middle root
    ROT_NODE_ID = 5 # pinky root

    # 1. Translate the whole set s.t. the root kp is located in the origin
    # trans = coords[:, ROOT_NODE_ID, :]
    # coords_t = coords - trans

    # 2. Rotate and scale keypoints such that the root bone is of unit length and aligned with the y axis
    p = coords[:, ALIGN_NODE_ID, :]  # thats the point we want to put on coord (0, 1, 0)
    # Rotate point into the yz-plane
    alpha = np.arctan2(p[:, 0], p[:, 1])
    rot_mat = left_handed_rot_mat('z', alpha)
    coords_t_r1 = np.matmul(coords, rot_mat)
    total_rot_mat = rot_mat

    # Rotate point within the yz-plane onto the xy-plane
    p = coords_t_r1[:, ALIGN_NODE_ID, :]
    beta = -np.arctan2(p[:, 2], p[:, 1])
    rot_mat = left_handed_rot_mat('x', beta + 3.141592653589793)
    coords_t_r2 = np.matmul(coords_t_r1, rot_mat)
    total_rot_mat = np.matmul(total_rot_mat, rot_mat)

    # 3. Rotate keypoints such that rotation along the y-axis is defined
    p = coords_t_r2[:, ROT_NODE_ID, :]
    gamma = np.arctan2(p[:, 2], p[:, 0])
    rot_mat = left_handed_rot_mat('y', gamma)
    coords_normed = np.matmul(coords_t_r2, rot_mat)
    total_rot_mat = np.matmul(total_rot_mat, rot_mat)

    return coords_normed, total_rot_mat

def write_to_lmdb(keys, dataset, save_dir, logger):
    data_size_per_img = dataset[keys[0]].nbytes
    logger.info('data size per image is: %i' %data_size_per_img)
    data_size = data_size_per_img * len(dataset)
    env = lmdb.open(save_dir, map_size=data_size * 10)

    if logger:
        logger.info('WRITING DATA: %s' %save_dir)

    with env.begin(write=True) as txn:  # txn is a Transaction object
        for key in tqdm(keys):
            key_byte = key.encode('ascii')
            data = dataset[key]
            txn.put(key_byte, data)
    logger.info('FINISH WRITING LMBD')

def write_to_lmdb_add_map_size(keys, dataset, save_dir, logger):
    data_size_per_img = dataset[keys[0]].nbytes
    logger.info('data size per image is: %i' %data_size_per_img)
    data_size = data_size_per_img * len(dataset)
    env = lmdb.open(save_dir, map_size=data_size * 10 + 100000)

    if logger:
        logger.info('WRITING DATA: %s' %save_dir)

    with env.begin(write=True) as txn:  # txn is a Transaction object
        for key in tqdm(keys):
            key_byte = key.encode('ascii')
            data = dataset[key]
            txn.put(key_byte, data)
    logger.info('FINISH WRITING LMBD')


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '█' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

def reading_data_worker_img_crop(file_name, uvd_hand_gt):
    color = Image.open(os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files', file_name))
    color = np.asarray(color, dtype='uint32')

    pad = 50
    x_max = int(np.amax(uvd_hand_gt[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_hand_gt[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_hand_gt[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_hand_gt[:,1])) - pad, 0)

    crop_hand = color[y_min:y_max, x_min:x_max, :]

    if crop_hand.size == 0:
        return None

    uvd_hand_gt[:, 0] = uvd_hand_gt[:, 0] - x_min
    uvd_hand_gt[:, 1] = uvd_hand_gt[:, 1] - y_min

    crop_hand_rsz = resize(crop_hand, (256, 256), order=3, preserve_range=True).astype('uint8')

    return file_name, crop_hand_rsz

def write_data_to_file_img_crop(file_name, xyz_gt, logger=None):
    # xyz_gt: (-1, 63)
    if logger:
        logger.info('write_data_to_file_img_crop')
    n_thread = 12
    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyzuvd.xyz2uvd_color(xyz_gt)
    fpha_path = os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark')
    scoremap = {}
    keys = []
    pbar = ProgressBar(uvd_gt.shape[0])
    def mycallback(arg):
        '''mp callback,
        get the image data and update pbar'''
        if arg is not None:
            file_name = arg[0]
            img_save_path = os.path.join(fpha_path, 'Video_files_256_crop', file_name)
            if os.path.isfile(img_save_path):
                print('exists', img_save_path, '%i/%i' %(i, len(result)))
            else:
                img = arg[1]
                img = Image.fromarray(img)
                img.save(img_save_path)
                pbar.update('Saving {}'.format(img_save_path))
        else:
            pbar.update('NO HAND, SKIPPING FRAME')

    pool = Pool(n_thread)
    for path, uvd_hand_gt in zip(file_name, uvd_gt):
        pool.apply_async(reading_data_worker_img_crop, args=(path, uvd_hand_gt), callback=mycallback)
    pool.close()
    pool.join()

def check_no_crop_images(file_name, xyz_gt, logger=None):
    # xyz_gt: (-1, 63)
    if logger:
        logger.info('check_no_crop_images')

    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyzuvd.xyz2uvd_color(xyz_gt)

    for i, (fname, uvd_hand_gt) in enumerate(zip(file_name, uvd_gt)):
        color = Image.open(os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files', fname))
        color = np.asarray(color, dtype='uint32')

        pad = 50
        x_max = int(np.amax(uvd_hand_gt[:,0])) + pad
        x_min = np.maximum(int(np.amin(uvd_hand_gt[:,0])) - pad, 0)
        y_max = int(np.amax(uvd_hand_gt[:,1])) + pad
        y_min = np.maximum(int(np.amin(uvd_hand_gt[:,1])) - pad, 0)

        crop_hand = color[y_min:y_max, x_min:x_max, :]

        if crop_hand.size == 0:
            print(i, fname)


def reading_data_worker_scoremap_32(file_name, uvd_hand_gt):
    color = Image.open(os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files', file_name))
    color = np.asarray(color, dtype='uint32')

    pad = 50
    x_max = int(np.amax(uvd_hand_gt[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_hand_gt[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_hand_gt[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_hand_gt[:,1])) - pad, 0)

    crop_hand = color[y_min:y_max, x_min:x_max, :]

    if crop_hand.size == 0:
        return None

    uvd_hand_gt[:, 0] = uvd_hand_gt[:, 0] - x_min
    uvd_hand_gt[:, 1] = uvd_hand_gt[:, 1] - y_min

    Rx = 256/crop_hand.shape[1]
    Ry = 256/crop_hand.shape[0]

    uvd_hand_gt_rsz = uvd_hand_gt.copy()
    uvd_hand_gt_rsz[:, 0] = uvd_hand_gt_rsz[:, 0]*Rx
    uvd_hand_gt_rsz[:, 1] = uvd_hand_gt_rsz[:, 1]*Ry
    scoremap = create_multiple_gaussian_map(uvd_hand_gt_rsz, (256,256))
    scoremap = sk_resize(scoremap, (32, 32))

    return file_name, scoremap.astype('float32')

def write_data_lmdb_mp_scoremap_32(file_name, xyz_gt, save_prefix, logger=None):
    # xyz_gt: (-1, 63)
    if logger:
        logger.info('PROCESSING DATA LMDB MULTIPROCESS SCOREMAP 32: %s' %save_prefix)
    n_thread = 12
    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyzuvd.xyz2uvd_color(xyz_gt)

    scoremap = {}
    keys = []
    pbar = ProgressBar(uvd_gt.shape[0])
    def mycallback(arg):
        '''mp callback,
        get the image data and update pbar'''
        if arg is not None:
            key = arg[0]
            keys.append(key)
            scoremap[key] = arg[1]
            pbar.update('Reading {}'.format(key))
        else:
            pbar.update('NO HAND, SKIPPING FRAME')

    pool = Pool(n_thread)
    for path, uvd_hand_gt in zip(file_name, uvd_gt):
        pool.apply_async(reading_data_worker_scoremap_32, args=(path, uvd_hand_gt), callback=mycallback)
    pool.close()
    pool.join()
    logger.info('Finish reading {} images.\nWrite lmdb...'.format(uvd_gt.shape[0]))

    write_to_lmdb(keys, scoremap, os.path.join(DATA_DIR, save_prefix + '_scoremap_32.lmdb'), logger)

    keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
    pickle.dump(keys, open(keys_cache_file, "wb"))
    logger.info('Finish creating lmdb keys cache.')

def reading_data_worker(file_name, xyz_hand_gt, uvd_hand_gt):
    color = Image.open(os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files', file_name))
    color = np.asarray(color, dtype='uint32')

    xyz_gt_save = xyz_hand_gt.copy()

    pad = 50
    x_max = int(np.amax(uvd_hand_gt[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_hand_gt[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_hand_gt[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_hand_gt[:,1])) - pad, 0)

    crop_hand = color[y_min:y_max, x_min:x_max, :]

    if crop_hand.size == 0:
        return None

    uvd_hand_gt[:, 0] = uvd_hand_gt[:, 0] - x_min
    uvd_hand_gt[:, 1] = uvd_hand_gt[:, 1] - y_min

    crop_hand_rsz = resize(crop_hand, (256, 256), order=3, preserve_range=True).astype('uint32')

    Rx = 256/crop_hand.shape[1]
    Ry = 256/crop_hand.shape[0]

    uvd_hand_gt_rsz = uvd_hand_gt.copy()
    uvd_hand_gt_rsz[:, 0] = uvd_hand_gt_rsz[:, 0]*Rx
    uvd_hand_gt_rsz[:, 1] = uvd_hand_gt_rsz[:, 1]*Ry

    xyz_hand_gt_norm, _ = norm_keypoint(np.expand_dims(xyz_hand_gt, axis=0))
    xyz_hand_gt_canon, inv_rot_mat_i = canonical_transform(xyz_hand_gt_norm)
    rot_mat_i = np.linalg.inv(inv_rot_mat_i)
    xyz_hand_gt_canon = np.squeeze(xyz_hand_gt_canon)

    uvd_unscale_param = np.asarray([Rx, Ry, x_min, y_min])

    return file_name, crop_hand_rsz, uvd_hand_gt_rsz.astype('float32'), xyz_gt_save.astype('float32'), xyz_hand_gt_canon.astype('float32'), rot_mat_i.astype('float32'), uvd_unscale_param.astype('float32')

def reading_data_worker_uvd_unscale(file_name, xyz_hand_gt, uvd_hand_gt):
    color = Image.open(os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files', file_name))
    color = np.asarray(color, dtype='uint32')

    pad = 50
    x_max = int(np.amax(uvd_hand_gt[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_hand_gt[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_hand_gt[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_hand_gt[:,1])) - pad, 0)

    crop_hand = color[y_min:y_max, x_min:x_max, :]

    if crop_hand.size == 0:
        return None

    uvd_hand_gt[:, 0] = uvd_hand_gt[:, 0] - x_min
    uvd_hand_gt[:, 1] = uvd_hand_gt[:, 1] - y_min

    Rx = 256/crop_hand.shape[1]
    Ry = 256/crop_hand.shape[0]

    uvd_hand_gt_rsz = uvd_hand_gt.copy()
    uvd_hand_gt_rsz[:, 0] = uvd_hand_gt_rsz[:, 0]*Rx
    uvd_hand_gt_rsz[:, 1] = uvd_hand_gt_rsz[:, 1]*Ry

    uvd_unscale_param = np.asarray([Rx, Ry, x_min, y_min])

    return file_name, uvd_unscale_param.astype('float32')

def read_lmdb_from_file(key, dataroot, dtype, reshape_dim):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
        data_flat = np.frombuffer(buf, dtype=dtype)
        data = data_flat.reshape(reshape_dim)
    return data

def read_all_lmdb_from_file(keys, dataroot, dtype, reshape_dim):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    all_data = []
    with env.begin(write=False) as txn:
        for key in keys:
            buf = txn.get(key.encode('ascii'))
            data_flat = np.frombuffer(buf, dtype=dtype)
            data = data_flat.reshape(reshape_dim)
            all_data.append(data)
    return all_data

def read_lmdb(key, env, dtype, reshape_dim):
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
        data_flat = np.frombuffer(buf, dtype=dtype)
        data = data_flat.reshape(reshape_dim)
    return data

def write_data_lmdb_mp(file_name, xyz_gt, save_prefix, logger=None):
    # xyz_gt: (-1, 63)
    if logger:
        logger.info('PROCESSING DATA LMDB MULTIPROCESS: %s' %save_prefix)
    n_thread = 12
    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyzuvd.xyz2uvd_color(xyz_gt)

    xyz_gt_canon = {}
    rot_mat = {}
    xyz_gt_save={}
    uvd_gt_scaled = {}
    imgs = {}
    uvd_unscale_param = {}
    keys = []
    pbar = ProgressBar(uvd_gt.shape[0])
    def mycallback(arg):
        '''mp callback,
        get the image data and update pbar'''
        if arg is not None:
            key = arg[0]
            keys.append(key)
            imgs[key] = arg[1]
            uvd_gt_scaled[key] = arg[2]
            xyz_gt_save[key] = arg[3]
            xyz_gt_canon[key] = arg[4]
            rot_mat[key] = arg[5]
            uvd_unscale_param[key] = arg[6]
            pbar.update('Reading {}'.format(key))
        else:
            pbar.update('NO HAND, SKIPPING FRAME')

    pool = Pool(n_thread)
    for path, xyz_hand_gt, uvd_hand_gt in zip(file_name, xyz_gt, uvd_gt):
        pool.apply_async(reading_data_worker, args=(path, xyz_hand_gt, uvd_hand_gt), callback=mycallback)
    pool.close()
    pool.join()
    logger.info('Finish reading {} images.\nWrite lmdb...'.format(uvd_gt.shape[0]))

    write_to_lmdb(keys, imgs, os.path.join(DATA_DIR, save_prefix + '_256_img.lmdb'), logger)
    write_to_lmdb(keys, uvd_gt_scaled, os.path.join(DATA_DIR, save_prefix + '_uvd_gt_scaled.lmdb'), logger)
    write_to_lmdb(keys, xyz_gt_save, os.path.join(DATA_DIR, save_prefix + '_xyz_gt.lmdb'), logger)
    write_to_lmdb(keys, xyz_gt_canon, os.path.join(DATA_DIR, save_prefix + '_xyz_gt_canon.lmdb'), logger)
    write_to_lmdb(keys, rot_mat, os.path.join(DATA_DIR, save_prefix + '_rot_mat.lmdb'), logger)
    write_to_lmdb(keys, uvd_unscale_param, os.path.join(DATA_DIR, save_prefix + '_uvd_unscale_param.lmdb'), logger)

    keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
    pickle.dump(keys, open(keys_cache_file, "wb"))
    logger.info('Finish creating lmdb keys cache.')

def write_data_lmdb_mp_uvd_unscale(file_name, xyz_gt, save_prefix, logger=None):
    # xyz_gt: (-1, 63)
    if logger:
        logger.info('PROCESSING DATA LMDB MULTIPROCESS: %s' %save_prefix)
    n_thread = 12
    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyzuvd.xyz2uvd_color(xyz_gt)

    uvd_unscale_param = {}
    keys = []
    pbar = ProgressBar(uvd_gt.shape[0])
    def mycallback(arg):
        '''mp callback,
        get the image data and update pbar'''
        if arg is not None:
            key = arg[0]
            keys.append(key)
            uvd_unscale_param[key] = arg[1]
            pbar.update('Reading {}'.format(key))
        else:
            pbar.update('NO HAND, SKIPPING FRAME')

    pool = Pool(n_thread)
    for path, xyz_hand_gt, uvd_hand_gt in zip(file_name, xyz_gt, uvd_gt):
        pool.apply_async(reading_data_worker_uvd_unscale, args=(path, xyz_hand_gt, uvd_hand_gt), callback=mycallback)
    pool.close()
    pool.join()
    logger.info('Finish reading {} images.\nWrite lmdb...'.format(uvd_gt.shape[0]))

    write_to_lmdb(keys, uvd_unscale_param, os.path.join(DATA_DIR, save_prefix + '_uvd_unscale_param.lmdb'), logger)

def write_data_lmdb(file_name, xyz_gt, save_prefix, logger=None):
    # xyz_gt: (-1, 63)
    if logger:
        logger.info('PROCESSING DATA LMDB: %s' %save_prefix)

    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyzuvd.xyz2uvd_color(xyz_gt)

    rsz_dim = 256

    xyz_gt_canon = {}
    rot_mat = {}
    xyz_gt_save={}
    keypoint_scale = {}
    uvd_gt_scaled = {}
    keys = []
    imgs = {}
    for i in tqdm(range(uvd_gt.shape[0])):
        color = Image.open(os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files', file_name[i]))
        color = np.asarray(color, dtype='uint32')

        xyz_hand_gt = xyz_gt[i].copy()
        uvd_hand_gt = uvd_gt[i].copy()

        pad = 50
        x_max = int(np.amax(uvd_hand_gt[:,0])) + pad
        x_min = np.maximum(int(np.amin(uvd_hand_gt[:,0])) - pad, 0)
        y_max = int(np.amax(uvd_hand_gt[:,1])) + pad
        y_min = np.maximum(int(np.amin(uvd_hand_gt[:,1])) - pad, 0)

        crop_hand = color[y_min:y_max, x_min:x_max, :]

        if crop_hand.size == 0:
            if logger:
                logger.info('NO HAND, SKIPPING FRAME: %i %s' %(i, file_name[i]))
            continue

        uvd_hand_gt[:, 0] = uvd_hand_gt[:, 0] - x_min
        uvd_hand_gt[:, 1] = uvd_hand_gt[:, 1] - y_min

        crop_hand_rsz = resize(crop_hand, (rsz_dim, rsz_dim), order=3, preserve_range=True).astype('uint32')

        Rx = rsz_dim/crop_hand.shape[1]
        Ry = rsz_dim/crop_hand.shape[0]

        uvd_hand_gt_rsz = uvd_hand_gt.copy()
        uvd_hand_gt_rsz[:, 0] = uvd_hand_gt_rsz[:, 0]*Rx
        uvd_hand_gt_rsz[:, 1] = uvd_hand_gt_rsz[:, 1]*Ry

        xyz_hand_gt_norm, root_bone_length = norm_keypoint(np.expand_dims(xyz_hand_gt, axis=0))
        xyz_hand_gt_canon, inv_rot_mat_i = canonical_transform(xyz_hand_gt_norm)
        rot_mat_i = np.linalg.inv(inv_rot_mat_i)
        xyz_hand_gt_canon = np.squeeze(xyz_hand_gt_canon)

        keys.append(file_name[i])
        imgs[file_name[i]] = crop_hand_rsz # -1, 256, 256, 3
        uvd_gt_scaled[file_name[i]] = uvd_hand_gt_rsz.astype('float32') # -1, 21, 3
        xyz_gt_save[file_name[i]] = xyz_gt[i].astype('float32') # -1, 21, 3
        xyz_gt_canon[file_name[i]] = xyz_hand_gt_canon.astype('float32') # -1, 21, 3
        rot_mat[file_name[i]] = rot_mat_i.astype('float32') # -1, 3, 3
        keypoint_scale[file_name[i]] = root_bone_length.astype('float32') # -1, 1

    write_to_lmdb(keys, imgs, os.path.join(DATA_DIR, save_prefix + '_256_img.lmdb'), logger)
    write_to_lmdb(keys, uvd_gt_scaled, os.path.join(DATA_DIR, save_prefix + '_uvd_gt_scaled.lmdb'), logger)
    write_to_lmdb(keys, xyz_gt_save, os.path.join(DATA_DIR, save_prefix + '_xyz_gt.lmdb'), logger)
    write_to_lmdb(keys, xyz_gt_canon, os.path.join(DATA_DIR, save_prefix + '_xyz_gt_canon.lmdb'), logger)
    write_to_lmdb(keys, rot_mat, os.path.join(DATA_DIR, save_prefix + '_rot_mat.lmdb'), logger)
    write_to_lmdb(keys, keypoint_scale, os.path.join(DATA_DIR, save_prefix + '_keypoint_scale.lmdb'), logger)

    keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
    pickle.dump(keys, open(keys_cache_file, "wb"))
    logger.info('Finish creating lmdb keys cache.')

def write_img_h5py(file_name, xyz_gt, save_prefix, logger=None):
    # xyz_gt: (-1, 63)
    if logger:
        logger.info('PROCESSING IMG H5PY: %s' %save_prefix)

    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyzuvd.xyz2uvd_color(xyz_gt)

    rsz_dim = 256

    imgs = []
    for i in tqdm(range(uvd_gt.shape[0])):
        color = Image.open(os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files', file_name[i]))
        color = np.asarray(color, dtype='uint32')

        uvd_hand_gt = uvd_gt[i].copy()

        pad = 50
        x_max = int(np.amax(uvd_hand_gt[:,0])) + pad
        x_min = np.maximum(int(np.amin(uvd_hand_gt[:,0])) - pad, 0)
        y_max = int(np.amax(uvd_hand_gt[:,1])) + pad
        y_min = np.maximum(int(np.amin(uvd_hand_gt[:,1])) - pad, 0)

        crop_hand = color[y_min:y_max, x_min:x_max, :]

        if crop_hand.size == 0:
            if logger:
                logger.info('NO HAND, SKIPPING FRAME: %i %s' %(i, file_name[i]))
            continue

        uvd_hand_gt[:, 0] = uvd_hand_gt[:, 0] - x_min
        uvd_hand_gt[:, 1] = uvd_hand_gt[:, 1] - y_min

        crop_hand_rsz = resize(crop_hand, (rsz_dim, rsz_dim), order=3, preserve_range=True).astype('uint32')

        img.append(crop_hand_rsz)

    save_dir = os.path.join(DATA_DIR, save_prefix + '_img.h5py')
    f = h5py.File(save_dir, 'w')
    f.create_dataset('img', data=img)
    f.close()

def move_channel_dim_3_to_1(img):
    '''
    input: (batch_size, h, w, channel) or (batch_size, h, w)
    output: (batch_size, channel, h, w)
    '''
    if len(img.shape) == 3:
        img = np.expand_dims(img, -1)
    return np.reshape(img, (img.shape[0], img.shape[3], img.shape[1], img.shape[2]))

def move_channel_dim_2_to_0(img):
    return np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

def detect_keypoints_from_scoremap(smap):
    # input: [H W C] 32x32
    scoremaps = resize(smap, (256, 256), order=3, preserve_range=True)
    s = scoremaps.shape
    keypoint_coords = np.zeros((s[2], 2))
    for i in range(s[2]):
        v, u = np.unravel_index(np.argmax(scoremaps[:, :, i]), (s[0], s[1]))
        keypoint_coords[i, 0] = v
        keypoint_coords[i, 1] = u
    out_keypoint_coords = np.empty_like(keypoint_coords)
    out_keypoint_coords[:, 0] = keypoint_coords[:, 1]
    out_keypoint_coords[:, 1] = keypoint_coords[:, 0]

    return out_keypoint_coords

def detect_keypoints_from_scoremap_forloop(scoremaps, exp, epoch, data_split):
    # xyz_gt: (-1, 63)

    kpts = []
    def mycallback(arg):
        '''mp callback,
        get the image data and update pbar'''
        keypoint = np.reshape(arg, (-1))
        kpts.append(keypoint)

    for smap in tqdm(scoremaps):
        arg = detect_keypoints_from_scoremap(smap)
        mycallback(arg)

    WRITE_DIR = os.path.join(DATA_DIR, exp, 'predict_%s_%s.txt' %(epoch, data_split))
    print('WRITING TO predict_%s_%s.txt' %(epoch, data_split))
    with open(WRITE_DIR, "w") as f:
        for pred in kpts:
            for jnt  in np.squeeze(pred):
                f.write(str(jnt) + ' ')
            f.write('\n')

def detect_keypoints_from_scoremap_mp(scoremap, exp, epoch, data_split):
    n_thread = 12

    kpts = []
    pbar = ProgressBar(len(scoremap))
    def mycallback(arg):
        '''mp callback,
        get the image data and update pbar'''
        keypoint = np.reshape(arg, (-1))
        kpts.append(keypoint)
        pbar.update('Reading {}'.format(len(kpts)))

    pool = Pool(n_thread)
    for smap in scoremap:
        print(len(kpts))
        pool.apply_async(detect_keypoints_from_scoremap, args=smap, callback=mycallback)
    pool.close()
    pool.join()

def uvd_unscale(uvd_scale, uvd_unscale_param):
    Rx, Ry, x_min, y_min = uvd_unscale_param

    uvd_unscale = uvd_scale.copy()

    uvd_unscale[:, 0] = uvd_unscale[:, 0]/Rx
    uvd_unscale[:, 1] = uvd_unscale[:, 1]/Ry

    uvd_unscale[:, 0] = uvd_unscale[:, 0] + x_min
    uvd_unscale[:, 1] = uvd_unscale[:, 1] + y_min

    return uvd_unscale

def _read_predict_kpt(pred_file):
    pred = []
    with open(pred_file, "r") as f:
        pred_line = f.readlines()
        for lines in pred_line:
            pred.append([float(i) for i in lines.strip().split()])
    return pred

def get_pred_uvd(pred_file, save_prefix):
    pred_uvd_scaled = np.reshape(_read_predict_kpt(pred_file), (-1, 21, 2))
    pred_uvd = uvd_unscale_all(pred_uvd_scaled, save_prefix)
    return pred_uvd

def get_pred_xyz_canon(pred_file, save_prefix):
    pred_xyz_canon = np.reshape(_read_predict_kpt(pred_file), (-1, 21, 3))
    return pred_xyz_canon

def get_pred_rot_mat(pred_file, save_prefix):
    pred_rot_mat = np.reshape(_read_predict_kpt(pred_file), (-1, 3, 3))
    return pred_rot_mat

def uvd_unscale_all(uvd_scale, save_prefix):
    dataroot = os.path.join(DATA_DIR, save_prefix + '_uvd_unscale_param.lmdb')
    keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
    keys = pickle.load(open(keys_cache_file, "rb"))
    uvd_unscale_param = read_all_lmdb_from_file(keys, dataroot, np.float32, (4, 1))

    unscaled = []
    for scaled, param in zip(uvd_scale, uvd_unscale_param):
        unscaled.append(uvd_unscale(scaled, param))
    return unscaled
