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
import itertools
import math

import utils.xyzuvd as xyzuvd
import utils.cam as cam
from utils.directory import DATASET_DIR, DATA_DIR

def get_hand_cell_idx(uvd_gt, pad=10):
    x_max = int(np.amax(uvd_gt[:, 0])) + pad
    x_min = np.maximum(int(np.amin(uvd_gt[:, 0])) - pad, 0)
    y_max = int(np.amax(uvd_gt[:, 1])) + pad
    y_min = np.maximum(int(np.amin(uvd_gt[:, 1])) - pad, 0)
    z_max = int(np.amax(uvd_gt[:, 2])) + pad
    z_min = np.maximum(int(np.amin(uvd_gt[:, 2])) - pad, 0)

    x_min_scale = x_min//32
    x_max_scale = np.ceil(x_max/32)
    y_min_scale = y_min//32
    y_max_scale = np.ceil(y_max/32)
    z_min_scale = z_min//200
    z_max_scale = np.ceil(z_max/200)

    if z_max_scale > 5:
        z_max_scale = 5
    if y_max_scale > 13:
        y_max_scale = 13
    if x_max_scale > 13:
        x_max_scale = 13

    comb = [list(i) for i in itertools.product(np.arange(x_min_scale, x_max_scale), \
                                                  np.arange(y_min_scale, y_max_scale), \
                                                  np.arange(z_min_scale, z_max_scale))]
    comb = np.asarray(comb, dtype=np.uint8)
    ravel_comb = []
    for c in comb:
        ravel_comb.append(np.ravel_multi_index(c, (13,13,5)))

    hand_cell_idx = np.zeros(845)
    hand_cell_idx[ravel_comb] = 1
    hand_cell_idx = hand_cell_idx.astype('uint8')
    return hand_cell_idx

def sk_resize(data, size):
    return resize(data, size, order=3, preserve_range=True)

def read_predict(pred_file):
    pred = []
    with open(pred_file, "r") as f:
        pred_line = f.readlines()
        for lines in pred_line:
            pred.append([float(i) for i in lines.strip().split()])
    return pred

def get_fpha_data_list(dataset_dir, modality='color'):
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

def get_fpha_data_list_val(dataset_dir, modality='color'):
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

def reading_data_worker(file_name, xyz_gt_i, uvd_gt_i):
    img = Image.open(os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files', file_name))
    img = np.asarray(img, dtype='uint32')
    img_resize = resize(img, (416, 416), order=3, preserve_range=True).astype('uint32')
    uvd_gt_i_resize = uvd_gt_i.copy()
    uvd_gt_i_resize[:, 0] *= 416/1920
    uvd_gt_i_resize[:, 1] *= 416/1080

    pad = 10
    x_max = int(np.amax(uvd_gt_i_resize[:, 0])) + pad
    x_min = np.maximum(int(np.amin(uvd_gt_i_resize[:, 0])) - pad, 0)
    y_max = int(np.amax(uvd_gt_i_resize[:, 1])) + pad
    y_min = np.maximum(int(np.amin(uvd_gt_i_resize[:, 1])) - pad, 0)
    z_max = int(np.amax(uvd_gt_i_resize[:, 2])) + pad
    z_min = np.maximum(int(np.amin(uvd_gt_i_resize[:, 2])) - pad, 0)

    x_min_scale = x_min//32
    x_max_scale = math.ceil(x_max/32)
    y_min_scale = y_min//32
    y_max_scale = math.ceil(y_max/32)
    z_min_scale = z_min//120
    z_max_scale = math.ceil(z_max/120)

    if z_max_scale >= 5:
        z_max_scale = 4
    if y_max_scale >= 13:
        y_max_scale = 12
    if x_max_scale >= 13:
        x_max_scale = 12

    comb_3d = [list(i) for i in itertools.product(np.arange(x_min_scale, x_max_scale), \
                                        np.arange(y_min_scale, y_max_scale), \
                                        np.arange(z_min_scale, z_max_scale))]

    ravel_comb = []
    for comb in comb_3d:
        ravel_comb.append(np.ravel_multi_index(comb, (13,13,5)))

    hand_cell_i = np.zeros(845)
    hand_cell_i[ravel_comb] = 1
    hand_cell_i = hand_cell_i.astype('bool')

    return file_name, xyz_gt_i.astype('float32'), uvd_gt_i_resize.astype('float32'), hand_cell_i

def reading_data_worker_normuvd(file_name, xyz_gt_i, uvd_gt_i, bbox_uvd, ref_z, bbsize, center_joint_idx):

    uvd_hand_gt = uvd_gt_i.copy()
    xyz_hand_gt = xyz_gt_i.copy()

    #mean calculation
    mean_z = xyz_hand_gt[center_joint_idx, 2]
    xyz_hand_gt[:,2] += ref_z - mean_z
    uvd_hand_gt = xyzuvd.xyz2uvd_color(xyz_hand_gt)
    mean_u = uvd_hand_gt[center_joint_idx, 0]
    mean_v = uvd_hand_gt[center_joint_idx, 1]

    #U
    uvd_hand_gt[:,0] = (uvd_hand_gt[:,0] - mean_u + bbox_uvd[0,0]/2 ) / bbox_uvd[0,0]
    uvd_hand_gt[np.where(uvd_hand_gt[:,0]>1),0]=1
    uvd_hand_gt[np.where(uvd_hand_gt[:,0]<0),0]=0

    #V
    uvd_hand_gt[:,1] =(uvd_hand_gt[:,1] - mean_v+bbox_uvd[0,1]/2) / bbox_uvd[0,1]
    uvd_hand_gt[ np.where(uvd_hand_gt[:,1]>1),1]=1
    uvd_hand_gt[ np.where(uvd_hand_gt[:,1]<0),1]=0

    # Z
    uvd_hand_gt[:,2] = (uvd_hand_gt[:,2] - ref_z + bbsize/2)/bbsize

    return file_name, uvd_hand_gt.astype('float32')

def write_data_lmdb_mp_normuvd(keys, xyz_gt, save_prefix):
    # xyz_gt: (-1, 63)
    print('PROCESSING DATA LMDB MULTIPROCESS NORM: %s' %save_prefix)
    keys = keys
    n_thread = 12
    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyzuvd.xyz2uvd_color(xyz_gt)
    uvd_norm_gt_dict = {}

    center_joint_idx = 3
    ref_z = 1000
    bbsize = 260
    u0 = cam.X0_COLOR
    v0 = cam.Y0_COLOR

    _, bbox_uvd = get_bbox(bbsize, ref_z, u0, v0)

    pbar = ProgressBar(uvd_gt.shape[0])
    def mycallback(arg):
        key = arg[0]
        uvd_norm_gt_dict[key] = arg[1]
        pbar.update('Reading {}'.format(key))

    pool = Pool(n_thread)
    for path, xyz, uvd in zip(keys, xyz_gt, uvd_gt):
        pool.apply_async(reading_data_worker_normuvd, args=(path, xyz, uvd, bbox_uvd, ref_z, bbsize, center_joint_idx), callback=mycallback)
    pool.close()
    pool.join()
    print('Finish reading {} images.\nWrite lmdb...'.format(uvd_gt.shape[0]))

    write_to_lmdb(keys, uvd_norm_gt_dict, os.path.join(DATA_DIR, save_prefix + '_uvd_norm_gt.lmdb'))

    # keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
    # pickle.dump(keys, open(keys_cache_file, "wb"))
    # print('Finish creating lmdb keys cache.')

def write_data_lmdb_mp(keys, xyz_gt, save_prefix):
    # xyz_gt: (-1, 63)
    print('PROCESSING DATA LMDB MULTIPROCESS: %s' %save_prefix)
    keys = keys
    n_thread = 12
    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyzuvd.xyz2uvd_color(xyz_gt)
    xyz_gt_dict = {}
    uvd_gt_dict = {}
    hand_cell_idx = {}

    pbar = ProgressBar(uvd_gt.shape[0])
    def mycallback(arg):
        key = arg[0]
        xyz_gt_dict[key] = arg[1]
        uvd_gt_dict[key] = arg[2]
        hand_cell_idx[key] = arg[3]
        pbar.update('Reading {}'.format(key))

    pool = Pool(n_thread)
    for path, xyz, uvd in zip(keys, xyz_gt, uvd_gt):
        pool.apply_async(reading_data_worker, args=(path, xyz, uvd), callback=mycallback)
    pool.close()
    pool.join()
    print('Finish reading {} images.\nWrite lmdb...'.format(uvd_gt.shape[0]))

    write_to_lmdb(keys, xyz_gt_dict, os.path.join(DATA_DIR, save_prefix + '_xyz_gt.lmdb'))
    write_to_lmdb(keys, uvd_gt_dict, os.path.join(DATA_DIR, save_prefix + '_uvd_gt_resize.lmdb'))
    write_to_lmdb(keys, hand_cell_idx, os.path.join(DATA_DIR, save_prefix + '_hand_cell_idx.lmdb'))

    keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
    pickle.dump(keys, open(keys_cache_file, "wb"))
    print('Finish creating lmdb keys cache.')

def write_data_lmdb(keys, xyz_gt, save_prefix):
    # xyz_gt: (-1, 63)
    print('PROCESSING DATA LMDB: %s' %save_prefix)
    xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
    uvd_gt = xyzuvd.xyz2uvd_color(xyz_gt)
    ccs_gt = xyzuvd.xyz2ccs_color(xyz_gt)
    xyz_gt_dict = {}
    uvd_gt_dict = {}
    ccs_gt_dict = {}
    hand_cell_idx = {}

    pbar = ProgressBar(uvd_gt.shape[0])
    def mycallback(arg):
        key = arg[0]
        xyz_gt_dict[key] = arg[1]
        uvd_gt_dict[key] = arg[2]
        ccs_gt_dict[key] = arg[3]
        hand_cell_idx[key] = arg[4]
        pbar.update('Reading {}'.format(key))

    for path, xyz, uvd, ccs in zip(keys, xyz_gt, uvd_gt, ccs_gt):
        arg = reading_data_worker(path, xyz, uvd, ccs)
        mycallback(arg)

    write_to_lmdb(keys, xyz_gt_dict, os.path.join(DATA_DIR, save_prefix + '_xyz_gt.lmdb'))
    write_to_lmdb(keys, uvd_gt_dict, os.path.join(DATA_DIR, save_prefix + '_uvd_gt.lmdb'))
    write_to_lmdb(keys, ccs_gt_dict, os.path.join(DATA_DIR, save_prefix + '_ccs_gt.lmdb'))
    write_to_lmdb(keys, hand_cell_idx, os.path.join(DATA_DIR, save_prefix + '_hand_cell_idx.lmdb'))

    keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
    pickle.dump(keys, open(keys_cache_file, "wb"))
    print('Finish creating lmdb keys cache.')

def write_to_lmdb(keys, dataset, save_dir):
    data_size_per_data = dataset[keys[0]].nbytes
    print('data size per image is: %i' %data_size_per_data)
    data_size = data_size_per_data * len(dataset)
    env = lmdb.open(save_dir, map_size=data_size * 10)

    print('WRITING DATA: %s' %save_dir)

    with env.begin(write=True) as txn:  # txn is a Transaction object
        for key in tqdm(keys):
            key_byte = key.encode('ascii')
            data = dataset[key]
            txn.put(key_byte, data)
    print('FINISH WRITING LMBD')

def read_lmdb(key, env, dtype, reshape_dim):
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
        data_flat = np.frombuffer(buf, dtype=dtype)
        data = data_flat.reshape(reshape_dim)
    return data

def read_all_lmdb_from_name(keys, save_prefix, name, dtype, reshape_dim):
    dataroot = os.path.join(DATA_DIR, save_prefix + '_' + name + '.lmdb')
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    all_data = []
    with env.begin(write=False) as txn:
        for key in keys:
            buf = txn.get(key.encode('ascii'))
            data_flat = np.frombuffer(buf, dtype=dtype)
            data = data_flat.reshape(reshape_dim)
            all_data.append(data)
    return all_data

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

def get_keys(save_prefix):
    keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
    keys = pickle.load(open(keys_cache_file, "rb"))
    return keys

def get_bbox(bbsize, ref_z, u0, v0):
    bbox_xyz = np.array([(bbsize,bbsize,ref_z)])
    bbox_uvd = xyzuvd.xyz2uvd_color(bbox_xyz)
    bbox_uvd[0,0] = np.ceil(bbox_uvd[0,0] - u0)
    bbox_uvd[0,1] = np.ceil(bbox_uvd[0,1] - v0)
    return bbox_xyz, bbox_uvd

def get_img(key):
    img = Image.open(os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark', 'Video_files', key))
    img = np.asarray(img, dtype='uint32')
    return img

def scale_annot_wh(annot, i_shape, o_shape):
    new_annot = annot.copy()
    new_annot[..., 0] *= o_shape[0]/i_shape[0]
    new_annot[..., 1] *= o_shape[1]/i_shape[1]

    return new_annot

def resize_img(img, new_size):
    img_resize = resize(img, new_size, order=3, preserve_range=True).astype('uint32')
    return img_resize
