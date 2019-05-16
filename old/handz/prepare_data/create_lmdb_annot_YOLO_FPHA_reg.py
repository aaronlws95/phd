import os
import sys
from PIL import Image
import numpy as np
from multiprocessing import Pool

sys.path.append(os.path.dirname(os.path.abspath("")))
from utils.lmdb_utils import  *
import utils.FPHA_utils as FPHA
import utils.YOLO_utils as YOLO
from progress_bar import ProgressBar

with open(os.path.join(FPHA.DIR, 'bad_imgs.txt'), 'r') as f:
    bad_imgs_list = f.readlines()
bad_imgs_list = [i.rstrip() for i in bad_imgs_list]

def data_worker_rootinimg(file_name, xyz_gt_i):
    uvd_gt_i = FPHA.xyz2uvd_color(xyz_gt_i)
    if FPHA.is_root_in_img(uvd_gt_i) and (file_name not in bad_imgs_list):
        return file_name, xyz_gt_i.astype('float32')
    else:
        return None

def data_worker(file_name, xyz_gt_i):
    img = Image.open(os.path.join(FPHA.DIR, 'Video_files', file_name))
    img = np.asarray(img, dtype='uint32')
    uvd_gt_i = FPHA.xyz2uvd_color(xyz_gt_i)
    return file_name, xyz_gt_i.astype('float32')

def write_data_lmdb_mp(keys, xyz_gt, save_prefix, save_dir, worker_fun, n_thread=12):
    print(f"PROCESSING DATA LMDB: {save_prefix}")
    print(f"THREADS: {n_thread}")
    xyz_gt_dict = {}
    saved_keys = []
    pbar = ProgressBar(len(xyz_gt))
    def my_callback(arg):
        if arg:
            key = arg[0]
            xyz_gt_dict[key] = arg[1]
            saved_keys.append(key)
            pbar.update("Reading {}".format(key))
        else:
            pbar.update("Not reading.")

    pool = Pool(n_thread)
    for path, xyz in zip(keys, xyz_gt):
        pool.apply_async(worker_fun, args=(path, xyz), callback=my_callback)
    pool.close()
    pool.join()
    print("Finish reading {} images.\nWrite lmdb...".format(len(xyz_gt)))

    write_to_lmdb(saved_keys, xyz_gt_dict, os.path.join(save_dir, save_prefix + "_xyz_gt.lmdb"))

    keys_cache_file = os.path.join(save_dir, save_prefix + "_keys_cache.p")
    pickle.dump(saved_keys, open(keys_cache_file, "wb"))
    print("Finish creating lmdb keys cache.")
    
if __name__ == "__main__":       
    train_file_name, test_file_name, train_xyz_gt, test_xyz_gt \
     = FPHA.get_train_test_pairs('color', FPHA.DIR)   
     
    write_data_lmdb_mp(train_file_name, train_xyz_gt, 'train_fpha_root', YOLO.FPHA_DIR, data_worker_rootinimg)
    write_data_lmdb_mp(test_file_name, test_xyz_gt, 'test_fpha_root', YOLO.FPHA_DIR, data_worker_rootinimg)