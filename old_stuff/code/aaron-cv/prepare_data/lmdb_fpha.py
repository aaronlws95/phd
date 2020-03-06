import sys
import numpy as np
import argparse
import pickle
from PIL import Image
from pathlib import Path
from multiprocessing import Pool

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from progress_bar import ProgressBar
from src.utils import FPHA, DATA_DIR, LMDB

with open(Path(DATA_DIR)/'First_Person_Action_Benchmark'/'bad_imgs.txt', 'r') as f:
    bad_imgs_list = f.readlines()
bad_imgs_list = [i.rstrip() for i in bad_imgs_list]

def data_worker(file_name, xyz_gt_i):
    if file_name not in bad_imgs_list:
        uvd_gt_i = FPHA.xyz2uvd_color(xyz_gt_i)
        bbox = FPHA.get_bbox(uvd_gt_i).astype('float32')
        return file_name, xyz_gt_i.astype('float32'), bbox
    else:
        return None

def data_worker_all(file_name, xyz_gt_i):
    uvd_gt_i = FPHA.xyz2uvd_color(xyz_gt_i)
    bbox = FPHA.get_bbox(uvd_gt_i).astype('float32')
    return file_name, xyz_gt_i.astype('float32'), bbox

def write_data_lmdb_mp(keys, xyz_gt, save_prefix, 
                       save_dir, worker_fun, n_thread=12):
    print(f"PROCESSING DATA LMDB: {save_prefix}")
    print(f"THREADS: {n_thread}")
    xyz_gt_dict = {}
    bbox_dict = {}
    saved_keys = []
    pbar = ProgressBar(len(xyz_gt))
    def my_callback(arg):
        if arg:
            key = arg[0]
            xyz_gt_dict[key] = arg[1]
            bbox_dict[key] = arg[2]
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

    LMDB.write_to_lmdb(saved_keys, 
                       xyz_gt_dict, 
                       str(Path(DATA_DIR)/save_dir/(save_prefix + "_xyz_gt.lmdb")))

    LMDB.write_to_lmdb(saved_keys, 
                       bbox_dict, 
                       str(Path(DATA_DIR)/save_dir/(save_prefix + "_bbox_gt.lmdb")))

    keys_cache_file = Path(DATA_DIR)/save_dir/(save_prefix + "_keys_cache.p")
    pickle.dump(saved_keys, open(keys_cache_file, "wb"))
    print("Finish creating lmdb keys cache.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True)
    p.add_argument("--save_prefix", type=str, required=True)
    args = p.parse_args()

    train_file_name, test_file_name, train_xyz_gt, test_xyz_gt \
     = FPHA.get_train_test_pairs('color', 
                                 Path(DATA_DIR)/'First_Person_Action_Benchmark')
    
    # Commented to ensure no accidents

    # write_data_lmdb_mp(train_file_name, train_xyz_gt, 
    #                    'train_{}'.format(args.save_prefix), 
    #                    args.dir, data_worker)
    # write_data_lmdb_mp(test_file_name, test_xyz_gt, 
    #                    'test_{}'.format(args.save_prefix), 
    #                    args.dir, data_worker)
    
    # write_data_lmdb_mp(train_file_name, train_xyz_gt, 
    #                    'train_{}'.format(args.save_prefix), 
    #                    args.dir, data_worker_all)
    # write_data_lmdb_mp(test_file_name, test_xyz_gt, 
    #                    'test_{}'.format(args.save_prefix), 
    #                    args.dir, data_worker_all)