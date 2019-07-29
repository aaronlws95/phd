import os
import sys
from PIL import Image
import numpy as np
import lmdb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath("")))
from utils.lmdb_utils import  *
import utils.FPHA_utils as FPHA

def write_lmdb_img(env, file_name):
    """
    Write img to lmdb one by one
    """
    print(f"WRITING LMDB IMG")
    div = 5000
    file_batch_num = len(file_name)//div
    for i in range(file_batch_num):
        file_name_i = file_name[i*div:(i+1)*div]
        with env.begin(write=True) as txn:
            # txn is a Transaction object
            for key in tqdm(file_name_i):
                key_byte = key.encode("ascii")
                img = np.asarray(Image.open(os.path.join(FPHA.DIR, 'Video_files', key)))
                txn.put(key_byte, img)

    remainder_batch = len(file_name)-div*file_batch_num
    file_name_i = file_name[-remainder_batch:]    
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for key in tqdm(file_name_i):
            key_byte = key.encode("ascii")
            img = np.asarray(Image.open(os.path.join(FPHA.DIR, 'Video_files', key)))
            txn.put(key_byte, img)
    
    print("FINISH WRITING LMBD")

if __name__ == "__main__":   
    print("DO YOU HAVE ENOUGH RAM/SPACE?")
    # train_file_name, test_file_name, train_xyz_gt, test_xyz_gt \
    #     = FPHA.get_train_test_pairs('color', FPHA.DIR)
        
    # file_name = [j for i in zip(train_file_name,test_file_name) for j in i]
    # data_eg = np.asarray(Image.open(os.path.join(FPHA.DIR, 'Video_files', file_name[0])))
    # data_size_per_data = data_eg.nbytes
    # data_size = data_size_per_data*len(file_name)
    # dataroot = os.path.join(FPHA.DIR, 'fpha_img.lmdb')
    # env = lmdb.open(dataroot, map_size=int(data_size*10))
    # write_lmdb_img(env, file_name)
    # env = get_env(dataroot)
    # print(env.stat())