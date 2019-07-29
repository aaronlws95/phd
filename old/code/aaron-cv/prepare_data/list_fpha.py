import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src.utils import FPHA, DATA_DIR

with open(Path(DATA_DIR)/'First_Person_Action_Benchmark'/'bad_imgs_with_outbounds.txt', 'r') as f:
    bad_imgs_list = f.readlines()
bad_imgs_list = [i.rstrip() for i in bad_imgs_list]

def write_data(file_name, xyz_gt, split):
    img_f = open(Path(DATA_DIR)/'First_Person_Action_Benchmark'/'{}_fpha_img.txt'.format(split), 'w')

    xyz_save = []
    for file, xyz in tqdm(zip(file_name, xyz_gt)):
        if file not in bad_imgs_list:
            img_f.write(file + '\n')
            xyz_save.append(xyz)
    np.savetxt(Path(DATA_DIR)/'First_Person_Action_Benchmark'/'{}_fpha_xyz.txt'.format(split),
               np.reshape(xyz_save, (-1, 63)))
    
if __name__ == "__main__":
    train_file_name, test_file_name, train_xyz_gt, test_xyz_gt \
     = FPHA.get_train_test_pairs('color', 
                                 Path(DATA_DIR)/'First_Person_Action_Benchmark')
    write_data(train_file_name, train_xyz_gt, 'train')
    write_data(test_file_name, test_xyz_gt, 'test')
