import os
import sys

import utils.prepare_data as pd
from utils.directory import DATA_DIR, DATASET_DIR
from utils.logger import get_logger
logger = get_logger()

dataset_dir = os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark')
save_dir = DATA_DIR
train_pairs, test_pairs = pd.get_fpha_data_list('color', dataset_dir)

file_name = [i for i,j in train_pairs]
xyz_gt = [j for i,j in train_pairs]
pd.write_data_no_crop_h5py(file_name, xyz_gt, os.path.join(save_dir, 'train_fpha_RGB_no_crop.h5'), logger)

file_name = [i for i,j in test_pairs]
xyz_gt = [j for i,j in test_pairs]
pd.write_data_no_crop_h5py(file_name, xyz_gt, os.path.join(save_dir, 'test_fpha_RGB_no_crop.h5'), logger)

