import os
import sys

import utils.prepare_data as pd
from utils.directory import DATA_DIR, DATASET_DIR
from utils.logger import get_logger
logger = get_logger()

train_pairs, test_pairs = pd.get_fpha_data_list_general('color', DATASET_DIR)

print('train')
file_name = [i for i,j in train_pairs]
xyz_gt = [j for i,j in train_pairs]
pd.check_no_crop_images(file_name, xyz_gt, logger)

print('test')
file_name = [i for i,j in test_pairs]
xyz_gt = [j for i,j in train_pairs]
pd.check_no_crop_images(file_name, xyz_gt, logger)
