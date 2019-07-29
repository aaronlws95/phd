import os
import sys

import utils.prepare_data as pd
from utils.directory import DATA_DIR, DATASET_DIR
from utils.logger import get_logger
logger = get_logger()

train_pairs, test_pairs = pd.get_fpha_data_list_general('color', DATASET_DIR)

file_name = [i for i,j in train_pairs]
xyz_gt = [j for i,j in train_pairs]
pd.write_data_lmdb_mp_scoremap_32(file_name, xyz_gt, 'train_fpha', logger)

file_name = [i for i,j in test_pairs]
xyz_gt = [j for i,j in test_pairs]
pd.write_data_lmdb_mp_scoremap_32(file_name, xyz_gt, 'test_fpha', logger)

# train_pairs, test_pairs, val_pairs = pd.get_fpha_data_list_general_val('color', DATASET_DIR)

# file_name = [i for i,j in val_pairs]
# xyz_gt = [j for i,j in val_pairs]
# pd.write_data_lmdb_mp(file_name, xyz_gt, 'val_fpha', logger)
