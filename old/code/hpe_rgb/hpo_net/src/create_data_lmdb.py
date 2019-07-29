import os
import sys

import utils.prepare_data as pd
from utils.directory import DATA_DIR, DATASET_DIR

if __name__ == '__main__':
    # train_pairs, test_pairs, val_pairs = pd.get_fpha_data_list_val(DATASET_DIR)

    # file_name = [i for i,j in val_pairs]
    # xyz_gt = [j for i,j in val_pairs]
    # pd.write_data_lmdb_mp(file_name, xyz_gt, 'val_fpha')

    train_pairs, test_pairs = pd.get_fpha_data_list(DATASET_DIR)

    file_name = [i for i,j in train_pairs]
    xyz_gt = [j for i,j in train_pairs]
    pd.write_data_lmdb_mp(file_name, xyz_gt, 'train_fpha')

    file_name = [i for i,j in test_pairs]
    xyz_gt = [j for i,j in test_pairs]
    pd.write_data_lmdb_mp(file_name, xyz_gt, 'test_fpha')


