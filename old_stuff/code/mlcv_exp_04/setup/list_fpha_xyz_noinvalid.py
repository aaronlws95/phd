import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT
from src.utils import *

def list_fpha_xyz_noinvalid():
    """ Prepare list for FPHA dataset """
    parent_dir = Path(__file__).absolute().parents[1]

    # Accumulate list
    train_img_list  = []
    test_img_list   = []
    train_xyz_list = []
    test_xyz_list  = []
    split = ['test', 'train']
    for spl in split:
        xyz_gt = np.loadtxt(parent_dir/'data'/'labels'/'fpha_xyz_{}.txt'.format(spl))
        with open(parent_dir/'data'/'labels'/'fpha_img_{}.txt'.format(spl), 'r') as f:
            img_paths = f.read().splitlines()
        with open(parent_dir/'data'/'labels'/'fpha_img_invalid.txt', 'r') as f:
            invalid_paths = f.readlines()
            invalid_paths = [i.rstrip() for i in invalid_paths]

        for i, b  in zip(img_paths, xyz_gt):
            if i not in invalid_paths:
                if spl == 'train':
                    train_img_list.append(i)
                    train_xyz_list.append(b)
                elif spl == 'test':
                    test_img_list.append(i)
                    test_xyz_list.append(b)

    # Save
    np.savetxt(parent_dir/'data'/'labels'/'fpha_xyz_noinvalid_train.txt', train_xyz_list)
    np.savetxt(parent_dir/'data'/'labels'/'fpha_xyz_noinvalid_test.txt', test_xyz_list)
    # with open(parent_dir/'data'/'labels'/'fpha_img_noinvalid_test.txt', 'w') as f:
    #     for i in test_img_list:
    #         f.write("%s\n" %i)
    # with open(parent_dir/'data'/'labels'/'fpha_img_noinvalid_train.txt', 'w') as f:
    #     for i in train_img_list:
    #         f.write("%s\n" %i)

if __name__ == '__main__':
    list_fpha_xyz_noinvalid()