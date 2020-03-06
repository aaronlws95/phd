import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT
from src.utils import *

def list_ho3d_hand():
    img_root = Path(ROOT)/'datasets'/'Task3'

    train_img_list  = []
    train_ccs_list  = []
    val_img_list    = []
    val_ccs_list    = []
    val_len = 400
    with open(img_root/'training_joint_annotation.txt') as f:
        lines = f.readlines()
        val_index = np.arange(0, len(lines), len(lines)//val_len)
        for i, l in enumerate(lines):
            words = l.split()
            if i in val_index:
                val_img_list.append(words[0])
                val_ccs_list.append([float(x) for x in words[1:]])
            else:
                train_img_list.append(words[0])
                train_ccs_list.append([float(x) for x in words[1:]])

    parent_dir = Path(__file__).absolute().parents[1]
    np.savetxt(parent_dir/'data'/'labels'/'ho3d_ccs_train.txt', train_ccs_list)
    np.savetxt(parent_dir/'data'/'labels'/'ho3d_ccs_val.txt', val_ccs_list)
    with open(parent_dir/'data'/'labels'/'ho3d_img_train.txt', 'w') as f:
        for i in train_img_list:
            f.write("%s\n" %i)
    with open(parent_dir/'data'/'labels'/'ho3d_img_val.txt', 'w') as f:
        for i in val_img_list:
            f.write("%s\n" %i)

if __name__ == '__main__':
    list_ho3d_hand()