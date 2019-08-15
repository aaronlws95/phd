import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT
from src.utils import *

def list_ho3d_object():
    img_root = Path(ROOT)/'datasets'/'Task3'

    train_corner_list   = []
    train_ccs_list      = []
    val_corner_list     = []
    val_ccs_list        = []
    val_len = 400

    model_path = Path(ROOT)/'datasets'/'Task3'/'object_models'
    corners = {}

    corners['003_cracker_box']      = np.load(model_path/'003_cracker_box'/'corners.npy')
    corners['004_sugar_box']        = np.load(model_path/'004_sugar_box'/'corners.npy')
    corners['006_mustard_bottle']   = np.load(model_path/'006_mustard_bottle'/'corners.npy')
    corners['019_pitcher_base']     = np.load(model_path/'019_pitcher_base'/'corners.npy')

    with open(img_root/'training_object_annotation.txt') as f:
        lines = f.readlines()
        val_index = np.arange(0, len(lines), len(lines)//val_len)
        for i, l in enumerate(lines):
            words = l.split()
            if i in val_index:
                val_corner_list.append(np.reshape([corners[words[1]]], -1))
                val_ccs_list.append([float(x) for x in words[2:]])
            else:
                train_corner_list.append(np.reshape([corners[words[1]]], -1))
                train_ccs_list.append([float(x) for x in words[2:]])

    parent_dir = Path(__file__).absolute().parents[1]
    np.savetxt(parent_dir/'data'/'labels'/'ho3d_obj6D_ccs_train.txt', train_ccs_list)
    np.savetxt(parent_dir/'data'/'labels'/'ho3d_obj6D_ccs_val.txt', val_ccs_list)
    np.savetxt(parent_dir/'data'/'labels'/'ho3d_corner_ccs_train.txt', train_corner_list)
    np.savetxt(parent_dir/'data'/'labels'/'ho3d_corner_ccs_val.txt', val_corner_list)

if __name__ == '__main__':
    list_ho3d_object()