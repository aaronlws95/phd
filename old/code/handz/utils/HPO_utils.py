import os
import numpy as np
import itertools

from utils.lmdb_utils import *
from utils.eval_utils import *
from utils.dir import dir_dict

DIR = dir_dict["HPO_DIR"]

# ========================================================
# PREDICTION UTILS
# ========================================================

def load_all_pred(exp, epoch, data_split):
    pred_file = os.path.join(DIR, exp, 'predict_%s_%s_best.txt' %(epoch, data_split))
    pred_uvd_best = load_pred(pred_file, (-1, 21, 3))
    
    pred_file = os.path.join(DIR, exp, 'predict_%s_%s_topk.txt' %(epoch, data_split))
    pred_uvd_topk = load_pred(pred_file, (-1, 10, 21, 3))
    
    pred_file = os.path.join(DIR, exp, 'predict_%s_%s_conf.txt' %(epoch, data_split))
    pred_uvd_conf = load_pred(pred_file)
    
    return pred_uvd_best, pred_uvd_topk, pred_uvd_conf

def load_pred(pred_dir, reshape=None):
    pred = np.loadtxt(pred_dir)
    if reshape:
        pred = np.reshape(pred, reshape)
    return pred
    
def write_pred(pred_dir, pred, reshape):
    np.savetxt(pred_dir, np.asarray(np.reshape(pred, reshape)))             