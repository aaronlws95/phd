import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from PIL import Image
import pickle
import h5py
import cv2
from skimage.transform import resize
from tqdm import tqdm
import argparse

from utils.directory import DATA_DIR, DATASET_DIR
import utils.prepare_data as pd
import utils.error as error
import utils.visualize as visual
import utils.convert_xyz_uvd as xyzuvd

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--exp', type=str, default='')
args = parser.parse_args()

for data_split in ['train', 'test']:
    f= h5py.File(os.path.join(DATA_DIR, args.exp, 'scoremap_%s_%s.h5' %(args.epoch, data_split)), 'r')
    scoremaps = f['scoremap'][...]
    f.close()

    pd.detect_keypoints_from_scoremap_forloop(scoremaps, args.exp, args.epoch, data_split)
