import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import multiprocessing
from keras.models import load_model
import argparse

import constants
import utils
from data_generator import DataGenerator_h5py

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', type=str, choices=['train', 'test'], required=True)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--exp', type=str, default='')
args = parser.parse_args()

data_split = args.data_split

if data_split == 'train':
    data_file_h5py = os.path.join(constants.DATA_DIR, 'train_fpha.h5')
else:
    data_file_h5py = os.path.join(constants.DATA_DIR, 'test_fpha.h5')

generator = DataGenerator_h5py(data_file_h5py)

epoch = args.epoch
# load model
model = load_model(os.path.join(constants.DATA_DIR, args.exp, 'ckpt', 'model-%02d.hdf5' %epoch))
# model.summary()

# predict
pred_normuvd = model.predict_generator(generator,
                                       workers=multiprocessing.cpu_count(),
                                       use_multiprocessing=True,
                                       verbose=1)

with open(os.path.join(constants.DATA_DIR, args.exp, 'predict_%s_%s.txt' %(epoch, data_split)), "w") as f:
    for pred in pred_normuvd:
        for jnt in pred:
            f.write(str(jnt) + ' ')
        f.write('\n')

