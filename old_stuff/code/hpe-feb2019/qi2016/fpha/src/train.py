import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
import multiprocessing
import argparse

import constants
import utils
from data_generator import DataGenerator_h5py
import multireso_net as mnet
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0)
args = parser.parse_args()

epoch = args.epoch

if epoch != 0:
    model = load_model(os.path.join(constants.CKPT_DIR, 'model-%02d.hdf5' %epoch))
else:
    model = mnet.multireso_net()

data_file_h5py = os.path.join(constants.DATA_DIR, 'train_fpha.h5')
train_generator = DataGenerator_h5py(data_file_h5py, shuffle=True)
data_file_h5py = os.path.join(constants.DATA_DIR, 'test_fpha.h5')
test_generator = DataGenerator_h5py(data_file_h5py, shuffle=True)

checkpoint = ModelCheckpoint(filepath= os.path.join(constants.CKPT_DIR, 'model-{epoch:02d}.hdf5'), monitor='loss', verbose=1, period=5)
tensorboard = TensorBoard(constants.LOG_DIR, batch_size=constants.BATCH_SIZE)
tensorboard.set_model(model)

callbacks = [checkpoint, tensorboard]

model.fit_generator(generator=train_generator,
                    validation_data=test_generator,
                    callbacks=callbacks,
                    epochs=constants.EPOCHS,
                    workers=multiprocessing.cpu_count(),
                    use_multiprocessing=True,
                    initial_epoch=epoch
                    )

