import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from keras.callbacks import ModelCheckpoint, TensorBoard
import multiprocessing

import constants
import utils
from data_generator import DataGenerator
from hponet import hponet_hpe_yad2k

model = hponet_hpe_yad2k()

train_pairs, test_pairs = utils.get_data_list()
train_generator = DataGenerator(train_pairs)

checkpoint = ModelCheckpoint(filepath=constants.CKPT_DIR, monitor='loss', verbose=1)
tensorboard = TensorBoard(constants.LOG_DIR, batch_size=constants.BATCH_SIZE)
tensorboard.set_model(model)

callbacks = [checkpoint, tensorboard]

model.fit_generator(generator=train_generator,
                    callbacks=callbacks,
                    epochs=constants.EPOCHS,
                    workers=multiprocessing.cpu_count(),
                    use_multiprocessing=True,
                    )
