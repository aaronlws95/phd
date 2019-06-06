import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from keras.preprocessing.sequence import pad_sequences
import read_meta as rm
import image_utils as imgut

root_path = '/media/aaron/DATA/ubuntu/egohands_data'
video_path = os.path.join(root_path, '_LABELLED_SAMPLES')
data_path = os.path.join(root_path, 'data')

def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

def convert(image_path_list, out_path, dim):

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    print("Converting: " + out_path)
    with tf.python_io.TFRecordWriter(out_path) as writer:
        for i, path in enumerate(image_path_list):

            print_progress(count=i, total=len(image_path_list)-1)
            path = path.strip()
            frames = rm.get_frames(path)
            img_path = os.path.join(video_path, path)
            for frame in frames:

                target = rm.create_target(path, frame, dim)
                target = imgut.preprocess_target(target, dim, 1)

                input_img = cv2.imread(os.path.join(img_path, 'frame_%04d.jpg' %frame))
                input_img = imgut.preprocess(input_img, dim, 3)

                # # check
                # imgut.info(target, 'target')
                # imgut.info(input_img, 'input_img')
                # imgut.show(imgut.restore(target, dim, 1), 1, 'gray')
                # imgut.show(imgut.restore(input_img, dim, 3), 3)

                feature = {
                            'image': _bytes_feature(tf.compat.as_bytes(input_img.tostring())),
                            'target': _bytes_feature(tf.compat.as_bytes(target.tostring()))
                            }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

def create_tfrecord(data_list, out_name, dim):
    data_list = os.path.join(data_path, data_list)
    tfrecord_out = os.path.join(data_path, '%s.tfrecord' %out_name)
    with open(data_list) as f:
        image_path_list = f.readlines()

        convert(image_path_list=image_path_list,
                out_path=tfrecord_out,
                dim=dim)

dim = (256, 256)
create_tfrecord('data_train.txt', 'train_%d' %dim[0], dim)
create_tfrecord('data_val.txt', 'val_%d' %dim[0], dim)
create_tfrecord('data_test.txt', 'test_%d' %dim[0], dim)
